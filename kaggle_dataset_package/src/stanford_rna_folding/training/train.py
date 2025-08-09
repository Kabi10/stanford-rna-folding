"""
Training module for RNA folding models.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# Add torch.cuda.amp for mixed precision training
from torch.cuda.amp import autocast, GradScaler

from ..data.data_processing import StanfordRNADataset, rna_collate_fn
from ..data.transforms import RNADataTransform
from ..models.rna_folding_model import RNAFoldingModel
from ..evaluation.metrics import batch_rmsd, batch_tm_score


def train_model(
    config: Dict,
    data_dir: Union[str, Path],
    save_dir: Union[str, Path],
    use_wandb: bool = True,
    device: Optional[str] = None,
):
    """
    Train an RNA folding model according to the given configuration.
    
    Args:
        config: Configuration dictionary
        data_dir: Directory containing the data
        save_dir: Directory to save model checkpoints
        use_wandb: Whether to log to Weights & Biases
        device: Device to train on (if None, automatically chosen)
        
    Returns:
        Tuple of (trained model, best validation RMSD, best validation TM-score)
    """
    # Set random seed for reproducibility
    if 'seed' in config:
        pl.seed_everything(config['seed'])
    
    # Determine device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Create directories if they don't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize WandB
    if use_wandb:
        wandb.init(
            project=config.get('project', 'stanford-rna-3d-folding'),
            name=config.get('run_name', None),
            config=config
        )
    
    # Create data directory path
    data_dir = Path(data_dir)
    
    # Initialize transforms
    # Use augmentation for training
    train_transform = RNADataTransform(
        normalize_coords=config.get('normalize', True),
        random_rotation=config.get('rotation', True),
        random_noise=config.get('noise', 0.05),
        jitter_strength=config.get('jitter', 0.1),
        atom_mask_prob=config.get('random_mask', 0.1),
    )
    # Mark training mode for transform augmentations
    train_transform.training = True

    # Initialize validation transforms (no augmentation)
    val_transform = RNADataTransform(
        normalize_coords=config.get('normalize', True),
        random_rotation=False,
        random_noise=0.0,
        jitter_strength=0.0,
        atom_mask_prob=0.0,
    )
    val_transform.training = False
    
    # Create datasets
    train_dataset = StanfordRNADataset(
        data_dir=str(data_dir),
        split="train",
        transform=train_transform,
    )

    val_dataset = StanfordRNADataset(
        data_dir=str(data_dir),
        split="validation",
        transform=val_transform,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        collate_fn=rna_collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        collate_fn=rna_collate_fn,
        pin_memory=True,
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create model
    model = RNAFoldingModel(
        vocab_size=5,
        embedding_dim=config.get('embedding_dim', 256),
        hidden_dim=config.get('hidden_dim', 512),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 8),
        dropout=config.get('dropout', 0.1),
        num_atoms=config.get('num_atoms', 1),
        multi_atom_mode=config.get('multi_atom_mode', False),
        coord_dims=config.get('coord_dims', 3),
        max_seq_len=config.get('max_seq_len', 1200),
        use_rna_constraints=config.get('use_rna_constraints', True),
        bond_length_weight=config.get('bond_length_weight', 0.3),
        bond_angle_weight=config.get('bond_angle_weight', 0.3),
        steric_clash_weight=config.get('steric_clash_weight', 0.5),
        watson_crick_weight=config.get('watson_crick_weight', 0.2),
        normalize_coords=config.get('model_normalize_coords', False),
        use_relative_attention=config.get('use_relative_attention', True),
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4),
    )
    
    # Learning rate scheduler - Enhanced with more options
    if config.get('scheduler_type', 'reduce_on_plateau') == 'cosine':
        # Cosine annealing scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('t_max', config['epochs']),
            eta_min=config.get('min_lr', 1e-7)
        )
    elif config.get('scheduler_type', 'reduce_on_plateau') == 'one_cycle':
        # One cycle learning rate
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['learning_rate'],
            total_steps=config['epochs'] * len(train_loader) // config.get('gradient_accumulation_steps', 1),
            pct_start=config.get('pct_start', 0.3),
            div_factor=config.get('div_factor', 25.0),
            final_div_factor=config.get('final_div_factor', 10000.0)
        )
    else:
        # Default: ReduceLROnPlateau
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('lr_factor', 0.5),
            patience=config.get('lr_patience', 5),
            min_lr=config.get('min_lr', 1e-7),
        )
    
    # Initialize best validation metrics
    best_val_rmsd = float('inf')
    best_val_tm_score = 0.0
    best_epoch = 0
    no_improve_count = 0
    
    # Set up checkpoint manager
    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Training loop
    for epoch in range(1, config['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        
        # Train
        train_losses = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            bond_length_weight=config.get('bond_length_weight', 0.2),
            bond_angle_weight=config.get('bond_angle_weight', 0.2),
            steric_clash_weight=config.get('steric_clash_weight', 0.3),
            use_mixed_precision=config.get('use_mixed_precision', True),
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
            gradient_clip_val=config.get('gradient_clip_val', None),
        )
        
        # Validate
        val_losses, val_rmsd, val_tm_score = validate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            bond_length_weight=config.get('bond_length_weight', 0.2),
            bond_angle_weight=config.get('bond_angle_weight', 0.2),
            steric_clash_weight=config.get('steric_clash_weight', 0.3),
            use_mixed_precision=config.get('use_mixed_precision', True),
        )
        
        # Log to WandB with enhanced metrics
        if use_wandb:
            log_dict = {
                'train/loss': train_losses['loss'],
                'train/mse_loss': train_losses['mse_loss'],
                'train/bond_length_loss': train_losses['bond_length_loss'],
                'train/bond_angle_loss': train_losses['bond_angle_loss'],
                'train/steric_clash_loss': train_losses['steric_clash_loss'],
                'val/loss': val_losses['loss'],
                'val/mse_loss': val_losses['mse_loss'],
                'val/bond_length_loss': val_losses['bond_length_loss'],
                'val/bond_angle_loss': val_losses['bond_angle_loss'],
                'val/steric_clash_loss': val_losses['steric_clash_loss'],
                'val/rmsd': val_rmsd,
                'val/tm_score': val_tm_score,
                'lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                # Add more detailed training stats
                'train/grad_norm': train_losses.get('grad_norm', 0.0),
                'train/batch_per_sec': train_losses.get('batches_per_second', 0.0),
                'train/samples_per_sec': train_losses.get('samples_per_second', 0.0),
            }
            wandb.log(log_dict)
        
        # Print metrics
        train_loss = train_losses['loss']
        val_loss = val_losses['loss']
        print(f"Train Loss: {train_loss:.4f} (MSE: {train_losses['mse_loss']:.4f}, "
              f"Length: {train_losses['bond_length_loss']:.4f}, "
              f"Angle: {train_losses['bond_angle_loss']:.4f}, "
              f"Clash: {train_losses['steric_clash_loss']:.4f})")
        print(f"Val Loss: {val_loss:.4f}, Val RMSD: {val_rmsd:.4f}, Val TM-score: {val_tm_score:.4f}")
        
        # Update learning rate scheduler
        if config.get('scheduler_type', 'reduce_on_plateau') == 'reduce_on_plateau':
            lr_scheduler.step(val_rmsd)
        else:
            lr_scheduler.step()
        
        # Save checkpoint - Enhanced with checkpoint rotation
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_rmsd': val_rmsd,
            'val_tm_score': val_tm_score,
            'config': config,
        }, checkpoint_path)
        
        # Checkpoint rotation (keep only last N checkpoints to save disk space)
        if config.get('keep_last_n_checkpoints', 5) > 0:
            all_checkpoints = sorted(list(checkpoint_dir.glob("checkpoint_epoch_*.pt")))
            if len(all_checkpoints) > config.get('keep_last_n_checkpoints', 5):
                # Remove oldest checkpoints
                for old_ckpt in all_checkpoints[:-config.get('keep_last_n_checkpoints', 5)]:
                    old_ckpt.unlink()
        
        # Update best model based on validation metrics
        improved = False
        improvement_threshold = config.get('improvement_threshold', 0.0001)
        
        # Check if this is the best model by RMSD (lower is better)
        if val_rmsd < best_val_rmsd * (1 - improvement_threshold):
            best_val_rmsd = val_rmsd
            improved = True
            print(f"New best model by RMSD: {val_rmsd:.6f}")

        # Check if this is the best model by TM-score (higher is better)
        # Because TM-score is the competition metric, we prioritize it
        if val_tm_score > best_val_tm_score * (1 + improvement_threshold):
            best_val_tm_score = val_tm_score
            improved = True
            print(f"New best model by TM-score: {val_tm_score:.6f}")
        
        if improved:
            best_epoch = epoch
            no_improve_count = 0
            
            # Save best model
            best_model_path = save_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_rmsd': val_rmsd,
                'val_tm_score': val_tm_score,
                'config': config,
            }, best_model_path)
            
            print(f"New best model saved at epoch {epoch}")
        else:
            no_improve_count += 1
            
            # Enhanced early stopping with a minimum number of epochs
            patience = config.get('patience', 10)
            min_epochs = config.get('min_epochs', 0)
            if no_improve_count >= patience and epoch >= min_epochs:
                print(f"No improvement for {no_improve_count} epochs. "
                      f"Best RMSD: {best_val_rmsd:.6f}, Best TM-score: {best_val_tm_score:.6f} at epoch {best_epoch}")
                break
    
    # Load best model
    best_model_path = save_dir / "best_model.pt"
    best_checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    print(f"\nTraining completed. Best RMSD: {best_val_rmsd:.6f}, "
          f"Best TM-score: {best_val_tm_score:.6f} at epoch {best_epoch}")
    
    if use_wandb:
        # Log the best model metrics as summary statistics
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.run.summary["best_rmsd"] = best_val_rmsd
        wandb.run.summary["best_tm_score"] = best_val_tm_score
        wandb.finish()
        
    return model, best_val_rmsd, best_val_tm_score


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    bond_length_weight: float = 0.2,
    bond_angle_weight: float = 0.2,
    steric_clash_weight: float = 0.3,
    use_mixed_precision: bool = True,
    gradient_accumulation_steps: int = 1,
    gradient_clip_val: Optional[float] = None,
) -> Dict[str, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to use for training
        epoch: Current epoch number
        bond_length_weight: Weight for bond length constraint loss
        bond_angle_weight: Weight for bond angle constraint loss
        steric_clash_weight: Weight for steric clash penalty loss
        use_mixed_precision: Whether to use mixed precision training (FP16)
        gradient_accumulation_steps: Number of steps to accumulate gradients
        gradient_clip_val: Max gradient norm for gradient clipping (None to disable)
        
    Returns:
        Dictionary of average loss values for the epoch
    """
    model.train()
    
    # Initialize running losses
    running_losses = {
        'loss': 0.0,
        'mse_loss': 0.0,
        'bond_length_loss': 0.0,
        'bond_angle_loss': 0.0,
        'steric_clash_loss': 0.0,
        'grad_norm': 0.0,
        'batches_per_second': 0.0,
        'samples_per_second': 0.0,
    }
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if use_mixed_precision and torch.cuda.is_available() else None
    
    num_batches = len(dataloader)
    total_samples = 0
    start_time = time.time()
    
    # Use tqdm for progress bar
    progress_bar = tqdm(enumerate(dataloader), total=num_batches, desc=f"Epoch {epoch}")
    
    for i, batch in progress_bar:
        # Move data to device
        sequences = batch['sequence'].to(device)
        coordinates = batch['coordinates'].to(device)
        lengths = batch['lengths'].to(device)
        
        batch_size = sequences.size(0)
        total_samples += batch_size
        
        # Forward pass with autocast for mixed precision
        with autocast(enabled=use_mixed_precision and torch.cuda.is_available()):
            # Forward pass
            pred_coords = model(sequences, lengths)

            # Align coordinate shapes if dataset has 5 atoms but model predicts 1 (or vice versa)
            coords_in = coordinates
            if pred_coords.shape[2] != coords_in.shape[2]:
                if pred_coords.shape[2] == 1 and coords_in.shape[2] >= 1:
                    coords_in = coords_in[:, :, :1, :]
                elif pred_coords.shape[2] == 5 and coords_in.shape[2] == 1:
                    coords_in = coords_in.repeat(1, 1, 5, 1)

            # Compute loss manually (mask-aware) with optional physics constraints
            mask = (sequences != 4).to(pred_coords.device)
            coord_mask = mask.unsqueeze(-1).unsqueeze(-1).float()
            # MSE over valid positions
            diff = (pred_coords - coords_in) * coord_mask
            mse_sum = (diff ** 2).sum()
            denom = coord_mask.sum() * pred_coords.shape[-1]  # valid positions * 3
            mse = mse_sum / (denom + 1e-8)
            rmsd_val = torch.sqrt(mse + 1e-8)
            # Physics constraints if available
            bl = model.compute_bond_length_loss(pred_coords, mask) if hasattr(model, 'compute_bond_length_loss') else torch.tensor(0.0, device=pred_coords.device)
            ba = model.compute_bond_angle_loss(pred_coords, mask) if hasattr(model, 'compute_bond_angle_loss') else torch.tensor(0.0, device=pred_coords.device)
            sc = model.compute_steric_clash_loss(pred_coords, mask) if hasattr(model, 'compute_steric_clash_loss') else torch.tensor(0.0, device=pred_coords.device)
            loss = rmsd_val + bond_length_weight * bl + bond_angle_weight * ba + steric_clash_weight * sc
            loss_components = {'rmsd': rmsd_val, 'bond_length': bl, 'bond_angle': ba, 'steric_clash': sc, 'total': loss}
            
            # loss already computed as total
            # (retain variable name 'loss' as scalar)

            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
        
        # Backward pass with gradient scaling
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Only step optimizer after accumulating gradients
        if ((i + 1) % gradient_accumulation_steps == 0) or (i + 1 == num_batches):
            # Gradient clipping
            if gradient_clip_val is not None:
                if scaler is not None:
                    # For mixed precision training, unscale gradients before clipping
                    scaler.unscale_(optimizer)
                
                # Compute and store gradient norm
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                running_losses['grad_norm'] += grad_norm.item()
            
            # Step optimizer with gradient scaling
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            # Zero the gradients
            optimizer.zero_grad()
        
        # Update running losses - scale back to account for gradient accumulation
        running_losses['loss'] += loss.item() * gradient_accumulation_steps
        running_losses['mse_loss'] += loss_components.get('rmsd', torch.tensor(0.0, device=device)).item() * gradient_accumulation_steps
        running_losses['bond_length_loss'] += loss_components.get('bond_length', torch.tensor(0.0, device=device)).item() * gradient_accumulation_steps
        running_losses['bond_angle_loss'] += loss_components.get('bond_angle', torch.tensor(0.0, device=device)).item() * gradient_accumulation_steps
        running_losses['steric_clash_loss'] += loss_components.get('steric_clash', torch.tensor(0.0, device=device)).item() * gradient_accumulation_steps
        
        # Update progress bar
        elapsed = time.time() - start_time
        if elapsed > 0:
            batches_per_sec = (i + 1) / elapsed
            samples_per_sec = total_samples / elapsed
            running_losses['batches_per_second'] = batches_per_sec
            running_losses['samples_per_second'] = samples_per_sec
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'b/s': f"{batches_per_sec:.2f}",
                's/s': f"{samples_per_sec:.2f}"
            })
    
    # Calculate average losses
    for k in running_losses:
        if k not in ['batches_per_second', 'samples_per_second']:
            running_losses[k] /= num_batches
    
    return running_losses


def validate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    bond_length_weight: float = 0.2,
    bond_angle_weight: float = 0.2,
    steric_clash_weight: float = 0.3,
    use_mixed_precision: bool = True,
) -> Tuple[Dict[str, float], float, float]:
    """
    Validate the model.
    
    Args:
        model: The model to validate
        dataloader: Validation data loader
        device: Device to use for validation
        bond_length_weight: Weight for bond length constraint loss
        bond_angle_weight: Weight for bond angle constraint loss
        steric_clash_weight: Weight for steric clash penalty loss
        use_mixed_precision: Whether to use mixed precision for validation
        
    Returns:
        Tuple of (average loss dictionary, average RMSD, average TM-score)
    """
    model.eval()
    
    # Initialize running losses and metrics
    running_losses = {
        'loss': 0.0,
        'mse_loss': 0.0,
        'bond_length_loss': 0.0,
        'bond_angle_loss': 0.0,
        'steric_clash_loss': 0.0
    }
    
    total_rmsd = 0.0
    total_tm_score = 0.0
    num_batches = len(dataloader)
    num_samples = 0
    
    # Use tqdm for progress bar
    progress_bar = tqdm(dataloader, total=num_batches, desc="Validation")
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move data to device
            sequences = batch['sequence'].to(device)
            coordinates = batch['coordinates'].to(device)
            lengths = batch['lengths'].to(device)
            
            batch_size = sequences.size(0)
            num_samples += batch_size
            
            # Use autocast for mixed precision
            with autocast(enabled=use_mixed_precision and torch.cuda.is_available()):
                # Forward pass
                pred_coords = model(sequences, lengths)

                # Align coordinate shapes analogous to training
                coords_in = coordinates
                if pred_coords.shape[2] != coords_in.shape[2]:
                    if pred_coords.shape[2] == 1 and coords_in.shape[2] >= 1:
                        coords_in = coords_in[:, :, :1, :]
                    elif pred_coords.shape[2] == 5 and coords_in.shape[2] == 1:
                        coords_in = coords_in.repeat(1, 1, 5, 1)

                # Compute loss manually (mask-aware) with optional physics constraints
                mask = (sequences != 4).to(pred_coords.device)
                coord_mask = mask.unsqueeze(-1).unsqueeze(-1).float()
                diff = (pred_coords - coords_in) * coord_mask
                mse_sum = (diff ** 2).sum()
                denom = coord_mask.sum() * pred_coords.shape[-1]
                mse = mse_sum / (denom + 1e-8)
                rmsd_val = torch.sqrt(mse + 1e-8)
                bl = model.compute_bond_length_loss(pred_coords, mask) if hasattr(model, 'compute_bond_length_loss') else torch.tensor(0.0, device=pred_coords.device)
                ba = model.compute_bond_angle_loss(pred_coords, mask) if hasattr(model, 'compute_bond_angle_loss') else torch.tensor(0.0, device=pred_coords.device)
                sc = model.compute_steric_clash_loss(pred_coords, mask) if hasattr(model, 'compute_steric_clash_loss') else torch.tensor(0.0, device=pred_coords.device)
                loss = rmsd_val + bond_length_weight * bl + bond_angle_weight * ba + steric_clash_weight * sc
                loss_components = {'rmsd': rmsd_val, 'bond_length': bl, 'bond_angle': ba, 'steric_clash': sc, 'total': loss}

            
            # Compute RMSD
            _, batch_mean_rmsd = batch_rmsd(
                pred_coords=pred_coords,
                true_coords=coordinates,
                lengths=lengths,
                align=True,
            )
            
            # Compute TM-score
            _, batch_mean_tm = batch_tm_score(
                pred_coords=pred_coords,
                true_coords=coordinates,
                lengths=lengths,
                align=True,
            )
            
            # Update running losses and metrics
            running_losses['loss'] += loss.item() * batch_size
            running_losses['mse_loss'] += loss_components.get('rmsd', torch.tensor(0.0, device=device)).item() * batch_size
            running_losses['bond_length_loss'] += loss_components.get('bond_length', torch.tensor(0.0, device=device)).item() * batch_size
            running_losses['bond_angle_loss'] += loss_components.get('bond_angle', torch.tensor(0.0, device=device)).item() * batch_size
            running_losses['steric_clash_loss'] += loss_components.get('steric_clash', torch.tensor(0.0, device=device)).item() * batch_size
                
            total_rmsd += batch_mean_rmsd.item() * batch_size
            total_tm_score += batch_mean_tm.item() * batch_size
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'rmsd': f"{batch_mean_rmsd.item():.4f}",
                'tm': f"{batch_mean_tm.item():.4f}"
            })
    
    # Calculate weighted averages by sample count
    for k in running_losses:
        running_losses[k] /= num_samples
        
    avg_rmsd = total_rmsd / num_samples
    avg_tm_score = total_tm_score / num_samples
    
    return running_losses, avg_rmsd, avg_tm_score
