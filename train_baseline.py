#!/usr/bin/env python3
"""
Simplified baseline training script for Stanford RNA 3D Structure Prediction.

This script provides a streamlined approach to train a baseline model using the
existing dataset and model infrastructure.
"""

import os
import sys
import argparse
import logging
import yaml
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from stanford_rna_folding.data.data_processing import StanfordRNADataset, rna_collate_fn
from stanford_rna_folding.models.rna_folding_model import RNAFoldingModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_datasets(config: Dict[str, Any]) -> tuple:
    """Create training and validation datasets."""
    data_dir = config['data_dir']
    
    # Create datasets
    train_dataset = StanfordRNADataset(
        data_dir=data_dir,
        split='train',
        transform=None
    )
    
    val_dataset = StanfordRNADataset(
        data_dir=data_dir,
        split='validation',
        transform=None
    )
    
    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, config: Dict[str, Any]) -> tuple:
    """Create data loaders."""
    batch_size = config.get('batch_size', 8)
    num_workers = config.get('num_workers', 2)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=rna_collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=rna_collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create and initialize the model."""
    model = RNAFoldingModel(
        vocab_size=config.get('vocab_size', 5),
        embedding_dim=config.get('embedding_dim', 64),
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 2),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('dropout', 0.1),
        num_atoms=1,  # Use single atom mode since we only have x_1, y_1, z_1
        coord_dims=config.get('coord_dims', 3),
        max_seq_len=config.get('max_seq_len', 500),
        use_rna_constraints=False,  # Disable constraints for baseline
        bond_length_weight=0.0,
        bond_angle_weight=0.0,
        steric_clash_weight=0.0,
    ).to(device)

    return model


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, 
                device: torch.device, epoch: int, scaler: Optional[GradScaler] = None) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_mse_loss = 0.0
    total_constraint_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        try:
            # Move data to device
            sequences = batch['sequence'].to(device)
            coordinates = batch['coordinates'].to(device)
            lengths = batch['lengths'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(enabled=scaler is not None):
                # Forward pass
                pred_coords = model(sequences, lengths)

                # Compute simple MSE loss
                mse_loss = nn.MSELoss()(pred_coords, coordinates)
                total_loss_batch = mse_loss
            
            # Backward pass
            if scaler is not None:
                scaler.scale(total_loss_batch).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss_batch.backward()
                optimizer.step()
            
            # Update metrics
            total_loss += total_loss_batch.item()
            total_mse_loss += mse_loss.item()
            total_constraint_loss += 0.0
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss_batch.item():.4f}",
                'mse': f"{mse_loss.item():.4f}"
            })
            
        except Exception as e:
            logger.warning(f"Error in batch: {e}")
            continue
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'mse_loss': total_mse_loss / max(num_batches, 1),
        'constraint_loss': total_constraint_loss / max(num_batches, 1)
    }


def validate_epoch(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    
    total_loss = 0.0
    total_mse_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            try:
                # Move data to device
                sequences = batch['sequence'].to(device)
                coordinates = batch['coordinates'].to(device)
                lengths = batch['lengths'].to(device)
                
                # Forward pass
                pred_coords = model(sequences, lengths)
                
                # Compute loss
                mse_loss = nn.MSELoss()(pred_coords, coordinates)
                
                total_loss += mse_loss.item()
                total_mse_loss += mse_loss.item()
                num_batches += 1
                
            except Exception as e:
                logger.warning(f"Error in validation batch: {e}")
                continue
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'mse_loss': total_mse_loss / max(num_batches, 1)
    }


def train_model(config: Dict[str, Any]):
    """Main training function."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets and dataloaders
    logger.info("Creating datasets...")
    train_dataset, val_dataset = create_datasets(config)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config, device)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 5e-4),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # Training loop
    num_epochs = config.get('num_epochs', 5)
    best_val_loss = float('inf')
    save_dir = Path(config.get('save_dir', 'models/baseline'))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{num_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, scaler)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device)
        
        # Update scheduler
        scheduler.step(val_metrics['loss'])
        
        # Log metrics
        logger.info(f"Train Loss: {train_metrics['loss']:.4f} (MSE: {train_metrics['mse_loss']:.4f})")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f} (MSE: {val_metrics['mse_loss']:.4f})")
        logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config
            }, save_dir / 'best_model.pth')
            logger.info(f"Saved best model with validation loss: {best_val_loss:.4f}")
    
    logger.info("Training completed!")
    return model, best_val_loss


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train baseline RNA folding model')
    parser.add_argument('--config', type=str, default='configs/quick_test_config.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Train model
    model, best_val_loss = train_model(config)
    
    logger.info(f"Training finished. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
