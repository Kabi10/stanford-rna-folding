"""
Training script for RNA folding model.
"""

import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from tqdm import tqdm

from ..models.rna_folding_model import RNAFoldingModel
from ..data.data_processing import create_data_loaders

def train_epoch(model, train_loader, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        optimizer (Optimizer): The optimizer
        device (torch.device): Device to train on
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    with tqdm(train_loader, desc='Training') as pbar:
        for batch in pbar:
            # Get data
            sequences = batch['sequence'].to(device)
            coordinates = batch['coordinates'].to(device)
            
            # Forward pass
            pred_coords = model(sequences)
            loss = model.compute_loss(pred_coords, coordinates)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    """
    Validate the model.
    
    Args:
        model (nn.Module): The model to validate
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to validate on
        
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        with tqdm(val_loader, desc='Validation') as pbar:
            for batch in pbar:
                # Get data
                sequences = batch['sequence'].to(device)
                coordinates = batch['coordinates'].to(device)
                
                # Forward pass
                pred_coords = model(sequences)
                loss = model.compute_loss(pred_coords, coordinates)
                
                # Update progress
                total_loss += loss.item()
                pbar.set_postfix({'val_loss': loss.item()})
    
    return total_loss / len(val_loader)

def train(config):
    """
    Main training function.
    
    Args:
        config (dict): Training configuration
    """
    # Set up wandb
    if config['use_wandb']:
        wandb.init(
            project="stanford-rna-folding",
            config=config
        )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Create model
    model = RNAFoldingModel(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        coord_hidden_dim=config['coord_hidden_dim'],
        coord_num_layers=config['coord_num_layers']
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        if config['use_wandb']:
            wandb.log(metrics)
            
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(config['model_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, model_path)
            print(f"Saved best model to {model_path}")
        
        # Save checkpoint
        if (epoch + 1) % config['checkpoint_freq'] == 0:
            checkpoint_path = os.path.join(
                config['model_dir'],
                f'checkpoint_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

def main():
    parser = argparse.ArgumentParser(description='Train RNA folding model')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create directories
    os.makedirs(config['model_dir'], exist_ok=True)
    
    # Start training
    train(config)

if __name__ == '__main__':
    main() 