#!/usr/bin/env python3
"""
Ultra-simplified baseline training script for Stanford RNA 3D Structure Prediction.

This script creates a minimal working baseline to establish the training pipeline
and get initial performance metrics.
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleRNADataset(Dataset):
    """Simplified RNA dataset that handles the actual data format."""
    
    def __init__(self, data_dir: str, split: str = "train", max_samples: int = 100):
        self.data_dir = data_dir
        self.split = split
        self.max_samples = max_samples
        
        # Load sequences
        seq_file = os.path.join(data_dir, f"{split}_sequences.csv")
        self.sequences_df = pd.read_csv(seq_file)
        
        # Load labels if not test
        if split != "test":
            labels_file = os.path.join(data_dir, f"{split}_labels.csv")
            self.labels_df = pd.read_csv(labels_file)
        else:
            self.labels_df = None
        
        # Nucleotide mapping
        self.nucleotide_to_idx = {"A": 0, "U": 1, "G": 2, "C": 3, "N": 4}
        
        # Limit samples for quick testing
        if len(self.sequences_df) > max_samples:
            self.sequences_df = self.sequences_df.head(max_samples)
            logger.info(f"Limited to {max_samples} samples for quick testing")
    
    def __len__(self):
        return len(self.sequences_df)
    
    def _encode_sequence(self, sequence: str) -> torch.Tensor:
        """Encode RNA sequence to tensor."""
        encoded = [self.nucleotide_to_idx.get(nt, 4) for nt in sequence]
        return torch.tensor(encoded, dtype=torch.long)
    
    def _get_coordinates(self, target_id: str) -> torch.Tensor:
        """Get coordinates for a target."""
        if self.labels_df is None:
            return torch.zeros(1, 3)  # Dummy coordinates for test
        
        # Filter coordinates for this target
        target_coords = self.labels_df[self.labels_df['ID'].str.startswith(target_id)]
        
        if len(target_coords) == 0:
            return torch.zeros(1, 3)
        
        # Extract x_1, y_1, z_1 coordinates
        coords = target_coords[['x_1', 'y_1', 'z_1']].values
        
        # Handle NaN values
        coords = np.nan_to_num(coords, nan=0.0)
        
        return torch.tensor(coords, dtype=torch.float32)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.sequences_df.iloc[idx]
        target_id = row['ID'] if 'ID' in row else row['target_id']
        sequence = row['sequence']
        
        # Encode sequence
        seq_tensor = self._encode_sequence(sequence)
        
        # Get coordinates
        coords = self._get_coordinates(target_id)
        
        return {
            'sequence': seq_tensor,
            'coordinates': coords,
            'target_id': target_id,
            'seq_length': len(seq_tensor)
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function to handle variable length sequences."""
    # Find max sequence length
    max_seq_len = max(item['seq_length'] for item in batch)
    max_coord_len = max(item['coordinates'].shape[0] for item in batch)
    
    batch_size = len(batch)
    
    # Pad sequences
    sequences = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    coordinates = torch.zeros(batch_size, max_coord_len, 3, dtype=torch.float32)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    target_ids = []
    
    for i, item in enumerate(batch):
        seq_len = item['seq_length']
        coord_len = item['coordinates'].shape[0]
        
        sequences[i, :seq_len] = item['sequence']
        coordinates[i, :coord_len] = item['coordinates']
        lengths[i] = seq_len
        target_ids.append(item['target_id'])
    
    return {
        'sequence': sequences,
        'coordinates': coordinates,
        'lengths': lengths,
        'target_ids': target_ids
    }


class SimpleRNAModel(nn.Module):
    """Ultra-simple RNA folding model for baseline."""
    
    def __init__(self, vocab_size: int = 5, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.coord_head = nn.Linear(hidden_dim * 2, 3)  # Predict x, y, z
        
    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Embed sequences
        embedded = self.embedding(sequences)  # [batch, seq_len, embed_dim]
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # [batch, seq_len, hidden_dim * 2]
        
        # Predict coordinates for each position
        coords = self.coord_head(lstm_out)  # [batch, seq_len, 3]
        
        return coords


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, 
                device: torch.device, epoch: int) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        try:
            sequences = batch['sequence'].to(device)
            coordinates = batch['coordinates'].to(device)
            lengths = batch['lengths'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_coords = model(sequences, lengths)
            
            # Compute loss - only on valid positions
            batch_size, max_seq_len, _ = pred_coords.shape
            _, max_coord_len, _ = coordinates.shape
            
            # Truncate predictions to match coordinate length
            min_len = min(max_seq_len, max_coord_len)
            pred_coords_truncated = pred_coords[:, :min_len, :]
            coords_truncated = coordinates[:, :min_len, :]
            
            # Simple MSE loss
            loss = nn.MSELoss()(pred_coords_truncated, coords_truncated)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        except Exception as e:
            logger.warning(f"Error in batch: {e}")
            continue
    
    return total_loss / max(num_batches, 1)


def validate_epoch(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            try:
                sequences = batch['sequence'].to(device)
                coordinates = batch['coordinates'].to(device)
                lengths = batch['lengths'].to(device)
                
                # Forward pass
                pred_coords = model(sequences, lengths)
                
                # Compute loss
                batch_size, max_seq_len, _ = pred_coords.shape
                _, max_coord_len, _ = coordinates.shape
                
                min_len = min(max_seq_len, max_coord_len)
                pred_coords_truncated = pred_coords[:, :min_len, :]
                coords_truncated = coordinates[:, :min_len, :]
                
                loss = nn.MSELoss()(pred_coords_truncated, coords_truncated)
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                logger.warning(f"Error in validation batch: {e}")
                continue
    
    return total_loss / max(num_batches, 1)


def main():
    """Main training function."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Data directory
    data_dir = "datasets/stanford-rna-3d-folding"
    
    # Create datasets (limited for quick testing)
    logger.info("Creating datasets...")
    train_dataset = SimpleRNADataset(data_dir, split='train', max_samples=50)
    val_dataset = SimpleRNADataset(data_dir, split='validation', max_samples=10)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn
    )
    
    # Create model
    logger.info("Creating model...")
    model = SimpleRNAModel().to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    num_epochs = 3
    best_val_loss = float('inf')
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, device)
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dir = Path("models/simple_baseline")
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_dir / 'best_model.pth')
            logger.info(f"Saved best model with validation loss: {best_val_loss:.4f}")
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    return model, best_val_loss


if __name__ == "__main__":
    main()
