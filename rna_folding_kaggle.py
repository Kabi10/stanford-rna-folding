#!/usr/bin/env python
"""
Stanford RNA 3D Folding - Kaggle Notebook

This notebook is designed to run the RNA 3D folding model training and evaluation 
on Kaggle's GPU environment.
"""

# Standard imports
import os
import sys
import json
import math
import time
import random
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List

# Data manipulation
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Define directories
DATA_DIR = "/kaggle/input/stanford-rna-3d-folding"
OUTPUT_DIR = "/kaggle/working"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
SUBMISSION_DIR = os.path.join(OUTPUT_DIR, "submissions")

# Create necessary directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Configure device - EXPLICITLY requesting GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu" and os.environ.get('KAGGLE_IS_COMPETITION_RERUN'):
    print("WARNING: GPU is not available but we're in a Kaggle competition environment!")
    print("Please make sure you've enabled GPU in the Kaggle notebook settings.")

print("Stanford RNA 3D Folding - Setup complete!")
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Check the competition data
print("\nExploring the competition data:")
print(f"Files in {DATA_DIR}:")
for f in os.listdir(DATA_DIR):
    print(f"- {f}")

# Quick test: Load sample data
try:
    train_seq = pd.read_csv(f"{DATA_DIR}/train_sequences.csv")
    print(f"\nTrain sequences shape: {train_seq.shape}")
    print(train_seq.head(3))
    
    sample_submission = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
    print(f"\nSample submission shape: {sample_submission.shape}")
    print(sample_submission.head(3))
except Exception as e:
    print(f"Error loading data: {e}")

# Create output directory for experiment results
os.makedirs(f"{OUTPUT_DIR}/rna_folding_results", exist_ok=True)

print("\nInitial setup and data exploration complete!")
print("Next steps would include:")
print("1. Building the transformer-based model")
print("2. Setting up the data pipeline")
print("3. Implementing training and validation loops")
print("4. Generating predictions for submission")

# ==============================
# 1. Define Model Architecture
# ==============================

class RNAFoldingModel(nn.Module):
    """
    Transformer-based model for RNA 3D structure prediction.
    
    Takes RNA sequences as input and predicts 3D coordinates for each nucleotide.
    Uses a Transformer encoder to process the sequence and an MLP to predict coordinates.
    """
    
    def __init__(
        self,
        vocab_size: int = 5,  # A, U, G, C, N (padding)
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_atoms: int = 5,  # Number of atoms per nucleotide to predict
        coord_dims: int = 3,  # 3D coordinates (x, y, z)
        max_seq_len: int = 500,
        use_rna_constraints: bool = False,
        bond_length_weight: float = 1.0,
        bond_angle_weight: float = 1.0,
        steric_clash_weight: float = 1.5,
        watson_crick_weight: float = 2.0,
    ):
        """Initialize the RNA folding model."""
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_atoms = num_atoms
        self.coord_dims = coord_dims
        self.max_seq_len = max_seq_len
        
        # Biophysics constraint flags and weights
        self.use_rna_constraints = use_rna_constraints
        self.bond_length_weight = bond_length_weight
        self.bond_angle_weight = bond_angle_weight
        self.steric_clash_weight = steric_clash_weight
        self.watson_crick_weight = watson_crick_weight
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=4,  # Assuming 4 is the padding token (N)
        )
        
        # Positional encoding
        self.register_buffer('positional_encoding', self._generate_positional_encoding(seq_len=None))
        
        # Transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,  # Input shape: (batch, seq, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers,
        )
        
        # MLP for coordinate prediction
        self.coordinate_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_atoms * coord_dims),  # Predict all atoms at once
        )
        
        # Initialize ideal bond lengths and angles for RNA nucleotides
        self.register_buffer('ideal_bond_lengths', torch.tensor([
            1.5,  # P-O5' (phosphate to sugar)
            1.6,  # O5'-C5' (sugar-sugar connection)
            1.5,  # C5'-C4' (ribose ring)
            1.4,  # C4'-C3' (ribose ring)
            1.4,  # C3'-O3' (connection to next nucleotide)
        ], dtype=torch.float32))
        
        # Bond length tolerances (how much deviation we allow)
        self.register_buffer('bond_length_tolerance', torch.tensor([0.2] * 5, dtype=torch.float32))
        
        # VdW radius for atoms to calculate steric clashes (in Angstroms)
        self.register_buffer('vdw_radii', torch.tensor([
            1.9,  # P (phosphate)
            1.5,  # O5' (oxygen)
            1.7,  # C5' (carbon)
            1.7,  # C4' (carbon)
            1.5,  # O3' (oxygen)
        ], dtype=torch.float32))
    
    def _generate_positional_encoding(self, seq_len: Optional[int] = None) -> torch.Tensor:
        """Generate sinusoidal positional encodings."""
        # Use specified sequence length or fall back to max_seq_len
        length = seq_len if seq_len is not None else self.max_seq_len
        
        pe = torch.zeros(length, self.embedding_dim)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2).float() * 
            (-math.log(10000.0) / self.embedding_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Shape: (1, length, embedding_dim)
    
    def create_padding_mask(self, sequence: torch.Tensor, pad_idx: int = 4) -> torch.Tensor:
        """Create a mask for padding tokens in the sequence."""
        return sequence == pad_idx  # Shape: (batch_size, seq_len)
    
    def forward(
        self, 
        sequence: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the model."""
        batch_size, seq_len = sequence.shape
        
        # Embed sequence tokens
        x = self.token_embedding(sequence)  # Shape: (batch_size, seq_len, embedding_dim)
        
        # Add positional encoding
        if seq_len > self.max_seq_len:
            # Generate positional encoding for the current sequence length
            pos_enc = self._generate_positional_encoding(seq_len=seq_len)
        else:
            pos_enc = self.positional_encoding[:, :seq_len, :]
        
        # Ensure positional encoding is on the same device as x
        pos_enc = pos_enc.to(x.device)
        
        x = x + pos_enc  # Shape: (batch_size, seq_len, embedding_dim)
        
        # Create padding mask for the transformer
        padding_mask = self.create_padding_mask(sequence)
        
        # Pass through transformer encoder
        encoded = self.transformer_encoder(
            src=x,
            src_key_padding_mask=padding_mask,
        )  # Shape: (batch_size, seq_len, embedding_dim)
        
        # Predict coordinates using MLP
        coords_flat = self.coordinate_mlp(encoded)  # Shape: (batch_size, seq_len, num_atoms * coord_dims)
        
        # Reshape to separate atoms and coordinates
        coords = coords_flat.view(batch_size, seq_len, self.num_atoms, self.coord_dims)
        
        return coords
    
    # We'll omit the physics-based constraint loss functions for this simplified version
    # In a full model, we would include compute_bond_length_loss, compute_bond_angle_loss, etc.
    
    def compute_bond_length_loss(self, coords, lengths=None):
        """
        Compute loss based on bond length constraints.
        Penalizes deviations from ideal bond lengths in RNA structure.
        """
        # In a full implementation, this would calculate bond length violations
        # For now, return zero
        return torch.tensor(0.0, device=coords.device)
        
    def compute_bond_angle_loss(self, coords, lengths=None):
        """
        Compute loss based on bond angle constraints.
        Penalizes deviations from ideal bond angles in RNA structure.
        """
        # In a full implementation, this would calculate bond angle violations
        # For now, return zero
        return torch.tensor(0.0, device=coords.device)
        
    def compute_steric_clash_loss(self, coords, lengths=None):
        """
        Compute loss based on steric clashes.
        Penalizes atom pairs that are too close to each other.
        """
        # In a full implementation, this would calculate steric clashes
        # For now, return zero
        return torch.tensor(0.0, device=coords.device)
        
    def compute_watson_crick_loss(self, coords, sequence, lengths=None):
        """
        Compute loss to encourage Watson-Crick base pairing.
        Rewards A-U and G-C pairs at appropriate distances.
        """
        # In a full implementation, this would calculate Watson-Crick bonding
        # For now, return zero
        return torch.tensor(0.0, device=coords.device)
    
    def compute_loss(self, pred_coords, true_coords, lengths=None, sequence=None):
        """Compute combined loss with biophysical constraints."""
        batch_size, max_len = pred_coords.shape[:2]
        
        # Create mask for valid positions based on sequence lengths
        mask = None
        if lengths is not None:
            mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) < lengths.unsqueeze(1)
            # Expand mask for MSE loss
            coord_mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(pred_coords).float()
        else:
            # If no lengths provided, assume all positions are valid
            coord_mask = torch.ones_like(pred_coords)
        
        # Compute MSE loss on valid coordinates
        squared_error = (pred_coords - true_coords) ** 2
        masked_squared_error = squared_error * coord_mask
        
        # Take mean over valid positions
        num_valid_positions = coord_mask.sum()
        if num_valid_positions > 0:
            mse_loss = masked_squared_error.sum() / num_valid_positions
        else:
            mse_loss = torch.tensor(0.0, device=pred_coords.device)
        
        # Initialize total loss with MSE
        total_loss = mse_loss
        loss_components = {"mse_loss": mse_loss}
        
        # Add biophysical constraints if enabled
        if self.use_rna_constraints:
            # Bond length constraints
            if self.bond_length_weight > 0:
                bond_loss = self.compute_bond_length_loss(pred_coords, lengths)
                total_loss = total_loss + self.bond_length_weight * bond_loss
                loss_components["bond_length_loss"] = bond_loss
            
            # Bond angle constraints
            if self.bond_angle_weight > 0:
                angle_loss = self.compute_bond_angle_loss(pred_coords, lengths)
                total_loss = total_loss + self.bond_angle_weight * angle_loss
                loss_components["bond_angle_loss"] = angle_loss
            
            # Steric clash constraints
            if self.steric_clash_weight > 0:
                clash_loss = self.compute_steric_clash_loss(pred_coords, lengths)
                total_loss = total_loss + self.steric_clash_weight * clash_loss
                loss_components["steric_clash_loss"] = clash_loss
            
            # Watson-Crick base pairing constraints
            if self.watson_crick_weight > 0 and sequence is not None:
                wc_loss = self.compute_watson_crick_loss(pred_coords, sequence, lengths)
                total_loss = total_loss + self.watson_crick_weight * wc_loss
                loss_components["watson_crick_loss"] = wc_loss
        
        # Return total loss and individual components
        loss_components["loss"] = total_loss
        return loss_components

# ==============================
# 2. Data Processing
# ==============================

class StanfordRNADataset(Dataset):
    """Dataset class for Stanford RNA 3D Structure Prediction competition."""
    
    def __init__(self, data_dir: str, split: str = "train", transform=None):
        """Initialize the Stanford RNA Structure dataset."""
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Define coordinate columns explicitly based on sample_submission.csv
        self.coord_cols = []
        for i in range(1, 6):
            self.coord_cols.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
        
        # Load sequence data
        self.sequences_df = self._load_sequences()
        
        if split == "test":
            # For test set, we only have sequences
            self.processed_labels = None 
        else:
            # Load and process label data
            raw_labels_df = self._load_labels()
            self.processed_labels = self._process_labels(raw_labels_df)
            
            # Filter sequences_df to only include targets present in processed_labels
            valid_targets = list(self.processed_labels.keys())
            self.sequences_df = self.sequences_df[self.sequences_df['target_id'].isin(valid_targets)].reset_index(drop=True)

        # Create mapping of nucleotides to integers
        self.nucleotide_to_idx = {"A": 0, "U": 1, "G": 2, "C": 3, "N": 4}  # N for unknown/padding
    
    def _load_sequences(self):
        """Load sequence data from CSV file."""
        filename = f"{self.split}_sequences.csv"
        filepath = os.path.join(self.data_dir, filename)
        return pd.read_csv(filepath)
    
    def _load_labels(self):
        """Load label data from CSV file if available."""
        if self.split == "test":
            return None
            
        filename = f"{self.split}_labels.csv"
        filepath = os.path.join(self.data_dir, filename)
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"Error loading labels file {filepath}: {e}")
            raise
    
    def _extract_base_target_id(self, complex_id: str) -> str:
        """Extracts the base target ID (e.g., '1SCL_A') from the full ID."""
        match = re.match(r"(.+)_\d+$", complex_id)
        if match:
            return match.group(1)
        else:
            return complex_id
            
    def _process_labels(self, labels_df: pd.DataFrame):
        """Process the raw labels dataframe to group coordinates by target ID."""
        processed = {}
        
        # Extract base target ID for grouping
        try:
            labels_df['target_id'] = labels_df['ID'].apply(self._extract_base_target_id)
        except KeyError:
            print("Error: 'ID' column not found in labels dataframe. Cannot process labels.")
            return {}
        
        # Check if coordinate columns exist
        missing_cols = [col for col in self.coord_cols if col not in labels_df.columns]
        if missing_cols:
            print(f"Error: Missing expected coordinate columns in labels_df: {missing_cols}")
            # Determine which atom's coordinates are available
            available_atoms = []
            for atom_idx in range(1, 6):
                atom_cols = [f'x_{atom_idx}', f'y_{atom_idx}', f'z_{atom_idx}']
                if all(col in labels_df.columns for col in atom_cols):
                    available_atoms.append(atom_idx)
            
            print(f"Available atoms: {available_atoms}")
            if available_atoms:
                print(f"Using coordinates for atoms: {available_atoms}")
                self.coord_cols = []
                for atom_idx in available_atoms:
                    self.coord_cols.extend([f'x_{atom_idx}', f'y_{atom_idx}', f'z_{atom_idx}'])
            else:
                raise ValueError("No complete atom coordinates found in the data")
        
        # Group by the base target ID
        grouped = labels_df.groupby('target_id')
        
        for target_id, group in grouped:
            # Ensure residues are sorted correctly by residue ID
            group = group.sort_values(by='resid')
            
            # Select coordinate columns and convert to numpy array
            coords_np = group[self.coord_cols].values
            
            # Check for NaN/missing values
            if np.isnan(coords_np).any():
                print(f"Warning: NaN values found in coordinates for target {target_id}.")
                # Replace NaNs with zeros
                coords_np = np.nan_to_num(coords_np)

            # Reshape: (SequenceLength, NumAtoms, NumCoords)
            try:
                num_atoms = len(self.coord_cols) // 3
                coords_reshaped = coords_np.reshape(-1, num_atoms, 3)
                
                # If we have less than 5 atoms, pad to 5 atoms for model compatibility
                if num_atoms < 5:
                    print(f"Padding coordinates from {num_atoms} atoms to 5 atoms for target {target_id}")
                    # Create a padded array filled with zeros
                    padded_coords = np.zeros((*coords_reshaped.shape[:-2], 5, 3))
                    # Copy the available atoms to the padded array
                    padded_coords[:, :num_atoms, :] = coords_reshaped
                    coords_reshaped = padded_coords
            except ValueError as e:
                print(f"Error reshaping coordinates for target {target_id}. Error: {e}")
                continue # Skip this target if reshaping fails
                
            # Convert to tensor
            processed[target_id] = torch.tensor(coords_reshaped, dtype=torch.float32)
            
        return processed

    def _encode_sequence(self, sequence: str) -> torch.Tensor:
        """Convert RNA sequence to tensor of integers."""
        return torch.tensor([self.nucleotide_to_idx.get(nt, self.nucleotide_to_idx["N"]) 
                           for nt in sequence], dtype=torch.long)
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences_df)
    
    def __getitem__(self, idx: int):
        """Get a single example from the dataset."""
        # Get sequence data row
        row = self.sequences_df.iloc[idx]
        target_id = row["target_id"]
        sequence = row["sequence"]
        
        # Encode sequence
        sequence_tensor = self._encode_sequence(sequence)
        sequence_length = len(sequence_tensor)
        
        # Create output dictionary
        output = {
            "sequence": sequence_tensor,
            "target_id": target_id
        }
        
        # Retrieve pre-processed coordinates if available (train/validation)
        if self.processed_labels is not None and target_id in self.processed_labels:
            coords_tensor = self.processed_labels[target_id]
            
            # Validate sequence length against coordinate length
            if sequence_length != coords_tensor.shape[0]:
                print(f"Warning: Sequence length mismatch with coordinate length for {target_id}.")
                
                # Handle mismatch by truncating or padding
                if coords_tensor.shape[0] > sequence_length:
                    # Truncate coordinates if longer
                    coords_tensor = coords_tensor[:sequence_length]
                else:
                    # Pad coordinates if shorter
                    pad_size = sequence_length - coords_tensor.shape[0]
                    atom_dims = coords_tensor.shape[1:]
                    padding = torch.zeros((pad_size, *atom_dims), dtype=coords_tensor.dtype)
                    coords_tensor = torch.cat([coords_tensor, padding], dim=0)
            
            output["coordinates"] = coords_tensor
        else:
            # For test set, add empty coordinates tensor
            output["coordinates"] = torch.zeros((sequence_length, 5, 3), dtype=torch.float32)

        # Apply transforms if specified
        if self.transform is not None:
            output = self.transform(output)
            
        return output

def rna_collate_fn(batch, pad_value_sequence: int = 4, pad_value_coords: float = 0.0):
    """Collate function for DataLoader to handle variable length sequences and coordinates."""
    # Separate components
    sequences = [item['sequence'] for item in batch]
    coordinates = [item['coordinates'] for item in batch]
    target_ids = [item['target_id'] for item in batch]
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)

    # Pad sequences
    padded_sequences = pad_sequence(
        sequences, 
        batch_first=True, 
        padding_value=pad_value_sequence
    )

    # Manual padding for coordinates
    max_len = lengths.max().item()
    batch_size = len(batch)
    num_atoms = coordinates[0].shape[1]  # Assuming same number of atoms for all
    num_dims = coordinates[0].shape[2]  # Assuming 3D coordinates
    
    # Initialize padded coordinates tensor
    padded_coordinates = torch.full(
        (batch_size, max_len, num_atoms, num_dims), 
        pad_value_coords, 
        dtype=coordinates[0].dtype
    )
    
    # Fill with actual coordinate data
    for i, coords in enumerate(coordinates):
        seq_len = coords.shape[0]
        if seq_len > 0:
            # Make sure coords length doesn't exceed max_len
            current_len = min(seq_len, max_len)
            padded_coordinates[i, :current_len, :, :] = coords[:current_len]

    return {
        'sequence': padded_sequences,
        'coordinates': padded_coordinates,
        'target_id': target_ids, 
        'lengths': lengths
    }

# Next, we'll implement data transforms and training configuration 

# ==============================
# 3. Training Configuration
# ==============================

class RNADataTransform:
    """Data transformations for RNA folding."""
    
    def __init__(self, 
                 normalize_coords=True,
                 random_rotation=False,
                 position_noise=0.0,
                 atom_mask_prob=0.0):
        """Initialize RNA data transform."""
        self.normalize_coords = normalize_coords
        self.random_rotation = random_rotation
        self.position_noise = position_noise
        self.atom_mask_prob = atom_mask_prob
        
    def _normalize_coordinates(self, coords):
        """Center coordinates at origin and normalize."""
        # Calculate the center of the RNA (mean position)
        center = coords.reshape(-1, 3).mean(dim=0, keepdim=True)
        
        # Center the coordinates
        centered_coords = coords - center
        
        # Normalize by the max distance from center for scale invariance
        max_dist = torch.norm(centered_coords.reshape(-1, 3), dim=1).max()
        if max_dist > 0:
            normalized_coords = centered_coords / max_dist
        else:
            normalized_coords = centered_coords
            
        return normalized_coords
    
    def _random_rotation_matrix(self):
        """Generate a random 3D rotation matrix."""
        # Random rotation angles
        alpha = torch.rand(1) * 2 * math.pi  # rotation around x
        beta = torch.rand(1) * 2 * math.pi   # rotation around y
        gamma = torch.rand(1) * 2 * math.pi  # rotation around z
        
        # Rotation matrices
        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(alpha), -torch.sin(alpha)],
            [0, torch.sin(alpha), torch.cos(alpha)]
        ])
        
        Ry = torch.tensor([
            [torch.cos(beta), 0, torch.sin(beta)],
            [0, 1, 0],
            [-torch.sin(beta), 0, torch.cos(beta)]
        ])
        
        Rz = torch.tensor([
            [torch.cos(gamma), -torch.sin(gamma), 0],
            [torch.sin(gamma), torch.cos(gamma), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation
        R = Rz @ Ry @ Rx
        return R
    
    def _apply_rotation(self, coords):
        """Apply a random 3D rotation to coordinates."""
        R = self._random_rotation_matrix()
        
        # Ensure rotation matrix is on the same device as coordinates
        R = R.to(coords.device)
        
        # Reshape for matrix multiplication, then rotate and reshape back
        original_shape = coords.shape
        reshaped_coords = coords.reshape(-1, 3)
        rotated_coords = torch.matmul(reshaped_coords, R.T)
        return rotated_coords.reshape(original_shape)
    
    def _add_position_noise(self, coords):
        """Add random noise to atom positions."""
        noise = torch.randn_like(coords) * self.position_noise
        return coords + noise
    
    def _apply_atom_masking(self, coords):
        """Randomly mask some atoms by setting their coordinates to zero."""
        mask = torch.rand(coords.shape[0], coords.shape[1]) > self.atom_mask_prob
        mask = mask.unsqueeze(-1).expand_as(coords)
        return coords * mask
    
    def __call__(self, sample):
        """Apply transformations to the sample."""
        # Get coordinates
        coords = sample['coordinates']
        
        # Only apply transforms if coordinates are not all zero (i.e., not test set)
        if torch.any(coords != 0):
            if self.normalize_coords:
                coords = self._normalize_coordinates(coords)
                
            if self.random_rotation:
                coords = self._apply_rotation(coords)
                
            if self.position_noise > 0:
                coords = self._add_position_noise(coords)
                
            if self.atom_mask_prob > 0:
                coords = self._apply_atom_masking(coords)
                
            # Update the sample with transformed coordinates
            sample['coordinates'] = coords
            
        return sample

def train_model(model, train_loader, val_loader, optimizer, scheduler, device, config):
    """Train the RNA folding model."""
    # Initialize tracking variables
    best_val_rmsd = float('inf')
    best_val_tm_score = 0.0
    best_epoch = 0
    early_stop_counter = 0
    patience = config.get('patience', 10)
    num_epochs = config.get('num_epochs', 30)
    
    # Set output directory
    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(exist_ok=True)
    
    # Track loss history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmsd': [],
        'val_tm_score': []
    }
    
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        
        print(f"Epoch {epoch}/{num_epochs}")
        for batch in train_loader:
            # Move data to device
            sequences = batch['sequence'].to(device)
            coordinates = batch['coordinates'].to(device)
            lengths = batch.get('lengths', None)
            if lengths is not None:
                lengths = lengths.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            pred_coords = model(sequences, lengths)
            
            # Compute loss
            loss = model.compute_loss(pred_coords, coordinates, lengths, sequences)
            if isinstance(loss, dict):
                loss = loss['loss']  # Extract main loss if it's a dictionary
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        val_loss, val_rmsd, val_tm_score = validate_model(model, val_loader, device)
        
        # Update learning rate
        if scheduler is not None:
            # Use TM-score as the metric for the scheduler (higher is better, so negate it)
            scheduler.step(-val_tm_score if scheduler.__class__.__name__ == 'ReduceLROnPlateau' else None)
        
        # Track history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmsd'].append(val_rmsd)
        history['val_tm_score'].append(val_tm_score)
        
        # Check for model improvement
        improved = False
        
        # Check if RMSD improved (lower is better)
        if val_rmsd < best_val_rmsd:
            best_val_rmsd = val_rmsd
            improved = True
            print(f"New best RMSD: {val_rmsd:.6f}")
        
        # Check if TM-score improved (higher is better)
        # Because TM-score is the competition metric, we prioritize it
        if val_tm_score > best_val_tm_score:
            best_val_tm_score = val_tm_score
            improved = True
            print(f"New best TM-score: {val_tm_score:.6f}")
            
            # Save the best model (by TM-score)
            best_model_path = model_dir / 'best_model_tm_score.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_rmsd': val_rmsd,
                'val_tm_score': val_tm_score,
                'config': config,
            }, best_model_path)
            print(f"Best model saved to {best_model_path}")
        
        if improved:
            best_epoch = epoch
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement.")
                print(f"Best RMSD: {best_val_rmsd:.6f}, Best TM-score: {best_val_tm_score:.6f} at epoch {best_epoch}")
                break
        
        # Save checkpoint after each epoch
        checkpoint_path = model_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_rmsd': val_rmsd,
            'val_tm_score': val_tm_score,
            'history': history,
            'config': config,
        }, checkpoint_path)
    
    print(f"Training completed. Best RMSD: {best_val_rmsd:.6f}, Best TM-score: {best_val_tm_score:.6f} at epoch {best_epoch}")
    
    return model, history, best_val_rmsd, best_val_tm_score

def validate_model(model, val_loader, device):
    """Validate the RNA folding model."""
    model.eval()
    total_loss = 0.0
    total_rmsd = 0.0
    total_tm_score = 0.0
    valid_count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            sequences = batch['sequence'].to(device)
            coordinates = batch['coordinates'].to(device)
            lengths = batch.get('lengths', None)
            if lengths is not None:
                lengths = lengths.to(device)
                
            # Forward pass
            pred_coords = model(sequences, lengths)
            
            # Compute loss
            loss = model.compute_loss(pred_coords, coordinates, lengths, sequences)
            if isinstance(loss, dict):
                loss = loss['loss']  # Extract main loss if it's a dictionary
            
            # Calculate RMSD
            batch_rmsd = calculate_batch_rmsd(pred_coords, coordinates, lengths)
            
            # Calculate TM-score
            batch_tm_score = calculate_batch_tm_score(pred_coords, coordinates, lengths)
            
            # Update running totals
            total_loss += loss.item()
            total_rmsd += batch_rmsd
            total_tm_score += batch_tm_score
            valid_count += 1
    
    # Average metrics
    avg_loss = total_loss / max(valid_count, 1)
    avg_rmsd = total_rmsd / max(valid_count, 1)
    avg_tm_score = total_tm_score / max(valid_count, 1)
    
    print(f"Validation: Loss: {avg_loss:.4f}, RMSD: {avg_rmsd:.4f}, TM-score: {avg_tm_score:.4f}")
    
    return avg_loss, avg_rmsd, avg_tm_score

def calculate_batch_rmsd(pred_coords, true_coords, lengths):
    """Calculate RMSD for a batch of structures."""
    batch_size = pred_coords.shape[0]
    total_rmsd = 0.0
    valid_count = 0

    for i in range(batch_size):
        try:
            # Extract valid coordinates (non-padded)
            if lengths is not None:
                length = lengths[i].item()
                pred = pred_coords[i, :length]
                true = true_coords[i, :length]
            else:
                pred = pred_coords[i]
                true = true_coords[i]
            
            # Check for NaN values
            if torch.isnan(pred).any() or torch.isnan(true).any():
                print(f"Warning: NaN values detected in batch item {i}. Skipping.")
                continue
            
            # Reshape coordinates to (N*A, 3) format for Kabsch alignment
            pred_flat = pred.reshape(-1, 3)
            true_flat = true.reshape(-1, 3)
            
            # Apply Kabsch alignment
            pred_aligned = kabsch_align(pred_flat, true_flat)
            
            # Reshape back to original format to compute RMSD
            pred_aligned = pred_aligned.reshape(pred.shape)
            
            # Calculate RMSD
            squared_diff = torch.sum((pred_aligned - true) ** 2)
            mse = squared_diff / (pred.shape[0] * pred.shape[1] * 3)  # Normalize by all coordinates
            rmsd = torch.sqrt(mse)
            
            total_rmsd += rmsd.item()
            valid_count += 1
        except Exception as e:
            print(f"Error calculating RMSD for batch item {i}: {e}")
            print(f"Pred shape: {pred.shape}, True shape: {true.shape}")
            continue
    
    # Average RMSD across the batch
    return total_rmsd / max(valid_count, 1)

def calculate_batch_tm_score(pred_coords, true_coords, lengths):
    """Calculate TM-score for a batch of structures."""
    batch_size = pred_coords.shape[0]
    total_tm_score = 0.0
    valid_count = 0

    for i in range(batch_size):
        try:
            # Extract valid coordinates (non-padded)
            if lengths is not None:
                length = lengths[i].item()
                pred = pred_coords[i, :length]
                true = true_coords[i, :length]
            else:
                pred = pred_coords[i]
                true = true_coords[i]
            
            # Check for NaN values
            if torch.isnan(pred).any() or torch.isnan(true).any():
                print(f"Warning: NaN values detected in batch item {i}. Skipping TM-score calculation.")
                continue
                
            # Reshape coordinates if needed for tm_score calculation
            # We don't need to flatten here as tm_score handles the reshaping internally
            
            # Calculate TM-score
            tm = tm_score(pred, true)
            
            total_tm_score += tm.item()
            valid_count += 1
        except Exception as e:
            print(f"Error calculating TM-score for batch item {i}: {e}")
            print(f"Pred shape: {pred.shape}, True shape: {true.shape}")
            continue
    
    # Average TM-score across the batch
    return total_tm_score / max(valid_count, 1)

def tm_score(pred_coords, true_coords, d0=None, align=True, eps=1e-8):
    """
    Compute Template Modeling score (TM-score) between predicted and true coordinates.
    
    TM-score = max[1/Lref ∑(i=1 to Lalign) 1/(1+(di/d0)²)]
    
    Args:
        pred_coords: Predicted coordinates, shape (N, A, 3)
        true_coords: True coordinates, shape (N, A, 3)
        d0: Distance scaling factor. If None, calculated as 1.24 * (Lref - 15)^(1/3) - 1.8
        align: Whether to align the structures before computing TM-score
        eps: Small value to avoid numerical issues
        
    Returns:
        TM-score value, ranges from 0 to 1 where 1 indicates perfect match
    """
    # Reshape to (N*A, 3) format
    pred_flat = pred_coords.reshape(-1, 3)
    true_flat = true_coords.reshape(-1, 3)
    
    # Calculate Lref (reference length)
    Lref = true_flat.shape[0]
    
    # Calculate d0 (scaling factor) if not provided
    if d0 is None:
        # Standard TM-score d0 formula, may need adjustment per competition rules
        d0 = 1.24 * (Lref - 15) ** (1/3) - 1.8
        d0 = max(d0, 0.5)  # Ensure d0 is at least 0.5
    
    # Align structures if requested
    if align:
        # Use an iterative alignment process to maximize TM-score
        pred_flat = tm_align(pred_flat, true_flat, d0)
    
    # Calculate distances between corresponding points
    distances = torch.sqrt(torch.sum((pred_flat - true_flat) ** 2, dim=-1) + eps)
    
    # Calculate TM-score components
    tm_components = 1.0 / (1.0 + (distances / d0) ** 2)
    
    # Final TM-score
    tm_value = tm_components.mean()
    
    return tm_value

def tm_align(coords1, coords2, d0, max_iterations=5, eps=1e-8):
    """
    Iterative alignment to maximize TM-score.
    
    Args:
        coords1: First set of coordinates, shape (N, 3)
        coords2: Second set of coordinates, shape (N, 3)
        d0: Distance scaling factor for TM-score calculation
        max_iterations: Maximum number of alignment iterations
        eps: Small value to avoid numerical issues
        
    Returns:
        coords1_aligned: First coordinates aligned to maximize TM-score
    """
    # Initial alignment using Kabsch
    coords1_aligned = kabsch_align(coords1, coords2)
    
    # Initialize weights uniformly
    weights = torch.ones(coords1.shape[0], device=coords1.device)
    
    # Iterative refinement
    for _ in range(max_iterations):
        # Calculate distances
        distances = torch.sqrt(torch.sum((coords1_aligned - coords2) ** 2, dim=-1) + eps)
        
        # Update weights based on TM-score formula - giving higher weight to better aligned regions
        weights = 1.0 / (1.0 + (distances / d0) ** 2)
        
        # Use the weights in a weighted Kabsch alignment
        # Step 1: Calculate weighted centroids
        weight_sum = weights.sum()
        weighted_coords1 = weights.unsqueeze(-1) * coords1_aligned
        weighted_coords2 = weights.unsqueeze(-1) * coords2
        
        centroid1 = weighted_coords1.sum(dim=0) / weight_sum
        centroid2 = weighted_coords2.sum(dim=0) / weight_sum
        
        # Step 2: Center coordinates based on weighted centroids
        centered_coords1 = coords1_aligned - centroid1
        centered_coords2 = coords2 - centroid2
        
        # Step 3: Calculate weighted covariance matrix
        weighted_centered1 = weights.unsqueeze(-1) * centered_coords1
        covariance = torch.matmul(weighted_centered1.T, centered_coords2)
        
        # Step 4: SVD to find optimal rotation
        u, _, v = torch.svd(covariance)
        
        # Step 5: Calculate rotation matrix
        rotation = torch.matmul(v, u.T)
        
        # Handle reflection case
        if torch.det(rotation) < 0:
            v_adjusted = v.clone()
            v_adjusted[:, -1] = -v_adjusted[:, -1]
            rotation = torch.matmul(v_adjusted, u.T)
        
        # Step 6: Apply rotation and translation
        coords1_aligned = torch.matmul(centered_coords1, rotation) + centroid2
    
    return coords1_aligned

def kabsch_align(coords1, coords2):
    """
    Align two sets of coordinates using the Kabsch algorithm.
    
    Args:
        coords1: First set of coordinates, shape (N, 3)
        coords2: Second set of coordinates, shape (N, 3)
        
    Returns:
        coords1_aligned: First coordinates aligned to coords2
    """
    try:
        # Check input dimensions
        if coords1.shape != coords2.shape:
            print(f"WARNING: Shape mismatch in kabsch_align - coords1: {coords1.shape}, coords2: {coords2.shape}")
            # Try to handle mismatched dimensions if possible
            min_length = min(coords1.shape[0], coords2.shape[0])
            coords1 = coords1[:min_length]
            coords2 = coords2[:min_length]
            print(f"Using first {min_length} coordinates for alignment")
            
        # Center the coordinates
        coords1_mean = coords1.mean(dim=0, keepdim=True)
        coords2_mean = coords2.mean(dim=0, keepdim=True)
        
        coords1_centered = coords1 - coords1_mean
        coords2_centered = coords2 - coords2_mean
        
        # Calculate the covariance matrix
        covariance = torch.matmul(coords1_centered.T, coords2_centered)
        
        # Perform SVD
        try:
            u, _, v = torch.svd(covariance)
        except Exception as e:
            print(f"SVD failed: {e}")
            print(f"Covariance shape: {covariance.shape}")
            print(f"Attempting fallback to CPU for SVD")
            # Fallback to CPU if GPU SVD fails
            u, _, v = torch.svd(covariance.cpu())
            u, v = u.to(coords1.device), v.to(coords1.device)
        
        # Calculate the rotation matrix
        rotation = torch.matmul(v, u.T)
        
        # Check for reflection
        if torch.det(rotation) < 0:
            v_adjusted = v.clone()
            v_adjusted[:, -1] = -v_adjusted[:, -1]
            rotation = torch.matmul(v_adjusted, u.T)
        
        # Apply rotation and translation
        coords1_aligned = torch.matmul(coords1_centered, rotation) + coords2_mean
        
        return coords1_aligned
    except Exception as e:
        print(f"ERROR in kabsch_align: {e}")
        print(f"coords1 shape: {coords1.shape}, coords2 shape: {coords2.shape}")
        print(f"coords1 device: {coords1.device}, coords2 device: {coords2.device}")
        # Return unaligned coordinates as fallback
        return coords1

def generate_predictions(model, test_loader, device):
    """Generate predictions for the test set."""
    model.eval()
    all_predictions = {}
    
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            sequences = batch['sequence'].to(device)
            lengths = batch['lengths'].to(device)
            target_ids = batch['target_id']
            
            # Forward pass
            pred_coords = model(sequences, lengths)
            
            # Store predictions by target ID
            for i, target_id in enumerate(target_ids):
                seq_len = lengths[i].item()
                # Extract predictions for this sequence (non-padded part)
                coords = pred_coords[i, :seq_len].cpu().numpy()
                all_predictions[target_id] = coords
    
    return all_predictions

def create_submission_file(predictions, output_file):
    """Create a CSV submission file from predictions."""
    # Read sample submission file to get the expected format
    sample_submission = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
    
    # Initialize a dataframe for predictions
    submission = sample_submission.copy()
    
    # Populate with predictions
    for target_id, coords in predictions.items():
        # Flatten prediction to match sample submission format
        flattened_coords = coords.reshape(-1)
        
        # Find rows for this target
        target_rows = submission['target_id'] == target_id
        
        # Ensure the prediction length matches the expected length
        expected_length = sum(target_rows)
        if flattened_coords.shape[0] != expected_length * 15:  # 5 atoms * 3 coords
            print(f"Warning: Prediction shape mismatch for {target_id}.")
            # Truncate or pad prediction if necessary
            if flattened_coords.shape[0] > expected_length * 15:
                flattened_coords = flattened_coords[:expected_length * 15]
            else:
                pad_length = expected_length * 15 - flattened_coords.shape[0]
                flattened_coords = np.pad(flattened_coords, (0, pad_length))
        
        # Assign predictions to corresponding columns
        for i, col in enumerate(sample_submission.columns[1:]):  # Skip target_id column
            if i < flattened_coords.shape[0]:
                submission.loc[target_rows, col] = flattened_coords[i]
    
    # Save to file
    submission.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
    return submission

def load_config(config_path=None):
    """
    Load configuration from a YAML file.
    
    Parameters:
    -----------
    config_path : str, default=None
        Path to the YAML configuration file. If None, uses default configuration.
        
    Returns:
    --------
    dict
        Configuration dictionary
    """
    # Default configuration
    default_config = {
        'data': {
            'train_sequence_file': 'datasets/stanford-rna-3d-folding/train_sequences.csv',
            'train_labels_file': 'datasets/stanford-rna-3d-folding/train_labels.csv',
            'val_sequence_file': 'datasets/stanford-rna-3d-folding/validation_sequences.csv',
            'val_labels_file': 'datasets/stanford-rna-3d-folding/validation_labels.csv',
            'test_sequence_file': 'datasets/stanford-rna-3d-folding/test_sequences.csv',
            'batch_size': 32,
            'num_workers': 0,  # No multiprocessing for Kaggle
            'sequence_bucketing': True,
        },
        'model': {
            'name': 'RNAFoldingModel',
            'params': {
                'vocab_size': 5,
                'embedding_dim': 128,
                'hidden_dim': 256,
                'num_layers': 4,
                'num_heads': 8,
                'dropout': 0.1,
                'num_atoms': 5,
                'coord_dims': 3,
                'max_seq_len': 500,
                'use_rna_constraints': False,
            }
        },
        'training': {
            'num_epochs': 20,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'lr_scheduler': 'plateau',
            'lr_scheduler_params': {
                'factor': 0.5,
                'patience': 3,
                'verbose': True
            },
            'gradient_clip_val': 1.0,
            'early_stopping': True,
            'patience': 5,
            'checkpoint_dir': 'models/stanford-rna-3d-folding',
        },
        'transforms': {
            'train': {
                'normalize_coords': True,
                'random_rotation': True,
                'position_noise': 0.05,
                'atom_mask_prob': 0.1
            },
            'val': {
                'normalize_coords': True,
                'random_rotation': False,
                'position_noise': 0.0,
                'atom_mask_prob': 0.0
            }
        },
        'misc': {
            'seed': 42,
        }
    }
    
    # If no config path provided, use default configuration
    if not config_path:
        print("Using default configuration")
        return default_config
    
    # Check if config file exists
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Config file {config_path} not found. Using default configuration.")
        return default_config
    
    # Load configuration from YAML file
    try:
        print(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            print("Configuration loaded successfully")
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default configuration")
        return default_config

# ==============================
# 4. Main Script
# ==============================

def main(config_path=None):
    """
    Main function to run the RNA folding model.
    
    Parameters:
    -----------
    config_path : str, default=None
        Path to the YAML configuration file
    """
    print("\nStarting Stanford RNA 3D Folding prediction...")
    
    # Load configuration
    config = load_config(config_path)
    
    # Print detailed device information
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Current GPU memory usage: {torch.cuda.memory_allocated(device) / 1e9:.1f} GB")
    
    # Set random seed for reproducibility
    seed = config['misc']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")
    
    # Get model configuration
    model_config = config['model']['params']
    
    # Training parameters
    train_params = {
        'batch_size': config['data']['batch_size'],
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'num_epochs': config['training']['num_epochs'],
        'patience': config['training']['patience'],
    }
    
    # Create model and explicitly move to device
    model = RNAFoldingModel(**model_config)
    model = model.to(device)
    
    # Verify model is on the correct device
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Create transforms
    train_transform = RNADataTransform(**config['transforms']['train'])
    val_transform = RNADataTransform(**config['transforms']['val'])
    
    # Create datasets
    train_dataset = StanfordRNADataset(
        data_dir=DATA_DIR,
        split="train",
        transform=train_transform
    )
    
    val_dataset = StanfordRNADataset(
        data_dir=DATA_DIR,
        split="validation",
        transform=val_transform
    )
    
    test_dataset = StanfordRNADataset(
        data_dir=DATA_DIR,
        split="test",
        transform=val_transform
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_params['batch_size'],
        shuffle=True,
        collate_fn=rna_collate_fn,
        num_workers=config['data']['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_params['batch_size'],
        shuffle=False,
        collate_fn=rna_collate_fn,
        num_workers=config['data']['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_params['batch_size'],
        shuffle=False,
        collate_fn=rna_collate_fn,
        num_workers=config['data']['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Set up optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_params['learning_rate'],
        weight_decay=train_params['weight_decay']
    )
    
    # Create scheduler based on configuration
    scheduler_type = config['training']['lr_scheduler'].lower()
    scheduler_params = config['training']['lr_scheduler_params']
    
    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            **scheduler_params
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs'],
            **scheduler_params
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            **scheduler_params
        )
    else:
        print(f"Warning: Unknown scheduler type '{scheduler_type}'. Using ReduceLROnPlateau.")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
    
    # Train model
    model, history, best_val_rmsd, best_val_tm_score = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=train_params
    )
    
    # Load best model for prediction
    checkpoint = torch.load(os.path.join(MODEL_DIR, 'best_model_tm_score.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model with validation RMSD: {best_val_rmsd:.6f} and TM-score: {best_val_tm_score:.6f}")
    
    # Generate predictions
    print("\nGenerating predictions for test set...")
    predictions = generate_predictions(model, test_loader, device)
    
    # Create submission file
    submission_path = os.path.join(SUBMISSION_DIR, 'submission.csv')
    submission = create_submission_file(predictions, submission_path)
    
    print("\nTraining and prediction complete!")
    print(f"Best validation RMSD: {best_val_rmsd:.6f}")
    print(f"Best validation TM-score: {best_val_tm_score:.6f}")
    print(f"Submission file created at: {submission_path}")
    
    return {
        'model': model,
        'history': history,
        'best_val_rmsd': best_val_rmsd,
        'best_val_tm_score': best_val_tm_score,
        'predictions': predictions,
        'submission': submission
    }

if __name__ == "__main__":
    # Check if a config file path is provided as a command-line argument
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        if not os.path.exists(config_path):
            print(f"Warning: Config file {config_path} not found. Using default configuration.")
            config_path = None
    
    # Look for biophysics_config.yaml in the current directory and configs directory
    if config_path is None:
        default_paths = [
            'biophysics_config.yaml',
            'configs/biophysics_config.yaml',
            os.path.join(os.path.dirname(__file__), 'configs/biophysics_config.yaml')
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                config_path = path
                print(f"Using configuration file: {config_path}")
                break
    
    # Run the main function with the specified or found config file
    result = main(config_path)
    