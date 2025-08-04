"""
Data processing utilities for RNA structure prediction.
"""

import os
import numpy as np
import pandas as pd
from Bio import PDB
from Bio.PDB import *
import torch
from torch.utils.data import Dataset, DataLoader

class RNAStructureDataset(Dataset):
    """Dataset class for RNA structure data."""
    
    def __init__(self, data_dir, split='train', transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Directory containing the data
            split (str): 'train' or 'test'
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Load sequence and structure data
        self.data = self._load_data()
        
    def _load_data(self):
        """Load RNA sequences and structures."""
        data_path = os.path.join(self.data_dir, f"{self.split}_data.csv")
        return pd.read_csv(data_path)
    
    def __len__(self):
        """Return the size of dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            dict: Sample containing sequence and structure information
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get sequence and structure
        sample = self.data.iloc[idx]
        
        # Convert sequence to numeric
        sequence = self._encode_sequence(sample['sequence'])
        
        # Get 3D coordinates if available
        if self.split == 'train':
            coords = self._get_coordinates(sample['structure_file'])
        else:
            coords = None
            
        item = {
            'sequence': sequence,
            'coordinates': coords,
            'id': sample['id']
        }
        
        if self.transform:
            item = self.transform(item)
            
        return item
    
    def _encode_sequence(self, sequence):
        """
        Convert RNA sequence to numeric encoding.
        
        Args:
            sequence (str): RNA sequence
            
        Returns:
            torch.Tensor: Encoded sequence
        """
        # Nucleotide to index mapping
        nuc_dict = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
        
        # Convert to numeric
        encoded = [nuc_dict[nuc] for nuc in sequence]
        return torch.tensor(encoded, dtype=torch.long)
    
    def _get_coordinates(self, structure_file):
        """
        Extract 3D coordinates from structure file.
        
        Args:
            structure_file (str): Path to structure file
            
        Returns:
            torch.Tensor: 3D coordinates
        """
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('RNA', 
                                      os.path.join(self.data_dir, 'structures', structure_file))
        
        # Extract coordinates
        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        coords.append(atom.get_coord())
                        
        return torch.tensor(coords, dtype=torch.float)

class RNADataTransform:
    """Transformations for RNA structure data."""
    
    def __init__(self, normalize=True, augment=False):
        """
        Initialize transforms.
        
        Args:
            normalize (bool): Whether to normalize coordinates
            augment (bool): Whether to apply data augmentation
        """
        self.normalize = normalize
        self.augment = augment
        
    def __call__(self, sample):
        """
        Apply the transforms.
        
        Args:
            sample (dict): Input sample
            
        Returns:
            dict: Transformed sample
        """
        if self.normalize and sample['coordinates'] is not None:
            sample['coordinates'] = self._normalize_coords(sample['coordinates'])
            
        if self.augment and sample['coordinates'] is not None:
            sample['coordinates'] = self._augment_coords(sample['coordinates'])
            
        return sample
    
    def _normalize_coords(self, coords):
        """Normalize coordinates to zero mean and unit variance."""
        mean = coords.mean(dim=0)
        std = coords.std(dim=0)
        return (coords - mean) / (std + 1e-7)
    
    def _augment_coords(self, coords):
        """Apply random rotation and small perturbations."""
        # Random rotation matrix
        theta = torch.rand(1) * 2 * np.pi
        rotation_matrix = torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1]
        ])
        
        # Apply rotation
        coords = torch.matmul(coords, rotation_matrix)
        
        # Add small random noise
        noise = torch.randn_like(coords) * 0.01
        coords = coords + noise
        
        return coords

def create_data_loaders(data_dir, batch_size=32, num_workers=4):
    """
    Create data loaders for training and validation.
    
    Args:
        data_dir (str): Data directory
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        
    Returns:
        tuple: Training and validation data loaders
    """
    # Create transforms
    train_transform = RNADataTransform(normalize=True, augment=True)
    val_transform = RNADataTransform(normalize=True, augment=False)
    
    # Create datasets
    train_dataset = RNAStructureDataset(
        data_dir=data_dir,
        split='train',
        transform=train_transform
    )
    
    val_dataset = RNAStructureDataset(
        data_dir=data_dir,
        split='val',
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 