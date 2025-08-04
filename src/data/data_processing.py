"""
Data processing utilities for RNA structure prediction.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RNAStructureDataset(Dataset):
    """Dataset class for RNA structure prediction."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[callable] = None,
    ):
        """
        Initialize the RNA Structure dataset.
        
        Args:
            data_dir: Path to the directory containing the data files
            split: One of ["train", "validation", "test"]
            transform: Optional transform to be applied to the data
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Load and process data
        self.sequences_df = self._load_sequences()
        self.labels_df = self._load_labels() if split != "test" else None
        
        # Create mapping of nucleotides to integers
        self.nucleotide_to_idx = {"A": 0, "U": 1, "G": 2, "C": 3, "N": 4}
        
    def _load_sequences(self) -> pd.DataFrame:
        """Load sequence data from CSV file."""
        filename = f"{self.split}_sequences.csv"
        filepath = os.path.join(self.data_dir, filename)
        return pd.read_csv(filepath)
    
    def _load_labels(self) -> Optional[pd.DataFrame]:
        """Load label data from CSV file if available."""
        if self.split == "test":
            return None
            
        filename = f"{self.split}_labels.csv"
        filepath = os.path.join(self.data_dir, filename)
        return pd.read_csv(filepath)
    
    def _encode_sequence(self, sequence: str) -> torch.Tensor:
        """
        Convert RNA sequence to tensor of integers.
        
        Args:
            sequence: RNA sequence string
            
        Returns:
            torch.Tensor: Encoded sequence as long tensor
        """
        return torch.tensor([self.nucleotide_to_idx[nt] for nt in sequence], dtype=torch.long)
    
    def _get_coordinates(self, target_id: str) -> torch.Tensor:
        """
        Get 3D coordinates for a target from labels dataframe.
        
        Args:
            target_id: ID of the target sequence
            
        Returns:
            torch.Tensor: 3D coordinates tensor of shape (L, 3) where L is sequence length
        """
        if self.labels_df is None:
            raise ValueError("Labels not available for test split")
            
        # Get coordinates for this target
        target_coords = self.labels_df[self.labels_df["target_id"] == target_id]
        
        # Convert coordinates to tensor
        coords = torch.tensor(
            target_coords[["x", "y", "z"]].values,
            dtype=torch.float32
        )
        
        return coords
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example from the dataset.
        
        Args:
            idx: Index of the example to get
            
        Returns:
            Dict containing:
                - sequence: Encoded sequence tensor
                - coordinates: 3D coordinates tensor (if available)
                - target_id: ID of the target
        """
        # Get sequence data
        row = self.sequences_df.iloc[idx]
        target_id = row["target_id"]
        sequence = row["sequence"]
        
        # Encode sequence
        sequence_tensor = self._encode_sequence(sequence)
        
        # Create output dictionary
        output = {
            "sequence": sequence_tensor,
            "target_id": target_id
        }
        
        # Add coordinates if available
        if self.split != "test":
            coords = self._get_coordinates(target_id)
            output["coordinates"] = coords
            
        # Apply transforms if specified
        if self.transform is not None:
            output = self.transform(output)
            
        return output 