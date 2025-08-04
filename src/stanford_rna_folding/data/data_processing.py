"""
Data processing utilities for Stanford RNA 3D Structure Prediction competition.
"""

import os
import re  # Added regex module
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class StanfordRNADataset(Dataset):
    """Dataset class for Stanford RNA 3D Structure Prediction competition."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[callable] = None,
    ):
        """
        Initialize the Stanford RNA Structure dataset.
        
        Args:
            data_dir: Path to the directory containing the competition data files
            split: One of ["train", "validation", "test"]
            transform: Optional transform to be applied to the data
        """
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
            # This handles cases where a sequence might be missing coordinates
            valid_targets = list(self.processed_labels.keys())
            self.sequences_df = self.sequences_df[self.sequences_df['target_id'].isin(valid_targets)].reset_index(drop=True)

        # Create mapping of nucleotides to integers
        self.nucleotide_to_idx = {"A": 0, "U": 1, "G": 2, "C": 3, "N": 4}  # N for unknown/padding
        
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
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"Error loading labels file {filepath}: {e}")
            raise
    
    def _extract_base_target_id(self, complex_id: str) -> str:
        """Extracts the base target ID (e.g., '1SCL_A') from the full ID (e.g., '1SCL_A_1')."""
        # Use regex to remove the trailing '_<number>'
        match = re.match(r"(.+)_\d+$", complex_id)
        if match:
            return match.group(1)
        else:
            # Handle cases where ID format might be different
            print(f"Warning: Could not extract base target ID from {complex_id}. Using full ID.")
            return complex_id
            
    def _process_labels(self, labels_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Processes the raw labels dataframe to group coordinates by target ID."""
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
            if len(missing_cols) > 10:  # If many columns are missing, likely only first atom coords are available
                print("Only atom 1 coordinates seem to be available. Proceeding with those only.")
                self.coord_cols = [col for col in self.coord_cols if col not in missing_cols]
            else:
                raise ValueError(f"Missing coordinate columns: {missing_cols}")
            
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
                # Replace NaNs with zeros (simple strategy)
                coords_np = np.nan_to_num(coords_np)

            # Reshape: (SequenceLength, NumAtoms, NumCoords)
            # For example, if we have 15 coordinate columns (5 atoms x 3 coords), 
            # reshape to (sequence_length, 5, 3)
            try:
                if len(self.coord_cols) == 15:  # Full 5 atoms
                    coords_reshaped = coords_np.reshape(-1, 5, 3)
                elif len(self.coord_cols) == 3:  # Only first atom
                    coords_reshaped = coords_np.reshape(-1, 1, 3)
                    # Might need to duplicate for model compatibility
                    coords_reshaped = np.repeat(coords_reshaped, 5, axis=1)
                else:
                    # Handle other cases (partial atoms)
                    num_atoms = len(self.coord_cols) // 3
                    coords_reshaped = coords_np.reshape(-1, num_atoms, 3)
            except ValueError as e:
                print(f"Error reshaping coordinates for target {target_id}. Error: {e}")
                continue # Skip this target if reshaping fails
                
            # Convert to tensor
            processed[target_id] = torch.tensor(coords_reshaped, dtype=torch.float32)
            
        return processed

    def _encode_sequence(self, sequence: str) -> torch.Tensor:
        """
        Convert RNA sequence to tensor of integers.
        
        Args:
            sequence: RNA sequence string
            
        Returns:
            torch.Tensor: Encoded sequence as long tensor
        """
        # Handle non-standard nucleotides by mapping to 'N'
        return torch.tensor([self.nucleotide_to_idx.get(nt, self.nucleotide_to_idx["N"]) 
                           for nt in sequence], dtype=torch.long)
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences_df)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
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
        if self.processed_labels is not None:
            if target_id in self.processed_labels:
                coords_tensor = self.processed_labels[target_id]
                
                # Validate sequence length against coordinate length
                if sequence_length != coords_tensor.shape[0]:
                    print(f"Warning: Sequence length ({sequence_length}) mismatch with coordinate length ({coords_tensor.shape[0]}) for {target_id}.")
                    
                    # Handle mismatch by truncating or padding
                    if coords_tensor.shape[0] > sequence_length:
                        # Truncate coordinates if longer
                        coords_tensor = coords_tensor[:sequence_length]
                    else:
                        # Pad coordinates if shorter
                        pad_size = sequence_length - coords_tensor.shape[0]
                        
                        # Get appropriate shape and dtype
                        atom_dims = coords_tensor.shape[1:]
                        padding = torch.zeros((pad_size, *atom_dims), dtype=coords_tensor.dtype)
                        
                        # Concatenate original tensor with padding
                        coords_tensor = torch.cat([coords_tensor, padding], dim=0)
                
                output["coordinates"] = coords_tensor
            else:
                print(f"Warning: No processed labels found for target {target_id} in __getitem__.")
                # Create zero-filled tensor for missing coordinates
                atom_shape = (5, 3)  # Default shape if we don't know
                if self.processed_labels and len(self.processed_labels) > 0:
                    # Get shape from first processed label
                    first_key = next(iter(self.processed_labels))
                    atom_shape = self.processed_labels[first_key].shape[1:]
                    
                output["coordinates"] = torch.zeros((sequence_length, *atom_shape), dtype=torch.float32)
        else:
            # For test set, add empty coordinates tensor
            output["coordinates"] = torch.zeros((sequence_length, 5, 3), dtype=torch.float32)

        # Apply transforms if specified
        if self.transform is not None:
            output = self.transform(output)
            
        return output

def rna_collate_fn(batch: List[Dict[str, Union[str, torch.Tensor]]], pad_value_sequence: int = 4, pad_value_coords: float = 0.0) -> Dict[str, Union[List[str], torch.Tensor]]:
    """
    Collate function for DataLoader to handle variable length sequences and coordinates.

    Pads sequences and coordinates to the maximum length in the batch.

    Args:
        batch: A list of dictionaries, where each dictionary is an output 
               from StanfordRNADataset.__getitem__.
        pad_value_sequence: The integer value to use for padding sequences (default: 4, index of 'N').
        pad_value_coords: The float value to use for padding coordinates (default: 0.0).

    Returns:
        A dictionary containing batched and padded tensors:
        - 'sequence': Padded sequence tensor (BatchSize, MaxLength)
        - 'coordinates': Padded coordinates tensor (BatchSize, MaxLength, 5, 3)
        - 'target_id': List of target IDs for the batch.
        - 'lengths': Tensor containing the original lengths of sequences in the batch.
    """
    # Separate components
    sequences = [item['sequence'] for item in batch]
    coordinates = [item['coordinates'] for item in batch]
    target_ids = [item['target_id'] for item in batch]
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)

    # Pad sequences
    # pad_sequence expects batch_first=False by default, but our sequences are (L)
    # We want the output to be (BatchSize, MaxLength), so set batch_first=True
    padded_sequences = pad_sequence(
        sequences, 
        batch_first=True, 
        padding_value=pad_value_sequence
    )

    # Pad coordinates
    # Coordinates are (L, 5, 3). We need to pad along the L dimension.
    # pad_sequence works on the first dimension by default if batch_first=False.
    # If we set batch_first=True, it pads dim 1. 
    # Let's handle padding manually for clarity or use pad_sequence carefully.
    
    # Manual padding approach:
    max_len = lengths.max().item()
    batch_size = len(batch)
    num_atoms = 5 # Assuming 5 atoms
    num_dims = 3 # Assuming 3D coordinates
    
    # Initialize padded coordinates tensor
    # Check if coordinates exist (might be test set or samples skipped due to errors)
    if coordinates[0] is not None and coordinates[0].numel() > 0:
        # Use the shape and dtype from the first valid coordinate tensor
        num_atoms = coordinates[0].shape[1]
        num_dims = coordinates[0].shape[2]
        padded_coordinates = torch.full(
            (batch_size, max_len, num_atoms, num_dims), 
            pad_value_coords, 
            dtype=coordinates[0].dtype
        )
        # Fill with actual coordinate data
        for i, coords in enumerate(coordinates):
            seq_len = coords.shape[0]
            if seq_len > 0: # Ensure coords are not empty
                 # Make sure coords length doesn't exceed max_len (can happen if __getitem__ didn't truncate)
                current_len = min(seq_len, max_len)
                padded_coordinates[i, :current_len, :, :] = coords[:current_len]
    else:
        # Handle cases where coordinates are missing (e.g., test set) or empty
        # Create a tensor of appropriate shape filled with padding value
        # We need dtype, let's default to float32
        padded_coordinates = torch.full(
            (batch_size, max_len, num_atoms, num_dims), 
            pad_value_coords, 
            dtype=torch.float32 
        )

    return {
        'sequence': padded_sequences,
        'coordinates': padded_coordinates,
        'target_id': target_ids, 
        'lengths': lengths
    }