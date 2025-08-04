"""
Data transforms for Stanford RNA 3D Structure Prediction competition.
"""

from typing import Dict, Optional, Union

import numpy as np
import torch


class RNADataTransform:
    """Transform class for RNA structure data."""
    
    def __init__(
        self,
        normalize_coords: bool = True,
        random_rotation: bool = False,
        random_noise: float = 0.0,
        jitter_strength: float = 0.0,
        atom_mask_prob: float = 0.0,
    ):
        """
        Initialize the transform.
        
        Args:
            normalize_coords: Whether to normalize coordinates to zero mean and unit variance
            random_rotation: Whether to apply random 3D rotations during training
            random_noise: Standard deviation of Gaussian noise to add to coordinates (0.0 = no noise)
            jitter_strength: Amount of position-dependent jitter to apply (useful for augmentation)
            atom_mask_prob: Probability of masking individual atoms during training (0.0 = no masking)
        """
        self.normalize_coords = normalize_coords
        self.random_rotation = random_rotation
        self.random_noise = random_noise
        self.jitter_strength = jitter_strength
        self.atom_mask_prob = atom_mask_prob
        
    def _normalize_coordinates(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Normalize coordinates to zero mean and unit variance.
        
        Args:
            coords: Coordinate tensor of shape (L, A, 3) where:
                   L = sequence length
                   A = number of atoms per residue (5 for this competition)
                   3 = xyz coordinates
            
        Returns:
            Normalized coordinates
        """
        # Flatten atoms dimension for centering/scaling across all atoms
        orig_shape = coords.shape
        coords_flat = coords.reshape(orig_shape[0], -1)
        
        # Center coordinates - compute mean across all dims except batch
        mean = coords_flat.mean(dim=0, keepdim=True)
        centered = coords_flat - mean
        
        # Scale to unit variance
        std = centered.std(dim=0, keepdim=True)
        std = torch.where(std > 1e-8, std, torch.ones_like(std))  # Avoid division by zero
        normalized = centered / std
        
        # Reshape back to original
        return normalized.reshape(orig_shape)
        
    def _random_rotation_matrix(self) -> torch.Tensor:
        """Generate a random 3D rotation matrix."""
        # Generate random rotation angles
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        z = np.random.uniform(0, 2 * np.pi)
        
        # Create rotation matrices for each axis
        Rx = torch.tensor([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ], dtype=torch.float32)
        
        Ry = torch.tensor([
            [np.cos(phi), 0, np.sin(phi)],
            [0, 1, 0],
            [-np.sin(phi), 0, np.cos(phi)]
        ], dtype=torch.float32)
        
        Rz = torch.tensor([
            [np.cos(z), -np.sin(z), 0],
            [np.sin(z), np.cos(z), 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        # Combine rotations
        R = Rz @ Ry @ Rx
        return R
        
    def _apply_rotation(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply random 3D rotation to coordinates.
        
        Args:
            coords: Coordinate tensor of shape (L, A, 3)
        
        Returns:
            Rotated coordinates with the same shape
        """
        R = self._random_rotation_matrix()
        
        # Apply rotation - need to reshape for batched matrix multiplication
        orig_shape = coords.shape
        # Reshape to (L*A, 3) for batched matrix multiplication
        coords_flat = coords.reshape(-1, 3)
        # Apply rotation
        rotated = coords_flat @ R
        # Reshape back
        return rotated.reshape(orig_shape)
        
    def _add_noise(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to coordinates.
        
        Args:
            coords: Coordinate tensor of shape (L, A, 3)
            
        Returns:
            Coordinates with added noise
        """
        if self.random_noise > 0:
            noise = torch.randn_like(coords) * self.random_noise
            coords = coords + noise
        return coords
        
    def _apply_jitter(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply position-dependent jitter to coordinates.
        
        This type of augmentation can help the model learn to be
        robust to small local variations in the structure.
        
        Args:
            coords: Coordinate tensor of shape (L, A, 3)
            
        Returns:
            Jittered coordinates with the same shape
        """
        if self.jitter_strength <= 0:
            return coords
            
        # Create position-dependent jitter - stronger at the sequence ends
        seq_len = coords.shape[0]
        if seq_len <= 1:
            return coords
            
        # Create linear effect that's stronger at ends
        position_factor = torch.abs(torch.linspace(-1, 1, seq_len))
        position_factor = position_factor.unsqueeze(-1).unsqueeze(-1)  # Shape (L, 1, 1)
        
        # Apply jitter
        jitter = torch.randn_like(coords) * position_factor * self.jitter_strength
        return coords + jitter
        
    def _apply_atom_masking(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Randomly mask (zero out) some atom coordinates.
        
        This helps the model learn to be robust to missing atom data.
        
        Args:
            coords: Coordinate tensor of shape (L, A, 3)
            
        Returns:
            Coordinates with some atoms masked
        """
        if self.atom_mask_prob <= 0:
            return coords
            
        # Create random mask - each atom has atom_mask_prob chance of being masked
        mask = torch.rand(coords.shape[0], coords.shape[1], 1) >= self.atom_mask_prob
        mask = mask.to(coords.device).float()  # Convert to float for multiplication
        
        # Expand mask for xyz coords
        mask = mask.expand(-1, -1, 3)
        
        # Apply mask - zeroes out masked atoms
        return coords * mask
        
    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply transforms to the data.
        
        Args:
            data: Dictionary containing:
                - sequence: Encoded sequence tensor
                - coordinates: 3D coordinates tensor (L, A, 3)
                - target_id: ID of the target
                
        Returns:
            Transformed data dictionary
        """
        # Only transform if we have coordinates
        if "coordinates" in data and data["coordinates"].numel() > 0:
            coords = data["coordinates"]
            
            # Apply normalization
            if self.normalize_coords:
                coords = self._normalize_coordinates(coords)
                
            # Training-time augmentations - only apply during training
            if self.training if hasattr(self, 'training') else True:
                # Apply random rotation
                if self.random_rotation:
                    coords = self._apply_rotation(coords)
                
                # Add gaussian noise
                coords = self._add_noise(coords)
                
                # Apply position-dependent jitter
                coords = self._apply_jitter(coords)
                
                # Apply atom masking
                coords = self._apply_atom_masking(coords)
            
            # Update coordinates in data dictionary
            data["coordinates"] = coords
            
        return data 