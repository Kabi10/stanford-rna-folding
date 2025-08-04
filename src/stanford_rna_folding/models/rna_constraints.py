"""
RNA-specific biological constraints for 3D structure prediction.

This module provides specialized constraints based on RNA biology, including:
- Watson-Crick and wobble base pairing rules
- RNA motif detection and constraints
- Secondary structure consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

class BaseRNAConstraint(nn.Module):
    """Base class for RNA constraints."""
    
    def __init__(self, weight: float = 1.0):
        """
        Initialize the constraint.
        
        Args:
            weight: Weight of this constraint in the loss function
        """
        super().__init__()
        self.weight = weight
    
    def forward(self, coords: torch.Tensor, sequence: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the constraint loss.
        
        Args:
            coords: Coordinate tensor of shape (batch_size, seq_len, num_atoms, 3)
            sequence: Sequence tensor of shape (batch_size, seq_len)
            mask: Mask tensor of shape (batch_size, seq_len) for valid positions
            
        Returns:
            Loss value for this constraint
        """
        raise NotImplementedError("Subclasses must implement forward method")


class WatsonCrickConstraint(BaseRNAConstraint):
    """
    Constraint enforcing Watson-Crick base pairing geometry.
    
    A-U and G-C base pairs should have specific distances and orientations.
    """
    
    def __init__(self, weight: float = 1.0, pair_distance: float = 10.0, tolerance: float = 2.0):
        """
        Initialize the Watson-Crick constraint.
        
        Args:
            weight: Weight of this constraint in the loss function
            pair_distance: Ideal distance between paired nucleotides (in Angstroms)
            tolerance: Acceptable deviation from ideal distance
        """
        super().__init__(weight)
        self.pair_distance = pair_distance
        self.tolerance = tolerance
        
        # Define Watson-Crick pairing rules (assuming 0=A, 1=U, 2=G, 3=C, 4=N)
        # 1 indicates a valid base pair, 0 indicates no pairing
        self.register_buffer('pairing_matrix', torch.tensor([
            [0, 1, 0, 0, 0],  # A pairs with U
            [1, 0, 0, 0, 0],  # U pairs with A
            [0, 0, 0, 1, 0],  # G pairs with C
            [0, 0, 1, 0, 0],  # C pairs with G
            [0, 0, 0, 0, 0],  # N doesn't pair
        ], dtype=torch.float32))
        
        # Also allow G-U wobble pairs
        self.register_buffer('wobble_matrix', torch.tensor([
            [0, 0, 0, 0, 0],  # A doesn't wobble
            [0, 0, 1, 0, 0],  # U wobbles with G
            [0, 1, 0, 0, 0],  # G wobbles with U
            [0, 0, 0, 0, 0],  # C doesn't wobble
            [0, 0, 0, 0, 0],  # N doesn't wobble
        ], dtype=torch.float32))
    
    def forward(self, coords: torch.Tensor, sequence: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the Watson-Crick constraint loss.
        
        Args:
            coords: Coordinate tensor of shape (batch_size, seq_len, num_atoms, 3)
            sequence: Sequence tensor of shape (batch_size, seq_len)
            mask: Mask tensor of shape (batch_size, seq_len) for valid positions
            
        Returns:
            Loss value for Watson-Crick base pairing
        """
        batch_size, seq_len, num_atoms, _ = coords.shape
        
        if mask is None:
            mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=coords.device)
        
        # We'll use the central atom (usually atom 2 or 3) as the reference point for base pairing
        reference_atom_idx = 2
        reference_coords = coords[:, :, reference_atom_idx, :]  # (batch_size, seq_len, 3)
        
        # Calculate pairwise distances between all nucleotides
        # Reshape to (batch_size, seq_len, 1, 3) and (batch_size, 1, seq_len, 3)
        a = reference_coords.unsqueeze(2)  # (batch_size, seq_len, 1, 3)
        b = reference_coords.unsqueeze(1)  # (batch_size, 1, seq_len, 3)
        
        # Calculate squared distances: (batch_size, seq_len, seq_len)
        squared_distances = torch.sum((a - b) ** 2, dim=-1)
        distances = torch.sqrt(squared_distances + 1e-8)
        
        # Get potential base pairs from sequence using pairing matrix
        seq_a = sequence.unsqueeze(2)  # (batch_size, seq_len, 1)
        seq_b = sequence.unsqueeze(1)  # (batch_size, 1, seq_len)
        
        # Look up valid Watson-Crick pairs: (batch_size, seq_len, seq_len)
        wc_pairs = self.pairing_matrix[seq_a, seq_b]
        
        # Look up valid wobble pairs: (batch_size, seq_len, seq_len)
        wobble_pairs = self.wobble_matrix[seq_a, seq_b]
        
        # Combine Watson-Crick and wobble pairs
        valid_pairs = wc_pairs + wobble_pairs
        
        # Avoid self-pairing and near-neighbor pairing (i to i+1, i+2, etc.)
        # Create a mask for pairs that are too close in sequence
        min_sequence_separation = 3
        seq_indices = torch.arange(seq_len, device=coords.device)
        seq_separation = torch.abs(seq_indices.unsqueeze(1) - seq_indices.unsqueeze(0))
        too_close_mask = seq_separation < min_sequence_separation
        too_close_mask = too_close_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply sequence separation mask to valid_pairs
        valid_pairs = valid_pairs * (~too_close_mask).float()
        
        # Apply original mask to restrict to valid nucleotides
        mask_2d = mask.unsqueeze(2) & mask.unsqueeze(1)  # (batch_size, seq_len, seq_len)
        valid_pairs = valid_pairs * mask_2d.float()
        
        # Calculate loss for valid pairs
        # For each nucleotide, find the closest valid pairing partner
        # Deviation from ideal distance for valid pairs
        distance_deviation = torch.abs(distances - self.pair_distance)
        
        # Apply smooth L1 loss (Huber loss)
        pair_loss = torch.where(
            distance_deviation < self.tolerance,
            0.5 * distance_deviation**2 / self.tolerance,
            distance_deviation - 0.5 * self.tolerance
        )
        
        # Only consider loss for valid pairs
        weighted_loss = pair_loss * valid_pairs
        
        # Sum loss and normalize
        total_valid_pairs = valid_pairs.sum() + 1e-8
        loss = weighted_loss.sum() / total_valid_pairs
        
        return self.weight * loss


class RNAMotifConstraint(BaseRNAConstraint):
    """
    Constraint enforcing common RNA structural motifs.
    
    This includes hairpins, internal loops, bulges, etc.
    """
    
    def __init__(self, weight: float = 1.0):
        """
        Initialize the RNA motif constraint.
        
        Args:
            weight: Weight of this constraint in the loss function
        """
        super().__init__(weight)
        
        # Define ideal geometries for common RNA motifs
        # These are approximate values based on known structures
        
        # Hairpin loop ideal parameters
        self.hairpin_min_length = 3  # Minimum nucleotides in hairpin loop
        # Typical hairpin loop radius in Angstroms
        self.register_buffer('hairpin_radius', torch.tensor([6.0], dtype=torch.float32))
        
        # Other motifs can be added similarly
    
    def detect_hairpins(self, sequence: torch.Tensor, secondary_structure: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Detect potential hairpin loops in the sequence.
        
        Args:
            sequence: Sequence tensor of shape (batch_size, seq_len)
            secondary_structure: Secondary structure tensor (0=unpaired, 1=paired)
                of shape (batch_size, seq_len)
                
        Returns:
            List of tuples (start, end) for each detected hairpin
        """
        # This is a placeholder for actual hairpin detection logic
        # In a real implementation, this would analyze the sequence and secondary structure
        # to identify hairpin loops
        
        # For now, return empty list as this is just a demonstration
        return []
    
    def forward(self, coords: torch.Tensor, sequence: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                secondary_structure: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the RNA motif constraint loss.
        
        Args:
            coords: Coordinate tensor of shape (batch_size, seq_len, num_atoms, 3)
            sequence: Sequence tensor of shape (batch_size, seq_len)
            mask: Mask tensor of shape (batch_size, seq_len) for valid positions
            secondary_structure: Optional secondary structure tensor
                
        Returns:
            Loss value for RNA motif constraints
        """
        # If secondary structure isn't provided, we can't enforce motif constraints
        if secondary_structure is None:
            return torch.tensor(0.0, device=coords.device)
            
        # This is a placeholder for the actual implementation
        # In practice, this would detect various RNA motifs and enforce their
        # characteristic geometries
        
        # For now, return a dummy loss of 0
        return torch.tensor(0.0, device=coords.device) * self.weight


class RNAConstraintManager:
    """
    Manager class for applying multiple RNA-specific constraints.
    """
    
    def __init__(self):
        """Initialize the constraint manager with default constraints."""
        self.constraints = {}
        
        # Add default constraints
        self.add_constraint("watson_crick", WatsonCrickConstraint(weight=1.0))
        self.add_constraint("rna_motif", RNAMotifConstraint(weight=0.5))
    
    def add_constraint(self, name: str, constraint: BaseRNAConstraint):
        """
        Add a constraint to the manager.
        
        Args:
            name: Name of the constraint
            constraint: Constraint object
        """
        self.constraints[name] = constraint
    
    def compute_all_constraints(self, coords: torch.Tensor, sequence: torch.Tensor,
                               mask: Optional[torch.Tensor] = None,
                               secondary_structure: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute all constraints and return individual losses.
        
        Args:
            coords: Coordinate tensor of shape (batch_size, seq_len, num_atoms, 3)
            sequence: Sequence tensor of shape (batch_size, seq_len)
            mask: Mask tensor of shape (batch_size, seq_len) for valid positions
            secondary_structure: Optional secondary structure tensor
            
        Returns:
            Dictionary mapping constraint names to loss values
        """
        losses = {}
        
        for name, constraint in self.constraints.items():
            if name == "rna_motif" and secondary_structure is not None:
                losses[name] = constraint(coords, sequence, mask, secondary_structure)
            else:
                losses[name] = constraint(coords, sequence, mask)
        
        return losses
    
    def compute_total_loss(self, coords: torch.Tensor, sequence: torch.Tensor,
                          mask: Optional[torch.Tensor] = None,
                          secondary_structure: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the total loss from all constraints.
        
        Args:
            coords: Coordinate tensor of shape (batch_size, seq_len, num_atoms, 3)
            sequence: Sequence tensor of shape (batch_size, seq_len)
            mask: Mask tensor of shape (batch_size, seq_len) for valid positions
            secondary_structure: Optional secondary structure tensor
            
        Returns:
            Total loss value
        """
        individual_losses = self.compute_all_constraints(
            coords, sequence, mask, secondary_structure
        )
        
        return sum(individual_losses.values()) 