"""
Distance-modulated attention for RNA structure prediction.

This module implements distance-modulated attention that incorporates geometric
information about the structure being predicted to improve attention mechanisms.
It scales attention scores based on the estimated physical distance between
nucleotides, creating a geometry-aware system.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceModulatedAttention(nn.Module):
    """
    Attention mechanism that modulates attention scores based on 
    distances between nucleotides in the predicted structure.
    
    This helps the model focus on structurally relevant interactions
    rather than being dominated by sequence proximity.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        distance_scaling: str = "inverse",  # "inverse", "gaussian", or "learned"
        max_distance: float = 30.0,  # Maximum distance in Angstroms
        min_distance: float = 1.0,   # Minimum distance in Angstroms
        temperature: float = 5.0,    # Temperature for softening distance effects
    ):
        """
        Initialize distance-modulated attention.
        
        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of attention heads
            dropout: Dropout probability
            distance_scaling: Type of distance scaling to use
            max_distance: Maximum distance in Angstroms
            min_distance: Minimum distance in Angstroms
            temperature: Temperature for softening distance effects
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.distance_scaling = distance_scaling
        self.max_distance = max_distance
        self.min_distance = min_distance
        self.temperature = temperature
        
        # Scaling factor
        self.scaling = self.head_dim ** -0.5
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Parameters for learned distance scaling
        if distance_scaling == "learned":
            # Create learned parameters for distance scaling
            self.distance_weights = nn.Parameter(torch.zeros(num_heads, 1, 1))
            self.distance_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))
            
            # Initialize with reasonable values
            nn.init.normal_(self.distance_weights, mean=1.0, std=0.1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def compute_distance_weights(
        self,
        distances: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention modulation weights based on distances.
        
        Args:
            distances: Pairwise distances between nucleotides (batch_size, seq_len, seq_len)
            mask: Optional mask to exclude certain positions (batch_size, seq_len, seq_len)
            
        Returns:
            Distance weights for attention (batch_size, num_heads, seq_len, seq_len)
        """
        # Clamp distances to sensible range
        distances = torch.clamp(distances, self.min_distance, self.max_distance)
        
        # Apply different scaling methods
        if self.distance_scaling == "inverse":
            # Simple inverse distance scaling (closer = stronger attention)
            weights = self.temperature / (distances + 1e-8)
            
        elif self.distance_scaling == "gaussian":
            # Gaussian scaling (strong in middle range, weak at very close or far distances)
            # This can be useful for emphasizing base-stacking and certain motifs
            mean_distance = (self.max_distance + self.min_distance) / 2
            weights = torch.exp(-(distances - mean_distance)**2 / (2 * self.temperature**2))
            
        elif self.distance_scaling == "learned":
            # Learned scaling function (more flexible)
            distances_expanded = distances.unsqueeze(1)  # (batch, 1, seq, seq)
            weights = torch.sigmoid(self.distance_weights * distances_expanded + self.distance_bias)
            
        else:
            raise ValueError(f"Unknown distance scaling method: {self.distance_scaling}")
        
        # Apply mask if provided
        if mask is not None:
            weights = weights.masked_fill(mask, 0.0)
        
        # Expand to heads dimension if needed
        if weights.dim() == 3:  # (batch, seq, seq)
            weights = weights.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            
        return weights
    
    def compute_pairwise_distances(
        self, 
        coords: torch.Tensor,
        reference_atom: int = 2,  # Typically atom 2 (C3') is used as reference
    ) -> torch.Tensor:
        """
        Compute pairwise distances between nucleotides.
        
        Args:
            coords: Predicted coordinates (batch_size, seq_len, num_atoms, 3)
            reference_atom: Which atom to use as reference for distances
            
        Returns:
            Pairwise distances (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, num_atoms, _ = coords.shape
        
        # Use specified atom as reference point
        ref_coords = coords[:, :, reference_atom, :]  # (batch_size, seq_len, 3)
        
        # Calculate pairwise distances
        a = ref_coords.unsqueeze(2)  # (batch, seq, 1, 3)
        b = ref_coords.unsqueeze(1)  # (batch, 1, seq, 3)
        
        # Compute squared Euclidean distance
        squared_distances = torch.sum((a - b) ** 2, dim=-1)  # (batch, seq, seq)
        
        # Take square root to get actual distances
        distances = torch.sqrt(squared_distances + 1e-8)  # Avoid sqrt(0)
        
        return distances
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        current_coords: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        distance_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for distance-modulated attention.
        
        Args:
            query: Query tensor (batch_size, tgt_len, embed_dim)
            key: Key tensor (batch_size, src_len, embed_dim)
            value: Value tensor (batch_size, src_len, embed_dim)
            current_coords: Current predicted coordinates (batch_size, seq_len, num_atoms, 3)
            key_padding_mask: Mask for padding (batch_size, src_len)
            attn_mask: Mask for attention (tgt_len, src_len)
            distance_mask: Mask for distances (batch_size, seq_len, seq_len)
            
        Returns:
            Tuple of (output, attention weights)
        """
        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]
        
        # Project and reshape for multi-head attention
        q = self.q_proj(query).reshape(batch_size, tgt_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).reshape(batch_size, src_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).reshape(batch_size, src_len, self.num_heads, self.head_dim)
        
        # Transpose for batch matrix multiplication
        q = q.transpose(1, 2)  # (batch_size, num_heads, tgt_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_heads, src_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, src_len, head_dim)
        
        # Compute standard attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Apply geometric information if coordinates are provided
        if current_coords is not None:
            # Compute pairwise distances between nucleotides
            distances = self.compute_pairwise_distances(current_coords)
            
            # Calculate distance-based attention weights
            distance_weights = self.compute_distance_weights(distances, distance_mask)
            
            # Apply distance modulation to attention scores
            attn_weights = attn_weights * distance_weights
        
        # Apply padding mask if provided
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply softmax to get probabilities
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        output = torch.matmul(attn_probs, v)  # (batch_size, num_heads, tgt_len, head_dim)
        
        # Transpose and reshape to original dimensions
        output = output.transpose(1, 2).reshape(batch_size, tgt_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(output)
        
        return output, attn_probs


class DistanceModulatedTransformerLayer(nn.Module):
    """
    Transformer encoder layer with distance-modulated attention.
    
    This layer continuously refines its attention based on the current
    coordinate predictions, creating a feedback loop between structure
    and attention.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        distance_scaling: str = "inverse",
    ):
        """
        Initialize the distance-modulated transformer layer.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            distance_scaling: Type of distance scaling to use
        """
        super().__init__()
        
        # Distance-modulated attention
        self.self_attn = DistanceModulatedAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            distance_scaling=distance_scaling,
        )
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation
        self.activation = F.gelu
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        current_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the distance-modulated transformer layer.
        
        Args:
            src: Input tensor (batch_size, seq_len, d_model)
            src_mask: Mask to prevent attention to certain positions (seq_len, seq_len)
            src_key_padding_mask: Mask for padding tokens (batch_size, seq_len)
            current_coords: Current predicted coordinates (batch_size, seq_len, num_atoms, 3)
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Self-attention block (with distance modulation)
        src2, _ = self.self_attn(
            query=self.norm1(src),
            key=self.norm1(src),
            value=self.norm1(src),
            current_coords=current_coords,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        
        # Residual connection
        src = src + self.dropout1(src2)
        
        # Feedforward block
        src2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
        
        # Residual connection
        src = src + self.dropout2(src2)
        
        return src


class DistanceModulatedTransformer(nn.Module):
    """
    Transformer encoder with distance-modulated attention.
    
    This encoder uses current coordinate predictions to modulate
    attention, creating a feedback loop that improves geometrical
    awareness in the model.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        distance_scaling: str = "inverse",
    ):
        """
        Initialize the distance-modulated transformer.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of encoder layers
            dim_feedforward: Dimension of feedforward networks
            dropout: Dropout rate
            distance_scaling: Type of distance scaling to use
        """
        super().__init__()
        
        # Transformer layers
        self.layers = nn.ModuleList([
            DistanceModulatedTransformerLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                distance_scaling=distance_scaling,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        current_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the distance-modulated transformer.
        
        Args:
            src: Input tensor (batch_size, seq_len, d_model)
            mask: Mask to prevent attention to certain positions (seq_len, seq_len)
            src_key_padding_mask: Mask for padding tokens (batch_size, seq_len)
            current_coords: Current predicted coordinates (batch_size, seq_len, num_atoms, 3)
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        output = src
        
        # Pass through each layer
        for layer in self.layers:
            output = layer(
                src=output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                current_coords=current_coords,
            )
        
        # Apply final layer normalization
        output = self.norm(output)
        
        return output


class IterativeDistanceRefinement(nn.Module):
    """
    Module for iterative refinement of RNA structure using distance-modulated attention.
    
    This performs multiple passes, refining the structure prediction in each iteration
    by using distance-modulated attention based on previous coordinate predictions.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        num_refinement_steps: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        distance_scaling: str = "inverse",
        num_atoms: int = 5,
        coord_dims: int = 3,
    ):
        """
        Initialize the iterative distance refinement module.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            num_refinement_steps: Number of iterative refinement steps
            dim_feedforward: Dimension of feedforward networks
            dropout: Dropout rate
            distance_scaling: Type of distance scaling to use
            num_atoms: Number of atoms per nucleotide
            coord_dims: Number of coordinate dimensions (typically 3)
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_refinement_steps = num_refinement_steps
        self.num_atoms = num_atoms
        self.coord_dims = coord_dims
        
        # Distance-modulated transformer
        self.transformer = DistanceModulatedTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            distance_scaling=distance_scaling,
        )
        
        # Initial coordinate predictor
        self.initial_coord_predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_atoms * coord_dims),
        )
        
        # Refinement coordinate predictor
        self.refine_coord_predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_atoms * coord_dims),
        )
    
    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with iterative refinement.
        
        Args:
            src: Input tensor (batch_size, seq_len, d_model)
            mask: Mask to prevent attention to certain positions (seq_len, seq_len)
            src_key_padding_mask: Mask for padding tokens (batch_size, seq_len)
            
        Returns:
            Predicted coordinates (batch_size, seq_len, num_atoms, coord_dims)
        """
        batch_size, seq_len, _ = src.shape
        
        # Initial coordinate prediction (without distance modulation)
        initial_output = self.transformer(
            src=src,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask,
            current_coords=None,  # No coordinates yet
        )
        
        # Predict initial coordinates
        initial_coords_flat = self.initial_coord_predictor(initial_output)
        current_coords = initial_coords_flat.view(
            batch_size, seq_len, self.num_atoms, self.coord_dims
        )
        
        # Iterative refinement
        for _ in range(self.num_refinement_steps):
            # Update embeddings using distance-modulated attention
            refined_output = self.transformer(
                src=src,
                mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                current_coords=current_coords,  # Use current coordinate predictions
            )
            
            # Predict refined coordinates
            refined_coords_flat = self.refine_coord_predictor(refined_output)
            refined_coords = refined_coords_flat.view(
                batch_size, seq_len, self.num_atoms, self.coord_dims
            )
            
            # Update current coordinates (with residual connection for stability)
            current_coords = current_coords + refined_coords
        
        return current_coords 