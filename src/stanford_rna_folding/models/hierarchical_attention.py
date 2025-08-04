"""
Hierarchical attention mechanisms for RNA structure prediction.

This module implements a hierarchical attention structure that models RNA at
multiple levels of organization: nucleotide level, secondary structure level,
and global tertiary structure level. This allows the model to better capture
both local and global structural patterns.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalRNAEncoder(nn.Module):
    """
    Hierarchical encoder for RNA structure prediction that models RNA at 
    multiple structural levels using specialized attention mechanisms.
    """
    
    def __init__(
        self,
        d_model: int,
        num_layers: int = 6,
        num_primary_heads: int = 4,
        num_secondary_heads: int = 4,
        num_tertiary_heads: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 500,
        pair_aware: bool = True,
    ):
        """
        Initialize the hierarchical RNA encoder.
        
        Args:
            d_model: Model dimension
            num_layers: Number of encoder layers
            num_primary_heads: Number of attention heads for primary structure (sequence)
            num_secondary_heads: Number of attention heads for secondary structure
            num_tertiary_heads: Number of attention heads for tertiary structure
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            pair_aware: Whether to use nucleotide pair-aware attention
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.pair_aware = pair_aware
        
        # Validate that the total number of heads works with the model dimension
        total_heads = num_primary_heads + num_secondary_heads + num_tertiary_heads
        assert d_model % total_heads == 0, "d_model must be divisible by total number of heads"
        
        # Create encoder layers
        self.layers = nn.ModuleList([
            HierarchicalRNAEncoderLayer(
                d_model=d_model,
                num_primary_heads=num_primary_heads,
                num_secondary_heads=num_secondary_heads,
                num_tertiary_heads=num_tertiary_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                pair_aware=pair_aware,
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Register base pairing matrix if using pair-aware attention
        if pair_aware:
            # Define pairing matrix (0=A, 1=U, 2=G, 3=C, 4=N)
            self.register_buffer('pairing_matrix', torch.tensor([
                [0, 1, 0, 0, 0],  # A pairs with U
                [1, 0, 1, 0, 0],  # U pairs with A and G (wobble)
                [0, 1, 0, 1, 0],  # G pairs with U (wobble) and C
                [0, 0, 1, 0, 0],  # C pairs with G
                [0, 0, 0, 0, 0],  # N doesn't pair
            ], dtype=torch.float32))
    
    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        sequence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the hierarchical RNA encoder.
        
        Args:
            src: Input tensor of shape (batch_size, seq_len, d_model)
            src_key_padding_mask: Mask for padding tokens (batch_size, seq_len)
            sequence: RNA sequence tensor for pair-aware attention (batch_size, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        output = src
        
        # Generate base pairing mask if using pair-aware attention and sequence is provided
        pairing_mask = None
        if self.pair_aware and sequence is not None:
            # Get sequence pairing matrix
            seq_i = sequence.unsqueeze(2)  # (batch, seq_len, 1)
            seq_j = sequence.unsqueeze(1)  # (batch, 1, seq_len)
            pairing_mask = self.pairing_matrix[seq_i, seq_j]  # (batch, seq_len, seq_len)
            
            # Apply sequence separation constraint (min 3 nucleotides separation)
            batch_size, seq_len = sequence.shape
            positions = torch.arange(seq_len, device=sequence.device)
            separation = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))
            separation_mask = separation >= 3
            
            pairing_mask = pairing_mask * separation_mask.unsqueeze(0).to(pairing_mask.dtype)
        
        # Apply transformer layers
        for layer in self.layers:
            output = layer(
                src=output,
                src_key_padding_mask=src_key_padding_mask,
                pairing_mask=pairing_mask,
            )
        
        # Apply layer normalization to output
        output = self.layer_norm(output)
        
        return output


class HierarchicalRNAEncoderLayer(nn.Module):
    """
    Encoder layer for hierarchical RNA structure modeling with specialized 
    attention mechanisms for different structural levels.
    """
    
    def __init__(
        self,
        d_model: int,
        num_primary_heads: int,
        num_secondary_heads: int,
        num_tertiary_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pair_aware: bool = True,
    ):
        """
        Initialize the hierarchical RNA encoder layer.
        
        Args:
            d_model: Model dimension
            num_primary_heads: Number of attention heads for primary structure
            num_secondary_heads: Number of attention heads for secondary structure
            num_tertiary_heads: Number of attention heads for tertiary structure
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            pair_aware: Whether to use nucleotide pair-aware attention
        """
        super().__init__()
        
        self.d_model = d_model
        total_heads = num_primary_heads + num_secondary_heads + num_tertiary_heads
        self.head_dim = d_model // total_heads
        
        # Define attention mechanisms for each level
        self.primary_attn = MultiheadAttention(
            embed_dim=num_primary_heads * self.head_dim,
            num_heads=num_primary_heads,
            dropout=dropout,
        )
        
        self.secondary_attn = PairAwareAttention(
            embed_dim=num_secondary_heads * self.head_dim,
            num_heads=num_secondary_heads,
            dropout=dropout,
            pair_aware=pair_aware,
        )
        
        self.tertiary_attn = MultiheadAttention(
            embed_dim=num_tertiary_heads * self.head_dim,
            num_heads=num_tertiary_heads,
            dropout=dropout,
        )
        
        # Input projections for each attention level
        self.primary_in_proj = nn.Linear(d_model, num_primary_heads * self.head_dim)
        self.secondary_in_proj = nn.Linear(d_model, num_secondary_heads * self.head_dim)
        self.tertiary_in_proj = nn.Linear(d_model, num_tertiary_heads * self.head_dim)
        
        # Output projection to combine attention outputs
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pairing_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the hierarchical RNA encoder layer.
        
        Args:
            src: Input tensor of shape (batch_size, seq_len, d_model)
            src_key_padding_mask: Mask for padding tokens (batch_size, seq_len)
            pairing_mask: Base pairing mask (batch_size, seq_len, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Apply layer normalization first (Pre-LN architecture for better training stability)
        src_norm = self.norm1(src)
        
        # Project input for each attention level
        primary_src = self.primary_in_proj(src_norm)
        secondary_src = self.secondary_in_proj(src_norm)
        tertiary_src = self.tertiary_in_proj(src_norm)
        
        # Primary structure attention (local, sequential context)
        primary_out = self.primary_attn(
            query=primary_src,
            key=primary_src,
            value=primary_src,
            key_padding_mask=src_key_padding_mask,
        )
        
        # Secondary structure attention (base pairing, stems, loops)
        secondary_out = self.secondary_attn(
            query=secondary_src,
            key=secondary_src,
            value=secondary_src,
            key_padding_mask=src_key_padding_mask,
            pairing_mask=pairing_mask,
        )
        
        # Tertiary structure attention (global 3D context)
        tertiary_out = self.tertiary_attn(
            query=tertiary_src,
            key=tertiary_src,
            value=tertiary_src,
            key_padding_mask=src_key_padding_mask,
        )
        
        # Concatenate outputs from different attention levels
        combined = torch.cat([primary_out, secondary_out, tertiary_out], dim=-1)
        
        # Project back to original dimension
        attn_output = self.out_proj(combined)
        
        # Residual connection
        src = src + self.dropout(attn_output)
        
        # Feedforward network
        src_norm = self.norm2(src)
        ff_output = self.feedforward(src_norm)
        
        # Residual connection
        src = src + self.dropout(ff_output)
        
        return src


class MultiheadAttention(nn.Module):
    """
    Multi-head attention mechanism with optimized implementation.
    Simplified version of PyTorch's MultiheadAttention.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Combined projections for efficiency
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Scaling factor
        self.scaling = self.head_dim ** -0.5
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, tgt_len, embed_dim)
            key: Key tensor of shape (batch_size, src_len, embed_dim)
            value: Value tensor of shape (batch_size, src_len, embed_dim)
            key_padding_mask: Mask for keys to ignore (batch_size, src_len)
            attn_mask: Mask to prevent attention to certain positions (tgt_len, src_len)
            
        Returns:
            Output tensor of shape (batch_size, tgt_len, embed_dim)
        """
        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]
        
        # Combined projection for Q, K, V
        qkv = self.qkv_proj(query)
        qkv = qkv.reshape(batch_size, tgt_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        
        # Separate Q, K, V
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        # (batch_size, num_heads, tgt_len, src_len)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply padding mask if provided
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
        
        # Apply softmax to get probabilities
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention weights to values
        # (batch_size, num_heads, tgt_len, head_dim)
        output = torch.matmul(attn_probs, v)
        
        # Reshape and transpose
        # (batch_size, tgt_len, embed_dim)
        output = output.transpose(1, 2).reshape(batch_size, tgt_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(output)
        
        return output


class PairAwareAttention(nn.Module):
    """
    Attention mechanism that is aware of RNA base-pairing rules.
    
    Enhances attention weights between nucleotides that can form base pairs
    according to Watson-Crick and wobble pairing rules.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        pair_bonus: float = 2.0,
        pair_aware: bool = True,
    ):
        """
        Initialize pair-aware attention.
        
        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of attention heads
            dropout: Dropout probability
            pair_bonus: Weight bonus for complementary base pairs
            pair_aware: Whether to use pair-aware attention
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.pair_aware = pair_aware
        self.pair_bonus = pair_bonus
        
        # Projections
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Scaling factor
        self.scaling = self.head_dim ** -0.5
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        pairing_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for pair-aware attention.
        
        Args:
            query: Query tensor of shape (batch_size, tgt_len, embed_dim)
            key: Key tensor of shape (batch_size, src_len, embed_dim)
            value: Value tensor of shape (batch_size, src_len, embed_dim)
            key_padding_mask: Mask for keys to ignore (batch_size, src_len)
            pairing_mask: Base pairing mask (batch_size, tgt_len, src_len)
            attn_mask: Mask to prevent attention to certain positions (tgt_len, src_len)
            
        Returns:
            Output tensor of shape (batch_size, tgt_len, embed_dim)
        """
        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]
        
        # Combined projection for Q, K, V
        qkv = self.qkv_proj(query)
        qkv = qkv.reshape(batch_size, tgt_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        
        # Separate Q, K, V
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        # (batch_size, num_heads, tgt_len, src_len)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Add base-pairing awareness if pairing mask is provided
        if self.pair_aware and pairing_mask is not None:
            # Add bonus to attention scores for valid pairs
            # Apply to all heads to encourage all of them to learn base-pairing patterns
            pair_bonus = pairing_mask.unsqueeze(1) * self.pair_bonus  # (batch, 1, tgt_len, src_len)
            attn_weights = attn_weights + pair_bonus
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply padding mask if provided
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
        
        # Apply softmax to get probabilities
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention weights to values
        # (batch_size, num_heads, tgt_len, head_dim)
        output = torch.matmul(attn_probs, v)
        
        # Reshape and transpose
        # (batch_size, tgt_len, embed_dim)
        output = output.transpose(1, 2).reshape(batch_size, tgt_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(output)
        
        return output 