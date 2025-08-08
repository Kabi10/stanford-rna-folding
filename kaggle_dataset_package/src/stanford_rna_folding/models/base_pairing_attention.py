"""
Base-pairing aware attention mechanism for RNA structure prediction.

This module implements attention mechanisms that are RNA-specific and recognize
complementary bases (A-U, G-C, and G-U wobble pairs), enhancing the model's
ability to learn RNA secondary structure.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasePairingAttention(nn.Module):
    """
    Attention mechanism that is aware of RNA base-pairing rules.
    
    This mechanism modifies attention weights to enhance potential base pairs
    according to Watson-Crick rules and wobble pairs.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        add_bias_kv: bool = False,
        pair_bonus: float = 2.0,  # Weight bonus for valid base pairs
        min_separation: int = 3,  # Minimum sequence separation for base pairs
    ):
        """
        Initialize base-pairing aware attention.
        
        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Add bias to input projections
            add_bias_kv: Add bias to key and value projections
            pair_bonus: Weight bonus to apply to valid base pairs
            min_separation: Minimum sequence separation to consider for base pairing
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.pair_bonus = pair_bonus
        self.min_separation = min_separation
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        
        # Define base-pairing matrix
        # Assuming nucleotide encoding: 0=A, 1=U, 2=G, 3=C, 4=N(padding)
        self.register_buffer('pairing_matrix', torch.tensor([
            [0, 1, 0, 0, 0],  # A pairs with U
            [1, 0, 1, 0, 0],  # U pairs with A and G (wobble)
            [0, 1, 0, 1, 0],  # G pairs with U (wobble) and C
            [0, 0, 1, 0, 0],  # C pairs with G
            [0, 0, 0, 0, 0],  # N doesn't pair
        ], dtype=torch.float32))
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        nucleotide_sequence: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rel_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for base-pairing aware attention.
        
        Args:
            query: Query embeddings of shape (batch_size, tgt_len, embed_dim)
            key: Key embeddings of shape (batch_size, src_len, embed_dim)
            value: Value embeddings of shape (batch_size, src_len, embed_dim)
            key_padding_mask: Mask for keys to ignore of shape (batch_size, src_len)
            nucleotide_sequence: Integer tensor of nucleotide indices (batch_size, src_len)
            attn_mask: Mask to prevent attention to certain positions, shape (tgt_len, src_len)
            rel_pos: Relative position encoding (optional)
            
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
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Add base-pairing awareness if sequence is provided
        if nucleotide_sequence is not None:
            # Create position indices
            positions = torch.arange(src_len, device=query.device)
            
            # Get sequence separation mask
            separation = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))
            separation_mask = separation >= self.min_separation
            
            # Get base-pairing mask from sequence
            seq_i = nucleotide_sequence.unsqueeze(2)  # (batch, src_len, 1)
            seq_j = nucleotide_sequence.unsqueeze(1)  # (batch, 1, src_len)
            pairing_mask = self.pairing_matrix[seq_i, seq_j]  # (batch, src_len, src_len)
            
            # Apply sequence separation constraint
            pairing_mask = pairing_mask * separation_mask.unsqueeze(0).to(pairing_mask.dtype)
            
            # Add bonus to attention scores for valid pairs
            # We add this to all heads to encourage all of them to learn base-pairing patterns
            pair_bonus = pairing_mask.unsqueeze(1) * self.pair_bonus  # (batch, 1, src_len, src_len)
            attn_weights = attn_weights + pair_bonus
        
        # Add relative positional encoding if provided
        if rel_pos is not None:
            attn_weights = attn_weights + rel_pos
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply padding mask if provided
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
        
        # Apply softmax to get probabilities
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout_layer(attn_probs)
        
        # Apply attention to values
        output = torch.matmul(attn_probs, v)  # (batch_size, num_heads, tgt_len, head_dim)
        
        # Reshape back to original dimensions
        output = output.transpose(1, 2).reshape(batch_size, tgt_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(output)
        
        return output, attn_probs


class HierarchicalRNAAttention(nn.Module):
    """
    Hierarchical attention for RNA structure prediction.
    
    This attention mechanism operates at multiple levels:
    1. Base-pair level: Captures interactions between complementary bases
    2. Local structure level: Focuses on small motifs (stems, loops)
    3. Global structure level: Models the overall 3D architecture
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_base_heads: int = 4,
        num_local_heads: int = 2,
        num_global_heads: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize hierarchical RNA attention.
        
        Args:
            embed_dim: Total embedding dimension
            num_base_heads: Number of heads for base-pair level attention
            num_local_heads: Number of heads for local structure attention
            num_global_heads: Number of heads for global structure attention
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_base_heads = num_base_heads
        self.num_local_heads = num_local_heads
        self.num_global_heads = num_global_heads
        total_heads = num_base_heads + num_local_heads + num_global_heads
        
        # Ensure the embedding dimension is divisible by the total number of heads
        assert embed_dim % total_heads == 0, "embed_dim must be divisible by total number of heads"
        
        self.head_dim = embed_dim // total_heads
        
        # Base-pair level attention
        self.base_attention = BasePairingAttention(
            embed_dim=num_base_heads * self.head_dim,
            num_heads=num_base_heads,
            dropout=dropout,
            pair_bonus=2.0,
        )
        
        # Local structure attention (stems, loops, etc.)
        local_dim = num_local_heads * self.head_dim
        self.local_attention = nn.MultiheadAttention(
            embed_dim=local_dim,
            num_heads=num_local_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Global structure attention (overall 3D architecture)
        global_dim = num_global_heads * self.head_dim
        self.global_attention = nn.MultiheadAttention(
            embed_dim=global_dim,
            num_heads=num_global_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Input projections for each attention level
        self.base_in_proj = nn.Linear(embed_dim, num_base_heads * self.head_dim)
        self.local_in_proj = nn.Linear(embed_dim, num_local_heads * self.head_dim)
        self.global_in_proj = nn.Linear(embed_dim, num_global_heads * self.head_dim)
        
        # Output projection to combine attention outputs
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        nucleotide_sequence: Optional[torch.Tensor] = None,
        rel_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for hierarchical RNA attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            padding_mask: Mask for padding tokens of shape (batch_size, seq_len)
            nucleotide_sequence: Integer tensor with nucleotide indices
            rel_pos: Relative position encoding
            
        Returns:
            Updated embeddings of shape (batch_size, seq_len, embed_dim)
        """
        # Apply layer normalization first (Pre-LN architecture for better training stability)
        x_norm = self.layer_norm(x)
        
        # Project input for each attention level
        base_x = self.base_in_proj(x_norm)
        local_x = self.local_in_proj(x_norm)
        global_x = self.global_in_proj(x_norm)
        
        # Base-pair level attention
        base_output, _ = self.base_attention(
            query=base_x,
            key=base_x,
            value=base_x,
            key_padding_mask=padding_mask,
            nucleotide_sequence=nucleotide_sequence,
            rel_pos=rel_pos,
        )
        
        # Local structure attention
        local_output, _ = self.local_attention(
            query=local_x,
            key=local_x,
            value=local_x,
            key_padding_mask=padding_mask,
        )
        
        # Global structure attention
        global_output, _ = self.global_attention(
            query=global_x,
            key=global_x,
            value=global_x,
            key_padding_mask=padding_mask,
        )
        
        # Concatenate outputs from different attention levels
        combined = torch.cat([base_output, local_output, global_output], dim=-1)
        
        # Project back to original dimension
        output = self.out_proj(combined)
        
        # Residual connection
        return x + output 