"""
RNA Structure Prediction model for the Stanford RNA 3D folding competition.
"""

import math
from typing import Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rna_constraints import RNAConstraintManager


class RNAFoldingModel(nn.Module):
    """
    Transformer-based model for RNA 3D structure prediction.
    
    Takes RNA sequences as input and predicts 3D coordinates for each nucleotide.
    Uses a Transformer encoder to process the sequence and an MLP to predict coordinates.
    """
    
    def __init__(
        self,
        vocab_size: int = 5,  # A, U, G, C, N (padding)
        embedding_dim: int = 256,  # Increased embedding dimension
        hidden_dim: int = 512,     # Increased hidden dimension
        num_layers: int = 6,      # Increased transformer layers
        num_heads: int = 8,
        dropout: float = 0.1,
        num_atoms: int = 1,      # Default to single-atom mode
        multi_atom_mode: bool = False,  # Flag to control multi-atom or single-atom mode
        coord_dims: int = 3,     # 3D coordinates (x, y, z)
        max_seq_len: int = 500,
        use_rna_constraints: bool = True,  # Whether to use RNA-specific constraints
        bond_length_weight: float = 1.0,   # Weight for bond length constraint
        bond_angle_weight: float = 1.0,    # Weight for bond angle constraint
        steric_clash_weight: float = 1.0,  # Weight for steric clash constraint
        watson_crick_weight: float = 1.0,  # Weight for Watson-Crick constraint
        normalize_coords: bool = True,     # Whether to normalize coordinates in the model
        use_relative_attention: bool = True,  # Whether to use relative attention
    ):
        """
        Initialize the RNA folding model.
        
        Args:
            vocab_size: Size of the RNA sequence vocabulary
            embedding_dim: Dimension of sequence embeddings
            hidden_dim: Hidden dimension of the Transformer encoder
            num_layers: Number of Transformer encoder layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            num_atoms: Number of atoms per nucleotide to predict coordinates for
            multi_atom_mode: If True, model will predict multiple atoms per nucleotide
            coord_dims: Number of coordinate dimensions (typically 3 for x, y, z)
            max_seq_len: Maximum sequence length
            use_rna_constraints: Whether to use RNA-specific constraints
            bond_length_weight: Weight for bond length constraint
            bond_angle_weight: Weight for bond angle constraint
            steric_clash_weight: Weight for steric clash constraint
            watson_crick_weight: Weight for Watson-Crick constraint
            normalize_coords: Whether to normalize coordinates in the model
            use_relative_attention: Whether to use relative attention mechanism
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.multi_atom_mode = multi_atom_mode
        self.num_atoms = 5 if multi_atom_mode else num_atoms  # Use 5 atoms if multi-atom mode is enabled
        self.coord_dims = coord_dims
        self.max_seq_len = max_seq_len
        self.normalize_coords = normalize_coords
        self.use_relative_attention = use_relative_attention
        
        # Constraint settings
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
        
        # RNA-specific relative positional encodings if enabled
        if self.use_relative_attention:
            self.relative_attention = RelativePositionEncoding(
                max_distance=max_seq_len,
                num_heads=num_heads,
                embedding_dim=embedding_dim // num_heads
            )
        
        # Transformer encoder 
        if self.use_relative_attention:
            # Custom transformer with relative position encoding
            self.transformer_layers = nn.ModuleList([
                RNATransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    dropout=dropout,
                    batch_first=True
                ) for _ in range(num_layers)
            ])
        else:
            # Standard PyTorch transformer
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
        
        # MLP for coordinate prediction - different architectures for single vs multi-atom
        if self.multi_atom_mode:
            # For multi-atom mode, predict all atoms at once
            self.coordinate_mlp = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_atoms * coord_dims),
            )
        else:
            # For single-atom mode, use a more focused architecture
            self.coordinate_mlp = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, self.num_atoms * coord_dims),
            )
        
        # Initialize ideal bond lengths and angles for RNA nucleotides
        # These are approximate values based on RNA structural biology
        # Values in Angstroms for distances and radians for angles
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
        
        # Initialize RNA constraint manager if enabled
        if self.use_rna_constraints:
            self.rna_constraint_manager = RNAConstraintManager()
            # Update Watson-Crick constraint weight
            if "watson_crick" in self.rna_constraint_manager.constraints:
                self.rna_constraint_manager.constraints["watson_crick"].weight = watson_crick_weight
    
    def _generate_positional_encoding(self, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Generate sinusoidal positional encodings.
        
        Args:
            seq_len: Length of sequence for which to generate encodings. If None, uses self.max_seq_len.
            
        Returns:
            Tensor of shape (1, seq_len, embedding_dim) with positional encodings
        """
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
        """
        Create a mask for padding tokens in the sequence.
        
        Args:
            sequence: Sequence tensor of shape (batch_size, seq_len)
            pad_idx: Index of the padding token
            
        Returns:
            Boolean mask tensor of shape (batch_size, seq_len) where True values
            indicate positions that should be masked (padding tokens)
        """
        return sequence == pad_idx  # Shape: (batch_size, seq_len)
    
    def _normalize_coordinate_output(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Normalize the predicted coordinates for better consistency.
        
        Args:
            coords: Coordinate tensor of shape (batch_size, seq_len, num_atoms, 3)
            
        Returns:
            Normalized coordinate tensor of same shape
        """
        if not self.normalize_coords:
            return coords
        
        batch_size, seq_len, num_atoms, coord_dims = coords.shape
        
        # Reshape for easier processing
        coords_flat = coords.view(batch_size, seq_len * num_atoms, coord_dims)
        
        # Calculate center of each structure (mean over all atoms)
        centers = coords_flat.mean(dim=1, keepdim=True)  # (batch_size, 1, 3)
        
        # Center coordinates
        centered_coords = coords_flat - centers
        
        # Calculate scale (max distance from center) for each structure
        dist_from_center = torch.norm(centered_coords, dim=2, keepdim=True)  # (batch_size, seq_len*num_atoms, 1)
        max_dist, _ = torch.max(dist_from_center, dim=1, keepdim=True)  # (batch_size, 1, 1)
        
        # Avoid division by zero
        max_dist = torch.clamp(max_dist, min=1e-10)
        
        # Scale coordinates
        normalized_coords = centered_coords / max_dist
        
        # Reshape back to original shape
        return normalized_coords.view(batch_size, seq_len, num_atoms, coord_dims)
    
    def forward(
        self, 
        sequence: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            sequence: RNA sequence tensor of shape (batch_size, seq_len)
            lengths: Tensor of sequence lengths (batch_size)
            
        Returns:
            Predicted 3D coordinates tensor of shape (batch_size, seq_len, num_atoms, coord_dims)
        """
        batch_size, seq_len = sequence.shape
        
        # Embed sequence tokens
        x = self.token_embedding(sequence)  # Shape: (batch_size, seq_len, embedding_dim)
        
        # Add positional encoding
        if seq_len > self.max_seq_len:
            # Generate positional encoding for the current sequence length
            pos_enc = self._generate_positional_encoding(seq_len=seq_len)
        else:
            pos_enc = self.positional_encoding[:, :seq_len, :]
        
        x = x + pos_enc.to(x.device)  # Shape: (batch_size, seq_len, embedding_dim)
        
        # Create padding mask for the transformer
        padding_mask = self.create_padding_mask(sequence)
        
        # Pass through transformer encoder
        if self.use_relative_attention:
            # Use custom transformer with relative position encoding
            for layer in self.transformer_layers:
                x = layer(x, padding_mask, self.relative_attention)
            encoded = x
        else:
            # Use standard transformer
            encoded = self.transformer_encoder(
                src=x,
                src_key_padding_mask=padding_mask,
            )  # Shape: (batch_size, seq_len, embedding_dim)
        
        # Predict coordinates using MLP
        coords_flat = self.coordinate_mlp(encoded)  # Shape: (batch_size, seq_len, num_atoms * coord_dims)
        
        # Reshape to separate atoms and coordinates
        coords = coords_flat.view(batch_size, seq_len, self.num_atoms, self.coord_dims)
        
        # Normalize coordinates if requested
        if self.normalize_coords:
            coords = self._normalize_coordinate_output(coords)
        
        return coords

# Add new RelativePositionEncoding class for RNA-specific attention
class RelativePositionEncoding(nn.Module):
    """
    Relative positional encoding for RNA attention mechanisms.
    
    This helps the model better understand the relative distances between nucleotides,
    which is crucial for RNA secondary structure formation.
    """
    
    def __init__(self, max_distance: int, num_heads: int, embedding_dim: int):
        super().__init__()
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        
        # Create relative position embeddings
        # We use 2*max_distance+1 to account for both positive and negative distances
        self.rel_embeddings = nn.Parameter(
            torch.randn(2 * max_distance + 1, num_heads, embedding_dim)
        )
        
        # Initialize with small values for stability
        nn.init.xavier_uniform_(self.rel_embeddings, gain=0.1)
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate relative position encodings for a sequence.
        
        Args:
            seq_len: Length of sequence
            device: Device to put tensors on
            
        Returns:
            Relative position encoding matrix of shape (seq_len, seq_len, num_heads, embedding_dim)
        """
        # Create position indices
        positions = torch.arange(seq_len, device=device)
        
        # Calculate relative positions
        # Shape: (seq_len, seq_len)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        
        # Clamp to max_distance
        relative_positions = torch.clamp(
            relative_positions,
            -self.max_distance,
            self.max_distance
        )
        
        # Shift to make all indices non-negative
        relative_positions = relative_positions + self.max_distance
        
        # Get the embeddings for each relative position
        # Shape: (seq_len, seq_len, num_heads, embedding_dim)
        return self.rel_embeddings[relative_positions]


class RNATransformerEncoderLayer(nn.Module):
    """
    Custom transformer encoder layer with RNA-specific relative attention.
    """
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, batch_first: bool = True):
        super().__init__()
        
        # Multi-head attention with relative position
        self.self_attn = RNAMultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation
        self.activation = F.gelu  # Using GELU instead of ReLU for better performance
    
    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor, rel_pos: RelativePositionEncoding) -> torch.Tensor:
        """
        Forward pass for the RNA transformer encoder layer.
        
        Args:
            src: Input tensor (batch_size, seq_len, d_model)
            src_key_padding_mask: Mask for padding (batch_size, seq_len)
            rel_pos: Relative position encoding module
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        seq_len = src.size(1)
        device = src.device
        
        # Get relative position encodings
        rel_pos_enc = rel_pos(seq_len, device)
        
        # Self-attention block with residual connection
        src2 = self.self_attn(src, src, src, src_key_padding_mask, rel_pos_enc)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward block with residual connection
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class RNAMultiheadAttention(nn.Module):
    """
    RNA-specific multi-head attention with relative position encoding.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Projection matrices
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        key_padding_mask: Optional[torch.Tensor] = None,
        rel_pos_enc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for RNA multi-head attention.
        
        Args:
            query: Query tensor (batch_size, seq_len, embed_dim)
            key: Key tensor (batch_size, seq_len, embed_dim)
            value: Value tensor (batch_size, seq_len, embed_dim)
            key_padding_mask: Mask for padding (batch_size, seq_len)
            rel_pos_enc: Relative position encodings (seq_len, seq_len, num_heads, head_dim)
            
        Returns:
            Output tensor (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = query.size()
        
        # Project and reshape for multi-head attention
        q = self.q_proj(query).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for batched matrix multiplication
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Calculate attention scores
        # (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add relative positional encoding if provided
        if rel_pos_enc is not None:
            # Reshape relative position encodings
            rel_pos_enc = rel_pos_enc.permute(2, 0, 1, 3)  # (num_heads, seq_len, seq_len, head_dim)
            
            # Calculate position-aware attention scores
            rel_scores = torch.matmul(
                q.unsqueeze(3),  # (batch_size, num_heads, seq_len, 1, head_dim)
                rel_pos_enc.transpose(-2, -1)  # (num_heads, seq_len, seq_len, head_dim, 1)
            ).squeeze(-1).squeeze(-1)  # (batch_size, num_heads, seq_len, seq_len)
            
            # Add to content-based scores
            scores = scores + rel_scores
        
        # Apply padding mask if provided
        if key_padding_mask is not None:
            # (batch_size, 1, 1, seq_len)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -1e9)
        
        # Apply attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Transpose and reshape
        attn_output = attn_output.transpose(1, 2)  # (batch_size, seq_len, num_heads, head_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        
        # Final projection
        return self.out_proj(attn_output)
    
    def compute_bond_length_loss(
        self, 
        coords: torch.Tensor, 
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss based on bond length constraints within each nucleotide.
        
        Args:
            coords: Coordinate tensor of shape (batch_size, seq_len, num_atoms, 3)
            mask: Mask tensor of shape (batch_size, seq_len) for valid positions
            
        Returns:
            Bond length loss
        """
        batch_size, seq_len, num_atoms, _ = coords.shape
        total_loss = 0.0
        
        # For each consecutive pair of atoms within a nucleotide
        for i in range(num_atoms - 1):
            # Extract coordinates for the current and next atom
            atom1 = coords[:, :, i, :]      # (batch_size, seq_len, 3)
            atom2 = coords[:, :, i+1, :]    # (batch_size, seq_len, 3)
            
            # Calculate distances between consecutive atoms
            diffs = atom2 - atom1
            distances = torch.sqrt(torch.sum(diffs**2, dim=-1) + 1e-8)  # (batch_size, seq_len)
            
            # Calculate loss based on deviation from ideal bond length
            ideal_length = self.ideal_bond_lengths[i]
            tolerance = self.bond_length_tolerance[i]
            
            # Use smooth L1 loss (Huber loss) for bond length deviations
            bond_deviation = torch.abs(distances - ideal_length)
            length_loss = torch.where(
                bond_deviation < tolerance,
                0.5 * bond_deviation**2 / tolerance,
                bond_deviation - 0.5 * tolerance
            )
            
            # Apply mask
            masked_loss = length_loss * mask.float()
            num_valid = mask.sum()
            
            if num_valid > 0:
                total_loss += masked_loss.sum() / num_valid
                
        # Normalize by number of bond types
        return total_loss / (num_atoms - 1)
    
    def compute_bond_angle_loss(
        self, 
        coords: torch.Tensor, 
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss based on bond angle constraints within each nucleotide.
        
        Args:
            coords: Coordinate tensor of shape (batch_size, seq_len, num_atoms, 3)
            mask: Mask tensor of shape (batch_size, seq_len) for valid positions
            
        Returns:
            Bond angle loss
        """
        batch_size, seq_len, num_atoms, _ = coords.shape
        
        # We need at least 3 atoms to compute angles
        if num_atoms < 3:
            return torch.tensor(0.0, device=coords.device)
        
        total_loss = 0.0
        
        # For each triplet of consecutive atoms within a nucleotide
        for i in range(num_atoms - 2):
            # Extract coordinates for three consecutive atoms
            atom1 = coords[:, :, i, :]      # (batch_size, seq_len, 3)
            atom2 = coords[:, :, i+1, :]    # (batch_size, seq_len, 3)
            atom3 = coords[:, :, i+2, :]    # (batch_size, seq_len, 3)
            
            # Calculate vectors between atoms
            v1 = atom1 - atom2  # Vector from atom2 to atom1
            v2 = atom3 - atom2  # Vector from atom2 to atom3
            
            # Normalize vectors
            v1_norm = F.normalize(v1, p=2, dim=-1)
            v2_norm = F.normalize(v2, p=2, dim=-1)
            
            # Calculate cosine of angle between vectors
            cos_angle = torch.sum(v1_norm * v2_norm, dim=-1)
            # Clamp to avoid numerical issues
            cos_angle = torch.clamp(cos_angle, min=-1.0, max=1.0)
            
            # Most bond angles in RNA are around 109.5 degrees (tetrahedral)
            # Convert to radians: 109.5 degrees ≈ 1.91 radians
            # cos(109.5°) ≈ -0.33
            ideal_cos_angle = -0.33
            
            # Calculate loss - we want the cosine to be close to ideal_cos_angle
            angle_loss = (cos_angle - ideal_cos_angle)**2
            
            # Apply mask
            masked_loss = angle_loss * mask.float()
            num_valid = mask.sum()
            
            if num_valid > 0:
                total_loss += masked_loss.sum() / num_valid
        
        # Normalize by number of angle types
        return total_loss / (num_atoms - 2)
    
    def compute_steric_clash_loss(
        self, 
        coords: torch.Tensor, 
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss based on steric clash constraints between atoms.
        
        Args:
            coords: Coordinate tensor of shape (batch_size, seq_len, num_atoms, 3)
            mask: Mask tensor of shape (batch_size, seq_len) for valid positions
            
        Returns:
            Steric clash loss
        """
        batch_size, seq_len, num_atoms, _ = coords.shape
        
        # Reshape coordinates to (batch_size, seq_len * num_atoms, 3)
        coords_flat = coords.view(batch_size, seq_len * num_atoms, 3)
        
        # Create atom type indices tensor to identify which VdW radius to use
        # Repeat atom indices for each position in sequence
        atom_indices = torch.arange(num_atoms, device=coords.device)
        atom_indices = atom_indices.repeat(seq_len, 1).t().reshape(-1)  # Shape: (seq_len * num_atoms)
        
        # Create a mask for valid atoms - reshape from (batch_size, seq_len) to (batch_size, seq_len * num_atoms)
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, num_atoms).reshape(batch_size, -1)
        
        total_loss = 0.0
        min_sequence_separation = 2  # Atoms in adjacent nucleotides can be close
        
        # Create sequence position tensor to track which nucleotide each atom belongs to
        seq_positions = torch.arange(seq_len, device=coords.device)
        seq_positions = seq_positions.repeat(num_atoms, 1).reshape(-1)  # Shape: (seq_len * num_atoms)
        
        # For each batch item
        for b in range(batch_size):
            # Get coordinates and mask for this batch item
            coords_b = coords_flat[b]  # (seq_len * num_atoms, 3)
            mask_b = mask_expanded[b]  # (seq_len * num_atoms)
            
            # Only consider valid atoms (not masked)
            valid_indices = torch.where(mask_b)[0]
            if len(valid_indices) <= 1:
                continue
            
            valid_coords = coords_b[valid_indices]  # (num_valid, 3)
            valid_atom_indices = atom_indices[valid_indices]  # (num_valid)
            valid_seq_positions = seq_positions[valid_indices]  # (num_valid)
            
            # Calculate pairwise distances between all valid atoms
            num_valid = len(valid_indices)
            diffs = valid_coords.unsqueeze(1) - valid_coords.unsqueeze(0)  # (num_valid, num_valid, 3)
            distances = torch.sqrt(torch.sum(diffs**2, dim=-1) + 1e-8)  # (num_valid, num_valid)
            
            # Get VdW radii for each atom
            radii1 = self.vdw_radii[valid_atom_indices].unsqueeze(1)  # (num_valid, 1)
            radii2 = self.vdw_radii[valid_atom_indices].unsqueeze(0)  # (1, num_valid)
            
            # Calculate minimum allowed distance (sum of VdW radii)
            min_distances = radii1 + radii2  # (num_valid, num_valid)
            
            # Create a mask for pairs of atoms to consider
            # Ignore self-pairs and pairs that are too close in sequence
            pair_mask = torch.ones((num_valid, num_valid), device=coords.device, dtype=torch.bool)
            pair_mask.fill_diagonal_(False)  # Ignore self-pairs
            
            # Calculate sequence separation
            seq_sep = torch.abs(valid_seq_positions.unsqueeze(1) - valid_seq_positions.unsqueeze(0))
            pair_mask = pair_mask & (seq_sep >= min_sequence_separation)
            
            # Calculate clash loss - only consider pairs where distance < min_distance
            # Use a smooth function that approaches zero as distance approaches min_distance
            clash_margin = 0.1  # Additional margin to avoid borderline clashes
            overlap = min_distances + clash_margin - distances  # (num_valid, num_valid)
            clash_loss = F.relu(overlap) ** 2  # Square to penalize larger overlaps more
            
            # Apply pair mask
            masked_clash_loss = clash_loss * pair_mask.float()
            
            # Sum and normalize
            if pair_mask.sum() > 0:
                total_loss += masked_clash_loss.sum() / pair_mask.sum()
        
        # Normalize by batch size
        return total_loss / batch_size
    
    def compute_loss(
        self, 
        pred_coords: torch.Tensor, 
        true_coords: torch.Tensor, 
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        secondary_structure: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the loss between predicted and true coordinates.
        
        Args:
            pred_coords: Predicted coordinates [batch_size, seq_len, num_atoms, 3]
            true_coords: True coordinates [batch_size, seq_len, num_atoms, 3]
            sequence: Input sequence [batch_size, seq_len]
            mask: Mask for valid positions [batch_size, seq_len]
            secondary_structure: Optional secondary structure information
            
        Returns:
            Dictionary with individual loss components and total loss
        """
        batch_size, seq_len = sequence.shape
        
        # Create mask if not provided
        if mask is None:
            mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=pred_coords.device)
            if 4 in sequence:  # 4 is the padding token index
                mask = (sequence != 4)
        
        # Calculate RMSD loss (main prediction loss)
        # Only consider positions that are not masked
        masked_pred = pred_coords * mask.unsqueeze(-1).unsqueeze(-1)
        masked_true = true_coords * mask.unsqueeze(-1).unsqueeze(-1)
        
        # Calculate MSE and RMSD
        mse_loss = F.mse_loss(masked_pred, masked_true, reduction='sum')
        valid_points = mask.sum() * pred_coords.shape[2] * pred_coords.shape[3]  # num_valid * num_atoms * 3
        if valid_points > 0:
            mse_loss = mse_loss / valid_points
        rmsd_loss = torch.sqrt(mse_loss + 1e-8)
        
        losses = {"rmsd": rmsd_loss}
        
        # Calculate physics-based constraints
        if self.bond_length_weight > 0:
            bond_length_loss = self.compute_bond_length_loss(pred_coords, mask)
            losses["bond_length"] = bond_length_loss * self.bond_length_weight
        
        if self.bond_angle_weight > 0:
            bond_angle_loss = self.compute_bond_angle_loss(pred_coords, mask)
            losses["bond_angle"] = bond_angle_loss * self.bond_angle_weight
        
        if self.steric_clash_weight > 0:
            steric_clash_loss = self.compute_steric_clash_loss(pred_coords, mask)
            losses["steric_clash"] = steric_clash_loss * self.steric_clash_weight
        
        # Calculate RNA-specific constraints if enabled
        if self.use_rna_constraints:
            rna_constraint_losses = self.rna_constraint_manager.compute_all_constraints(
                pred_coords, sequence, mask, secondary_structure
            )
            losses.update(rna_constraint_losses)
        
        # Calculate total loss
        total_loss = rmsd_loss
        for name, loss in losses.items():
            if name != "rmsd":
                total_loss += loss
        
        losses["total"] = total_loss
        
        return losses 