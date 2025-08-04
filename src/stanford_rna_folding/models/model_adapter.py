"""
Model adapter for integrating specialized attention mechanisms with RNA folding model.

This module provides adapters that extend the base RNAFoldingModel with 
specialized attention mechanisms, such as hierarchical attention and 
distance-modulated attention, for improved RNA structure prediction.
"""

import logging
from typing import Dict, Optional, Union, Any

import torch
import torch.nn as nn

from stanford_rna_folding.models.rna_folding_model import RNAFoldingModel
from stanford_rna_folding.models.hierarchical_attention import HierarchicalRNAEncoder
from stanford_rna_folding.models.distance_modulated_attention import (
    DistanceModulatedTransformer, 
    IterativeDistanceRefinement
)


logger = logging.getLogger(__name__)


class ModelAdapter:
    """Base adapter for configuring and creating RNA folding models with specialized components."""
    
    @staticmethod
    def create_model_from_config(config: Dict[str, Any]) -> nn.Module:
        """
        Create an RNA folding model from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configured RNA folding model
        """
        model_type = "base"
        
        # Determine model type from configuration
        if config.get("use_hierarchical_attention", False):
            model_type = "hierarchical"
        elif config.get("use_distance_modulation", False):
            model_type = "distance_modulated"
        elif config.get("use_gradient_checkpointing", False):
            model_type = "memory_optimized"
            
        logger.info(f"Creating {model_type} RNA folding model")
        
        # Create the appropriate model based on the type
        if model_type == "hierarchical":
            return HierarchicalAttentionAdapter.create_model(config)
        elif model_type == "distance_modulated":
            return DistanceModulatedAdapter.create_model(config)
        elif model_type == "memory_optimized":
            return MemoryOptimizedAdapter.create_model(config)
        else:
            return RNAFoldingModel(
                vocab_size=config.get("vocab_size", 5),
                embedding_dim=config.get("embedding_dim", 256),
                hidden_dim=config.get("hidden_dim", 512),
                num_layers=config.get("num_layers", 6),
                num_heads=config.get("num_heads", 8),
                dropout=config.get("dropout", 0.1),
                num_atoms=config.get("num_atoms", 5),
                multi_atom_mode=True,
                coord_dims=config.get("coord_dims", 3),
                max_seq_len=config.get("max_seq_len", 500),
                use_rna_constraints=True,
                bond_length_weight=config.get("bond_length_weight", 1.0),
                bond_angle_weight=config.get("bond_angle_weight", 1.0),
                steric_clash_weight=config.get("steric_clash_weight", 1.0),
                watson_crick_weight=config.get("watson_crick_weight", 1.0),
                normalize_coords=config.get("normalize_coords", True),
                use_relative_attention=True,
            )


class HierarchicalAttentionAdapter:
    """Adapter for creating RNA folding models with hierarchical attention."""
    
    @staticmethod
    def create_model(config: Dict[str, Any]) -> nn.Module:
        """
        Create an RNA folding model with hierarchical attention.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            RNA folding model with hierarchical attention
        """
        # Create base model with standard parameters
        model = RNAFoldingModel(
            vocab_size=config.get("vocab_size", 5),
            embedding_dim=config.get("embedding_dim", 256),
            hidden_dim=config.get("hidden_dim", 512),
            num_layers=config.get("num_layers", 6),
            num_heads=config.get("num_heads", 8),
            dropout=config.get("dropout", 0.1),
            num_atoms=config.get("num_atoms", 5),
            multi_atom_mode=True,
            coord_dims=config.get("coord_dims", 3),
            max_seq_len=config.get("max_seq_len", 500),
            use_rna_constraints=True,
            bond_length_weight=config.get("bond_length_weight", 1.0),
            bond_angle_weight=config.get("bond_angle_weight", 1.0),
            steric_clash_weight=config.get("steric_clash_weight", 1.0),
            watson_crick_weight=config.get("watson_crick_weight", 1.0),
            normalize_coords=config.get("normalize_coords", True),
            use_relative_attention=False,  # Disable relative attention since we're replacing it
        )
        
        # Replace the encoder with hierarchical attention encoder
        logger.info("Replacing standard encoder with hierarchical attention encoder")
        model.transformer_encoder = HierarchicalRNAEncoder(
            d_model=config.get("hidden_dim", 512),
            num_layers=config.get("num_layers", 6),
            num_primary_heads=config.get("num_primary_heads", 4),
            num_secondary_heads=config.get("num_secondary_heads", 6),
            num_tertiary_heads=config.get("num_tertiary_heads", 6),
            dim_feedforward=config.get("hidden_dim", 512) * 4,
            dropout=config.get("dropout", 0.1),
            max_seq_len=config.get("max_seq_len", 500),
            pair_aware=config.get("pair_aware", True),
        )
        
        # Add gradient checkpointing if specified
        if config.get("use_gradient_checkpointing", False):
            from stanford_rna_folding.models.gradient_checkpointing import (
                add_gradient_checkpointing_to_model
            )
            model = add_gradient_checkpointing_to_model(model.__class__)(
                *model.__init__.__code__.co_varnames[1:],
                use_gradient_checkpointing=True
            )
        
        return model


class DistanceModulatedAdapter:
    """Adapter for creating RNA folding models with distance-modulated attention."""
    
    @staticmethod
    def create_model(config: Dict[str, Any]) -> nn.Module:
        """
        Create an RNA folding model with distance-modulated attention.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            RNA folding model with distance-modulated attention
        """
        # Create base model with standard parameters
        model = RNAFoldingModel(
            vocab_size=config.get("vocab_size", 5),
            embedding_dim=config.get("embedding_dim", 256),
            hidden_dim=config.get("hidden_dim", 512),
            num_layers=config.get("num_layers", 6),
            num_heads=config.get("num_heads", 8),
            dropout=config.get("dropout", 0.1),
            num_atoms=config.get("num_atoms", 5),
            multi_atom_mode=True,
            coord_dims=config.get("coord_dims", 3),
            max_seq_len=config.get("max_seq_len", 500),
            use_rna_constraints=True,
            bond_length_weight=config.get("bond_length_weight", 1.0),
            bond_angle_weight=config.get("bond_angle_weight", 1.0),
            steric_clash_weight=config.get("steric_clash_weight", 1.0),
            watson_crick_weight=config.get("watson_crick_weight", 1.0),
            normalize_coords=config.get("normalize_coords", True),
            use_relative_attention=False,  # Disable relative attention
        )
        
        # Modify the model architecture for iterative refinement
        use_iterative_refinement = config.get("num_refinement_steps", 0) > 0
        
        if use_iterative_refinement:
            logger.info(f"Using iterative refinement with {config.get('num_refinement_steps')} steps")
            # Replace the encoder with iterative distance refinement
            model.transformer_encoder = IterativeDistanceRefinement(
                d_model=config.get("hidden_dim", 512),
                nhead=config.get("num_heads", 8),
                num_layers=config.get("num_layers", 6),
                num_refinement_steps=config.get("num_refinement_steps", 3),
                dim_feedforward=config.get("hidden_dim", 512) * 4,
                dropout=config.get("dropout", 0.1),
                distance_scaling=config.get("distance_scaling", "inverse"),
                num_atoms=config.get("num_atoms", 5),
                coord_dims=config.get("coord_dims", 3),
            )
            
            # Override the forward method to use the iterative refinement directly
            original_forward = model.forward
            
            def new_forward(self, sequence, lengths=None):
                # Get embeddings
                mask = self.create_padding_mask(sequence) if lengths is None else self._create_mask_from_lengths(lengths)
                embeddings = self.embedding(sequence)
                pos_encoding = self._generate_positional_encoding(embeddings.size(1))
                embeddings = embeddings + pos_encoding.to(embeddings.device)
                
                # Use iterative refinement directly for coordinate prediction
                coords = self.transformer_encoder(
                    src=embeddings,
                    mask=None,
                    src_key_padding_mask=mask,
                )
                
                # Optional coordinate normalization
                if self.normalize_coords:
                    coords = self._normalize_coordinate_output(coords)
                
                return coords
            
            # Patch the forward method
            model.forward = lambda seq, lengths=None: new_forward(model, seq, lengths)
            
        else:
            logger.info("Using standard distance-modulated transformer")
            # Replace the encoder with distance-modulated transformer
            model.transformer_encoder = DistanceModulatedTransformer(
                d_model=config.get("hidden_dim", 512),
                nhead=config.get("num_heads", 8),
                num_layers=config.get("num_layers", 6),
                dim_feedforward=config.get("hidden_dim", 512) * 4,
                dropout=config.get("dropout", 0.1),
                distance_scaling=config.get("distance_scaling", "inverse"),
            )
        
        # Add gradient checkpointing if specified
        if config.get("use_gradient_checkpointing", False):
            from stanford_rna_folding.models.gradient_checkpointing import (
                add_gradient_checkpointing_to_model
            )
            model = add_gradient_checkpointing_to_model(model.__class__)(
                *model.__init__.__code__.co_varnames[1:],
                use_gradient_checkpointing=True
            )
        
        return model


class MemoryOptimizedAdapter:
    """Adapter for creating memory-optimized RNA folding models."""
    
    @staticmethod
    def create_model(config: Dict[str, Any]) -> nn.Module:
        """
        Create a memory-optimized RNA folding model.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Memory-optimized RNA folding model
        """
        from stanford_rna_folding.models.gradient_checkpointing import (
            add_gradient_checkpointing_to_model
        )
        
        # Create base model with gradient checkpointing enabled
        model_class = add_gradient_checkpointing_to_model(RNAFoldingModel)
        
        model = model_class(
            vocab_size=config.get("vocab_size", 5),
            embedding_dim=config.get("embedding_dim", 256),
            hidden_dim=config.get("hidden_dim", 512),
            num_layers=config.get("num_layers", 6),
            num_heads=config.get("num_heads", 8),
            dropout=config.get("dropout", 0.1),
            num_atoms=config.get("num_atoms", 5),
            multi_atom_mode=True,
            coord_dims=config.get("coord_dims", 3),
            max_seq_len=config.get("max_seq_len", 500),
            use_rna_constraints=True,
            bond_length_weight=config.get("bond_length_weight", 1.0),
            bond_angle_weight=config.get("bond_angle_weight", 1.0),
            steric_clash_weight=config.get("steric_clash_weight", 1.0),
            watson_crick_weight=config.get("watson_crick_weight", 1.0),
            normalize_coords=config.get("normalize_coords", True),
            use_relative_attention=True,
            use_gradient_checkpointing=True,
        )
        
        return model