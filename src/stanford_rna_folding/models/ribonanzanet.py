"""
RibonanzaNet integration module for RNA 3D structure prediction.

This module provides functionality to leverage the pre-trained RibonanzaNet model
for enhancing RNA 3D structure prediction through transfer learning and feature
extraction.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class RibonanzaNetFeatureExtractor(nn.Module):
    """
    Feature extractor for RibonanzaNet embeddings and attention maps.
    """
    
    def __init__(
        self,
        pretrained_path: str,
        embedding_dim: int = 768,
        freeze_base: bool = True,
        layer_pooling: str = "last",  # Options: "last", "all", "weighted"
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Initialize the RibonanzaNet feature extractor.
        
        Args:
            pretrained_path: Path to pretrained RibonanzaNet model
            embedding_dim: Dimension of RibonanzaNet embeddings
            freeze_base: Whether to freeze the base RibonanzaNet model initially
            layer_pooling: Method for pooling features from different layers
            device: Device to run the model on
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.freeze_base_initially = freeze_base  # Store initial freeze state
        self.layer_pooling = layer_pooling
        self.device = device
        
        # Load pretrained model
        self.load_pretrained_model(pretrained_path)
        
        if self.freeze_base_initially:
            self.freeze_parameters()
        
        # Layer-wise feature weights (if using weighted pooling)
        if layer_pooling == "weighted":
            # Ensure base_model is loaded before accessing config
            if hasattr(self, 'base_model') and hasattr(self.base_model, 'config'):
                num_layers = self.base_model.config.num_hidden_layers
                self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
            else:
                logger.warning("Base model not loaded, cannot initialize layer weights for weighted pooling.")
                self.layer_weights = None
    
    def load_pretrained_model(self, pretrained_path: str):
        """Load the pretrained RibonanzaNet model."""
        try:
            # Import here to avoid dependency if not using RibonanzaNet
            from transformers import AutoModel, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
            self.base_model = AutoModel.from_pretrained(pretrained_path)
            self.base_model.to(self.device)
            
            logger.info(f"Successfully loaded RibonanzaNet from {pretrained_path}")
            
        except ImportError:
            logger.error("Transformers library not installed. Cannot load RibonanzaNet.")
            raise
        except Exception as e:
            logger.error(f"Failed to load RibonanzaNet from {pretrained_path}: {str(e)}")
            # Potentially fall back or raise a more specific error
            raise
    
    def freeze_parameters(self):
        """Freeze all parameters of the base model."""
        if hasattr(self, 'base_model'):
            for param in self.base_model.parameters():
                param.requires_grad = False
            logger.info("Froze all RibonanzaNet base model parameters.")
        else:
            logger.warning("Base model not loaded, cannot freeze parameters.")
    
    def unfreeze_layers(self, num_layers_to_unfreeze: int):
        """Unfreeze the top N layers of the RibonanzaNet base model."""
        if not hasattr(self, 'base_model') or not hasattr(self.base_model, 'encoder') or not hasattr(self.base_model.encoder, 'layer'):
            logger.warning("RibonanzaNet base model or its layers not found. Cannot unfreeze.")
            return

        total_layers = len(self.base_model.encoder.layer)
        if num_layers_to_unfreeze > total_layers:
            logger.warning(f"Requested to unfreeze {num_layers_to_unfreeze} layers, but model only has {total_layers}. Unfreezing all layers.")
            num_layers_to_unfreeze = total_layers
        elif num_layers_to_unfreeze < 0:
            logger.warning("Number of layers to unfreeze cannot be negative. Keeping all layers frozen.")
            num_layers_to_unfreeze = 0
        
        # Freeze all first (if not initially frozen)
        if not self.freeze_base_initially:
             self.freeze_parameters()

        # Unfreeze the top N layers
        if num_layers_to_unfreeze > 0:
            for layer in self.base_model.encoder.layer[-num_layers_to_unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True
            logger.info(f"Unfroze the top {num_layers_to_unfreeze} layers of the RibonanzaNet base model.")

        # Ensure the embedding layer remains frozen/unfrozen based on initial setting
        if hasattr(self.base_model, 'embeddings'):
             for param in self.base_model.embeddings.parameters():
                 param.requires_grad = not self.freeze_base_initially

    def forward(
        self,
        sequences: List[str],
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract features from RNA sequences using RibonanzaNet.
        
        Args:
            sequences: List of RNA sequences
            return_attention: Whether to return attention maps
            
        Returns:
            Embeddings tensor of shape (batch_size, seq_len, embedding_dim)
            Optional attention maps of shape (batch_size, num_heads, seq_len, seq_len)
        """
        if not hasattr(self, 'base_model'):
             raise RuntimeError("RibonanzaNet base model has not been loaded.")
             
        # Tokenize sequences
        inputs = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Get model outputs
        # Determine if gradients should be enabled based on whether any part is unfrozen
        is_training = any(p.requires_grad for p in self.base_model.parameters())
        with torch.set_grad_enabled(is_training):
            outputs = self.base_model(
                **inputs,
                output_hidden_states=True,
                output_attentions=return_attention,
            )
        
        # Process hidden states based on pooling strategy
        embeddings = None
        if self.layer_pooling == "last":
            embeddings = outputs.last_hidden_state
        elif self.layer_pooling == "all":
            # Average across all layers
            hidden_states = torch.stack(outputs.hidden_states, dim=0)
            embeddings = torch.mean(hidden_states, dim=0)
        elif self.layer_pooling == "weighted" and self.layer_weights is not None:
            # Weighted sum of layer features
            hidden_states = torch.stack(outputs.hidden_states, dim=0)
            weights = F.softmax(self.layer_weights, dim=0)
            # Ensure weights are on the correct device
            weights = weights.to(hidden_states.device)
            embeddings = torch.sum(hidden_states * weights.view(-1, 1, 1, 1), dim=0)
        else:
            logger.warning(f"Invalid layer_pooling strategy '{self.layer_pooling}' or weights not initialized. Falling back to last hidden state.")
            embeddings = outputs.last_hidden_state

        # Pad/truncate embeddings to match expected rna_model input length if necessary
        # This requires knowing the target length expected by rna_model
        # target_seq_len = self.rna_model.config.max_seq_len # Example: Get target length from config
        # current_seq_len = embeddings.shape[1]
        # if current_seq_len < target_seq_len:
        #     padding = torch.zeros(embeddings.shape[0], target_seq_len - current_seq_len, embeddings.shape[2], device=self.device)
        #     embeddings = torch.cat([embeddings, padding], dim=1)
        # elif current_seq_len > target_seq_len:
        #     embeddings = embeddings[:, :target_seq_len, :]

        if return_attention:
            # Average attention maps across layers and heads
            # Ensure attentions are available
            if outputs.attentions is None:
                 logger.warning("Attention maps requested but not available in model output.")
                 attention_maps = None
            else:
                 attention_maps = torch.stack(outputs.attentions, dim=1)  # (batch, layers, heads, seq, seq)
                 attention_maps = attention_maps.mean(dim=[1, 2])  # (batch, seq, seq)
            return embeddings, attention_maps
        
        return embeddings

class RibonanzaHybridModel(nn.Module):
    """
    Hybrid model combining RibonanzaNet features with physics-based constraints.
    """
    
    def __init__(
        self,
        feature_extractor: RibonanzaNetFeatureExtractor,
        rna_model: nn.Module,
        integration_mode: str = "concat",  # Options: "concat", "add", "gate"
        use_attention_guide: bool = True,
    ):
        """
        Initialize the hybrid model.
        
        Args:
            feature_extractor: RibonanzaNet feature extractor
            rna_model: Base RNA folding model
            integration_mode: How to integrate RibonanzaNet features
            use_attention_guide: Whether to use attention maps for guidance
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.rna_model = rna_model
        self.integration_mode = integration_mode
        self.use_attention_guide = use_attention_guide
        
        # Feature integration layers
        if integration_mode == "concat":
            self.feature_proj = nn.Linear(
                feature_extractor.embedding_dim + rna_model.embedding_dim,
                rna_model.embedding_dim
            )
        elif integration_mode == "gate":
            self.gate = nn.Sequential(
                nn.Linear(feature_extractor.embedding_dim, rna_model.embedding_dim),
                nn.Sigmoid()
            )
    
    def unfreeze_ribonanza_layers(self, num_layers_to_unfreeze: int):
         """Helper method to call unfreeze_layers on the feature extractor."""
         self.feature_extractor.unfreeze_layers(num_layers_to_unfreeze)

    def forward(
        self,
        sequences: List[str],
        sequence_encoding: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the hybrid model.
        
        Args:
            sequences: List of RNA sequences for RibonanzaNet
            sequence_encoding: Encoded sequences for RNA model
            
        Returns:
            Predicted 3D coordinates
        """
        # Get RibonanzaNet features
        if self.use_attention_guide:
            ribonanza_features, attention_maps = self.feature_extractor(
                sequences, return_attention=True
            )
        else:
            ribonanza_features = self.feature_extractor(sequences)
            attention_maps = None
        
        # Integrate features based on selected mode
        if self.integration_mode == "concat":
            # Concatenate and project features
            combined_features = torch.cat([ribonanza_features, sequence_encoding], dim=-1)
            integrated_features = self.feature_proj(combined_features)
            
        elif self.integration_mode == "add":
            # Simple addition (assuming dimensions match)
            integrated_features = ribonanza_features + sequence_encoding
            
        elif self.integration_mode == "gate":
            # Gated integration
            gates = self.gate(ribonanza_features)
            integrated_features = gates * sequence_encoding + (1 - gates) * ribonanza_features
        
        # Forward pass through RNA model with integrated features
        if self.use_attention_guide and attention_maps is not None:
            coords = self.rna_model(integrated_features, attention_guide=attention_maps)
        else:
            coords = self.rna_model(integrated_features)
        
        return coords 