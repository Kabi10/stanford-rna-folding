"""
Gradient checkpointing implementation for memory-efficient RNA folding model.

This module provides functions to modify the RNAFoldingModel to use gradient
checkpointing, which reduces memory usage during training at the cost of
some additional computation time.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable

def apply_gradient_checkpointing_to_transformer(
    model: nn.Module,
    is_training: bool,
    use_gradient_checkpointing: bool
) -> nn.Module:
    """
    Apply gradient checkpointing to a PyTorch transformer model.
    
    Args:
        model: The transformer model
        is_training: Whether model is in training mode
        use_gradient_checkpointing: Whether to use gradient checkpointing
        
    Returns:
        Modified model with gradient checkpointing
    """
    # Only apply in training mode and when checkpointing is enabled
    if not (is_training and use_gradient_checkpointing):
        return model
    
    if hasattr(model, "transformer_encoder"):
        # Standard PyTorch transformer
        orig_forward = model.transformer_encoder.forward
        
        def checkpointed_forward(*args, **kwargs):
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            return torch.utils.checkpoint.checkpoint(
                create_custom_forward(orig_forward),
                *args, **kwargs
            )
        
        model.transformer_encoder.forward = checkpointed_forward
    
    elif hasattr(model, "transformer_layers"):
        # Custom transformer with relative position encoding
        for i, layer in enumerate(model.transformer_layers):
            orig_forward = layer.forward
            
            def make_checkpointed_forward(orig_fn, idx):
                def checkpointed_forward(*args, **kwargs):
                    def custom_forward(*inputs):
                        return orig_fn(*inputs)
                    
                    return torch.utils.checkpoint.checkpoint(
                        custom_forward, *args, **kwargs
                    )
                return checkpointed_forward
            
            layer.forward = make_checkpointed_forward(orig_forward, i)
    
    return model

def enable_gradient_checkpointing_for_mlp(
    model: nn.Module,
    is_training: bool,
    use_gradient_checkpointing: bool
) -> nn.Module:
    """
    Apply gradient checkpointing to MLP layers in the model.
    
    Args:
        model: The model containing MLP layers
        is_training: Whether model is in training mode
        use_gradient_checkpointing: Whether to use gradient checkpointing
    
    Returns:
        Modified model with gradient checkpointing for MLP
    """
    # Only apply in training mode and when checkpointing is enabled
    if not (is_training and use_gradient_checkpointing):
        return model
    
    if hasattr(model, "coordinate_mlp") and isinstance(model.coordinate_mlp, nn.Sequential):
        # Wrap the MLP in checkpoint
        orig_forward = model.coordinate_mlp.forward
        
        def checkpointed_forward(x):
            def custom_forward(x_inner):
                return orig_forward(x_inner)
            
            return torch.utils.checkpoint.checkpoint(custom_forward, x)
        
        model.coordinate_mlp.forward = checkpointed_forward
    
    return model

def add_gradient_checkpointing_to_model(
    model_class: type,
    checkpoint_transformer: bool = True,
    checkpoint_mlp: bool = False
) -> type:
    """
    Modify a model class to include gradient checkpointing functionality.
    
    Args:
        model_class: The PyTorch model class to modify
        checkpoint_transformer: Whether to checkpoint the transformer
        checkpoint_mlp: Whether to checkpoint the MLP
        
    Returns:
        Modified model class with gradient checkpointing capabilities
    """
    orig_init = model_class.__init__
    orig_forward = model_class.forward
    
    # Override __init__ to add the checkpointing parameter
    def new_init(self, *args, use_gradient_checkpointing=False, **kwargs):
        orig_init(self, *args, **kwargs)
        self.use_gradient_checkpointing = use_gradient_checkpointing
    
    # Override forward to use checkpointing when enabled
    def new_forward(self, *args, **kwargs):
        if self.use_gradient_checkpointing and self.training:
            # Apply checkpointing to transformer
            if checkpoint_transformer:
                apply_gradient_checkpointing_to_transformer(
                    self, self.training, self.use_gradient_checkpointing
                )
            
            # Apply checkpointing to MLP
            if checkpoint_mlp:
                enable_gradient_checkpointing_for_mlp(
                    self, self.training, self.use_gradient_checkpointing
                )
        
        # Call original forward
        return orig_forward(self, *args, **kwargs)
    
    # Apply the overrides to the class
    model_class.__init__ = new_init
    model_class.forward = new_forward
    
    return model_class 