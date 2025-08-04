#!/usr/bin/env python
"""
Training script for RNA 3D structure prediction models.

This script provides a command-line interface for training RNA folding models
with different configurations, including memory-optimized models, hierarchical
attention models, and distance-modulated attention models.
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Local imports
from stanford_rna_folding.models.model_adapter import ModelAdapter
from stanford_rna_folding.training.profiling import measure_memory_usage, profile_model
from stanford_rna_folding.data.dataset import RNAFoldingDataset
from stanford_rna_folding.utils.logger import setup_logging
from stanford_rna_folding.training.trainer import Trainer


logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_experiment_directory(config: Dict[str, Any]) -> Path:
    """
    Set up directory for experiment results.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Path to the experiment directory
    """
    experiment_name = config.get('experiment_name', 'rna_folding_experiment')
    save_dir = config.get('save_dir', 'models/stanford-rna-3d-folding')
    
    # Create experiment directory
    experiment_dir = Path(save_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    checkpoints_dir = experiment_dir / 'checkpoints'
    logs_dir = experiment_dir / 'logs'
    profiling_dir = experiment_dir / 'profiling'
    
    checkpoints_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    profiling_dir.mkdir(exist_ok=True)
    
    return experiment_dir


def create_dataloaders(config: Dict[str, Any]) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of data loaders
    """
    data_dir = Path(config.get('data_dir', 'datasets/stanford-rna-3d-folding'))
    batch_size = config.get('batch_size', 16)
    num_workers = config.get('num_workers', 4)
    
    # Data augmentation settings
    augmentation = {
        'normalize_coords': config.get('normalize_coords', True),
        'random_rotation': config.get('random_rotation', True),
        'random_noise': config.get('random_noise', 0.05),
        'jitter_strength': config.get('jitter_strength', 0.02),
        'atom_mask_prob': config.get('atom_mask_prob', 0.1),
    }
    
    # Create datasets
    train_dataset = RNAFoldingDataset(
        data_dir=data_dir / 'train',
        **augmentation,
        split='train',
    )
    
    val_dataset = RNAFoldingDataset(
        data_dir=data_dir / 'validation',
        normalize_coords=augmentation['normalize_coords'],
        split='validation',
    )
    
    test_dataset = RNAFoldingDataset(
        data_dir=data_dir / 'test',
        normalize_coords=augmentation['normalize_coords'],
        split='test',
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return {
        'train': train_loader,
        'validation': val_loader,
        'test': test_loader,
    }


def setup_optimizer(model: nn.Module, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Set up optimizer and scheduler for training.
    
    Args:
        model: Model to optimize
        config: Configuration dictionary
        
    Returns:
        Dictionary with optimizer and scheduler
    """
    learning_rate = config.get('learning_rate', 3e-4)
    weight_decay = config.get('weight_decay', 1e-5)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Set up scheduler
    scheduler_type = config.get('scheduler_type', 'one_cycle')
    
    if scheduler_type == 'one_cycle':
        # Get number of steps
        batch_size = config.get('batch_size', 16)
        num_epochs = config.get('num_epochs', 100)
        steps_per_epoch = config.get('steps_per_epoch', 1000)  # This will be overridden
        total_steps = num_epochs * steps_per_epoch
        
        # One cycle learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=config.get('pct_start', 0.3),
            div_factor=config.get('div_factor', 25.0),
            final_div_factor=config.get('final_div_factor', 10000.0),
        )
        
    elif scheduler_type == 'cosine':
        # Cosine annealing
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('num_epochs', 100),
            eta_min=learning_rate / 100,
        )
        
    elif scheduler_type == 'linear':
        # Linear decrease
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=config.get('num_epochs', 100),
        )
        
    else:
        # No scheduler
        scheduler = None
    
    return {
        'optimizer': optimizer,
        'scheduler': scheduler,
        'scheduler_type': scheduler_type,
    }


def train_model(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """
    Train an RNA folding model with the given configuration.
    
    Args:
        config: Configuration dictionary
        args: Command-line arguments
    """
    # Set up experiment directory
    experiment_dir = setup_experiment_directory(config)
    
    # Set up logging
    log_file = experiment_dir / 'logs' / 'training.log'
    setup_logging(log_file)
    
    logger.info(f"Starting training with configuration: {config['experiment_name']}")
    
    # Set random seed for reproducibility
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    
    # Determine device
    device_name = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_name)
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    dataloaders = create_dataloaders(config)
    logger.info(f"Created dataloaders with {len(dataloaders['train'])} training batches")
    
    # Update steps per epoch for scheduler
    config['steps_per_epoch'] = len(dataloaders['train'])
    
    # Create model
    model = ModelAdapter.create_model_from_config(config)
    model = model.to(device)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Print model summary
    if args.verbose:
        logger.info(f"Model architecture:\n{model}")
    
    # Set up optimizer and scheduler
    optim_config = setup_optimizer(model, config)
    optimizer = optim_config['optimizer']
    scheduler = optim_config['scheduler']
    
    # Create gradient scaler for mixed precision training
    use_mixed_precision = config.get('use_mixed_precision', True)
    scaler = GradScaler() if use_mixed_precision and torch.cuda.is_available() else None
    
    # Set up gradient accumulation
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        experiment_dir=experiment_dir,
        num_epochs=config.get('num_epochs', 100),
        use_mixed_precision=use_mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        scaler=scaler,
        patience=config.get('patience', 20),
        scheduler_type=optim_config['scheduler_type'],
        keep_last_n_checkpoints=config.get('keep_last_n_checkpoints', 3),
    )
    
    # Profile model if requested
    if args.profile:
        logger.info("Profiling model performance...")
        profile_dir = experiment_dir / 'profiling'
        profile_model(
            model=model,
            dataloader=dataloaders['train'],
            device=device,
            profile_dir=profile_dir,
            num_batches=config.get('profiling_batch_range', 10),
            warmup_batches=3,
            use_autocast=use_mixed_precision,
        )
        
        # Log memory usage
        memory_usage = measure_memory_usage()
        logger.info(f"Memory usage: {memory_usage}")
    
    # Train model
    logger.info("Starting training...")
    best_model_path = trainer.train()
    
    # Log best model path
    logger.info(f"Training completed. Best model saved at: {best_model_path}")
    
    # Test best model
    logger.info("Evaluating best model on test set...")
    test_metrics = trainer.evaluate(dataloaders['test'], best_model_path)
    
    # Log test metrics
    logger.info(f"Test metrics: {test_metrics}")


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(description='Train RNA folding model')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML file')
    parser.add_argument('--profile', action='store_true', help='Profile model performance')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Train model
    train_model(config, args)


if __name__ == '__main__':
    main()