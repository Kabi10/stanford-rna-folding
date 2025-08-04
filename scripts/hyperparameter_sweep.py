#!/usr/bin/env python
"""
Hyperparameter Sweep for RNA 3D Folding

This script runs a systematic hyperparameter optimization using Weights & Biases Sweeps.
It defines a search space for model architecture, training parameters, and loss function
weights to find optimal configurations for RNA structure prediction.
"""

import os
import sys
import argparse
import yaml
import random
import numpy as np
import torch
from pathlib import Path
import wandb

# Add parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stanford_rna_folding.training.train import train_model_with_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RNA Folding Hyperparameter Sweep")
    
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to base configuration file"
    )
    
    parser.add_argument(
        "--sweep-config",
        type=str,
        default="configs/sweep_config.yaml",
        help="Path to sweep configuration file (optional)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/hyperparameter_sweep",
        help="Directory to save model outputs"
    )
    
    parser.add_argument(
        "--project",
        type=str,
        default="rna_folding_sweep",
        help="W&B project name"
    )
    
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity (team) name"
    )
    
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of runs to execute"
    )
    
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use (if multiple available)"
    )
    
    parser.add_argument(
        "--optimize-metric",
        type=str,
        choices=["rmsd", "tm_score"],
        default="tm_score",
        help="Metric to optimize (rmsd or tm_score)"
    )
    
    return parser.parse_args()


def create_sweep_config(base_config_path, optimize_metric="tm_score"):
    """
    Create a default sweep configuration if none is provided.
    
    Args:
        base_config_path: Path to base configuration file
        optimize_metric: Metric to optimize (rmsd or tm_score)
        
    Returns:
        Dictionary with sweep configuration
    """
    # Load base configuration
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Define metric to optimize
    if optimize_metric == "tm_score":
        # Higher is better for TM-score
        metric = {"name": "val_tm_score", "goal": "maximize"}
    else:
        # Lower is better for RMSD
        metric = {"name": "val_rmsd", "goal": "minimize"}
    
    # Define hyperparameter search space
    sweep_config = {
        "method": "bayes",  # Bayesian optimization
        "metric": metric,
        "parameters": {
            # Model architecture parameters
            "embedding_dim": {
                "values": [96, 128, 160, 192, 224, 256]
            },
            "hidden_dim": {
                "values": [192, 256, 320, 384, 448, 512]
            },
            "num_layers": {
                "values": [3, 4, 5, 6, 7, 8]
            },
            "num_heads": {
                "values": [4, 8, 10, 12, 14, 16]
            },
            "dropout": {
                "distribution": "uniform",
                "min": 0.05,
                "max": 0.3
            },
            
            # Training parameters
            "batch_size": {
                "values": [8, 12, 16, 20, 24, 32]
            },
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-3
            },
            "weight_decay": {
                "distribution": "log_uniform_values",
                "min": 1e-6,
                "max": 1e-4
            },
            "gradient_accumulation_steps": {
                "values": [1, 2, 4, 8]
            },
            "use_mixed_precision": {
                "values": [True, False]
            },
            
            # Optimizer and scheduler parameters
            "optimizer": {
                "values": ["adam", "adamw", "radam"]
            },
            "scheduler": {
                "values": ["reduce_on_plateau", "cosine_annealing", "one_cycle"]
            },
            
            # Physics-based constraint weights
            "bond_length_weight": {
                "distribution": "uniform",
                "min": 0.1,
                "max": 0.5
            },
            "bond_angle_weight": {
                "distribution": "uniform",
                "min": 0.1,
                "max": 0.5
            },
            "steric_clash_weight": {
                "distribution": "uniform",
                "min": 0.2,
                "max": 0.6
            },
            "watson_crick_weight": {
                "distribution": "uniform",
                "min": 1.0,
                "max": 3.0
            },
            
            # RNA-specific architecture parameters
            "use_rna_constraints": {
                "values": [True, False]
            },
            "multi_atom_mode": {
                "values": [True, False]
            },
            
            # Data augmentation
            "random_rotation": {
                "values": [True, False]
            },
            "random_noise": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 0.1
            },
            "jitter_strength": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 0.05
            },
            "atom_mask_prob": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 0.2
            }
        }
    }
    
    return sweep_config


def train_function():
    """
    Training function executed by wandb.agent for each sweep run.
    """
    # Set seeds for reproducibility
    seed = wandb.config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, f"run_{wandb.run.id}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load base configuration and update with sweep parameters
    with open(args.base_config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with sweep parameters
    for key, value in wandb.config.items():
        if key in config:
            config[key] = value
    
    # Update derived parameters
    config['experiment_name'] = f"sweep_{wandb.run.id}"
    config['save_dir'] = output_dir
    
    # Save the configuration for reference
    config_path = os.path.join(output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Train the model with this configuration
    model_path = train_model_with_config(
        config=config,
        wandb_run=wandb.run  # Pass existing wandb run instead of creating a new one
    )
    
    # Log the model path as an artifact
    model_artifact = wandb.Artifact(f"model_{wandb.run.id}", type="model")
    model_artifact.add_file(model_path)
    wandb.log_artifact(model_artifact)
    
    return model_path


def main(args):
    """Main function to run hyperparameter sweep."""
    # Set device
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU: {args.gpu}")
    else:
        print("No GPU available, using CPU")
    
    # Initialize wandb
    wandb.login()
    
    # Create sweep configuration (either from file or default)
    if os.path.exists(args.sweep_config):
        with open(args.sweep_config, 'r') as f:
            sweep_config = yaml.safe_load(f)
        print(f"Loaded sweep configuration from {args.sweep_config}")
    else:
        sweep_config = create_sweep_config(args.base_config, args.optimize_metric)
        print("Created default sweep configuration")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the sweep configuration for reference
    sweep_config_path = os.path.join(args.output_dir, 'sweep_config.yaml')
    with open(sweep_config_path, 'w') as f:
        yaml.dump(sweep_config, f, default_flow_style=False)
    
    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep_config,
        project=args.project,
        entity=args.entity
    )
    
    print(f"Created sweep with ID: {sweep_id}")
    print(f"Starting agent to run {args.count} trials...")
    
    # Start sweep agent
    wandb.agent(sweep_id, function=train_function, count=args.count)
    
    print("Sweep completed!")
    print(f"Results saved to {args.output_dir}")
    
    # Optionally, analyze the sweep results and suggest the best config
    print("\nAnalyzing sweep results...")
    api = wandb.Api()
    sweep = api.sweep(f"{args.project}/{sweep_id}" if args.entity is None else f"{args.entity}/{args.project}/{sweep_id}")
    
    # Find the best run
    if args.optimize_metric == "tm_score":
        best_run = max(sweep.runs, key=lambda run: run.summary.get("val_tm_score", 0))
    else:
        best_run = min(sweep.runs, key=lambda run: run.summary.get("val_rmsd", float('inf')))
    
    print(f"\nBest run: {best_run.name} (ID: {best_run.id})")
    print(f"Best {args.optimize_metric}: {best_run.summary.get(f'val_{args.optimize_metric}', 'N/A')}")
    print("\nBest hyperparameters:")
    for key, value in best_run.config.items():
        print(f"  {key}: {value}")
    
    # Save the best configuration
    best_config_path = os.path.join(args.output_dir, 'best_config.yaml')
    with open(best_config_path, 'w') as f:
        yaml.dump(best_run.config, f, default_flow_style=False)
    
    print(f"\nBest configuration saved to {best_config_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args) 