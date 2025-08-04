#!/usr/bin/env python
"""
Training script for Stanford RNA 3D Folding competition.
"""

import argparse
import os
import yaml
from pathlib import Path

import torch
from src.stanford_rna_folding.training.train import train_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a model for the Stanford RNA 3D Folding competition."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/base_config.yaml",
        help="Path to the configuration file."
    )
    parser.add_argument(
        "--no-wandb", 
        action="store_true", 
        help="Disable Weights & Biases logging."
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force using CPU even if GPU is available."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (overrides config)."
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Override seed if provided
    if args.seed is not None:
        config["seed"] = args.seed
    
    # Determine device
    if args.cpu:
        device = "cpu"
    else:
        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    # Print configuration
    print("\n=== Configuration ===")
    print(f"Config file: {args.config}")
    print(f"Device: {device}")
    print(f"Use W&B: {not args.no_wandb}")
    print(f"Random seed: {config['seed']}")
    print("====================\n")
    
    # Create save directory if it doesn't exist
    os.makedirs(config["save_dir"], exist_ok=True)
    
    # Train the model
    model, best_rmsd = train_model(
        config=config,
        data_dir=config["data_dir"],
        save_dir=config["save_dir"],
        use_wandb=not args.no_wandb,
        device=device,
    )
    
    print(f"\nTraining completed! Best validation RMSD: {best_rmsd:.6f}")
    print(f"Model saved to: {os.path.join(config['save_dir'], 'best_model.pt')}")


if __name__ == "__main__":
    main() 