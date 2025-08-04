#!/usr/bin/env python
"""
Train RNA Folding Model

This script provides a command-line interface for training the RNA folding model
with different configurations (single-atom or multi-atom modes).
"""

import os
import argparse
import sys
from pathlib import Path

# Add this directory to path so we can import local modules
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import the main training function
from rna_folding_kaggle import main as train_main


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RNA Folding Model")
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "multi"],
        default="single",
        help="Model mode: 'single' for single-atom or 'multi' for multi-atom prediction"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config file (overrides mode selection)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/output",
        help="Directory to save model outputs"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (overrides config file)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for training (overrides config file)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate (overrides config file)"
    )
    
    parser.add_argument(
        "--embedding-dim",
        type=int,
        help="Embedding dimension (overrides config file)"
    )
    
    parser.add_argument(
        "--hidden-dim",
        type=int, 
        help="Hidden dimension (overrides config file)"
    )
    
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Number of transformer layers (overrides config file)"
    )
    
    return parser.parse_args()


def main():
    """Main function to parse args and start training."""
    args = parse_args()
    
    # Determine config file path based on mode if not specified
    if args.config is None:
        config_dir = Path("configs")
        if args.mode == "single":
            config_path = config_dir / "single_atom_model.yaml"
        else:  # multi
            config_path = config_dir / "multi_atom_model.yaml"
    else:
        config_path = Path(args.config)
    
    # Check if config file exists
    if not config_path.exists():
        print(f"Error: Config file {config_path} not found")
        return 1
    
    # Call the main training function with the selected config
    print(f"Training RNA folding model with configuration: {config_path}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set environment variables for any command-line overrides
    if args.epochs:
        os.environ["RNA_TRAINING_EPOCHS"] = str(args.epochs)
    if args.batch_size:
        os.environ["RNA_BATCH_SIZE"] = str(args.batch_size)
    if args.learning_rate:
        os.environ["RNA_LEARNING_RATE"] = str(args.learning_rate)
    if args.embedding_dim:
        os.environ["RNA_EMBEDDING_DIM"] = str(args.embedding_dim)
    if args.hidden_dim:
        os.environ["RNA_HIDDEN_DIM"] = str(args.hidden_dim)
    if args.num_layers:
        os.environ["RNA_NUM_LAYERS"] = str(args.num_layers)
    
    # Start training
    train_main(config_path=str(config_path))
    return 0


if __name__ == "__main__":
    sys.exit(main()) 