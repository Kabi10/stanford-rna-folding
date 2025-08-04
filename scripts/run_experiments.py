#!/usr/bin/env python
"""
Experiment runner script for Stanford RNA 3D Folding competition.

This script runs training with different configurations sequentially
and tracks results for comparison.
"""

import argparse
import os
import sys
import yaml
import time
from datetime import datetime
from pathlib import Path
import subprocess
import json

import torch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run experiments for the Stanford RNA 3D Folding competition."
    )
    
    parser.add_argument(
        "--configs", 
        type=str, 
        nargs="+",
        default=["configs/base_config.yaml"],
        help="List of configuration files to run experiments with."
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
        default=42,
        help="Base random seed for reproducibility."
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment_results",
        help="Directory to save experiment results summary."
    )
    
    return parser.parse_args()


def run_experiment(config_path, use_wandb, device, seed):
    """
    Run a single experiment with the given configuration.
    
    Args:
        config_path: Path to the configuration file
        use_wandb: Whether to use Weights & Biases
        device: Device to use for training
        seed: Random seed
        
    Returns:
        Tuple of (success, best_rmsd, save_dir, elapsed_time)
    """
    start_time = time.time()
    
    # Run the training script
    cmd = [
        sys.executable,  # Use the current Python interpreter
        "scripts/train_rna_model.py",
        "--config", config_path,
        "--seed", str(seed)
    ]
    
    if not use_wandb:
        cmd.append("--no-wandb")
        
    if device == "cpu":
        cmd.append("--cpu")
    
    # Run the command
    print(f"\n\n{'='*80}")
    print(f"RUNNING EXPERIMENT: {config_path}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        success = True
        
        # Extract best RMSD from output
        output = result.stdout
        
        # Load config to get save_dir
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        save_dir = config.get("save_dir", "models/stanford-rna-3d-folding")
        
        # Try to find the best RMSD in the output
        best_rmsd = None
        
        for line in output.splitlines():
            if "Best validation RMSD:" in line:
                try:
                    best_rmsd = float(line.split("RMSD:")[1].split()[0].strip())
                except (IndexError, ValueError):
                    pass
                    
        # If we couldn't find RMSD in output, try to load from best model checkpoint
        if best_rmsd is None:
            best_model_path = Path(save_dir) / "best_model.pt"
            if best_model_path.exists():
                try:
                    checkpoint = torch.load(best_model_path, map_location="cpu")
                    best_rmsd = checkpoint.get("val_rmsd", None)
                except:
                    pass
                    
        elapsed_time = time.time() - start_time
        
        print(f"\nExperiment completed in {elapsed_time:.2f} seconds")
        print(f"Best validation RMSD: {best_rmsd}")
        
        return success, best_rmsd, save_dir, elapsed_time
    
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Experiment failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        
        elapsed_time = time.time() - start_time
        return False, None, None, elapsed_time


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Determine device
    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a timestamp for this experiment run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Results summary
    results = {
        "timestamp": timestamp,
        "device": device,
        "seed_base": args.seed,
        "experiments": []
    }
    
    # Run experiments
    for i, config_path in enumerate(args.configs):
        # Use a different seed for each experiment (but deterministic based on base seed)
        seed = args.seed + i
        
        print(f"\nRunning experiment {i+1}/{len(args.configs)}: {config_path}")
        print(f"Using seed: {seed}")
        
        success, best_rmsd, save_dir, elapsed_time = run_experiment(
            config_path=config_path,
            use_wandb=not args.no_wandb,
            device=device,
            seed=seed
        )
        
        # Add to results
        experiment_result = {
            "config_path": config_path,
            "success": success,
            "best_rmsd": best_rmsd,
            "save_dir": save_dir,
            "elapsed_time": elapsed_time,
            "seed": seed
        }
        
        results["experiments"].append(experiment_result)
        
        # Save incrementally
        results_file = os.path.join(args.output_dir, f"experiment_results_{timestamp}.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
            
    # Final summary
    print("\n\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    successful_experiments = [exp for exp in results["experiments"] if exp["success"]]
    failed_experiments = [exp for exp in results["experiments"] if not exp["success"]]
    
    print(f"Total experiments: {len(args.configs)}")
    print(f"Successful: {len(successful_experiments)}")
    print(f"Failed: {len(failed_experiments)}")
    
    if successful_experiments:
        # Sort by best RMSD (ascending)
        sorted_experiments = sorted(
            successful_experiments, 
            key=lambda x: float('inf') if x["best_rmsd"] is None else x["best_rmsd"]
        )
        
        print("\nTop results (lowest RMSD first):")
        print("-" * 80)
        print(f"{'Rank':<6}{'Config':<30}{'RMSD':<10}{'Time (s)':<15}")
        print("-" * 80)
        
        for i, exp in enumerate(sorted_experiments):
            if exp["best_rmsd"] is not None:
                config_name = Path(exp["config_path"]).stem
                print(f"{i+1:<6}{config_name:<30}{exp['best_rmsd']:<10.6f}{exp['elapsed_time']:<15.2f}")
    
    print("\nResults saved to:", results_file)


if __name__ == "__main__":
    main() 