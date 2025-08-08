#!/usr/bin/env python3
"""
Run full training for physics-enhanced transformer model on the full dataset.
This wraps the training utilities and integrates experiment tracking.
"""

from pathlib import Path

import argparse
import os
import sys
# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathlib import Path
import yaml

from src.stanford_rna_folding.training.train import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/full_run_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    exp_name = config.get("experiment_name", "rnafold_exp")
    save_dir = config.get("save_dir", f"experiments/{exp_name}")
    data_dir = config.get("data_dir", "datasets/stanford-rna-3d-folding")

    # Ensure directories
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Train
    model, best_rmsd, best_tm = train_model(
        config=config,
        data_dir=data_dir,
        save_dir=save_dir,
        use_wandb=True,
        device=config.get("device", None)
    )

    print(f"Best RMSD: {best_rmsd:.4f}, Best TM: {best_tm:.4f}")


if __name__ == "__main__":
    main()

