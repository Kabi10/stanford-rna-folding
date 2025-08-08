# Kaggle Script Kernel Entry Point
# This script sets up the environment, attaches the dataset, and runs training

import os
import sys
from pathlib import Path

print("=== Kaggle Script Kernel: Stanford RNA 3D Folding ===")

# Ensure dataset is attached
DATASET_PATH = Path("/kaggle/input/stanford-rna-3d-folding")
if not DATASET_PATH.exists():
    raise RuntimeError("Dataset '/kaggle/input/stanford-rna-3d-folding' not found. Attach it in the kernel settings.")
print(f"Dataset found: {DATASET_PATH}")

# Add dataset root (contains src/, configs/, scripts/) to sys.path
sys.path.insert(0, str(DATASET_PATH))

# Show GPU
try:
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"Torch import issue: {e}")

# Run Kaggle training script main()
from scripts.rna_folding_kaggle_train import main as kaggle_main

if __name__ == "__main__":
    model, results = kaggle_main()
    print("Kernel training completed. Results:")
    print(results)

