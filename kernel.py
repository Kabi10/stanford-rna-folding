# Kaggle Script Kernel Entry Point
# This script sets up the environment, attaches the dataset, and runs training

import os
import sys
import json
import traceback
import subprocess
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("/kaggle/working/stanford-rna-3d-folding-gpu-training.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

# Redirect stdout/stderr to also write to file
_log_file = open(LOG_PATH, "a", encoding="utf-8")
sys.stdout = Tee(sys.stdout, _log_file)
sys.stderr = Tee(sys.stderr, _log_file)

print("=== Kaggle Script Kernel: Stanford RNA 3D Folding ===")
print(f"Start time: {datetime.utcnow().isoformat()}Z")
print(f"Python: {sys.version}")
print(f"CWD: {os.getcwd()}")

# Debug environment
print("=== Environment Debug ===")
print(f"/kaggle/input contents: {os.listdir('/kaggle/input') if os.path.exists('/kaggle/input') else 'N/A'}")

# Investigate all available datasets
if os.path.exists('/kaggle/input'):
    for dataset_name in os.listdir('/kaggle/input'):
        dataset_path = Path(f"/kaggle/input/{dataset_name}")
        print(f"Dataset '{dataset_name}' contents: {os.listdir(dataset_path) if dataset_path.exists() else 'N/A'}")

        # Check deeper into the directory structure
        if dataset_path.exists():
            contents = os.listdir(dataset_path)
            for item in contents:
                item_path = dataset_path / item
                if item_path.is_dir():
                    print(f"  Subdirectory '{item}' contents: {os.listdir(item_path) if item_path.exists() else 'N/A'}")
                    # Check if this subdirectory contains our dataset
                    if item_path.exists():
                        sub_contents = os.listdir(item_path)
                        for sub_item in sub_contents:
                            sub_item_path = item_path / sub_item
                            if sub_item_path.is_dir():
                                print(f"    Sub-subdirectory '{sub_item}' contents: {os.listdir(sub_item_path) if sub_item_path.exists() else 'N/A'}")

        # Check if this could be our RNA dataset
        if dataset_path.exists():
            contents = os.listdir(dataset_path)
            if any('rna' in item.lower() or 'train' in item.lower() or 'data' in item.lower() for item in contents):
                print(f"Potential RNA dataset found in '{dataset_name}': {contents}")

# Ensure dataset is attached
DATASET_PATH = Path("/kaggle/input/stanford-rna-3d-folding")
if not DATASET_PATH.exists():
    # Try to find any RNA-related dataset
    if os.path.exists('/kaggle/input'):
        datasets = [d for d in os.listdir('/kaggle/input') if 'rna' in d.lower() or 'stanford' in d.lower()]
        if not datasets:
            # Check if any dataset contains RNA-related files or nested structure
            for dataset_name in os.listdir('/kaggle/input'):
                dataset_path = Path(f"/kaggle/input/{dataset_name}")
                if dataset_path.exists():
                    contents = os.listdir(dataset_path)
                    # Check direct contents
                    if any('rna' in item.lower() or 'train' in item.lower() or 'sequence' in item.lower() for item in contents):
                        datasets.append(dataset_name)
                        print(f"Found dataset '{dataset_name}' with RNA-related content: {contents}")
                    # Check nested structure (like d/kabitharma/stanford-rna-3d-folding)
                    else:
                        for item in contents:
                            item_path = dataset_path / item
                            if item_path.is_dir():
                                sub_contents = os.listdir(item_path)
                                for sub_item in sub_contents:
                                    if 'stanford-rna' in sub_item.lower() or 'rna' in sub_item.lower():
                                        potential_path = item_path / sub_item
                                        if potential_path.exists() and potential_path.is_dir():
                                            # Check if this contains our data structure
                                            final_contents = os.listdir(potential_path)
                                            if any('data' in fc.lower() or 'src' in fc.lower() for fc in final_contents):
                                                DATASET_PATH = potential_path
                                                print(f"Found nested dataset at: {DATASET_PATH}")
                                                print(f"Contents: {final_contents}")
                                                break
                                if DATASET_PATH.exists():
                                    break

        if not DATASET_PATH.exists() and datasets:
            DATASET_PATH = Path(f"/kaggle/input/{datasets[0]}")
            print(f"Using dataset: {DATASET_PATH}")

        if not DATASET_PATH.exists():
            print("[ERROR] No RNA or Stanford dataset found in /kaggle/input")
            print(f"Available datasets: {os.listdir('/kaggle/input')}")
            # Don't exit immediately - let's see what's in the available datasets
            for dataset_name in os.listdir('/kaggle/input'):
                dataset_path = Path(f"/kaggle/input/{dataset_name}")
                if dataset_path.exists():
                    print(f"Contents of '{dataset_name}': {os.listdir(dataset_path)}")
            print(f"LOG_SAVED: {LOG_PATH}")
            sys.exit(0)
    else:
        print("[ERROR] /kaggle/input directory not found")
        print(f"LOG_SAVED: {LOG_PATH}")
        sys.exit(0)

print(f"Dataset found: {DATASET_PATH}")
if DATASET_PATH.exists():
    print(f"Dataset contents: {os.listdir(DATASET_PATH)}")

# Add dataset root to sys.path
sys.path.insert(0, str(DATASET_PATH))

# Install dependencies
print("=== Installing Dependencies ===")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "wandb", "biopython"],
                         capture_output=True, text=True)
    print("Dependencies installed successfully")
except Exception as e:
    print(f"[WARN] Dependency installation failed: {e}")

# Show GPU
try:
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU total memory (GB): {getattr(props, 'total_memory', 0)/1e9:.2f}")
except Exception as e:
    print(f"[WARN] Torch import/GPU info issue: {e}")
    traceback.print_exc()

# Import and run training
try:
    from scripts.rna_folding_kaggle_train import main as kaggle_main
    print("Successfully imported kaggle_main")
except Exception as e:
    print("[ERROR] Failed to import kaggle_main:")
    traceback.print_exc()
    print(f"LOG_SAVED: {LOG_PATH}")
    sys.exit(0)

if __name__ == "__main__":
    try:
        model, results = kaggle_main()
        print("Kernel training completed. Results:")
        try:
            print(json.dumps(results, indent=2))
        except Exception:
            print(results)
    except Exception as e:
        print("[ERROR] Exception during training execution:")
        traceback.print_exc()
        results = {"status": "error", "message": str(e)}
        try:
            print(json.dumps(results, indent=2))
        except Exception:
            print(results)
    finally:
        print(f"End time: {datetime.utcnow().isoformat()}Z")
        print(f"LOG_SAVED: {LOG_PATH}")
        try:
            _log_file.flush()
            _log_file.close()
        except Exception:
            pass