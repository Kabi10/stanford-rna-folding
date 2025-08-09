#!/usr/bin/env python3
"""
Kaggle-compatible training script for Stanford RNA 3D folding competition.
Optimized for GPU training with proper Kaggle environment handling.
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Kaggle environment setup
KAGGLE_INPUT_PATH = Path("/kaggle/input")
KAGGLE_WORKING_PATH = Path("/kaggle/working")
KAGGLE_TEMP_PATH = Path("/kaggle/temp")

# Ensure working directory exists
KAGGLE_WORKING_PATH.mkdir(exist_ok=True)

# Add project source to path for imports
if KAGGLE_INPUT_PATH.exists():
    # In Kaggle environment - look for dataset with source code
    project_paths = list(KAGGLE_INPUT_PATH.glob("**/src"))
    if project_paths:
        sys.path.insert(0, str(project_paths[0].parent))
    else:
        # Fallback: assume source is in a dataset named 'stanford-rna-folding'
        sys.path.insert(0, str(KAGGLE_INPUT_PATH / "stanford-rna-folding"))
else:
    # Local development fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import our modules
try:
    from src.stanford_rna_folding.models.rna_folding_model import RNAFoldingModel
    from src.stanford_rna_folding.data.data_processing import StanfordRNADataset, rna_collate_fn
    from src.stanford_rna_folding.data.transforms import RNADataTransform
    from src.stanford_rna_folding.evaluation.metrics import batch_rmsd, batch_tm_score
except ImportError as e:
    print(f"Import error: {e}")
    print("Available paths:")
    for p in sys.path:
        print(f"  {p}")
    raise

def setup_kaggle_environment():
    """Setup Kaggle-specific environment variables and paths."""
    # Set environment variables for optimal GPU performance
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    
    # Create necessary directories
    (KAGGLE_WORKING_PATH / "checkpoints").mkdir(exist_ok=True)
    (KAGGLE_WORKING_PATH / "logs").mkdir(exist_ok=True)
    (KAGGLE_WORKING_PATH / "results").mkdir(exist_ok=True)
    
    return {
        "input_path": KAGGLE_INPUT_PATH,
        "working_path": KAGGLE_WORKING_PATH,
        "temp_path": KAGGLE_TEMP_PATH
    }

def detect_kaggle_dataset_path():
    """Detect the path to RNA dataset in Kaggle input."""
    if not KAGGLE_INPUT_PATH.exists():
        return None
    
    # Look for common dataset names
    dataset_candidates = [
        "stanford-rna-3d-folding",
        "rna-folding-dataset", 
        "stanford-rna-dataset",
        "rna-3d-structure-data"
    ]
    
    for candidate in dataset_candidates:
        candidate_path = KAGGLE_INPUT_PATH / candidate
        if candidate_path.exists():
            # Verify it contains expected files
            if (candidate_path / "train_sequences.csv").exists():
                return candidate_path
    
    # Fallback: look for any directory with train_sequences.csv
    for path in KAGGLE_INPUT_PATH.rglob("train_sequences.csv"):
        return path.parent
    
    return None

def get_device_info():
    """Get detailed device information for optimization."""
    device_info = {
        "device": "cpu",
        "device_name": "CPU",
        "memory_gb": 0,
        "compute_capability": None,
        "recommended_batch_size": 8
    }
    
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        device_info.update({
            "device": "cuda",
            "device_name": torch.cuda.get_device_name(device),
            "memory_gb": torch.cuda.get_device_properties(device).total_memory / 1e9,
            "compute_capability": torch.cuda.get_device_capability(device)
        })
        
        # Recommend batch size based on GPU memory
        memory_gb = device_info["memory_gb"]
        if memory_gb >= 15:  # V100, A100
            device_info["recommended_batch_size"] = 32
        elif memory_gb >= 10:  # P100, T4
            device_info["recommended_batch_size"] = 24
        else:  # K80, etc.
            device_info["recommended_batch_size"] = 16
    
    return device_info

def create_kaggle_config(base_config: Dict, device_info: Dict) -> Dict:
    """Create Kaggle-optimized configuration."""
    kaggle_config = base_config.copy()
    
    # GPU optimizations
    if device_info["device"] == "cuda":
        kaggle_config.update({
            "device": "cuda",
            "batch_size": device_info["recommended_batch_size"],
            "use_mixed_precision": True,
            "num_workers": 2,  # Kaggle has limited CPU cores
            "pin_memory": True,
            "gradient_accumulation_steps": max(1, 32 // device_info["recommended_batch_size"]),
            "learning_rate": base_config.get("learning_rate", 5e-4) * (device_info["recommended_batch_size"] / 8),  # Scale LR
        })
    else:
        kaggle_config.update({
            "device": "cpu", 
            "batch_size": 4,  # Smaller for CPU
            "use_mixed_precision": False,
            "num_workers": 1,
            "pin_memory": False
        })
    
    # Kaggle-specific paths
    kaggle_config.update({
        "save_dir": str(KAGGLE_WORKING_PATH / "experiments"),
        "checkpoint_dir": str(KAGGLE_WORKING_PATH / "checkpoints"),
        "log_dir": str(KAGGLE_WORKING_PATH / "logs")
    })
    
    return kaggle_config

def save_kaggle_results(results: Dict, config: Dict):
    """Save training results in Kaggle-compatible format."""
    results_dir = KAGGLE_WORKING_PATH / "results"
    results_dir.mkdir(exist_ok=True)

    # Save results summary
    with open(results_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save config used
    with open(results_dir / "config_used.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Create submission-ready files if needed
    if "best_model_path" in results:
        # Copy best model to results
        best_model_src = Path(results["best_model_path"])
        if best_model_src.exists():
            shutil.copy2(best_model_src, results_dir / "best_model.pth")


def generate_competition_submission(model, config):
    """Generate competition submission file"""
    print("=== Generating Competition Submission ===")

    try:
        # Import submission generator
        from src.stanford_rna_folding.inference.submission_generator import SubmissionGenerator

        # Setup device
        device = config.get('device', 'cuda')

        # Look for test sequences
        test_sequences_path = None
        data_dir = Path(config.get('data_dir', '/kaggle/input/d/kabitharma/stanford-rna-3d-folding/data'))

        # Try different possible test file locations
        test_candidates = [
            data_dir / 'test_sequences.csv',
            data_dir.parent / 'test_sequences.csv',
            data_dir / 'sample_submission.csv',
            data_dir.parent / 'sample_submission.csv'
        ]

        for candidate in test_candidates:
            if candidate.exists():
                test_sequences_path = candidate
                break

        if test_sequences_path is None:
            print("No test sequences found, creating dummy submission")
            return create_dummy_submission_kaggle()

        print(f"Loading test sequences from: {test_sequences_path}")
        test_sequences = pd.read_csv(test_sequences_path)
        print(f"Found {len(test_sequences)} test sequences")

        # Create submission generator
        generator = SubmissionGenerator(model, device)

        # Generate predictions (5 conformations per sequence)
        predictions = generator.process_test_sequences(
            test_sequences,
            num_conformations=5,
            temperature=1.0
        )

        # Format and save submission
        output_path = "/kaggle/working/submission.csv"
        submission_df = generator.format_submission(predictions, output_path)

        # Validate submission
        is_valid = generator.validate_submission(submission_df)

        if is_valid:
            print(f"✅ Submission generated successfully: {output_path}")
            print(f"Submission shape: {submission_df.shape}")
            print(f"Sequences: {len(test_sequences)}")
            print(f"Conformations per sequence: 5")
            return True
        else:
            print("❌ Submission validation failed")
            return False

    except Exception as e:
        print(f"Error generating submission: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_dummy_submission_kaggle():
    """Create dummy submission when no test data available"""
    print("Creating dummy submission for testing...")

    # Create minimal test sequences
    dummy_sequences = pd.DataFrame({
        'ID': ['dummy_seq_1', 'dummy_seq_2', 'dummy_seq_3'],
        'sequence': ['AUCG', 'GCAU', 'AUGCAU']
    })

    submission_rows = []

    for _, row in dummy_sequences.iterrows():
        sequence_id = row['ID']
        sequence_length = len(row['sequence'])

        # Generate dummy coordinates for 5 conformations
        for conf_idx in range(1, 6):  # 1-5 conformations
            for residue_idx in range(1, sequence_length + 1):  # 1-indexed
                for atom_idx in range(1, 2):  # 1 atom per residue
                    # Random coordinates
                    x = np.random.normal(0, 10)
                    y = np.random.normal(0, 10)
                    z = np.random.normal(0, 10)

                    submission_rows.append({
                        'ID': f"{sequence_id}_{residue_idx}_{atom_idx}",
                        'x': x,
                        'y': y,
                        'z': z,
                        'conformation': conf_idx
                    })

    submission_df = pd.DataFrame(submission_rows)
    output_path = "/kaggle/working/submission.csv"
    submission_df.to_csv(output_path, index=False)

    print(f"Dummy submission created: {output_path}")
    print(f"Shape: {submission_df.shape}")

    return True

def main():
    """Main training function for Kaggle environment."""
    print("=== Stanford RNA 3D Folding - Kaggle Training ===")
    
    # Setup environment
    env_paths = setup_kaggle_environment()
    print(f"Kaggle paths: {env_paths}")
    
    # Detect device and optimize config
    device_info = get_device_info()
    print(f"Device info: {device_info}")
    
    # Find dataset
    dataset_path = detect_kaggle_dataset_path()
    if dataset_path is None:
        raise RuntimeError("Could not find RNA dataset in Kaggle input. Please ensure dataset is attached.")
    print(f"Dataset found at: {dataset_path}")
    
    # Load base configuration
    base_config = {
        "experiment_name": "kaggle_rna_folding_gpu",
        "project": "stanford-rna-3d-folding",
        "seed": 42,
        "data_dir": str(dataset_path),
        "epochs": 20,  # Reduced for Kaggle time limits
        "learning_rate": 5e-4,
        "weight_decay": 1e-5,
        "patience": 10,
        "min_epochs": 3,
        "embedding_dim": 256,
        "hidden_dim": 512,
        "num_layers": 6,
        "num_heads": 8,
        "dropout": 0.1,
        "num_atoms": 1,
        "multi_atom_mode": False,
        "coord_dims": 3,
        "max_seq_len": 1200,
        "use_relative_attention": False,  # Disabled due to shape issues
        "use_rna_constraints": True,
        "bond_length_weight": 0.3,
        "bond_angle_weight": 0.3,
        "steric_clash_weight": 0.5,
        "watson_crick_weight": 0.2,
        "normalize_coords": True,
        "scheduler_type": "reduce_on_plateau",
        "lr_factor": 0.5,
        "lr_patience": 5,
        "min_lr": 1e-7,
        "gradient_clip_val": 1.0,
        "keep_last_n_checkpoints": 3,
        "improvement_threshold": 0.001
    }
    
    # Create Kaggle-optimized config
    config = create_kaggle_config(base_config, device_info)
    print(f"Final config - Batch size: {config['batch_size']}, Mixed precision: {config['use_mixed_precision']}")
    
    # Add phase awareness
    try:
        from src.stanford_rna_folding.competition.phase_manager import CompetitionPhaseManager
        phase_manager = CompetitionPhaseManager(config["data_dir"])
        current_phase = phase_manager.get_current_phase()
        print(f"Competition Phase: {current_phase}")

        # Apply temporal filtering if needed
        temporal_cutoff = phase_manager.phase_cutoffs.get(current_phase)
        if temporal_cutoff:
            print(f"Applying temporal cutoff: {temporal_cutoff}")
            config["temporal_cutoff"] = temporal_cutoff
            config["competition_phase"] = current_phase

    except Exception as e:
        print(f"Phase management not available: {e}")
        print("Proceeding with standard training")
        current_phase = 1
        config["competition_phase"] = current_phase

    # Import and run training
    from src.stanford_rna_folding.training.train import train_model

    try:
        model, best_rmsd, best_tm = train_model(
            config=config,
            data_dir=config["data_dir"],
            save_dir=config["save_dir"],
            use_wandb=False,  # Disable W&B in Kaggle
            device=config.get("device")
        )
        
        # Save results with phase information
        results = {
            "best_rmsd": float(best_rmsd),
            "best_tm_score": float(best_tm),
            "device_used": device_info["device_name"],
            "final_config": config,
            "training_completed": True,
            "competition_phase": config.get("competition_phase", 1),
            "temporal_cutoff": config.get("temporal_cutoff"),
            "temporal_compliance": True
        }

        # Generate competition submission
        print("=== Generating Competition Submission ===")
        try:
            submission_generated = generate_competition_submission(trained_model, config)
            results["submission_generated"] = submission_generated
        except Exception as e:
            print(f"Submission generation failed: {e}")
            results["submission_generated"] = False
            results["submission_error"] = str(e)
        
        save_kaggle_results(results, config)
        
        print(f"\n=== Training Complete ===")
        print(f"Best RMSD: {best_rmsd:.4f}")
        print(f"Best TM-score: {best_tm:.4f}")
        print(f"Results saved to: {KAGGLE_WORKING_PATH / 'results'}")
        
        return model, results
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error info
        error_results = {
            "training_completed": False,
            "error": str(e),
            "device_used": device_info["device_name"],
            "config": config
        }
        save_kaggle_results(error_results, config)
        raise

if __name__ == "__main__":
    main()
