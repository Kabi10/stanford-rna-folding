"""
Prediction script for RNA 3D structure prediction.

This module contains functionality to load a trained model and generate
predictions for the test set in the correct submission format.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import tqdm

from ..data.data_processing import StanfordRNADataset, rna_collate_fn
from ..data.transforms import RNADataTransform
from ..models.rna_folding_model import RNAFoldingModel


def load_model(
    checkpoint_path: Union[str, Path], 
    device: Optional[str] = None
) -> tuple[RNAFoldingModel, Dict]:
    """
    Load a trained model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on (default: None, will use CUDA if available)
        
    Returns:
        Tuple of (loaded_model, config_dict)
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    config = checkpoint.get("config", {})
    
    # Create model
    model = RNAFoldingModel(
        vocab_size=config.get("vocab_size", 5),
        embedding_dim=config.get("embedding_dim", 128),
        hidden_dim=config.get("hidden_dim", 256),
        num_layers=config.get("num_layers", 4),
        num_heads=config.get("num_heads", 8),
        dropout=config.get("dropout", 0.1),
        num_atoms=config.get("num_atoms", 5),
        coord_dims=config.get("coord_dims", 3),
        max_seq_len=config.get("max_seq_len", 500),
    )
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, config


def predict_coordinates(
    model: RNAFoldingModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Generate coordinate predictions for test sequences.
    
    Args:
        model: Trained model
        dataloader: DataLoader containing test sequences
        device: Device to run prediction on
        
    Returns:
        Dictionary mapping target_id to predicted coordinates
    """
    predictions = {}
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Predicting"):
            # Get batch data
            sequences = batch["sequence"].to(device)
            lengths = batch["lengths"].to(device)
            target_ids = batch["target_id"]
            
            # Forward pass
            pred_coords = model(sequences, lengths)
            
            # Process predictions for each example in the batch
            for i, target_id in enumerate(target_ids):
                # Get actual sequence length
                seq_len = lengths[i].item()
                
                # Get coordinates for this sequence (remove padding)
                coords = pred_coords[i, :seq_len].cpu().numpy()
                
                # Store in predictions dictionary
                predictions[target_id] = coords
    
    return predictions


def format_predictions_for_submission(
    predictions: Dict[str, np.ndarray],
    sample_submission_path: Union[str, Path],
    output_path: Union[str, Path],
) -> None:
    """
    Format predictions according to the competition submission format.
    
    Args:
        predictions: Dictionary mapping target_id to predicted coordinates
        sample_submission_path: Path to the sample submission file
        output_path: Path to save the formatted predictions
    """
    # Load sample submission to get the required format
    sample_df = pd.read_csv(sample_submission_path)
    
    # Initialize submission dataframe with the same index
    submission_df = pd.DataFrame(index=sample_df.index)
    submission_df["id"] = sample_df["id"]
    
    # Dictionary to map atom indices to column names
    coord_cols = []
    for i in range(1, 6):
        for axis in ["x", "y", "z"]:
            coord_cols.append(f"{axis}_{i}")
    
    # Fill submission dataframe with predictions
    for i, row in tqdm.tqdm(sample_df.iterrows(), total=len(sample_df), desc="Formatting"):
        target_id = row["id"].split("_")[0]  # Extract base target ID
        resid = int(row["id"].split("_")[1])  # Extract residue ID
        
        # Get predictions for this target
        if target_id in predictions:
            coords = predictions[target_id]
            
            # Check if this residue exists in our predictions
            if resid <= coords.shape[0]:
                # Get coordinates for this residue
                residue_coords = coords[resid - 1]  # Convert to 0-indexed
                
                # Flatten into a single row and add to submission dataframe
                flat_coords = residue_coords.reshape(-1)
                
                # Add to submission dataframe
                for j, col in enumerate(coord_cols):
                    if j < len(flat_coords):
                        submission_df.loc[i, col] = flat_coords[j]
    
    # Save submission file
    submission_df.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")


def generate_predictions(
    checkpoint_path: Union[str, Path],
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    sample_submission_path: Optional[Union[str, Path]] = None,
    batch_size: int = 16,
    device: Optional[str] = None,
) -> str:
    """
    Generate predictions for the test set and save in submission format.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        data_dir: Directory containing the competition data
        output_dir: Directory to save the predictions
        sample_submission_path: Path to the sample submission file (if None, will look in data_dir)
        batch_size: Batch size for prediction
        device: Device to use for prediction
        
    Returns:
        Path to the generated submission file
    """
    # Set paths
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    checkpoint_path = Path(checkpoint_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine sample submission path
    if sample_submission_path is None:
        sample_submission_path = data_dir / "sample_submission.csv"
    
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {checkpoint_path}")
    model, config = load_model(checkpoint_path, device)
    
    # Initialize transform for test data (no augmentation)
    test_transform = RNADataTransform(
        normalize_coords=config.get("normalize_coords", True),
        random_rotation=False,
        random_noise=0.0,
        jitter_strength=0.0,
        atom_mask_prob=0.0,
    )
    test_transform.training = False
    
    # Load test dataset
    print(f"Loading test data from {data_dir}")
    test_dataset = StanfordRNADataset(
        data_dir=data_dir,
        split="test",
        transform=test_transform,
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=rna_collate_fn,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )
    
    # Generate predictions
    print("Generating predictions...")
    predictions = predict_coordinates(model, test_loader, device)
    
    # Format predictions for submission
    submission_file = output_dir / f"submission_{checkpoint_path.stem}.csv"
    format_predictions_for_submission(predictions, sample_submission_path, submission_file)
    
    return str(submission_file)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate predictions for the test set.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--data-dir", type=str, default="datasets/stanford-rna-3d-folding", help="Directory containing the competition data")
    parser.add_argument("--output-dir", type=str, default="submissions", help="Directory to save the predictions")
    parser.add_argument("--sample-submission", type=str, help="Path to the sample submission file")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for prediction")
    parser.add_argument("--cpu", action="store_true", help="Force using CPU")
    
    args = parser.parse_args()
    
    device = "cpu" if args.cpu else None
    
    generate_predictions(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sample_submission_path=args.sample_submission,
        batch_size=args.batch_size,
        device=device,
    ) 