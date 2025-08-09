#!/usr/bin/env python3
"""
Kaggle Submission Generator for Stanford RNA 3D Folding Competition
Generates submission.csv from trained model checkpoints
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_kaggle_paths():
    """Setup paths for Kaggle environment"""
    paths = {
        'input_path': Path('/kaggle/input'),
        'working_path': Path('/kaggle/working'),
        'dataset_path': None,
        'test_sequences_path': None,
        'checkpoint_path': None
    }
    
    # Find dataset path
    if paths['input_path'].exists():
        # Look for our dataset
        for item in paths['input_path'].iterdir():
            if 'stanford-rna' in item.name.lower():
                paths['dataset_path'] = item
                break
        
        # If not found, try nested structure
        if paths['dataset_path'] is None:
            for item in paths['input_path'].iterdir():
                if item.is_dir():
                    for subitem in item.iterdir():
                        if 'stanford-rna' in subitem.name.lower():
                            paths['dataset_path'] = subitem
                            break
                    if paths['dataset_path']:
                        break
    
    if paths['dataset_path']:
        # Look for test sequences
        test_candidates = [
            paths['dataset_path'] / 'data' / 'test_sequences.csv',
            paths['dataset_path'] / 'test_sequences.csv',
            paths['dataset_path'] / 'data' / 'sample_submission.csv'
        ]
        
        for candidate in test_candidates:
            if candidate.exists():
                paths['test_sequences_path'] = candidate
                break
    
    # Look for checkpoint in working directory
    if paths['working_path'].exists():
        checkpoint_candidates = [
            paths['working_path'] / 'checkpoints' / 'best_model.pt',
            paths['working_path'] / 'best_model.pt',
            paths['working_path'] / 'model_checkpoint.pt'
        ]
        
        for candidate in checkpoint_candidates:
            if candidate.exists():
                paths['checkpoint_path'] = candidate
                break
    
    return paths


def load_model_from_checkpoint(checkpoint_path: Path, device: str = 'cuda'):
    """Load trained model from checkpoint"""
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Add src to path for imports
    if '/kaggle/input' in str(checkpoint_path):
        # Find dataset path and add to sys.path
        dataset_root = None
        current = checkpoint_path
        while current.parent != current:
            if 'stanford-rna' in current.name.lower():
                dataset_root = current
                break
            current = current.parent
        
        if dataset_root:
            sys.path.insert(0, str(dataset_root))
    
    try:
        from src.stanford_rna_folding.models.rna_folding_model import RNAFoldingModel
        from src.stanford_rna_folding.inference.submission_generator import SubmissionGenerator
    except ImportError:
        logger.error("Could not import model classes. Check dataset structure.")
        raise
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        config = checkpoint.get('config', {})
    else:
        # Checkpoint might be just the state dict
        model_state = checkpoint
        config = {}
    
    # Initialize model with saved or default configuration
    model = RNAFoldingModel(
        vocab_size=5,  # A, C, G, U, padding
        hidden_dim=config.get('hidden_dim', 512),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 8),
        dropout=0.0  # No dropout during inference
    )
    
    # Load weights
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def generate_kaggle_submission(checkpoint_path: Optional[Path] = None,
                             test_sequences_path: Optional[Path] = None,
                             output_path: str = "/kaggle/working/submission.csv",
                             num_conformations: int = 5,
                             temperature: float = 1.0):
    """Generate submission for Kaggle competition"""
    
    # Setup paths
    paths = setup_kaggle_paths()
    
    # Use provided paths or auto-detected ones
    if checkpoint_path is None:
        checkpoint_path = paths['checkpoint_path']
    if test_sequences_path is None:
        test_sequences_path = paths['test_sequences_path']
    
    logger.info(f"Paths detected:")
    logger.info(f"  Dataset: {paths['dataset_path']}")
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  Test sequences: {test_sequences_path}")
    
    # Validate paths
    if checkpoint_path is None or not checkpoint_path.exists():
        logger.error(f"No valid checkpoint found. Looked in: {paths['working_path']}")
        # Create a dummy submission for testing
        return create_dummy_submission(test_sequences_path, output_path)
    
    if test_sequences_path is None or not test_sequences_path.exists():
        logger.error(f"No test sequences found. Looked in: {paths['dataset_path']}")
        raise FileNotFoundError("Test sequences file not found")
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model_from_checkpoint(checkpoint_path, device)
    
    # Load test sequences
    logger.info(f"Loading test sequences from {test_sequences_path}")
    test_sequences = pd.read_csv(test_sequences_path)
    logger.info(f"Found {len(test_sequences)} test sequences")
    
    # Add src to path for submission generator
    if paths['dataset_path']:
        sys.path.insert(0, str(paths['dataset_path']))
    
    from src.stanford_rna_folding.inference.submission_generator import SubmissionGenerator
    
    # Create submission generator
    generator = SubmissionGenerator(model, device)
    
    # Generate predictions
    logger.info(f"Generating {num_conformations} conformations per sequence...")
    predictions = generator.process_test_sequences(
        test_sequences,
        num_conformations=num_conformations,
        temperature=temperature
    )
    
    # Format and save submission
    submission_df = generator.format_submission(predictions, output_path)
    
    # Validate submission
    is_valid = generator.validate_submission(submission_df)
    if not is_valid:
        logger.error("Generated submission failed validation")
        return None
    
    # Save metadata
    metadata = {
        'num_sequences': len(test_sequences),
        'num_conformations': num_conformations,
        'temperature': temperature,
        'device': device,
        'checkpoint_path': str(checkpoint_path),
        'submission_shape': list(submission_df.shape),
        'model_parameters': sum(p.numel() for p in model.parameters())
    }
    
    metadata_path = Path(output_path).parent / 'submission_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Submission generation completed successfully!")
    logger.info(f"Submission saved to: {output_path}")
    logger.info(f"Metadata saved to: {metadata_path}")
    
    return submission_df


def create_dummy_submission(test_sequences_path: Optional[Path], 
                          output_path: str = "/kaggle/working/submission.csv"):
    """Create dummy submission for testing when no trained model is available"""
    logger.warning("Creating dummy submission - no trained model found")
    
    # Try to load test sequences
    if test_sequences_path and test_sequences_path.exists():
        test_sequences = pd.read_csv(test_sequences_path)
    else:
        # Create minimal test data
        test_sequences = pd.DataFrame({
            'ID': ['test_seq_1', 'test_seq_2'],
            'sequence': ['AUCG', 'GCAU']
        })
    
    submission_rows = []
    
    for _, row in test_sequences.iterrows():
        sequence_id = row['ID']
        sequence_length = len(row['sequence'])
        
        # Generate dummy coordinates for 5 conformations
        for conf_idx in range(1, 6):  # 1-5 conformations
            for residue_idx in range(1, sequence_length + 1):  # 1-indexed
                for atom_idx in range(1, 2):  # 1 atom per residue for simplicity
                    # Random coordinates around origin
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
    submission_df.to_csv(output_path, index=False)
    
    logger.info(f"Dummy submission created: {output_path}")
    logger.info(f"Shape: {submission_df.shape}")
    
    return submission_df


def main():
    """Main entry point for Kaggle submission generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Kaggle submission for RNA folding competition")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--test-sequences", type=str, help="Path to test sequences CSV")
    parser.add_argument("--output", type=str, default="/kaggle/working/submission.csv", 
                       help="Output path for submission CSV")
    parser.add_argument("--conformations", type=int, default=5, 
                       help="Number of conformations per sequence")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature for diversity")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    test_sequences_path = Path(args.test_sequences) if args.test_sequences else None
    
    submission_df = generate_kaggle_submission(
        checkpoint_path=checkpoint_path,
        test_sequences_path=test_sequences_path,
        output_path=args.output,
        num_conformations=args.conformations,
        temperature=args.temperature
    )
    
    if submission_df is not None:
        print(f"Submission generated successfully: {args.output}")
        print(f"Submission shape: {submission_df.shape}")
    else:
        print("Submission generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
