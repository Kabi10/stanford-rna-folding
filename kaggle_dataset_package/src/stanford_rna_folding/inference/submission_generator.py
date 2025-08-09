"""
Submission Generator for Stanford RNA 3D Folding Competition
Generates submission.csv with 5 diverse conformations per sequence
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SubmissionGenerator:
    """Generates competition submission files with multiple conformations"""
    
    def __init__(self, model, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def generate_diverse_conformations(self, 
                                     sequence: torch.Tensor, 
                                     length: int,
                                     num_conformations: int = 5,
                                     temperature: float = 1.0) -> torch.Tensor:
        """Generate diverse conformations for a single sequence
        
        Args:
            sequence: (L,) tensor of sequence tokens
            length: actual sequence length (excluding padding)
            num_conformations: number of conformations to generate
            temperature: sampling temperature for diversity
            
        Returns:
            conformations: (num_conformations, L, num_atoms, 3) tensor
        """
        conformations = []
        
        with torch.no_grad():
            for i in range(num_conformations):
                # Add noise for diversity if temperature > 0
                if temperature > 0 and i > 0:
                    # Add small random noise to input embeddings for diversity
                    noise_scale = temperature * 0.1
                    noise = torch.randn_like(sequence.float()) * noise_scale
                    noisy_sequence = sequence.float() + noise
                    noisy_sequence = noisy_sequence.long().clamp(0, 4)  # Keep valid tokens
                else:
                    noisy_sequence = sequence
                
                # Forward pass
                coords = self.model(noisy_sequence.unsqueeze(0))  # (1, L, num_atoms, 3)
                
                # Extract valid length and add to conformations
                valid_coords = coords[0, :length]  # (L, num_atoms, 3)
                conformations.append(valid_coords)
        
        return torch.stack(conformations)  # (num_conformations, L, num_atoms, 3)
    
    def process_test_sequences(self, 
                             test_sequences: pd.DataFrame,
                             num_conformations: int = 5,
                             temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """Process all test sequences and generate conformations
        
        Args:
            test_sequences: DataFrame with 'ID' and 'sequence' columns
            num_conformations: number of conformations per sequence
            temperature: sampling temperature for diversity
            
        Returns:
            predictions: Dict mapping sequence_id -> conformations tensor
        """
        predictions = {}
        
        logger.info(f"Processing {len(test_sequences)} test sequences...")
        
        for idx, row in tqdm(test_sequences.iterrows(), total=len(test_sequences)):
            sequence_id = row['ID']
            sequence_str = row['sequence']
            
            # Convert sequence to tokens (A=0, C=1, G=2, U=3, padding=4)
            sequence_tokens = self._sequence_to_tokens(sequence_str)
            sequence_tensor = torch.tensor(sequence_tokens, device=self.device)
            
            # Generate conformations
            conformations = self.generate_diverse_conformations(
                sequence_tensor, 
                len(sequence_str),
                num_conformations,
                temperature
            )
            
            predictions[sequence_id] = conformations.cpu()
            
        logger.info(f"Generated predictions for {len(predictions)} sequences")
        return predictions
    
    def _sequence_to_tokens(self, sequence: str) -> List[int]:
        """Convert RNA sequence string to token indices"""
        token_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        return [token_map.get(base.upper(), 4) for base in sequence]  # 4 = unknown/padding
    
    def format_submission(self, 
                         predictions: Dict[str, torch.Tensor],
                         output_path: str = "/kaggle/working/submission.csv") -> pd.DataFrame:
        """Format predictions into competition submission format
        
        Args:
            predictions: Dict mapping sequence_id -> conformations tensor
            output_path: Path to save submission CSV
            
        Returns:
            submission_df: Formatted submission DataFrame
        """
        submission_rows = []
        
        logger.info("Formatting submission...")
        
        for sequence_id, conformations in tqdm(predictions.items()):
            # conformations shape: (num_conformations, L, num_atoms, 3)
            num_conformations, seq_length, num_atoms, _ = conformations.shape
            
            for conf_idx in range(num_conformations):
                for residue_idx in range(seq_length):
                    for atom_idx in range(num_atoms):
                        x, y, z = conformations[conf_idx, residue_idx, atom_idx]
                        
                        # Create row for submission
                        row = {
                            'ID': f"{sequence_id}_{residue_idx+1}_{atom_idx+1}",  # 1-indexed
                            'x': float(x),
                            'y': float(y), 
                            'z': float(z),
                            'conformation': conf_idx + 1  # 1-indexed conformations
                        }
                        submission_rows.append(row)
        
        # Create DataFrame
        submission_df = pd.DataFrame(submission_rows)
        
        # Sort by ID and conformation for consistency
        submission_df = submission_df.sort_values(['ID', 'conformation']).reset_index(drop=True)
        
        # Save to CSV
        submission_df.to_csv(output_path, index=False)
        logger.info(f"Submission saved to {output_path}")
        logger.info(f"Submission shape: {submission_df.shape}")
        logger.info(f"Unique sequences: {len(set(row['ID'].split('_')[0] for _, row in submission_df.iterrows()))}")
        logger.info(f"Conformations per sequence: {submission_df['conformation'].nunique()}")
        
        return submission_df
    
    def validate_submission(self, submission_df: pd.DataFrame) -> bool:
        """Validate submission format meets competition requirements
        
        Args:
            submission_df: Submission DataFrame to validate
            
        Returns:
            is_valid: True if submission format is valid
        """
        logger.info("Validating submission format...")
        
        # Check required columns
        required_columns = ['ID', 'x', 'y', 'z', 'conformation']
        missing_columns = set(required_columns) - set(submission_df.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check data types
        if not pd.api.types.is_numeric_dtype(submission_df['x']):
            logger.error("Column 'x' must be numeric")
            return False
        if not pd.api.types.is_numeric_dtype(submission_df['y']):
            logger.error("Column 'y' must be numeric")
            return False
        if not pd.api.types.is_numeric_dtype(submission_df['z']):
            logger.error("Column 'z' must be numeric")
            return False
        
        # Check conformations are 1-5
        conf_values = submission_df['conformation'].unique()
        if not all(1 <= conf <= 5 for conf in conf_values):
            logger.error(f"Conformations must be 1-5, found: {conf_values}")
            return False
        
        # Check for NaN values
        if submission_df.isnull().any().any():
            logger.error("Submission contains NaN values")
            return False
        
        # Check coordinate ranges (basic sanity check)
        coord_cols = ['x', 'y', 'z']
        for col in coord_cols:
            if submission_df[col].abs().max() > 1000:  # Reasonable coordinate range
                logger.warning(f"Large coordinate values detected in {col}: max={submission_df[col].abs().max()}")
        
        logger.info("Submission validation passed!")
        return True


def create_submission_from_checkpoint(checkpoint_path: str,
                                    test_sequences_path: str,
                                    output_path: str = "/kaggle/working/submission.csv",
                                    device: str = 'cuda',
                                    num_conformations: int = 5,
                                    temperature: float = 1.0) -> pd.DataFrame:
    """Complete pipeline to create submission from trained model checkpoint
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        test_sequences_path: Path to test sequences CSV
        output_path: Path to save submission CSV
        device: Device for inference
        num_conformations: Number of conformations per sequence
        temperature: Sampling temperature for diversity
        
    Returns:
        submission_df: Generated submission DataFrame
    """
    # Load model from checkpoint
    logger.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model (this would need to match your model architecture)
    from stanford_rna_folding.models.rna_folding_model import RNAFoldingModel
    model = RNAFoldingModel(
        vocab_size=5,
        hidden_dim=checkpoint.get('hidden_dim', 512),
        num_layers=checkpoint.get('num_layers', 6),
        num_heads=checkpoint.get('num_heads', 8),
        dropout=0.0  # No dropout during inference
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Load test sequences
    logger.info(f"Loading test sequences from {test_sequences_path}")
    test_sequences = pd.read_csv(test_sequences_path)
    
    # Create submission generator
    generator = SubmissionGenerator(model, device)
    
    # Generate predictions
    predictions = generator.process_test_sequences(
        test_sequences, 
        num_conformations, 
        temperature
    )
    
    # Format submission
    submission_df = generator.format_submission(predictions, output_path)
    
    # Validate submission
    is_valid = generator.validate_submission(submission_df)
    if not is_valid:
        raise ValueError("Generated submission failed validation")
    
    return submission_df
