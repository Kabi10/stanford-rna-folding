"""
Enhanced prediction module for generating multiple diverse RNA structures.

This module extends the standard prediction functionality to generate multiple
diverse structure predictions for each RNA sequence, as required by the competition.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

# Import necessary modules
from ..data.data_processing import StanfordRNADataset
from ..models.rna_folding_model import RNAFoldingModel
from .diversity import StructuralDiversityEvaluator

logger = logging.getLogger(__name__)


class MultiStructurePredictor:
    """
    Predictor for generating multiple diverse RNA structure predictions.
    """
    
    def __init__(
        self,
        model: RNAFoldingModel,
        device: torch.device,
        num_structures: int = 5,
        diversity_method: str = "stochastic",  # Options: "stochastic", "perturbation", "fragment", "guided"
        diversity_strength: float = 0.2,
        diversity_evaluator: Optional[StructuralDiversityEvaluator] = None,
    ):
        """
        Initialize the multi-structure predictor.
        
        Args:
            model: The RNA folding model to use for prediction
            device: Device to run the model on (CPU or GPU)
            num_structures: Number of structures to generate per sequence
            diversity_method: Method for generating diverse structures
            diversity_strength: Strength of diversity (higher = more diverse but potentially less accurate)
            diversity_evaluator: Optional evaluator for measuring structure diversity
        """
        self.model = model
        self.device = device
        self.num_structures = num_structures
        self.diversity_method = diversity_method
        self.diversity_strength = diversity_strength
        
        # Initialize diversity evaluator if not provided
        if diversity_evaluator is None:
            self.diversity_evaluator = StructuralDiversityEvaluator(
                clustering_method="hierarchical",
                distance_metric="rmsd",
                cluster_threshold=0.3,
            )
        else:
            self.diversity_evaluator = diversity_evaluator
    
    def generate_structures_stochastic(
        self, 
        sequence_encoding: torch.Tensor,
        sequence_length: int,
        num_structures: int
    ) -> List[torch.Tensor]:
        """
        Generate multiple structures using stochastic sampling.
        
        Uses temperature scaling to control the diversity vs. accuracy trade-off.
        
        Args:
            sequence_encoding: Encoded RNA sequence (batch_size, seq_len)
            sequence_length: Actual length of the sequence (without padding)
            num_structures: Number of structures to generate
            
        Returns:
            List of predicted structures, each of shape (seq_len, num_atoms, 3)
        """
        self.model.eval()
        structures = []
        
        # Set model to slightly different sampling modes for each prediction
        for i in range(num_structures):
            # Calculate temperature for this prediction
            # Start with lower temperature (more accurate) and gradually increase
            temperature = 1.0 + (i * self.diversity_strength / num_structures)
            
            with torch.no_grad():
                # Forward pass with temperature control for stochastic sampling
                pred_coords = self.model(
                    sequence_encoding, 
                    sampling_temperature=temperature,
                    use_stochastic_sampling=True
                )
                
                # Extract the actual sequence length (without padding)
                pred_coords = pred_coords[:, :sequence_length, :, :]
                
                # Add to list of structures
                structures.append(pred_coords.squeeze(0).cpu())
        
        return structures
    
    def generate_structures_perturbation(
        self, 
        sequence_encoding: torch.Tensor,
        sequence_length: int,
        num_structures: int
    ) -> List[torch.Tensor]:
        """
        Generate multiple structures using geometric perturbation.
        
        Applies small random perturbations to the backbone torsion angles
        within physically valid ranges.
        
        Args:
            sequence_encoding: Encoded RNA sequence (batch_size, seq_len)
            sequence_length: Actual length of the sequence (without padding)
            num_structures: Number of structures to generate
            
        Returns:
            List of predicted structures, each of shape (seq_len, num_atoms, 3)
        """
        self.model.eval()
        structures = []
        
        # First, generate a base structure
        with torch.no_grad():
            base_coords = self.model(sequence_encoding)
            base_coords = base_coords[:, :sequence_length, :, :]
            base_coords = base_coords.squeeze(0).cpu()
        
        # Add the base structure to the list
        structures.append(base_coords)
        
        # Generate additional structures with perturbations
        for i in range(1, num_structures):
            # Create a copy of the sequence encoding with noise
            perturbed_encoding = sequence_encoding.clone()
            
            # Add small random noise to the sequence embedding
            # This will propagate through the model to create different structures
            noise_scale = self.diversity_strength * (i / num_structures)
            noise = torch.randn_like(perturbed_encoding, device=self.device) * noise_scale
            perturbed_encoding = perturbed_encoding + noise
            
            with torch.no_grad():
                perturbed_coords = self.model(perturbed_encoding)
                perturbed_coords = perturbed_coords[:, :sequence_length, :, :]
                perturbed_coords = perturbed_coords.squeeze(0).cpu()
            
            structures.append(perturbed_coords)
        
        return structures
    
    def generate_structures_guided(
        self, 
        sequence_encoding: torch.Tensor,
        sequence_length: int,
        num_structures: int
    ) -> List[torch.Tensor]:
        """
        Generate multiple structures using guided diversity.
        
        After generating each structure, adds a repulsion term to guide
        the model away from previously generated structures.
        
        Args:
            sequence_encoding: Encoded RNA sequence (batch_size, seq_len)
            sequence_length: Actual length of the sequence (without padding)
            num_structures: Number of structures to generate
            
        Returns:
            List of predicted structures, each of shape (seq_len, num_atoms, 3)
        """
        self.model.eval()
        structures = []
        
        # Generate first structure normally
        with torch.no_grad():
            base_coords = self.model(sequence_encoding)
            base_coords = base_coords[:, :sequence_length, :, :]
            base_coords = base_coords.squeeze(0).cpu()
        
        structures.append(base_coords)
        
        # For each additional structure, add a term to guide away from previous structures
        for i in range(1, num_structures):
            # Start with a slightly perturbed encoding
            perturbed_encoding = sequence_encoding.clone()
            noise_scale = 0.05  # Small noise to break symmetry
            noise = torch.randn_like(perturbed_encoding, device=self.device) * noise_scale
            perturbed_encoding = perturbed_encoding + noise
            
            # Use the model's embedding as a starting point
            self.model.zero_grad()
            
            # Enable gradients for this pass
            with torch.set_grad_enabled(True):
                # Get initial prediction
                coords = self.model(perturbed_encoding)
                coords = coords[:, :sequence_length, :, :]
                
                # Compute repulsion loss from previous structures
                repulsion_loss = 0
                for prev_struct in structures:
                    # Convert previous structure to tensor on device
                    prev_struct_device = prev_struct.to(self.device).unsqueeze(0)
                    
                    # Calculate similarity - we want to minimize this
                    # (Lower similarity = higher diversity)
                    similarity = -torch.mean((coords - prev_struct_device) ** 2)
                    repulsion_loss += similarity
                
                # Scale the repulsion loss
                repulsion_loss *= self.diversity_strength
                
                # Backpropagate to get gradients
                repulsion_loss.backward()
                
                # Get gradients
                for param in self.model.parameters():
                    if param.grad is not None:
                        # Apply small update in the opposite direction of gradient
                        # This pushes the model away from previous structures
                        param.data -= 0.01 * param.grad
            
            # Now generate the new structure with the updated model
            with torch.no_grad():
                new_coords = self.model(sequence_encoding)
                new_coords = new_coords[:, :sequence_length, :, :]
                new_coords = new_coords.squeeze(0).cpu()
            
            structures.append(new_coords)
            
            # Reset model gradients
            self.model.zero_grad()
        
        return structures
    
    def generate_multiple_structures(
        self, 
        sequence_encoding: torch.Tensor,
        sequence_length: int
    ) -> List[torch.Tensor]:
        """
        Generate multiple diverse structure predictions for a single RNA sequence.
        
        Args:
            sequence_encoding: Encoded RNA sequence (batch_size, seq_len)
            sequence_length: Actual length of the sequence (without padding)
            
        Returns:
            List of predicted structures, each of shape (seq_len, num_atoms, 3)
        """
        if self.diversity_method == "stochastic":
            structures = self.generate_structures_stochastic(
                sequence_encoding, sequence_length, self.num_structures
            )
        elif self.diversity_method == "perturbation":
            structures = self.generate_structures_perturbation(
                sequence_encoding, sequence_length, self.num_structures
            )
        elif self.diversity_method == "guided":
            structures = self.generate_structures_guided(
                sequence_encoding, sequence_length, self.num_structures
            )
        else:
            # Default to stochastic method
            logger.warning(f"Unknown diversity method: {self.diversity_method}. Using stochastic instead.")
            structures = self.generate_structures_stochastic(
                sequence_encoding, sequence_length, self.num_structures
            )
        
        return structures
    
    def select_diverse_representatives(
        self, 
        structures: List[torch.Tensor],
        n_representatives: int = 5
    ) -> List[torch.Tensor]:
        """
        Select representative structures from the generated structures.
        
        Args:
            structures: List of predicted structures, each of shape (seq_len, num_atoms, 3)
            n_representatives: Number of representatives to select
            
        Returns:
            List of selected representative structures
        """
        # If we have fewer structures than requested representatives, return all structures
        if len(structures) <= n_representatives:
            return structures
        
        # Generate more structures than needed, then select diverse representatives
        indices = self.diversity_evaluator.select_representatives(
            structures, n_representatives=n_representatives
        )
        
        # Select the structures based on the indices
        selected_structures = [structures[i] for i in indices]
        
        return selected_structures
    
    def predict_multiple_structures(
        self, 
        dataloader: DataLoader,
        output_file: str = "submission.csv"
    ) -> None:
        """
        Generate multiple structure predictions for each sequence in the dataset,
        and save to a competition submission file.
        
        Args:
            dataloader: DataLoader containing test sequences
            output_file: Path to output CSV file
        """
        self.model.eval()
        
        # Dictionary to store predictions
        predictions = {}
        
        # Process each batch
        for batch in dataloader:
            # Extract relevant information
            sequence_encoding = batch["sequence_encoding"].to(self.device)
            sequence_lengths = batch["sequence_length"]
            target_ids = batch["target_id"]
            
            # Process each sequence in the batch
            for i in range(len(target_ids)):
                target_id = target_ids[i]
                seq_encoding = sequence_encoding[i:i+1]  # Convert to batch of size 1
                seq_length = sequence_lengths[i].item()
                
                # Generate multiple structures
                structures = self.generate_multiple_structures(seq_encoding, seq_length)
                
                # Select diverse representatives
                if len(structures) > self.num_structures:
                    structures = self.select_diverse_representatives(
                        structures, n_representatives=self.num_structures
                    )
                
                # Store the predictions for this target
                predictions[target_id] = structures
        
        # Convert predictions to submission format and save
        self._save_predictions_to_csv(predictions, output_file)
    
    def _save_predictions_to_csv(
        self, 
        predictions: Dict[str, List[torch.Tensor]], 
        output_file: str
    ) -> None:
        """
        Save predictions to a CSV file in the competition submission format.
        
        Args:
            predictions: Dictionary mapping target_id to list of predicted structures
            output_file: Path to output CSV file
        """
        with open(output_file, "w") as f:
            # Write header
            atom_headers = []
            for struct_idx in range(self.num_structures):
                for atom_idx in range(1, 6):  # 5 atoms per nucleotide
                    for coord in ["x", "y", "z"]:
                        atom_headers.append(f"{coord}_{atom_idx}_{struct_idx+1}")
            
            f.write("target_id," + ",".join(atom_headers) + "\n")
            
            # Write predictions
            for target_id, structures in predictions.items():
                row = [target_id]
                
                # Ensure we have exactly num_structures structures
                assert len(structures) <= self.num_structures, f"Too many structures for {target_id}: {len(structures)}"
                
                # If we have fewer structures than required, duplicate the last one
                while len(structures) < self.num_structures:
                    structures.append(structures[-1])
                
                # For each structure, add coordinates for each atom
                for struct in structures:
                    # Ensure structure is on CPU and convert to numpy
                    if isinstance(struct, torch.Tensor):
                        struct = struct.detach().cpu().numpy()
                    
                    # Add coordinates for each atom
                    for atom_idx in range(struct.shape[1]):  # num_atoms dimension
                        for coord_idx in range(struct.shape[2]):  # xyz dimension
                            coords = struct[:, atom_idx, coord_idx]
                            row.extend([f"{c:.6f}" for c in coords])
                
                f.write(",".join(row) + "\n")
        
        logger.info(f"Saved predictions to {output_file}") 