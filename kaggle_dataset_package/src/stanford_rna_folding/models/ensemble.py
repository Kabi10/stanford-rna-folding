"""
Ensemble module for combining multiple RNA structure prediction models.

This module provides functionality to combine predictions from multiple models,
either by averaging or more sophisticated methods, to improve prediction accuracy.
It also provides methods for generating diverse structure predictions.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..models.rna_folding_model import RNAFoldingModel
from ..evaluation.diversity import StructuralDiversityEvaluator

logger = logging.getLogger(__name__)


class StructureEnsembler:
    """
    Ensemble multiple models for RNA structure prediction.
    
    This class provides methods to combine predictions from multiple models,
    with various weighting strategies to improve overall accuracy.
    """
    
    def __init__(
        self,
        models: List[RNAFoldingModel],
        model_weights: Optional[List[float]] = None,
        ensemble_method: str = "weighted_average",  # Options: weighted_average, confidence_weighted, bayesian
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Initialize the structure ensembler.
        
        Args:
            models: List of RNA folding models to ensemble
            model_weights: Optional list of weights for each model (defaults to equal weights)
            ensemble_method: Method for combining model predictions
            device: Device to run models on
        """
        self.models = models
        self.device = device
        self.ensemble_method = ensemble_method
        
        # Move models to the correct device
        for model in self.models:
            model.to(self.device)
            model.eval()  # Set models to evaluation mode
        
        # Set model weights (default to equal weights if not provided)
        if model_weights is None:
            self.model_weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights to sum to 1
            total_weight = sum(model_weights)
            self.model_weights = [w / total_weight for w in model_weights]
    
    def _predict_single_model(
        self, 
        model: RNAFoldingModel, 
        sequence_encoding: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        """
        Generate a prediction using a single model.
        
        Args:
            model: The RNA folding model to use
            sequence_encoding: Encoded RNA sequence (batch_size, seq_len)
            sequence_length: Actual length of the sequence (without padding)
            
        Returns:
            Predicted coordinates (batch_size, seq_len, num_atoms, 3)
        """
        with torch.no_grad():
            # Forward pass
            pred_coords = model(sequence_encoding)
            
            # Extract the actual sequence (without padding)
            pred_coords = pred_coords[:, :sequence_length, :, :]
        
        return pred_coords
    
    def predict_weighted_average(
        self, 
        sequence_encoding: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        """
        Generate a prediction by weighted averaging of all model predictions.
        
        Args:
            sequence_encoding: Encoded RNA sequence (batch_size, seq_len)
            sequence_length: Actual length of the sequence (without padding)
            
        Returns:
            Averaged predicted coordinates (batch_size, seq_len, num_atoms, 3)
        """
        # Initialize with zeros
        weighted_prediction = None
        
        # Generate predictions from each model and combine with weights
        for i, model in enumerate(self.models):
            # Get prediction from this model
            pred = self._predict_single_model(model, sequence_encoding, sequence_length)
            
            # Initialize weighted prediction with the first model's shape
            if weighted_prediction is None:
                weighted_prediction = torch.zeros_like(pred)
            
            # Add weighted prediction
            weighted_prediction += self.model_weights[i] * pred
        
        return weighted_prediction
    
    def predict_confidence_weighted(
        self, 
        sequence_encoding: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        """
        Generate a prediction using confidence-weighted averaging.
        
        Models with higher confidence (lower predicted uncertainty) for a given
        position are given higher weight in the ensemble for that position.
        
        Args:
            sequence_encoding: Encoded RNA sequence (batch_size, seq_len)
            sequence_length: Actual length of the sequence (without padding)
            
        Returns:
            Confidence-weighted prediction (batch_size, seq_len, num_atoms, 3)
        """
        # Get all model predictions and their estimated uncertainties
        all_predictions = []
        all_uncertainties = []
        
        for model in self.models:
            with torch.no_grad():
                # Forward pass with uncertainty estimation
                # Note: This assumes the model returns both predictions and uncertainties
                # If your model doesn't support this, you'll need to modify the code
                pred, uncertainty = model(sequence_encoding, return_uncertainty=True)
                
                # Extract the actual sequence (without padding)
                pred = pred[:, :sequence_length, :, :]
                uncertainty = uncertainty[:, :sequence_length, :, :]
                
                all_predictions.append(pred)
                all_uncertainties.append(uncertainty)
        
        # Stack all predictions and uncertainties
        all_predictions = torch.stack(all_predictions, dim=0)  # (num_models, batch_size, seq_len, num_atoms, 3)
        all_uncertainties = torch.stack(all_uncertainties, dim=0)  # (num_models, batch_size, seq_len, num_atoms, 3)
        
        # Convert uncertainties to weights (lower uncertainty = higher weight)
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-8
        confidence_weights = 1.0 / (all_uncertainties + epsilon)
        
        # Normalize weights across models
        confidence_weights = confidence_weights / torch.sum(confidence_weights, dim=0, keepdim=True)
        
        # Apply confidence weights to predictions
        weighted_predictions = all_predictions * confidence_weights
        
        # Sum across models
        final_prediction = torch.sum(weighted_predictions, dim=0)
        
        return final_prediction
    
    def predict_bayesian(
        self, 
        sequence_encoding: torch.Tensor,
        sequence_length: int,
        num_samples: int = 10,
    ) -> torch.Tensor:
        """
        Generate a prediction using Bayesian model averaging.
        
        Combines predictions from multiple models using Bayesian principles,
        accounting for model uncertainty.
        
        Args:
            sequence_encoding: Encoded RNA sequence (batch_size, seq_len)
            sequence_length: Actual length of the sequence (without padding)
            num_samples: Number of samples to draw per model
            
        Returns:
            Bayesian averaged prediction (batch_size, seq_len, num_atoms, 3)
        """
        all_samples = []
        
        # For each model, generate multiple samples
        for model in self.models:
            model_samples = []
            
            # Enable dropout during inference for MC Dropout
            model.train()  # Set to training mode to enable dropout
            
            for _ in range(num_samples):
                with torch.no_grad():
                    # Forward pass with dropout enabled
                    pred = model(sequence_encoding)
                    
                    # Extract the actual sequence (without padding)
                    pred = pred[:, :sequence_length, :, :]
                    
                    model_samples.append(pred)
            
            # Reset model to evaluation mode
            model.eval()
            
            # Combine samples from this model
            all_samples.extend(model_samples)
        
        # Stack all samples
        all_samples = torch.stack(all_samples, dim=0)  # (num_models*num_samples, batch_size, seq_len, num_atoms, 3)
        
        # Calculate mean prediction
        mean_prediction = torch.mean(all_samples, dim=0)
        
        return mean_prediction
    
    def predict_feature_level(
        self, 
        sequence_encoding: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        """
        Generate a prediction using feature-level integration.
        
        Instead of averaging final predictions, this method combines intermediate
        features from multiple models before making a final prediction.
        
        Args:
            sequence_encoding: Encoded RNA sequence (batch_size, seq_len)
            sequence_length: Actual length of the sequence (without padding)
            
        Returns:
            Feature-level integrated prediction (batch_size, seq_len, num_atoms, 3)
        """
        # NOTE: This is a placeholder implementation and assumes models 
        # have a method to extract features. You may need to modify this
        # based on your model architecture.
        
        all_features = []
        
        # Extract features from each model
        for model in self.models:
            with torch.no_grad():
                # Extract features (intermediate representations)
                features = model.extract_features(sequence_encoding)
                all_features.append(features)
        
        # Combine features
        combined_features = torch.cat(all_features, dim=-1)  # Concatenate along feature dimension
        
        # Use the first model as the "head" to make predictions from combined features
        head_model = self.models[0]
        with torch.no_grad():
            # Make prediction from combined features
            pred = head_model.predict_from_features(combined_features)
            
            # Extract the actual sequence (without padding)
            pred = pred[:, :sequence_length, :, :]
        
        return pred
    
    def predict(
        self, 
        sequence_encoding: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        """
        Generate an ensemble prediction using the specified method.
        
        Args:
            sequence_encoding: Encoded RNA sequence (batch_size, seq_len)
            sequence_length: Actual length of the sequence (without padding)
            
        Returns:
            Ensemble prediction (batch_size, seq_len, num_atoms, 3)
        """
        # Ensure the input is on the correct device
        sequence_encoding = sequence_encoding.to(self.device)
        
        # Use the appropriate ensemble method
        if self.ensemble_method == "weighted_average":
            return self.predict_weighted_average(sequence_encoding, sequence_length)
        elif self.ensemble_method == "confidence_weighted":
            return self.predict_confidence_weighted(sequence_encoding, sequence_length)
        elif self.ensemble_method == "bayesian":
            return self.predict_bayesian(sequence_encoding, sequence_length)
        elif self.ensemble_method == "feature_level":
            return self.predict_feature_level(sequence_encoding, sequence_length)
        else:
            logger.warning(f"Unknown ensemble method: {self.ensemble_method}. Using weighted average instead.")
            return self.predict_weighted_average(sequence_encoding, sequence_length)


class DiverseStructureGenerator:
    """
    Generate diverse RNA structure predictions from a single model.
    
    This class provides methods to generate multiple diverse structure predictions
    for a single RNA sequence, which can then be combined or selected based on
    confidence or other criteria.
    """
    
    def __init__(
        self,
        model: RNAFoldingModel,
        num_structures: int = 5,
        diversity_method: str = "temperature",  # Options: temperature, perturbation, hierarchical
        diversity_strength: float = 0.2,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Initialize the diverse structure generator.
        
        Args:
            model: The RNA folding model to use
            num_structures: Number of structures to generate per sequence
            diversity_method: Method for generating diverse structures
            diversity_strength: Strength of diversity (higher = more diverse but potentially less accurate)
            device: Device to run the model on
        """
        self.model = model
        self.device = device
        self.num_structures = num_structures
        self.diversity_method = diversity_method
        self.diversity_strength = diversity_strength
        
        # Move model to the correct device
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode
        
        # Initialize diversity evaluator
        self.diversity_evaluator = StructuralDiversityEvaluator(
            clustering_method="hierarchical",
            distance_metric="rmsd",
            cluster_threshold=0.3,
        )
    
    def generate_temperature_sampling(
        self, 
        sequence_encoding: torch.Tensor,
        sequence_length: int,
    ) -> List[torch.Tensor]:
        """
        Generate diverse structures using temperature sampling.
        
        Higher temperature leads to more diverse but potentially less accurate predictions.
        
        Args:
            sequence_encoding: Encoded RNA sequence (batch_size, seq_len)
            sequence_length: Actual length of the sequence (without padding)
            
        Returns:
            List of predicted structures, each of shape (seq_len, num_atoms, 3)
        """
        structures = []
        
        # Generate structures with different temperatures
        for i in range(self.num_structures):
            # Calculate temperature for this prediction
            # Start with lower temperature (more accurate) and gradually increase
            temperature = 1.0 + (i * self.diversity_strength)
            
            with torch.no_grad():
                # Forward pass with temperature control
                pred_coords = self.model(
                    sequence_encoding, 
                    sampling_temperature=temperature,
                    use_stochastic_sampling=True
                )
                
                # Extract the actual sequence (without padding)
                pred_coords = pred_coords[:, :sequence_length, :, :]
                pred_coords = pred_coords.squeeze(0).cpu()  # Remove batch dimension and move to CPU
                
                structures.append(pred_coords)
        
        return structures
    
    def generate_perturbation(
        self, 
        sequence_encoding: torch.Tensor,
        sequence_length: int,
    ) -> List[torch.Tensor]:
        """
        Generate diverse structures by perturbing the input.
        
        Adds random noise to the input sequence encoding to generate diverse predictions.
        
        Args:
            sequence_encoding: Encoded RNA sequence (batch_size, seq_len)
            sequence_length: Actual length of the sequence (without padding)
            
        Returns:
            List of predicted structures, each of shape (seq_len, num_atoms, 3)
        """
        structures = []
        
        # First, generate a base structure
        with torch.no_grad():
            base_coords = self.model(sequence_encoding)
            base_coords = base_coords[:, :sequence_length, :, :]
            base_coords = base_coords.squeeze(0).cpu()  # Remove batch dimension and move to CPU
        
        structures.append(base_coords)
        
        # Generate additional structures with perturbations
        for i in range(1, self.num_structures):
            # Create a copy of the sequence encoding with noise
            perturbed_encoding = sequence_encoding.clone()
            
            # Add small random noise to the sequence embedding
            noise_scale = self.diversity_strength * (i / self.num_structures)
            noise = torch.randn_like(perturbed_encoding) * noise_scale
            perturbed_encoding = perturbed_encoding + noise
            
            with torch.no_grad():
                perturbed_coords = self.model(perturbed_encoding)
                perturbed_coords = perturbed_coords[:, :sequence_length, :, :]
                perturbed_coords = perturbed_coords.squeeze(0).cpu()  # Remove batch dimension and move to CPU
            
            structures.append(perturbed_coords)
        
        return structures
    
    def generate_hierarchical(
        self, 
        sequence_encoding: torch.Tensor,
        sequence_length: int,
    ) -> List[torch.Tensor]:
        """
        Generate diverse structures using a hierarchical approach.
        
        First generates a base structure, then varies different hierarchical levels
        of the RNA structure (backbone, nucleotides, local regions).
        
        Args:
            sequence_encoding: Encoded RNA sequence (batch_size, seq_len)
            sequence_length: Actual length of the sequence (without padding)
            
        Returns:
            List of predicted structures, each of shape (seq_len, num_atoms, 3)
        """
        structures = []
        
        # First, generate a base structure
        with torch.no_grad():
            base_coords = self.model(sequence_encoding)
            base_coords = base_coords[:, :sequence_length, :, :]
            base_coords = base_coords.squeeze(0).cpu()  # Remove batch dimension and move to CPU
        
        structures.append(base_coords)
        
        # Number of hierarchical levels to vary
        num_levels = min(3, self.num_structures - 1)
        
        # Generate additional structures by varying different hierarchical levels
        for level in range(num_levels):
            # Different perturbation strategies for each level
            if level == 0:
                # Level 1: Vary backbone structure
                with torch.no_grad():
                    # Perturb with specific backbone-focused noise
                    perturbed_encoding = sequence_encoding.clone()
                    noise = torch.randn_like(perturbed_encoding) * (self.diversity_strength * 0.5)
                    perturbed_encoding = perturbed_encoding + noise
                    
                    # Generate with backbone variation
                    level_coords = self.model(
                        perturbed_encoding,
                        vary_backbone=True  # Hypothetical parameter to focus on backbone variation
                    )
                    level_coords = level_coords[:, :sequence_length, :, :]
                    level_coords = level_coords.squeeze(0).cpu()
                
                structures.append(level_coords)
            
            elif level == 1:
                # Level 2: Vary nucleotide orientations
                with torch.no_grad():
                    # Perturb with nucleotide-focused noise
                    perturbed_encoding = sequence_encoding.clone()
                    noise = torch.randn_like(perturbed_encoding) * (self.diversity_strength * 0.3)
                    perturbed_encoding = perturbed_encoding + noise
                    
                    # Generate with nucleotide variation
                    level_coords = self.model(
                        perturbed_encoding,
                        vary_nucleotides=True  # Hypothetical parameter to focus on nucleotide variation
                    )
                    level_coords = level_coords[:, :sequence_length, :, :]
                    level_coords = level_coords.squeeze(0).cpu()
                
                structures.append(level_coords)
            
            elif level == 2:
                # Level 3: Vary local regions
                with torch.no_grad():
                    # Perturb with region-focused noise
                    perturbed_encoding = sequence_encoding.clone()
                    noise = torch.randn_like(perturbed_encoding) * (self.diversity_strength * 0.2)
                    perturbed_encoding = perturbed_encoding + noise
                    
                    # Generate with regional variation
                    level_coords = self.model(
                        perturbed_encoding,
                        vary_regions=True  # Hypothetical parameter to focus on regional variation
                    )
                    level_coords = level_coords[:, :sequence_length, :, :]
                    level_coords = level_coords.squeeze(0).cpu()
                
                structures.append(level_coords)
        
        # If we need more structures, add random perturbations
        while len(structures) < self.num_structures:
            with torch.no_grad():
                # Random perturbation
                perturbed_encoding = sequence_encoding.clone()
                noise = torch.randn_like(perturbed_encoding) * self.diversity_strength
                perturbed_encoding = perturbed_encoding + noise
                
                random_coords = self.model(perturbed_encoding)
                random_coords = random_coords[:, :sequence_length, :, :]
                random_coords = random_coords.squeeze(0).cpu()
            
            structures.append(random_coords)
        
        return structures
    
    def generate_structures(
        self, 
        sequence_encoding: torch.Tensor,
        sequence_length: int,
    ) -> List[torch.Tensor]:
        """
        Generate diverse structure predictions for a single RNA sequence.
        
        Args:
            sequence_encoding: Encoded RNA sequence (batch_size, seq_len)
            sequence_length: Actual length of the sequence (without padding)
            
        Returns:
            List of predicted structures, each of shape (seq_len, num_atoms, 3)
        """
        # Ensure the input is on the correct device
        sequence_encoding = sequence_encoding.to(self.device)
        
        # Use the appropriate diversity method
        if self.diversity_method == "temperature":
            return self.generate_temperature_sampling(sequence_encoding, sequence_length)
        elif self.diversity_method == "perturbation":
            return self.generate_perturbation(sequence_encoding, sequence_length)
        elif self.diversity_method == "hierarchical":
            return self.generate_hierarchical(sequence_encoding, sequence_length)
        else:
            logger.warning(f"Unknown diversity method: {self.diversity_method}. Using temperature sampling instead.")
            return self.generate_temperature_sampling(sequence_encoding, sequence_length)
    
    def select_diverse_subset(
        self, 
        structures: List[torch.Tensor],
        n_representatives: int = 5,
    ) -> List[torch.Tensor]:
        """
        Select a diverse subset of structures from a larger set.
        
        Uses clustering to identify representative structures.
        
        Args:
            structures: List of predicted structures, each of shape (seq_len, num_atoms, 3)
            n_representatives: Number of representatives to select
            
        Returns:
            List of selected representative structures
        """
        # If we have fewer structures than requested representatives, return all structures
        if len(structures) <= n_representatives:
            return structures
        
        # Select diverse representatives
        indices = self.diversity_evaluator.select_representatives(
            structures, n_representatives=n_representatives
        )
        
        # Select the structures based on the indices
        selected_structures = [structures[i] for i in indices]
        
        return selected_structures


class RNAEnsembleModel(nn.Module):
    """
    Ensemble model for RNA structure prediction.
    
    This model combines multiple RNA folding models and provides a unified
    interface for prediction, either with single or multiple diverse outputs.
    """
    
    def __init__(
        self,
        models: List[RNAFoldingModel],
        model_weights: Optional[List[float]] = None,
        ensemble_method: str = "weighted_average",
        num_structures: int = 5,
        diversity_method: str = "temperature",
        diversity_strength: float = 0.2,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Initialize the RNA ensemble model.
        
        Args:
            models: List of RNA folding models to ensemble
            model_weights: Optional list of weights for each model (defaults to equal weights)
            ensemble_method: Method for combining model predictions
            num_structures: Number of structures to generate per sequence
            diversity_method: Method for generating diverse structures
            diversity_strength: Strength of diversity
            device: Device to run models on
        """
        super().__init__()
        
        self.device = device
        self.num_structures = num_structures
        
        # Create ensembler and diverse structure generator
        self.ensembler = StructureEnsembler(
            models=models,
            model_weights=model_weights,
            ensemble_method=ensemble_method,
            device=device,
        )
        
        # Use the first model for diverse structure generation
        self.diverse_generator = DiverseStructureGenerator(
            model=models[0],
            num_structures=num_structures,
            diversity_method=diversity_method,
            diversity_strength=diversity_strength,
            device=device,
        )
        
        # Move to device
        self.to(device)
    
    def forward(
        self, 
        sequence_encoding: torch.Tensor,
        sequence_length: Optional[int] = None,
        generate_multiple: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for the ensemble model.
        
        Args:
            sequence_encoding: Encoded RNA sequence (batch_size, seq_len)
            sequence_length: Actual length of the sequence (without padding)
            generate_multiple: Whether to generate multiple diverse structures
            
        Returns:
            Either a single ensemble prediction (torch.Tensor) or multiple 
            diverse predictions (List[torch.Tensor])
        """
        # Determine sequence length if not provided
        if sequence_length is None:
            # Assuming padding is zeros and sequence is non-zero
            # This is a simplification and may need to be adapted to your data
            non_zero_mask = (sequence_encoding.sum(dim=-1) != 0)
            sequence_length = non_zero_mask.sum(dim=-1).item()
        
        if generate_multiple:
            # Generate multiple diverse structures
            structures = self.diverse_generator.generate_structures(
                sequence_encoding, sequence_length
            )
            return structures
        else:
            # Generate a single ensemble prediction
            ensemble_pred = self.ensembler.predict(
                sequence_encoding, sequence_length
            )
            return ensemble_pred
    
    def predict_single(
        self, 
        sequence_encoding: torch.Tensor,
        sequence_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate a single ensemble prediction.
        
        Args:
            sequence_encoding: Encoded RNA sequence (batch_size, seq_len)
            sequence_length: Actual length of the sequence (without padding)
            
        Returns:
            Ensemble prediction (batch_size, seq_len, num_atoms, 3)
        """
        return self.forward(sequence_encoding, sequence_length, generate_multiple=False)
    
    def predict_multiple(
        self, 
        sequence_encoding: torch.Tensor,
        sequence_length: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        Generate multiple diverse structure predictions.
        
        Args:
            sequence_encoding: Encoded RNA sequence (batch_size, seq_len)
            sequence_length: Actual length of the sequence (without padding)
            
        Returns:
            List of predicted structures, each of shape (seq_len, num_atoms, 3)
        """
        return self.forward(sequence_encoding, sequence_length, generate_multiple=True) 