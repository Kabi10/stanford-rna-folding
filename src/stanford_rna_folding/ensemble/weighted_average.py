"""
Weighted averaging for RNA ensemble prediction.

This module provides tools to combine multiple structure predictions using
weighted averaging techniques, including:
1. Confidence-weighted averaging
2. Hierarchical consensus building
3. Bayesian model averaging
4. Feature-level integration
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import json
from scipy.spatial import distance_matrix

from ..models.rna_folding_model import RNAFoldingModel
from ..evaluation.metrics import batch_rmsd, batch_tm_score


class StructureEnsembler:
    """
    Combine multiple predicted structures into an ensemble prediction.
    """
    
    def __init__(
        self,
        ensemble_method: str = "weighted_average",
        confidence_method: str = "tm_score",
        temperature: float = 1.0,
        use_median: bool = False,
        hierarchical_threshold: float = 2.5,
        refinement_iterations: int = 3,
        device: str = "cpu",
    ):
        """
        Initialize the structure ensembler.
        
        Args:
            ensemble_method: Method for combining structures 
                             ('weighted_average', 'hierarchical', 'bayesian', 'feature')
            confidence_method: Method for computing confidence scores
                              ('tm_score', 'energy', 'model_confidence')
            temperature: Temperature parameter for softmax weighting
            use_median: Use median instead of weighted mean (more robust to outliers)
            hierarchical_threshold: RMSD threshold for hierarchical clustering (Angstroms)
            refinement_iterations: Number of iterations for post-ensemble refinement
            device: Computation device ('cpu' or 'cuda')
        """
        self.ensemble_method = ensemble_method
        self.confidence_method = confidence_method
        self.temperature = temperature
        self.use_median = use_median
        self.hierarchical_threshold = hierarchical_threshold
        self.refinement_iterations = refinement_iterations
        self.device = device
        
        # Storage for confidence scores
        self.confidence_scores = None
    
    def ensemble(
        self, 
        structures: torch.Tensor,
        confidence_scores: Optional[torch.Tensor] = None,
        sequences: Optional[torch.Tensor] = None,
        models: Optional[List[RNAFoldingModel]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Create an ensemble prediction from multiple structures.
        
        Args:
            structures: Tensor of shape (num_models, seq_len, num_atoms, 3) with predictions
            confidence_scores: Optional pre-computed confidence scores for each structure
            sequences: Optional RNA sequences corresponding to the structures
            models: Optional list of models that generated the structures
            
        Returns:
            Tuple of (ensemble structure, metadata dictionary)
        """
        # Move structures to device
        structures = structures.to(self.device)
        
        # Calculate confidence scores if not provided
        if confidence_scores is None:
            confidence_scores = self._compute_confidence_scores(
                structures=structures,
                sequences=sequences,
                models=models
            )
        
        self.confidence_scores = confidence_scores
        
        # Apply ensemble method
        if self.ensemble_method == "weighted_average":
            ensemble_structure, metadata = self._weighted_average_ensemble(structures, confidence_scores)
        
        elif self.ensemble_method == "hierarchical":
            ensemble_structure, metadata = self._hierarchical_ensemble(structures, confidence_scores)
        
        elif self.ensemble_method == "bayesian":
            ensemble_structure, metadata = self._bayesian_ensemble(structures, confidence_scores)
        
        elif self.ensemble_method == "feature":
            ensemble_structure, metadata = self._feature_level_ensemble(
                structures, confidence_scores, models
            )
            
        else:
            raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")
        
        # Apply post-ensemble refinement if requested
        if self.refinement_iterations > 0:
            ensemble_structure = self._refine_structure(
                ensemble_structure, structures, sequences
            )
            metadata["refinement_applied"] = True
        
        return ensemble_structure, metadata
    
    def _weighted_average_ensemble(
        self, 
        structures: torch.Tensor,
        confidence_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Create an ensemble by weighted averaging of structures.
        
        Args:
            structures: Tensor of shape (num_models, seq_len, num_atoms, 3)
            confidence_scores: Tensor of shape (num_models,)
            
        Returns:
            Tuple of (ensemble structure, metadata dictionary)
        """
        # Apply softmax to get normalized weights
        weights = torch.softmax(confidence_scores / self.temperature, dim=0)
        
        # Get weighted mean or median
        if self.use_median:
            # Sort structures by confidence
            sorted_indices = torch.argsort(confidence_scores, descending=True)
            sorted_structures = structures[sorted_indices]
            
            # Take the median structure (middle of the sorted array)
            middle_idx = len(sorted_indices) // 2
            ensemble_structure = sorted_structures[middle_idx:middle_idx+1]
            
            metadata = {
                "method": "confidence_median",
                "median_structure_idx": sorted_indices[middle_idx].item(),
                "confidence_scores": confidence_scores.cpu().numpy().tolist(),
                "weights": weights.cpu().numpy().tolist()
            }
        else:
            # Weighted average (expand weights for broadcasting)
            weights_expanded = weights.view(-1, 1, 1, 1)
            ensemble_structure = torch.sum(structures * weights_expanded, dim=0, keepdim=True)
            
            metadata = {
                "method": "weighted_average",
                "weights": weights.cpu().numpy().tolist(),
                "confidence_scores": confidence_scores.cpu().numpy().tolist(),
                "temperature": self.temperature
            }
        
        return ensemble_structure, metadata
    
    def _hierarchical_ensemble(
        self, 
        structures: torch.Tensor,
        confidence_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Create an ensemble using hierarchical consensus building.
        
        Args:
            structures: Tensor of shape (num_models, seq_len, num_atoms, 3)
            confidence_scores: Tensor of shape (num_models,)
            
        Returns:
            Tuple of (ensemble structure, metadata dictionary)
        """
        num_structures = structures.shape[0]
        seq_len = structures.shape[1]
        num_atoms = structures.shape[2]
        
        # Sort structures by confidence
        sorted_indices = torch.argsort(confidence_scores, descending=True)
        sorted_structures = structures[sorted_indices]
        
        # Initialize the ensemble with the highest confidence structure
        ensemble = sorted_structures[0:1].clone()
        included_indices = [sorted_indices[0].item()]
        excluded_indices = []
        
        # Track metadata for diagnostics
        rmsd_values = []
        
        # Iteratively add structures if they're sufficiently different
        for i in range(1, num_structures):
            # Calculate RMSD between this structure and current ensemble
            structure_i = sorted_structures[i:i+1]
            rmsd = batch_rmsd(ensemble, structure_i).item()
            rmsd_values.append(rmsd)
            
            if rmsd > self.hierarchical_threshold:
                # Structure is different enough to add to ensemble
                # Update ensemble as weighted average of current ensemble and new structure
                weight_new = confidence_scores[sorted_indices[i]] / (
                    confidence_scores[sorted_indices[i]] + 
                    torch.sum(confidence_scores[included_indices])
                )
                weight_current = 1.0 - weight_new
                
                ensemble = weight_current * ensemble + weight_new * structure_i
                included_indices.append(sorted_indices[i].item())
            else:
                # Structure is too similar, exclude it
                excluded_indices.append(sorted_indices[i].item())
        
        metadata = {
            "method": "hierarchical_consensus",
            "included_indices": included_indices,
            "excluded_indices": excluded_indices,
            "rmsd_values": rmsd_values,
            "threshold": self.hierarchical_threshold,
            "confidence_scores": confidence_scores.cpu().numpy().tolist()
        }
        
        return ensemble, metadata
    
    def _bayesian_ensemble(
        self, 
        structures: torch.Tensor,
        confidence_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Create an ensemble using Bayesian model averaging.
        
        Args:
            structures: Tensor of shape (num_models, seq_len, num_atoms, 3)
            confidence_scores: Tensor of shape (num_models,)
            
        Returns:
            Tuple of (ensemble structure, metadata dictionary)
        """
        # Convert confidence scores to probabilities using softmax
        log_weights = confidence_scores / self.temperature
        log_weights = log_weights - torch.max(log_weights)  # For numerical stability
        weights = torch.exp(log_weights)
        weights = weights / weights.sum()
        
        # Bayesian model averaging with uncertainty estimation
        weights_expanded = weights.view(-1, 1, 1, 1)
        
        # Mean structure (weighted average)
        mean_structure = torch.sum(structures * weights_expanded, dim=0, keepdim=True)
        
        # Calculate variance/uncertainty
        # For each position, compute weighted variance across models
        squared_diff = (structures - mean_structure) ** 2
        variance = torch.sum(squared_diff * weights_expanded, dim=0, keepdim=True)
        uncertainty = torch.sqrt(variance)
        
        # Calculate effective sample size (ESS)
        ess = 1.0 / torch.sum(weights ** 2)
        
        metadata = {
            "method": "bayesian_averaging",
            "weights": weights.cpu().numpy().tolist(),
            "temperature": self.temperature,
            "effective_sample_size": ess.item(),
            "avg_uncertainty": uncertainty.mean().item()
        }
        
        return mean_structure, metadata
    
    def _feature_level_ensemble(
        self, 
        structures: torch.Tensor,
        confidence_scores: torch.Tensor,
        models: Optional[List[RNAFoldingModel]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Create an ensemble at the feature level rather than final structure level.
        
        Args:
            structures: Tensor of shape (num_models, seq_len, num_atoms, 3)
            confidence_scores: Tensor of shape (num_models,)
            models: List of models that generated the structures (needed for feature extraction)
            
        Returns:
            Tuple of (ensemble structure, metadata dictionary)
        """
        # If models not provided, fall back to weighted averaging
        if models is None or len(models) != structures.shape[0]:
            return self._weighted_average_ensemble(structures, confidence_scores)
        
        # First, get high-confidence regions from all models
        num_structures = structures.shape[0]
        seq_len = structures.shape[1]
        num_atoms = structures.shape[2]
        
        # Calculate variance across models at each position to identify agreement
        mean_structure = torch.mean(structures, dim=0, keepdim=True)
        position_variance = torch.mean(
            torch.sum((structures - mean_structure) ** 2, dim=3),  # Sum over x,y,z
            dim=2  # Average over atoms
        )
        
        # Identify positions with high agreement (low variance)
        agreement_mask = position_variance < torch.median(position_variance)
        
        # For high-agreement regions, use weighted average
        weights = torch.softmax(confidence_scores / self.temperature, dim=0)
        weights_expanded = weights.view(-1, 1, 1, 1)
        weighted_avg = torch.sum(structures * weights_expanded, dim=0, keepdim=True)
        
        # For low-agreement regions, use the highest confidence model
        best_model_idx = torch.argmax(confidence_scores).item()
        best_structure = structures[best_model_idx:best_model_idx+1]
        
        # Combine using the agreement mask
        # Expand mask to match structure dimensionality
        mask_expanded = agreement_mask.view(1, seq_len, 1, 1).expand(1, seq_len, num_atoms, 3)
        
        # Where agreement is high, use weighted average; otherwise use best model
        ensemble_structure = torch.where(
            mask_expanded,
            weighted_avg,
            best_structure
        )
        
        metadata = {
            "method": "feature_level_integration",
            "agreement_percentage": agreement_mask.float().mean().item() * 100,
            "best_model_idx": best_model_idx,
            "weights": weights.cpu().numpy().tolist(),
            "confidence_scores": confidence_scores.cpu().numpy().tolist()
        }
        
        return ensemble_structure, metadata
    
    def _compute_confidence_scores(
        self,
        structures: torch.Tensor,
        sequences: Optional[torch.Tensor] = None,
        models: Optional[List[RNAFoldingModel]] = None
    ) -> torch.Tensor:
        """
        Compute confidence scores for each predicted structure.
        
        Args:
            structures: Tensor of shape (num_models, seq_len, num_atoms, 3)
            sequences: Optional RNA sequences corresponding to the structures
            models: Optional list of models that generated the structures
            
        Returns:
            Tensor of confidence scores for each structure
        """
        num_structures = structures.shape[0]
        
        if self.confidence_method == "tm_score":
            # Use average TM-score against all other structures as confidence
            confidence_scores = torch.zeros(num_structures, device=self.device)
            
            for i in range(num_structures):
                tm_scores = []
                for j in range(num_structures):
                    if i != j:
                        tm_score = batch_tm_score(
                            structures[i:i+1],
                            structures[j:j+1]
                        ).item()
                        tm_scores.append(tm_score)
                
                # Average TM-score against all other structures
                confidence_scores[i] = torch.tensor(
                    np.mean(tm_scores), 
                    device=self.device
                )
            
        elif self.confidence_method == "energy":
            # Use physics-based energy as confidence (lower is better)
            if models is None or sequences is None:
                # Fall back to mutual agreement if models or sequences not provided
                return self._compute_confidence_scores(
                    structures=structures, 
                    confidence_method="tm_score"
                )
            
            # Compute energy for each structure
            energy_scores = torch.zeros(num_structures, device=self.device)
            
            for i, model in enumerate(models):
                if hasattr(model, 'compute_energy'):
                    energy = model.compute_energy(
                        structures[i:i+1], 
                        sequences
                    )
                    # Negative energy (lower is better, so negate for confidence)
                    energy_scores[i] = -energy
                else:
                    # If model doesn't have energy function, use default value
                    energy_scores[i] = torch.tensor(0.0, device=self.device)
            
            # Normalize energy scores
            if torch.max(energy_scores) > torch.min(energy_scores):
                energy_scores = (energy_scores - torch.min(energy_scores)) / (
                    torch.max(energy_scores) - torch.min(energy_scores)
                )
            
            confidence_scores = energy_scores
            
        elif self.confidence_method == "model_confidence":
            # Use model's own confidence scores
            if models is None or sequences is None:
                # Fall back to mutual agreement if models or sequences not provided
                return self._compute_confidence_scores(
                    structures=structures, 
                    confidence_method="tm_score"
                )
            
            confidence_scores = torch.zeros(num_structures, device=self.device)
            
            for i, model in enumerate(models):
                if hasattr(model, 'estimate_confidence'):
                    confidence = model.estimate_confidence(
                        structures[i:i+1], 
                        sequences
                    )
                    confidence_scores[i] = confidence
                else:
                    # Default confidence based on model index (assuming models are sorted by quality)
                    confidence_scores[i] = torch.tensor(
                        1.0 - (i / num_structures), 
                        device=self.device
                    )
        
        else:
            raise ValueError(f"Unsupported confidence method: {self.confidence_method}")
        
        return confidence_scores
    
    def _refine_structure(
        self,
        ensemble_structure: torch.Tensor,
        original_structures: torch.Tensor,
        sequences: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Refine the ensemble structure to improve local geometry.
        
        Args:
            ensemble_structure: Initial ensemble structure of shape (1, seq_len, num_atoms, 3)
            original_structures: Individual predicted structures
            sequences: Optional RNA sequences
            
        Returns:
            Refined ensemble structure
        """
        # Simple refinement: for each position, find the structure with closest
        # atom positions to the ensemble, and use its local geometry
        
        seq_len = ensemble_structure.shape[1]
        num_atoms = ensemble_structure.shape[2]
        num_structures = original_structures.shape[0]
        
        # Initialize refined structure with ensemble
        refined_structure = ensemble_structure.clone()
        
        # For each position, find the best matching structure
        for pos in range(seq_len):
            # Extract ensemble coordinates at this position
            pos_coords = ensemble_structure[0, pos]  # Shape: (num_atoms, 3)
            
            # Calculate distances to all structures at this position
            distances = torch.zeros(num_structures, device=self.device)
            
            for i in range(num_structures):
                # Calculate RMSD between ensemble and this structure at this position
                struct_coords = original_structures[i, pos]  # Shape: (num_atoms, 3)
                distances[i] = torch.sqrt(torch.mean((pos_coords - struct_coords) ** 2))
            
            # Find structure with minimum distance
            best_idx = torch.argmin(distances).item()
            
            # Replace this position in the refined structure
            refined_structure[0, pos] = original_structures[best_idx, pos]
        
        # Additional refinement iterations could be applied here
        # For simplicity, we'll just return the position-wise refined structure
        return refined_structure


def ensemble_structures(
    structures: torch.Tensor,
    ensemble_method: str = "weighted_average",
    confidence_method: str = "tm_score",
    temperature: float = 1.0,
    use_median: bool = False,
    sequences: Optional[torch.Tensor] = None,
    models: Optional[List[RNAFoldingModel]] = None,
    output_dir: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Create an ensemble prediction from multiple structures.
    
    Args:
        structures: Tensor of shape (num_models, seq_len, num_atoms, 3)
        ensemble_method: Method for combining structures
        confidence_method: Method for computing confidence scores
        temperature: Temperature parameter for softmax weighting
        use_median: Use median instead of weighted mean
        sequences: Optional RNA sequences
        models: Optional list of models that generated the structures
        output_dir: Directory to save metadata
        device: Computation device
        
    Returns:
        Tuple of (ensemble structure, metadata dictionary)
    """
    ensembler = StructureEnsembler(
        ensemble_method=ensemble_method,
        confidence_method=confidence_method,
        temperature=temperature,
        use_median=use_median,
        device=device
    )
    
    # Create ensemble
    ensemble_structure, metadata = ensembler.ensemble(
        structures=structures,
        sequences=sequences,
        models=models
    )
    
    # Save metadata if output directory provided
    if output_dir:
        output_path = Path(output_dir) / f"ensemble_metadata_{ensemble_method}.json"
        # Convert numpy arrays to lists for JSON serialization
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                metadata[key] = value.tolist()
                
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return ensemble_structure, metadata 