"""
Competition Phase Management System
Handles the three-phase structure of the Stanford RNA 3D Folding competition
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class CompetitionPhaseManager:
    """Manages data and models across competition phases"""
    
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self.current_phase = 1
        self.phase_cutoffs = {
            1: "2022-05-27",  # Safe cutoff for CASP15 validation
            2: "2024-09-18",  # CASP16 competition end
            3: None           # No cutoff for final phase
        }
        self.phase_transition_date = "2025-04-23"
        
    def get_current_phase(self) -> int:
        """Determine current competition phase based on date"""
        today = date.today().isoformat()
        if today < self.phase_transition_date:
            return 1
        else:
            # Phase 2 or 3 - would need additional logic for Phase 3 detection
            return 2
    
    def filter_by_temporal_cutoff(self, 
                                  data: pd.DataFrame, 
                                  cutoff_date: str,
                                  date_column: str = 'temporal_cutoff') -> pd.DataFrame:
        """Filter data by temporal cutoff to prevent data leakage"""
        if date_column not in data.columns:
            logger.warning(f"Date column '{date_column}' not found. Returning original data.")
            return data
            
        # Convert to datetime for comparison
        data_dates = pd.to_datetime(data[date_column])
        cutoff = pd.to_datetime(cutoff_date)
        
        filtered_data = data[data_dates <= cutoff].copy()
        
        logger.info(f"Filtered data: {len(filtered_data)}/{len(data)} samples "
                   f"before cutoff {cutoff_date}")
        
        return filtered_data
    
    def get_phase_training_data(self, phase: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get appropriate training data for specified phase"""
        
        # Load base training data
        train_sequences = pd.read_csv(self.data_path / "train_sequences.csv")
        train_labels = pd.read_csv(self.data_path / "train_labels.csv")
        
        if phase == 1:
            # Phase 1: Use only pre-CASP15 data for safe validation
            cutoff = self.phase_cutoffs[1]
            train_sequences_filtered = self.filter_by_temporal_cutoff(
                train_sequences, cutoff
            )
            
            # Filter labels to match filtered sequences
            valid_ids = set(train_sequences_filtered['target_id'])
            # Extract target_id from label ID (format: target_id_residue_number)
            train_labels_filtered = train_labels[
                train_labels['ID'].str.extract(r'^([^_]+_[^_]+)_')[0].isin(valid_ids)
            ]
            
            logger.info(f"Phase 1 data: {len(train_sequences_filtered)} sequences, "
                       f"{len(train_labels_filtered)} labels")
            
            return train_sequences_filtered, train_labels_filtered
            
        elif phase == 2:
            # Phase 2: Use expanded dataset (includes Phase 1 public test)
            # This would be implemented when Phase 2 begins
            cutoff = self.phase_cutoffs[2]
            train_sequences_filtered = self.filter_by_temporal_cutoff(
                train_sequences, cutoff
            )
            
            # In Phase 2, we would also integrate previous public test data
            # For now, return the temporally filtered data
            valid_ids = set(train_sequences_filtered['target_id'])
            # Extract target_id from label ID (format: target_id_residue_number)
            train_labels_filtered = train_labels[
                train_labels['ID'].str.extract(r'^([^_]+_[^_]+)_')[0].isin(valid_ids)
            ]
            
            logger.info(f"Phase 2 data: {len(train_sequences_filtered)} sequences, "
                       f"{len(train_labels_filtered)} labels")
            
            return train_sequences_filtered, train_labels_filtered
            
        else:
            # Phase 3: Use all available data
            logger.info(f"Phase 3 data: {len(train_sequences)} sequences, "
                       f"{len(train_labels)} labels")
            return train_sequences, train_labels
    
    def get_validation_data(self, phase: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get appropriate validation data for specified phase"""
        
        # CASP15 targets are always safe for validation
        val_sequences = pd.read_csv(self.data_path / "validation_sequences.csv")
        val_labels = pd.read_csv(self.data_path / "validation_labels.csv")
        
        logger.info(f"Validation data: {len(val_sequences)} sequences, "
                   f"{len(val_labels)} labels")
        
        return val_sequences, val_labels
    
    def validate_temporal_compliance(self, 
                                   data: pd.DataFrame, 
                                   phase: int,
                                   date_column: str = 'temporal_cutoff') -> bool:
        """Validate that data complies with temporal cutoff for given phase"""
        
        if phase >= 3:  # No restrictions in final phase
            return True
            
        cutoff = self.phase_cutoffs[phase]
        if cutoff is None:
            return True
            
        if date_column not in data.columns:
            logger.warning(f"Cannot validate temporal compliance: "
                          f"'{date_column}' column not found")
            return False
        
        data_dates = pd.to_datetime(data[date_column])
        cutoff_date = pd.to_datetime(cutoff)
        
        violations = data_dates > cutoff_date
        num_violations = violations.sum()
        
        if num_violations > 0:
            logger.error(f"Temporal compliance violation: {num_violations} samples "
                        f"after cutoff {cutoff} in phase {phase}")
            return False
        
        logger.info(f"Temporal compliance validated for phase {phase}")
        return True


class ModelVersionManager:
    """Manages model versions across competition phases"""
    
    def __init__(self, models_path: Path):
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.model_registry = {}
        self.load_registry()
    
    def save_model(self, 
                   model: Any, 
                   phase: int, 
                   model_name: str,
                   metrics: Dict[str, float],
                   metadata: Optional[Dict] = None) -> str:
        """Save model with phase and performance metadata"""
        
        timestamp = datetime.now().isoformat()
        model_id = f"phase{phase}_{model_name}_{timestamp}"
        
        model_info = {
            'model_id': model_id,
            'phase': phase,
            'model_name': model_name,
            'metrics': metrics,
            'metadata': metadata or {},
            'timestamp': timestamp,
            'file_path': str(self.models_path / f"{model_id}.pt")
        }
        
        # Save model file (implementation depends on model type)
        # torch.save(model.state_dict(), model_info['file_path'])
        
        # Update registry
        self.model_registry[model_id] = model_info
        self.save_registry()
        
        logger.info(f"Saved model {model_id} for phase {phase}")
        return model_id
    
    def get_phase_models(self, phase: int) -> List[Dict]:
        """Get all models for specified phase"""
        phase_models = [
            model_info for model_info in self.model_registry.values()
            if model_info['phase'] == phase
        ]
        
        # Sort by performance (assuming higher is better)
        phase_models.sort(
            key=lambda x: x['metrics'].get('val_rmsd', float('inf'))
        )
        
        return phase_models
    
    def get_best_models(self, 
                       phase: int, 
                       metric: str = 'val_rmsd',
                       top_k: int = 5) -> List[Dict]:
        """Get top-k best models for specified phase"""
        phase_models = self.get_phase_models(phase)
        
        # Sort by specified metric (assuming lower is better for RMSD)
        phase_models.sort(
            key=lambda x: x['metrics'].get(metric, float('inf'))
        )
        
        return phase_models[:top_k]
    
    def save_registry(self):
        """Save model registry to disk"""
        registry_path = self.models_path / "model_registry.json"
        with open(registry_path, 'w') as f:
            json.dump(self.model_registry, f, indent=2)
    
    def load_registry(self):
        """Load model registry from disk"""
        registry_path = self.models_path / "model_registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                self.model_registry = json.load(f)
        else:
            self.model_registry = {}


class PhaseAwareEnsemble:
    """Ensemble strategy that combines models across phases"""
    
    def __init__(self, model_manager: ModelVersionManager):
        self.model_manager = model_manager
        self.ensemble_weights = {}
    
    def create_ensemble(self, 
                       phase1_models: List[str],
                       phase2_models: Optional[List[str]] = None,
                       weights: Optional[Dict[str, float]] = None) -> Dict:
        """Create ensemble from models across phases"""
        
        ensemble_config = {
            'phase1_models': phase1_models,
            'phase2_models': phase2_models or [],
            'weights': weights or {},
            'created_at': datetime.now().isoformat()
        }
        
        # Validate that all models exist
        all_models = phase1_models + (phase2_models or [])
        for model_id in all_models:
            if model_id not in self.model_manager.model_registry:
                raise ValueError(f"Model {model_id} not found in registry")
        
        return ensemble_config
    
    def predict_ensemble(self, 
                        ensemble_config: Dict,
                        sequences: List[str]) -> np.ndarray:
        """Generate ensemble predictions (placeholder implementation)"""
        
        # This would implement actual ensemble prediction logic
        # For now, return placeholder
        logger.info(f"Generating ensemble predictions for {len(sequences)} sequences")
        
        # Placeholder: return random coordinates
        num_sequences = len(sequences)
        max_length = max(len(seq) for seq in sequences)
        
        # Return shape: (num_sequences, max_length, 5, 3) for 5 conformations
        predictions = np.random.randn(num_sequences, max_length, 5, 3)
        
        return predictions


def get_phase_manager(data_path: str) -> CompetitionPhaseManager:
    """Factory function to create phase manager"""
    return CompetitionPhaseManager(Path(data_path))


def get_model_manager(models_path: str) -> ModelVersionManager:
    """Factory function to create model version manager"""
    return ModelVersionManager(Path(models_path))
