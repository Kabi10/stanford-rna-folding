#!/usr/bin/env python3
"""
Phase-Aware Training Script for Stanford RNA 3D Folding Competition
Handles training across the three competition phases with temporal compliance
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
import pandas as pd
from datetime import date

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stanford_rna_folding.competition.phase_manager import (
    CompetitionPhaseManager, 
    ModelVersionManager,
    PhaseAwareEnsemble
)
from stanford_rna_folding.training.train import train_model
from stanford_rna_folding.models.rna_folding_model import RNAFoldingModel
from stanford_rna_folding.data.data_processing import StanfordRNADataset, rna_collate_fn
from stanford_rna_folding.evaluation.metrics import batch_rmsd, batch_tm_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_phase_config(config_path: str) -> dict:
    """Load phase-aware configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_phase_managers(config: dict) -> tuple:
    """Setup phase and model managers"""
    data_path = Path(config['data']['paths']['train_sequences']).parent
    models_path = Path(config['models']['save_path'])
    
    phase_manager = CompetitionPhaseManager(data_path)
    model_manager = ModelVersionManager(models_path)
    
    return phase_manager, model_manager


def get_phase_specific_config(config: dict, phase: int) -> dict:
    """Get configuration specific to the current phase"""
    phase_key = f"phase{phase}"
    
    if phase_key not in config['training']:
        logger.warning(f"No specific config for phase {phase}, using phase1 config")
        phase_key = "phase1"
    
    phase_config = config['training'][phase_key].copy()
    phase_config['competition_phase'] = phase
    
    return phase_config


def validate_temporal_compliance(phase_manager: CompetitionPhaseManager, 
                               data: pd.DataFrame, 
                               phase: int) -> bool:
    """Validate temporal compliance for the given phase"""
    is_compliant = phase_manager.validate_temporal_compliance(data, phase)
    
    if not is_compliant:
        logger.error(f"Temporal compliance violation detected for phase {phase}")
        if phase <= 2:  # Strict enforcement for phases 1 and 2
            raise ValueError("Temporal compliance violation - training aborted")
    
    return is_compliant


def create_phase_datasets(phase_manager: CompetitionPhaseManager, 
                         phase: int,
                         config: dict) -> tuple:
    """Create datasets appropriate for the current phase"""
    
    # Get phase-appropriate data
    train_sequences, train_labels = phase_manager.get_phase_training_data(phase)
    val_sequences, val_labels = phase_manager.get_validation_data(phase)
    
    # Validate temporal compliance
    validate_temporal_compliance(phase_manager, train_sequences, phase)
    
    logger.info(f"Phase {phase} datasets created:")
    logger.info(f"  Training: {len(train_sequences)} sequences, {len(train_labels)} labels")
    logger.info(f"  Validation: {len(val_sequences)} sequences, {len(val_labels)} labels")
    
    # Create datasets
    train_dataset = StanfordRNADataset(
        sequences_df=train_sequences,
        labels_df=train_labels,
        max_length=config.get('max_sequence_length', 512)
    )
    
    val_dataset = StanfordRNADataset(
        sequences_df=val_sequences,
        labels_df=val_labels,
        max_length=config.get('max_sequence_length', 512)
    )
    
    return train_dataset, val_dataset


def train_phase_model(phase: int, 
                     config: dict,
                     phase_manager: CompetitionPhaseManager,
                     model_manager: ModelVersionManager) -> str:
    """Train a model for the specified phase"""
    
    logger.info(f"Starting training for Phase {phase}")
    
    # Get phase-specific configuration
    phase_config = get_phase_specific_config(config, phase)
    
    # Create datasets
    train_dataset, val_dataset = create_phase_datasets(
        phase_manager, phase, phase_config
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=phase_config['training']['batch_size'],
        shuffle=True,
        collate_fn=rna_collate_fn,
        num_workers=config['resources']['optimization'].get('dataloader_workers', 4)
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=phase_config['training']['batch_size'],
        shuffle=False,
        collate_fn=rna_collate_fn,
        num_workers=config['resources']['optimization'].get('dataloader_workers', 4)
    )
    
    # Create model
    model_config = phase_config['model']
    model = RNAFoldingModel(
        vocab_size=5,  # A, C, G, U, padding
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        dropout=model_config['dropout']
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    trained_model, best_rmsd, best_tm = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=phase_config['training']['num_epochs'],
        learning_rate=phase_config['training']['learning_rate'],
        device=device,
        use_mixed_precision=config['resources']['optimization']['mixed_precision'],
        gradient_clip=phase_config['training']['gradient_clip'],
        patience=phase_config['training']['patience']
    )
    
    # Save model with phase information
    metrics = {
        'val_rmsd': best_rmsd,
        'val_tm_score': best_tm,
        'phase': phase
    }
    
    metadata = {
        'temporal_cutoff': phase_manager.phase_cutoffs[phase],
        'training_samples': len(train_dataset),
        'validation_samples': len(val_dataset),
        'config': phase_config
    }
    
    model_id = model_manager.save_model(
        model=trained_model,
        phase=phase,
        model_name=f"rna_folding_phase{phase}",
        metrics=metrics,
        metadata=metadata
    )
    
    logger.info(f"Phase {phase} training completed. Model saved as {model_id}")
    logger.info(f"Best RMSD: {best_rmsd:.4f}, Best TM-score: {best_tm:.4f}")
    
    return model_id


def create_ensemble_for_phase3(config: dict,
                              model_manager: ModelVersionManager) -> dict:
    """Create ensemble for Phase 3 final evaluation"""
    
    logger.info("Creating ensemble for Phase 3 evaluation")
    
    # Get best models from each phase
    phase1_models = model_manager.get_best_models(
        phase=1, 
        metric='val_rmsd',
        top_k=config['models']['ensemble']['max_models_per_phase']
    )
    
    phase2_models = model_manager.get_best_models(
        phase=2,
        metric='val_rmsd', 
        top_k=config['models']['ensemble']['max_models_per_phase']
    )
    
    # Create ensemble
    ensemble_manager = PhaseAwareEnsemble(model_manager)
    
    phase1_model_ids = [m['model_id'] for m in phase1_models]
    phase2_model_ids = [m['model_id'] for m in phase2_models]
    
    # Create weighted ensemble based on phase performance
    weights = {}
    phase1_weight = config['training']['phase3']['ensemble']['phase1_weight']
    phase2_weight = config['training']['phase3']['ensemble']['phase2_weight']
    
    for model_id in phase1_model_ids:
        weights[model_id] = phase1_weight / len(phase1_model_ids)
    
    for model_id in phase2_model_ids:
        weights[model_id] = phase2_weight / len(phase2_model_ids)
    
    ensemble_config = ensemble_manager.create_ensemble(
        phase1_models=phase1_model_ids,
        phase2_models=phase2_model_ids,
        weights=weights
    )
    
    logger.info(f"Ensemble created with {len(phase1_model_ids)} Phase 1 models "
               f"and {len(phase2_model_ids)} Phase 2 models")
    
    return ensemble_config


def main():
    parser = argparse.ArgumentParser(description="Phase-aware training for RNA folding competition")
    parser.add_argument("--config", default="configs/phase_aware_config.yaml",
                       help="Path to phase-aware configuration file")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3],
                       help="Competition phase to train for (auto-detected if not specified)")
    parser.add_argument("--force-phase", action="store_true",
                       help="Force training for specified phase regardless of date")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_phase_config(args.config)
    
    # Setup managers
    phase_manager, model_manager = setup_phase_managers(config)
    
    # Determine current phase
    if args.phase:
        current_phase = args.phase
        if not args.force_phase:
            detected_phase = phase_manager.get_current_phase()
            if current_phase != detected_phase:
                logger.warning(f"Specified phase {current_phase} differs from "
                             f"detected phase {detected_phase}")
    else:
        current_phase = phase_manager.get_current_phase()
    
    logger.info(f"Training for Competition Phase {current_phase}")
    
    # Execute phase-specific training
    if current_phase in [1, 2]:
        model_id = train_phase_model(
            phase=current_phase,
            config=config,
            phase_manager=phase_manager,
            model_manager=model_manager
        )
        logger.info(f"Phase {current_phase} training completed. Model: {model_id}")
        
    elif current_phase == 3:
        ensemble_config = create_ensemble_for_phase3(config, model_manager)
        logger.info("Phase 3 ensemble created and ready for final evaluation")
        
    else:
        logger.error(f"Invalid phase: {current_phase}")
        sys.exit(1)


if __name__ == "__main__":
    main()
