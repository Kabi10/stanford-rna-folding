#!/usr/bin/env python3
"""
Experiment tracking system for Stanford RNA 3D Structure Prediction.

This module provides comprehensive experiment tracking, model versioning,
and performance monitoring capabilities for iterative model development.
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import yaml

import torch
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Comprehensive experiment tracking system for RNA folding models.
    """
    
    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for storing experiments
        """
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / experiment_name
        
        # Create experiment directory structure
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "checkpoints").mkdir(exist_ok=True)
        (self.experiment_dir / "logs").mkdir(exist_ok=True)
        (self.experiment_dir / "plots").mkdir(exist_ok=True)
        (self.experiment_dir / "configs").mkdir(exist_ok=True)
        
        # Initialize tracking data
        self.start_time = time.time()
        self.metrics_history = []
        self.best_metrics = {}
        self.config = {}
        
        # Setup logging
        self.setup_logging()
        
        logger.info(f"Initialized experiment tracker: {experiment_name}")
    
    def setup_logging(self):
        """Setup experiment-specific logging."""
        log_file = self.experiment_dir / "logs" / f"{self.experiment_name}.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.config = config.copy()
        
        # Save config to file
        config_file = self.experiment_dir / "configs" / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Logged configuration: {len(config)} parameters")
    
    def log_model_info(self, model: torch.nn.Module):
        """Log model architecture information."""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            'model_class': model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        }
        
        # Save model info
        info_file = self.experiment_dir / "configs" / "model_info.json"
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Model: {total_params:,} total params, {trainable_params:,} trainable")
        
        return model_info
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float], phase: str = "train"):
        """Log training/validation metrics for an epoch."""
        timestamp = time.time()
        
        metric_entry = {
            'epoch': epoch,
            'phase': phase,
            'timestamp': timestamp,
            'elapsed_time': timestamp - self.start_time,
            **metrics
        }
        
        self.metrics_history.append(metric_entry)
        
        # Update best metrics
        if phase == "validation":
            for metric_name, value in metrics.items():
                if metric_name.endswith('_loss'):
                    # For loss metrics, lower is better
                    if metric_name not in self.best_metrics or value < self.best_metrics[metric_name]['value']:
                        self.best_metrics[metric_name] = {
                            'value': value,
                            'epoch': epoch,
                            'timestamp': timestamp
                        }
                else:
                    # For other metrics, higher is better
                    if metric_name not in self.best_metrics or value > self.best_metrics[metric_name]['value']:
                        self.best_metrics[metric_name] = {
                            'value': value,
                            'epoch': epoch,
                            'timestamp': timestamp
                        }
        
        # Log to file
        logger.info(f"Epoch {epoch} ({phase}): {metrics}")
    
    def save_checkpoint(self, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'timestamp': time.time()
        }
        
        # Save regular checkpoint
        checkpoint_file = self.experiment_dir / "checkpoints" / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_file)
        
        # Save best checkpoint
        if is_best:
            best_file = self.experiment_dir / "checkpoints" / "best_model.pth"
            torch.save(checkpoint, best_file)
            logger.info(f"Saved best model at epoch {epoch}")
        
        logger.info(f"Saved checkpoint: epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint
    
    def export_metrics(self) -> pd.DataFrame:
        """Export metrics history as DataFrame."""
        if not self.metrics_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.metrics_history)
        
        # Save to CSV
        csv_file = self.experiment_dir / "logs" / "metrics_history.csv"
        df.to_csv(csv_file, index=False)
        
        return df
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate experiment summary."""
        total_time = time.time() - self.start_time
        
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'total_duration_seconds': total_time,
            'total_duration_formatted': self._format_duration(total_time),
            'total_epochs': len([m for m in self.metrics_history if m['phase'] == 'train']),
            'best_metrics': self.best_metrics,
            'config_summary': {
                'model_type': self.config.get('model_type', 'unknown'),
                'learning_rate': self.config.get('learning_rate', 'unknown'),
                'batch_size': self.config.get('batch_size', 'unknown'),
                'num_epochs': self.config.get('num_epochs', 'unknown')
            }
        }
        
        # Save summary
        summary_file = self.experiment_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """Clean up old checkpoints, keeping only the last N."""
        checkpoint_dir = self.experiment_dir / "checkpoints"
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        
        if len(checkpoint_files) <= keep_last_n:
            return
        
        # Sort by epoch number
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        # Remove old checkpoints
        for checkpoint_file in checkpoint_files[:-keep_last_n]:
            checkpoint_file.unlink()
            logger.info(f"Removed old checkpoint: {checkpoint_file.name}")


class ModelVersionManager:
    """
    Model version management system for tracking model evolution.
    """
    
    def __init__(self, base_dir: str = "model_versions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.versions_file = self.base_dir / "versions.json"
        self.versions = self._load_versions()
    
    def _load_versions(self) -> Dict[str, Any]:
        """Load version history."""
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_versions(self):
        """Save version history."""
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions, f, indent=2, default=str)
    
    def register_version(self, version_name: str, model_path: str, 
                        metrics: Dict[str, float], description: str = ""):
        """Register a new model version."""
        version_info = {
            'version_name': version_name,
            'model_path': model_path,
            'metrics': metrics,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'file_size_mb': Path(model_path).stat().st_size / (1024 * 1024)
        }
        
        self.versions[version_name] = version_info
        self._save_versions()
        
        logger.info(f"Registered model version: {version_name}")
    
    def get_best_version(self, metric_name: str, higher_is_better: bool = False) -> Optional[str]:
        """Get the best model version based on a specific metric."""
        if not self.versions:
            return None
        
        best_version = None
        best_value = float('-inf') if higher_is_better else float('inf')
        
        for version_name, version_info in self.versions.items():
            if metric_name in version_info['metrics']:
                value = version_info['metrics'][metric_name]
                
                if higher_is_better and value > best_value:
                    best_value = value
                    best_version = version_name
                elif not higher_is_better and value < best_value:
                    best_value = value
                    best_version = version_name
        
        return best_version
    
    def list_versions(self) -> pd.DataFrame:
        """List all model versions as DataFrame."""
        if not self.versions:
            return pd.DataFrame()
        
        data = []
        for version_name, version_info in self.versions.items():
            row = {
                'version': version_name,
                'timestamp': version_info['timestamp'],
                'description': version_info['description'],
                'file_size_mb': version_info['file_size_mb']
            }
            row.update(version_info['metrics'])
            data.append(row)
        
        return pd.DataFrame(data)


# Example usage
if __name__ == "__main__":
    # Initialize tracker
    tracker = ExperimentTracker("baseline_v1")
    
    # Log configuration
    config = {
        'model_type': 'SimpleRNAModel',
        'learning_rate': 1e-3,
        'batch_size': 4,
        'num_epochs': 3
    }
    tracker.log_config(config)
    
    # Simulate training metrics
    for epoch in range(1, 4):
        train_metrics = {'loss': 1000 - epoch * 100, 'accuracy': epoch * 0.1}
        val_metrics = {'loss': 1200 - epoch * 50, 'accuracy': epoch * 0.08}
        
        tracker.log_metrics(epoch, train_metrics, 'train')
        tracker.log_metrics(epoch, val_metrics, 'validation')
    
    # Generate summary
    summary = tracker.generate_summary()
    print("Experiment Summary:", summary)
