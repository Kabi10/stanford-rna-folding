"""
Kaggle-optimized checkpoint management for RNA folding model training.
Handles saving, loading, and exporting models in Kaggle environment.
"""

import os
import json
import shutil
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

class KaggleCheckpointManager:
    """Manages model checkpoints in Kaggle environment."""
    
    def __init__(
        self,
        checkpoint_dir: str = "/kaggle/working/checkpoints",
        max_checkpoints: int = 3,
        save_optimizer: bool = True,
        save_scheduler: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        
        self.best_metrics = {}
        self.checkpoint_history = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        metrics: Dict[str, float],
        config: Dict,
        is_best: bool = False,
        checkpoint_type: str = "regular"
    ) -> Path:
        """Save model checkpoint with metadata."""
        
        # Create checkpoint data
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "config": config,
            "timestamp": datetime.now().isoformat(),
            "checkpoint_type": checkpoint_type
        }
        
        # Add optimizer and scheduler if requested
        if self.save_optimizer and optimizer is not None:
            checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()
        
        if self.save_scheduler and scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()
        
        # Determine filename
        if is_best:
            filename = f"best_model_epoch_{epoch}.pth"
        else:
            filename = f"checkpoint_epoch_{epoch}.pth"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Save checkpoint
        try:
            torch.save(checkpoint_data, checkpoint_path)
            self.logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Update history
            self.checkpoint_history.append({
                "path": str(checkpoint_path),
                "epoch": epoch,
                "metrics": metrics,
                "is_best": is_best,
                "timestamp": checkpoint_data["timestamp"]
            })
            
            # Save metadata
            self._save_checkpoint_metadata()
            
            # Clean up old checkpoints
            if not is_best:
                self._cleanup_old_checkpoints()
            
            return checkpoint_path
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda"
    ) -> Dict:
        """Load checkpoint and restore model state."""
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Restore model state
            model.load_state_dict(checkpoint["model_state_dict"])
            
            # Restore optimizer state
            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            # Restore scheduler state
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
            
            return {
                "epoch": checkpoint["epoch"],
                "metrics": checkpoint["metrics"],
                "config": checkpoint.get("config", {}),
                "timestamp": checkpoint.get("timestamp", "unknown")
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def get_best_checkpoint(self, metric: str = "rmsd", minimize: bool = True) -> Optional[Path]:
        """Get path to best checkpoint based on metric."""
        
        if not self.checkpoint_history:
            return None
        
        best_checkpoint = None
        best_value = float('inf') if minimize else float('-inf')
        
        for checkpoint_info in self.checkpoint_history:
            if metric in checkpoint_info["metrics"]:
                value = checkpoint_info["metrics"][metric]
                
                if minimize and value < best_value:
                    best_value = value
                    best_checkpoint = checkpoint_info["path"]
                elif not minimize and value > best_value:
                    best_value = value
                    best_checkpoint = checkpoint_info["path"]
        
        return Path(best_checkpoint) if best_checkpoint else None
    
    def export_for_kaggle_submission(
        self,
        model: torch.nn.Module,
        config: Dict,
        output_dir: str = "/kaggle/working/submission"
    ) -> Dict[str, str]:
        """Export model for Kaggle submission or sharing."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # Export model state dict only (smaller file)
        model_path = output_dir / "rna_folding_model.pth"
        torch.save(model.state_dict(), model_path)
        exported_files["model_state"] = str(model_path)
        
        # Export full model (includes architecture)
        full_model_path = output_dir / "rna_folding_model_full.pth"
        torch.save(model, full_model_path)
        exported_files["full_model"] = str(full_model_path)
        
        # Export config
        config_path = output_dir / "model_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        exported_files["config"] = str(config_path)
        
        # Export model info
        model_info = {
            "model_class": model.__class__.__name__,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "model_size_mb": model_path.stat().st_size / (1024 * 1024),
            "export_timestamp": datetime.now().isoformat()
        }
        
        info_path = output_dir / "model_info.json"
        with open(info_path, "w") as f:
            json.dump(model_info, f, indent=2)
        exported_files["model_info"] = str(info_path)
        
        # Create README
        readme_content = f"""# RNA Folding Model Export

## Model Information
- Parameters: {model_info['num_parameters']:,}
- Model size: {model_info['model_size_mb']:.2f} MB
- Export date: {model_info['export_timestamp']}

## Files
- `rna_folding_model.pth`: Model state dict (for loading into existing architecture)
- `rna_folding_model_full.pth`: Complete model (includes architecture)
- `model_config.json`: Training configuration used
- `model_info.json`: Model metadata

## Usage
```python
import torch
from src.stanford_rna_folding.models.rna_folding_model import RNAFoldingModel

# Load model
model = RNAFoldingModel(**config)
model.load_state_dict(torch.load('rna_folding_model.pth'))
model.eval()

# Or load full model
model = torch.load('rna_folding_model_full.pth')
model.eval()
```
"""
        
        readme_path = output_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)
        exported_files["readme"] = str(readme_path)
        
        self.logger.info(f"Model exported to: {output_dir}")
        return exported_files
    
    def _save_checkpoint_metadata(self):
        """Save checkpoint history metadata."""
        metadata_path = self.checkpoint_dir / "checkpoint_metadata.json"
        
        metadata = {
            "checkpoint_history": self.checkpoint_history,
            "best_metrics": self.best_metrics,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save space."""
        
        # Get regular checkpoints (not best)
        regular_checkpoints = [
            info for info in self.checkpoint_history 
            if not info["is_best"]
        ]
        
        # Sort by epoch (newest first)
        regular_checkpoints.sort(key=lambda x: x["epoch"], reverse=True)
        
        # Remove old checkpoints
        if len(regular_checkpoints) > self.max_checkpoints:
            for checkpoint_info in regular_checkpoints[self.max_checkpoints:]:
                checkpoint_path = Path(checkpoint_info["path"])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    self.logger.info(f"Removed old checkpoint: {checkpoint_path}")
                
                # Remove from history
                self.checkpoint_history = [
                    info for info in self.checkpoint_history 
                    if info["path"] != checkpoint_info["path"]
                ]
    
    def get_checkpoint_summary(self) -> Dict:
        """Get summary of all checkpoints."""
        
        summary = {
            "total_checkpoints": len(self.checkpoint_history),
            "best_checkpoints": [info for info in self.checkpoint_history if info["is_best"]],
            "latest_checkpoint": self.checkpoint_history[-1] if self.checkpoint_history else None,
            "checkpoint_dir": str(self.checkpoint_dir),
            "disk_usage_mb": sum(
                Path(info["path"]).stat().st_size 
                for info in self.checkpoint_history 
                if Path(info["path"]).exists()
            ) / (1024 * 1024)
        }
        
        return summary

def create_kaggle_checkpoint_manager(config: Dict) -> KaggleCheckpointManager:
    """Factory function to create checkpoint manager from config."""
    
    return KaggleCheckpointManager(
        checkpoint_dir=config.get("checkpoint_dir", "/kaggle/working/checkpoints"),
        max_checkpoints=config.get("keep_last_n_checkpoints", 3),
        save_optimizer=config.get("save_optimizer_state", True),
        save_scheduler=config.get("save_scheduler_state", True)
    )
