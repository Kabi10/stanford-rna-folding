# Additional cells for the Kaggle notebook
# These can be added manually to the notebook or run as separate scripts

# Cell: Results Analysis and Visualization
def analyze_training_results():
    """Analyze and visualize training results."""
    import json
    from pathlib import Path
    
    results_dir = Path('/kaggle/working/results')
    
    if results_dir.exists():
        # Load training results
        results_file = results_dir / 'training_results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                training_results = json.load(f)
            
            print('Final Training Results:')
            print(f'  Best RMSD: {training_results["best_rmsd"]:.4f}')
            print(f'  Best TM-score: {training_results["best_tm_score"]:.4f}')
            print(f'  Device used: {training_results["device_used"]}')
            print(f'  Training completed: {training_results["training_completed"]}')
        
        # List all result files
        print('\nGenerated files:')
        for file_path in results_dir.rglob('*'):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f'  {file_path.name}: {size_mb:.2f} MB')
    else:
        print('No results directory found')

# Cell: Performance Summary
def create_performance_summary():
    """Create comprehensive performance summary."""
    import json
    from pathlib import Path
    
    summary = {
        'gpu_info': gpu_config if 'gpu_config' in globals() else {},
        'training_config': training_config if 'training_config' in globals() else {},
        'dataset_info': dataset_analysis if 'dataset_analysis' in globals() else {},
        'training_duration_hours': training_duration / 3600 if 'training_duration' in globals() else None,
        'results': results if 'results' in globals() else None
    }
    
    # Save summary
    summary_path = Path('/kaggle/working/performance_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f'Performance summary saved to: {summary_path}')
    return summary

# Cell: Model Export
def export_trained_model():
    """Export trained model for submission or further use."""
    import torch
    import json
    from pathlib import Path
    
    if 'model' in globals() and model is not None:
        print('Exporting trained model...')
        
        # Create export directory
        export_dir = Path('/kaggle/working/model_export')
        export_dir.mkdir(exist_ok=True)
        
        # Export model state dict
        model_path = export_dir / 'rna_folding_model.pth'
        torch.save(model.state_dict(), model_path)
        print(f'Model state dict saved: {model_path}')
        
        # Export full model
        full_model_path = export_dir / 'rna_folding_model_full.pth'
        torch.save(model, full_model_path)
        print(f'Full model saved: {full_model_path}')
        
        # Export configuration
        config_path = export_dir / 'model_config.json'
        with open(config_path, 'w') as f:
            json.dump(training_config if 'training_config' in globals() else {}, f, indent=2)
        print(f'Configuration saved: {config_path}')
        
        # Model info
        model_info = {
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'model_size_mb': model_path.stat().st_size / (1024 * 1024),
            'architecture': 'Physics-Enhanced Transformer',
            'prediction_mode': 'Single-atom coordinates (x_1, y_1, z_1)',
            'constraints': ['bond_length', 'bond_angle', 'steric_clash', 'watson_crick']
        }
        
        info_path = export_dir / 'model_info.json'
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f'Model info saved: {info_path}')
        
        print(f'\nModel Export Summary:')
        print(f'  Parameters: {model_info["num_parameters"]:,}')
        print(f'  Model size: {model_info["model_size_mb"]:.2f} MB')
        print(f'  Export directory: {export_dir}')
    else:
        print('No trained model available for export')

# Cell: Resource Usage Analysis
def calculate_resource_usage():
    """Calculate Kaggle resource usage and remaining quota."""
    if 'training_duration' in globals():
        hours_used = training_duration / 3600
        remaining_gpu_hours = 30 - hours_used  # Kaggle GPU quota
        
        print('=== Kaggle Resource Usage Summary ===')
        print(f'GPU time used: {hours_used:.2f} hours')
        print(f'Remaining GPU quota: {remaining_gpu_hours:.2f} hours')
        print(f'Quota utilization: {(hours_used/30)*100:.1f}%')
        
        # Estimate additional experiments possible
        if hours_used > 0:
            additional_runs = int(remaining_gpu_hours / hours_used)
            print(f'Additional full training runs possible: {additional_runs}')
        
        # Recommendations
        print('\nRecommendations:')
        if remaining_gpu_hours > 20:
            print('  - Plenty of quota remaining for hyperparameter tuning')
            print('  - Consider running 2-3 more experiments with different configs')
        elif remaining_gpu_hours > 10:
            print('  - Moderate quota remaining')
            print('  - Consider 1-2 more focused experiments')
        elif remaining_gpu_hours > 5:
            print('  - Limited quota remaining')
            print('  - Consider shorter experiments or model export only')
        else:
            print('  - Very limited quota remaining')
            print('  - Focus on model export and analysis')
    else:
        print('Training duration not available')

# Cell: Hyperparameter Tuning Suggestions
def suggest_hyperparameter_experiments():
    """Suggest follow-up experiments based on results."""
    print('=== Suggested Follow-up Experiments ===')
    
    if 'results' in globals() and results is not None:
        best_rmsd = results.get('best_rmsd', float('inf'))
        best_tm = results.get('best_tm_score', 0.0)
        
        print(f'Current best RMSD: {best_rmsd:.4f}')
        print(f'Current best TM-score: {best_tm:.4f}')
        
        # Suggest experiments based on performance
        if best_rmsd > 8.0:
            print('\nHigh RMSD - Suggested improvements:')
            print('  1. Increase model size (hidden_dim: 768, num_layers: 8)')
            print('  2. Lower learning rate (2e-4)')
            print('  3. Increase physics constraint weights')
            print('  4. Add learning rate warmup')
        elif best_rmsd > 5.0:
            print('\nModerate RMSD - Suggested refinements:')
            print('  1. Fine-tune physics constraint weights')
            print('  2. Try different optimizers (AdamW)')
            print('  3. Experiment with dropout rates')
            print('  4. Enable relative attention (after fixing)')
        else:
            print('\nGood RMSD - Suggested optimizations:')
            print('  1. Ensemble multiple models')
            print('  2. Multi-atom prediction mode')
            print('  3. Advanced data augmentation')
            print('  4. Post-processing refinement')
        
        # Specific config suggestions
        print('\nSpecific configuration suggestions:')
        
        # Experiment 1: Larger model
        exp1_config = {
            'embedding_dim': 320,
            'hidden_dim': 640,
            'num_layers': 8,
            'learning_rate': 3e-4,
            'batch_size': 16,  # Reduced for larger model
            'epochs': 30
        }
        print(f'  Experiment 1 (Larger Model): {exp1_config}')
        
        # Experiment 2: Different physics weights
        exp2_config = {
            'bond_length_weight': 0.5,
            'bond_angle_weight': 0.4,
            'steric_clash_weight': 0.8,
            'watson_crick_weight': 0.3,
            'learning_rate': 4e-4
        }
        print(f'  Experiment 2 (Tuned Physics): {exp2_config}')
        
    else:
        print('No results available for analysis')

# Cell: Competition Submission Preparation
def prepare_competition_submission():
    """Prepare files for competition submission."""
    from pathlib import Path
    import shutil
    
    submission_dir = Path('/kaggle/working/submission')
    submission_dir.mkdir(exist_ok=True)
    
    print('=== Preparing Competition Submission ===')
    
    # Copy best model
    model_export_dir = Path('/kaggle/working/model_export')
    if model_export_dir.exists():
        for file_path in model_export_dir.glob('*.pth'):
            shutil.copy2(file_path, submission_dir)
            print(f'Copied: {file_path.name}')
    
    # Create submission README
    readme_content = """# RNA 3D Folding Model Submission

## Model Description
- Architecture: Physics-Enhanced Transformer
- Parameters: ~4M
- Prediction: Single-atom coordinates (x_1, y_1, z_1)
- Constraints: Bond length, bond angle, steric clash, Watson-Crick

## Performance
- Best RMSD: {best_rmsd:.4f}
- Best TM-score: {best_tm:.4f}
- Training time: {training_time:.2f} hours

## Usage
```python
import torch
model = torch.load('rna_folding_model_full.pth')
model.eval()

# Predict coordinates
with torch.no_grad():
    coords = model(sequences, lengths)
```

## Files
- rna_folding_model_full.pth: Complete trained model
- rna_folding_model.pth: Model state dict only
- model_config.json: Training configuration
- model_info.json: Model metadata
"""
    
    # Fill in actual values if available
    if 'results' in globals() and results is not None:
        readme_content = readme_content.format(
            best_rmsd=results.get('best_rmsd', 0.0),
            best_tm=results.get('best_tm_score', 0.0),
            training_time=training_duration/3600 if 'training_duration' in globals() else 0.0
        )
    
    with open(submission_dir / 'README.md', 'w') as f:
        f.write(readme_content)
    
    print(f'Submission prepared in: {submission_dir}')
    print('Files ready for download or competition submission')

# Main execution functions
if __name__ == "__main__":
    print("Additional notebook cells loaded. Available functions:")
    print("  - analyze_training_results()")
    print("  - create_performance_summary()")
    print("  - export_trained_model()")
    print("  - calculate_resource_usage()")
    print("  - suggest_hyperparameter_experiments()")
    print("  - prepare_competition_submission()")
