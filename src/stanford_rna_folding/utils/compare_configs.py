"""
Utility for comparing different RNA folding model configurations.

This script provides tools to compare different model configurations,
benchmark their performance, and visualize the results.
"""

import os
import sys
import argparse
import logging
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader

# Local imports
from stanford_rna_folding.models.model_adapter import ModelAdapter
from stanford_rna_folding.training.profiling import (
    measure_memory_usage,
    measure_throughput,
    profile_model,
    compare_model_configurations,
)
from stanford_rna_folding.data.dataset import RNAFoldingDataset
from stanford_rna_folding.utils.logger import setup_logging


logger = logging.getLogger(__name__)


def load_configs(config_paths: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Load multiple configuration files.
    
    Args:
        config_paths: List of paths to configuration files
        
    Returns:
        Dictionary mapping configuration names to configuration dictionaries
    """
    configs = {}
    
    for path in config_paths:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
            
        name = config.get('experiment_name', Path(path).stem)
        configs[name] = config
    
    return configs


def create_sample_dataloader(config: Dict[str, Any], num_samples: int = 5) -> DataLoader:
    """
    Create a small dataloader for benchmarking.
    
    Args:
        config: Configuration dictionary
        num_samples: Number of samples to include
        
    Returns:
        DataLoader with a small number of samples
    """
    data_dir = Path(config.get('data_dir', 'datasets/stanford-rna-3d-folding'))
    batch_size = config.get('batch_size', 16)
    
    # Create dataset
    dataset = RNAFoldingDataset(
        data_dir=data_dir / 'train',
        normalize_coords=config.get('normalize_coords', True),
        split='train',
    )
    
    # Create subset
    subset = torch.utils.data.Subset(dataset, list(range(min(num_samples, len(dataset)))))
    
    # Create dataloader
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Use single process for benchmarking
        pin_memory=True,
    )
    
    return dataloader


def format_metrics(metrics: Dict[str, float]) -> Dict[str, str]:
    """
    Format metrics for display.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Dictionary of formatted metrics
    """
    formatted = {}
    
    for key, value in metrics.items():
        if 'time' in key and 'ms' in key:
            formatted[key] = f"{value:.2f} ms"
        elif 'memory' in key or 'gb' in key.lower():
            formatted[key] = f"{value:.2f} GB"
        elif 'per_second' in key:
            formatted[key] = f"{value:.2f}/s"
        elif 'param_count' in key:
            if value > 1e6:
                formatted[key] = f"{value/1e6:.2f}M"
            elif value > 1e3:
                formatted[key] = f"{value/1e3:.2f}K"
            else:
                formatted[key] = f"{value:.0f}"
        else:
            formatted[key] = f"{value:.4f}"
    
    return formatted


def benchmark_configs(
    configs: Dict[str, Dict[str, Any]],
    model_class,
    output_dir: Path,
    device: torch.device,
    num_samples: int = 5,
    num_batches: int = 10,
    use_mixed_precision: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark multiple configurations.
    
    Args:
        configs: Dictionary mapping configuration names to configuration dictionaries
        model_class: Model class to instantiate
        output_dir: Directory to save benchmark results
        device: Device to use for benchmarking
        num_samples: Number of samples to use for benchmarking
        num_batches: Number of batches to measure for throughput
        use_mixed_precision: Whether to use mixed precision
        
    Returns:
        Dictionary mapping configuration names to benchmark results
    """
    results = {}
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample dataloader (use first config as reference)
    first_config = next(iter(configs.values()))
    dataloader = create_sample_dataloader(first_config, num_samples)
    
    # Benchmark each configuration
    for name, config in configs.items():
        logger.info(f"Benchmarking configuration: {name}")
        
        # Create model
        model = ModelAdapter.create_model_from_config(config)
        model = model.to(device)
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)  # MB
        
        # Measure memory before batch
        pre_mem = measure_memory_usage()
        
        # Measure model throughput
        throughput = measure_throughput(
            model=model,
            dataloader=dataloader,
            device=device,
            num_batches=num_batches,
            use_autocast=use_mixed_precision,
        )
        
        # Measure memory after batch
        post_mem = measure_memory_usage()
        
        # Combine results
        metrics = {
            'param_count': param_count,
            'param_count_millions': param_count / 1e6,
            'param_size_mb': param_size,
            **throughput,
            'gpu_memory_usage_gb': post_mem['gpu_memory_gb'],
            'memory_increase_gb': post_mem['gpu_memory_gb'] - pre_mem['gpu_memory_gb'],
        }
        
        results[name] = {
            'raw_metrics': metrics,
            'formatted_metrics': format_metrics(metrics),
            'config': config,
        }
        
        # Log results
        logger.info(f"Results for {name}:")
        for k, v in results[name]['formatted_metrics'].items():
            logger.info(f"  {k}: {v}")
    
    # Save results to file
    results_file = output_dir / 'benchmark_results.json'
    with open(results_file, 'w') as f:
        json.dump({name: result['formatted_metrics'] for name, result in results.items()}, f, indent=2)
    
    # Create comparison table
    comparison_df = pd.DataFrame({
        name: result['formatted_metrics'] 
        for name, result in results.items()
    })
    
    # Save comparison table
    comparison_file = output_dir / 'benchmark_comparison.csv'
    comparison_df.to_csv(comparison_file)
    
    return results


def plot_comparison(
    results: Dict[str, Dict[str, Any]],
    output_dir: Path,
    metrics_to_plot: List[str] = None,
) -> None:
    """
    Plot comparison of benchmark results.
    
    Args:
        results: Dictionary mapping configuration names to benchmark results
        output_dir: Directory to save plots
        metrics_to_plot: List of metrics to plot (default: key performance metrics)
    """
    if metrics_to_plot is None:
        metrics_to_plot = [
            'samples_per_second',
            'gpu_memory_usage_gb',
            'param_count_millions',
            'avg_batch_time_ms',
        ]
    
    # Create figure directory
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)
    
    # Extract raw metrics for plotting
    plot_data = {
        name: {
            metric: result['raw_metrics'].get(metric, 0)
            for metric in metrics_to_plot
            if metric in result['raw_metrics']
        }
        for name, result in results.items()
    }
    
    # Create bar charts for each metric
    for metric in metrics_to_plot:
        if not any(metric in result['raw_metrics'] for result in results.values()):
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Get values for this metric
        values = [result['raw_metrics'].get(metric, 0) for result in results.values()]
        names = list(results.keys())
        
        # Create bar chart
        bars = plt.bar(names, values)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height * 1.01,
                f"{height:.2f}",
                ha='center',
                va='bottom',
                rotation=0,
            )
        
        # Format plot
        metric_name = metric.replace('_', ' ').title()
        plt.title(f'Comparison of {metric_name}')
        plt.ylabel(metric_name)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(fig_dir / f"{metric}_comparison.png", dpi=300)
        plt.close()
    
    # Create radar chart for overall comparison
    metrics = [
        'samples_per_second',
        'param_count_millions',
        'memory_increase_gb',
    ]
    
    # Normalize metrics to 0-1 scale for radar chart
    normalized_data = {}
    for metric in metrics:
        if not any(metric in result['raw_metrics'] for result in results.values()):
            continue
            
        values = [result['raw_metrics'].get(metric, 0) for result in results.values()]
        min_val = min(values)
        max_val = max(values)
        
        # Handle case where all values are the same
        if max_val == min_val:
            normalized = [1.0 for _ in values]
        else:
            # Normalize, with direction based on whether higher is better
            if metric in ['samples_per_second']:
                # Higher is better
                normalized = [(val - min_val) / (max_val - min_val) for val in values]
            else:
                # Lower is better
                normalized = [1 - (val - min_val) / (max_val - min_val) for val in values]
                
        normalized_data[metric] = dict(zip(results.keys(), normalized))
    
    # Create radar chart if we have at least 3 metrics
    if len(normalized_data) >= 3:
        # Radar chart
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Set up the radar chart
        metrics_for_radar = list(normalized_data.keys())
        num_metrics = len(metrics_for_radar)
        angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Plot each configuration
        for name in results.keys():
            values = [normalized_data[metric][name] for metric in metrics_for_radar]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, label=name)
            ax.fill(angles, values, alpha=0.1)
        
        # Set up chart appearance
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_for_radar])
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.set_rlabel_position(0)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title('Configuration Comparison (Higher is Better)')
        plt.tight_layout()
        
        # Save radar chart
        plt.savefig(fig_dir / 'radar_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Compare RNA folding model configurations')
    parser.add_argument('--configs', nargs='+', required=True, help='Paths to configuration YAML files')
    parser.add_argument('--output-dir', type=str, default='benchmark_results', help='Directory to save results')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples for benchmarking')
    parser.add_argument('--num-batches', type=int, default=20, help='Number of batches for throughput measurement')
    parser.add_argument('--no-mixed-precision', action='store_true', help='Disable mixed precision')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Load configurations
    configs = load_configs(args.configs)
    logger.info(f"Loaded {len(configs)} configurations: {', '.join(configs.keys())}")
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    
    # Benchmark configurations
    results = benchmark_configs(
        configs=configs,
        model_class=None,  # Not needed since we use ModelAdapter
        output_dir=output_dir,
        device=device,
        num_samples=args.num_samples,
        num_batches=args.num_batches,
        use_mixed_precision=not args.no_mixed_precision,
    )
    
    # Plot comparison
    plot_comparison(results, output_dir)
    
    logger.info(f"Benchmarking completed. Results saved to {output_dir}")


if __name__ == '__main__':
    main()