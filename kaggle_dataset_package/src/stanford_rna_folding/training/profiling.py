"""
Profiling and benchmarking utilities for RNA folding model training.

This module provides functions to profile and benchmark the performance of
the RNA folding model training process, including memory usage, throughput,
and detailed profiling with PyTorch Profiler.
"""

import os
import time
from typing import Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import psutil


def measure_memory_usage() -> Dict[str, float]:
    """
    Measure current memory usage (CPU and GPU if available).
    
    Returns:
        Dictionary with memory usage statistics in GB
    """
    # CPU memory
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / (1024 ** 3)  # GB
    
    # GPU memory if available
    gpu_mem = 0.0
    gpu_mem_allocated = 0.0
    gpu_mem_reserved = 0.0
    
    if torch.cuda.is_available():
        gpu_mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        gpu_mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
        # Get max memory from torch.cuda.max_memory_allocated()
        gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
    
    return {
        'cpu_memory_gb': cpu_mem,
        'gpu_memory_gb': gpu_mem,
        'gpu_allocated_gb': gpu_mem_allocated,
        'gpu_reserved_gb': gpu_mem_reserved,
    }


def measure_throughput(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: int = 50,
    warmup_batches: int = 5,
    use_autocast: bool = False,
) -> Dict[str, float]:
    """
    Measure model throughput (samples per second, batches per second).
    
    Args:
        model: The model to benchmark
        dataloader: The dataloader to use for benchmarking
        device: The device to run the benchmark on
        num_batches: Number of batches to measure
        warmup_batches: Number of warmup batches (not included in measurement)
        use_autocast: Whether to use automatic mixed precision (FP16)
        
    Returns:
        Dictionary with throughput metrics
    """
    model.eval()  # Set model to evaluation mode
    
    # Stats to track
    total_samples = 0
    total_time = 0.0
    batch_times = []
    batch_sizes = []
    
    # Get a fresh dataloader iterator
    data_iter = iter(dataloader)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_batches):
            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart data iterator if we run out of data
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Forward pass with optional mixed precision
            if use_autocast and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    model(batch['sequence'], batch.get('lengths', None))
            else:
                model(batch['sequence'], batch.get('lengths', None))
    
    # Synchronize if using CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    with torch.no_grad():
        for i in range(num_batches):
            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart data iterator if we run out of data
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Track batch size
            batch_size = batch['sequence'].size(0)
            batch_sizes.append(batch_size)
            
            # Synchronize before timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Measure time
            start_time = time.time()
            
            # Forward pass with optional mixed precision
            if use_autocast and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    model(batch['sequence'], batch.get('lengths', None))
            else:
                model(batch['sequence'], batch.get('lengths', None))
            
            # Synchronize after forward pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Record time
            end_time = time.time()
            batch_time = end_time - start_time
            batch_times.append(batch_time)
            
            # Accumulate stats
            total_samples += batch_size
            total_time += batch_time
    
    # Calculate metrics
    samples_per_sec = total_samples / total_time
    batches_per_sec = num_batches / total_time
    avg_batch_time = np.mean(batch_times)
    std_batch_time = np.std(batch_times)
    
    # Per sample time
    per_sample_times = [t / s for t, s in zip(batch_times, batch_sizes)]
    avg_sample_time = np.mean(per_sample_times)
    std_sample_time = np.std(per_sample_times)
    
    return {
        'samples_per_second': samples_per_sec,
        'batches_per_second': batches_per_sec,
        'avg_batch_time_ms': avg_batch_time * 1000,
        'std_batch_time_ms': std_batch_time * 1000,
        'avg_sample_time_ms': avg_sample_time * 1000,
        'std_sample_time_ms': std_sample_time * 1000,
    }


def profile_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    profile_dir: Union[str, Path],
    num_batches: int = 10,
    warmup_batches: int = 3,
    use_autocast: bool = False,
    activities: Optional[List[ProfilerActivity]] = None,
) -> None:
    """
    Profile model using PyTorch Profiler.
    
    Args:
        model: The model to profile
        dataloader: The dataloader to use for profiling
        device: The device to run the profiling on
        profile_dir: Directory to save profiling results
        num_batches: Number of batches to profile
        warmup_batches: Number of warmup batches (not included in profiling)
        use_autocast: Whether to use automatic mixed precision (FP16)
        activities: List of profiler activities to track (default: CPU and CUDA)
    """
    # Create profile directory
    profile_dir = Path(profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    # Set default activities if not provided
    if activities is None:
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
    
    # Set model to evaluation mode for profiling
    model.eval()
    
    # Get a fresh dataloader iterator
    data_iter = iter(dataloader)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_batches):
            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart data iterator if we run out of data
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Forward pass with optional mixed precision
            if use_autocast and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    model(batch['sequence'], batch.get('lengths', None))
            else:
                model(batch['sequence'], batch.get('lengths', None))
    
    # Profile with torch.profiler
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir)),
    ) as prof:
        with torch.no_grad():
            for i in range(num_batches):
                with record_function(f"batch_{i}"):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        # Restart data iterator if we run out of data
                        data_iter = iter(dataloader)
                        batch = next(data_iter)
                    
                    # Move to device
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(device)
                    
                    # Forward pass with optional mixed precision
                    if use_autocast and torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            model(batch['sequence'], batch.get('lengths', None))
                    else:
                        model(batch['sequence'], batch.get('lengths', None))
                
                # Record profiling data
                prof.step()
    
    # Print some statistics
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    # Save profiling results as text summary
    with open(profile_dir / 'profile_summary.txt', 'w') as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))


def compare_model_configurations(
    model_configs: Dict[str, Dict],
    dataloader: DataLoader,
    device: torch.device,
    model_class: type,
    num_batches: int = 20,
    use_autocast: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Compare different model configurations for performance.
    
    Args:
        model_configs: Dictionary of model configurations to compare
            {config_name: {config_params}}
        dataloader: The dataloader to use for benchmarking
        device: The device to run the benchmark on
        model_class: The model class to instantiate
        num_batches: Number of batches to measure
        use_autocast: Whether to use automatic mixed precision (FP16)
        
    Returns:
        Dictionary with configuration names and their metrics
    """
    results = {}
    
    for name, config in model_configs.items():
        print(f"Benchmarking configuration: {name}")
        
        # Create model from configuration
        model = model_class(**config).to(device)
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        
        # Measure memory before batch
        pre_mem = measure_memory_usage()
        
        # Measure model throughput
        throughput = measure_throughput(
            model=model,
            dataloader=dataloader,
            device=device,
            num_batches=num_batches,
            use_autocast=use_autocast,
        )
        
        # Measure memory after batch
        post_mem = measure_memory_usage()
        
        # Combine results
        metrics = {
            'param_count': param_count,
            'param_count_millions': param_count / 1e6,
            **throughput,
            **post_mem,
            'memory_increase_gb': post_mem['gpu_memory_gb'] - pre_mem['gpu_memory_gb'],
        }
        
        results[name] = metrics
        
        print(f"Results for {name}:")
        print(f"  Parameters: {param_count / 1e6:.2f}M")
        print(f"  Throughput: {throughput['samples_per_second']:.2f} samples/sec")
        print(f"  GPU Memory: {post_mem['gpu_memory_gb']:.2f} GB")
        print(f"  Memory Increase: {metrics['memory_increase_gb']:.2f} GB")
        print("")
    
    return results 