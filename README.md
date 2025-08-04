# Stanford RNA 3D Structure Prediction

This repository contains models and utilities for predicting the 3D structure of RNA molecules using deep learning with specialized attention mechanisms.

## Overview

RNA structure prediction is crucial for understanding RNA function and designing RNA-based therapeutics. This project implements transformers with specialized attention mechanisms tailored for RNA structure prediction:

1. **Hierarchical Attention** - Models RNA at multiple organizational levels:
   - Primary structure (sequence)
   - Secondary structure (base pairs, stems, loops)
   - Tertiary structure (global 3D architecture)

2. **Distance-Modulated Attention** - Uses a geometry-aware attention mechanism:
   - Scales attention based on physical distances between nucleotides
   - Iteratively refines predictions through multiple steps
   - Supports different distance scaling methods (inverse, Gaussian, learned)

3. **Memory-Optimized Models** - Enables training on longer RNA sequences:
   - Gradient checkpointing to reduce memory footprint
   - Mixed precision training
   - Gradient accumulation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stanford-rna-3d-folding.git
cd stanford-rna-3d-folding

# Install dependencies
pip install -r requirements.txt
```

## Dataset

The RNA 3D structure dataset should be organized as follows:

```
datasets/
  stanford-rna-3d-folding/
    train/
    validation/
    test/
```

Each split should contain RNA sequences and their corresponding 3D coordinates.

## Model Configurations

Pre-defined configuration files are available in the `configs/` directory:

- `hierarchical_attention_config.yaml` - Configuration for the hierarchical attention model
- `distance_modulated_config.yaml` - Configuration for the distance-modulated attention model
- `memory_optimized_config.yaml` - Configuration for the memory-optimized model

## Usage

### Training

To train a model with a specific configuration:

```bash
python -m stanford_rna_folding.train --config configs/hierarchical_attention_config.yaml
```

Optional arguments:
- `--profile`: Profile model performance
- `--verbose`: Enable verbose logging

### Comparing Model Configurations

To benchmark and compare different model configurations:

```bash
python -m stanford_rna_folding.utils.compare_configs --configs configs/hierarchical_attention_config.yaml configs/distance_modulated_config.yaml configs/memory_optimized_config.yaml --output-dir benchmark_results
```

This will generate benchmark metrics and visualizations comparing the different models.

## Model Architecture

The base model uses a transformer-based architecture to predict RNA 3D coordinates. The specialized attention mechanisms enhance this architecture:

### Hierarchical Attention

This approach uses different attention heads to capture different levels of RNA organization:
- Primary heads focus on local sequence patterns
- Secondary heads emphasize base-pairing patterns using Watson-Crick and wobble pairing rules
- Tertiary heads capture global structural context

### Distance-Modulated Attention

This approach incorporates geometric awareness into the attention mechanism:
- Uses current coordinate predictions to modulate attention weights
- Attention weights are scaled based on predicted distances
- Iterative refinement steps improve structural accuracy

## Performance Tips

1. **Memory Optimization**:
   - Use gradient checkpointing for longer sequences
   - Enable mixed precision training
   - Use gradient accumulation for larger effective batch sizes

2. **Training Speed**:
   - Use the one-cycle learning rate schedule
   - Start with a smaller model and gradually increase size

3. **Model Selection**:
   - For shorter RNAs (<100 nucleotides), the distance-modulated model often performs best
   - For longer RNAs, the hierarchical attention model may capture more structural patterns
   - For very long sequences, use the memory-optimized configuration

## Citation

If you find this work useful, please cite:

```
@article{yourarticle,
  title={Specialized Attention Mechanisms for RNA 3D Structure Prediction},
  author={Your Name},
  journal={Journal Name},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 