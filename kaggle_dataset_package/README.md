# Stanford RNA 3D Folding Dataset with Physics-Enhanced Transformer

## Overview
This dataset contains the Stanford RNA 3D folding competition data along with a complete physics-enhanced transformer implementation for RNA structure prediction.

## Dataset Statistics

### train_sequences.csv
- Rows: 844
- Columns: 5
- Size: 2.92 MB
- Average sequence length: 162.4
- Max sequence length: 4298

### train_labels.csv
- Rows: 137,095
- Columns: 6
- Size: 9.34 MB
- Coordinate columns: 3
- NaN coordinates: 18,435

### validation_sequences.csv
- Rows: 12
- Columns: 5
- Size: 0.01 MB
- Average sequence length: 209.6
- Max sequence length: 720

### validation_labels.csv
- Rows: 2,515
- Columns: 123
- Size: 2.38 MB
- Coordinate columns: 120
- NaN coordinates: 0

### test_sequences.csv
- Rows: 12
- Columns: 5
- Size: 0.01 MB
- Average sequence length: 209.6
- Max sequence length: 720

## Model Architecture
- Physics-enhanced transformer with biophysical constraints
- Single-atom coordinate prediction (x_1, y_1, z_1)
- RMSD and TM-score evaluation metrics
- Bond length, bond angle, and steric clash constraints

## Usage
```python
# Load the training script
exec(open('/kaggle/input/stanford-rna-3d-folding/scripts/rna_folding_kaggle_train.py').read())
```

## Files Structure
- `data/`: RNA sequence and coordinate data
- `src/`: Complete source code for the model
- `configs/`: GPU-optimized training configurations
- `scripts/`: Kaggle-compatible training scripts

## Performance
- Expected 10-20x speedup on GPU vs CPU
- Optimized for V100/P100 instances
- Mixed precision training enabled
- Batch sizes: 12-48 depending on GPU memory

## Citation
Stanford RNA 3D Folding Competition
