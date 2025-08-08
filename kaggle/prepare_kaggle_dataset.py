#!/usr/bin/env python3
"""
Prepare Stanford RNA 3D folding dataset for Kaggle private dataset upload.
This script packages the dataset and source code for Kaggle environment.
"""

import os
import shutil
import zipfile
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

def create_kaggle_dataset_structure(source_dir: Path, output_dir: Path):
    """Create Kaggle-compatible dataset structure."""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset structure
    dataset_structure = {
        "data": output_dir / "data",
        "src": output_dir / "src", 
        "configs": output_dir / "configs",
        "scripts": output_dir / "scripts"
    }
    
    for dir_path in dataset_structure.values():
        dir_path.mkdir(exist_ok=True)
    
    return dataset_structure

def copy_source_code(source_dir: Path, target_dir: Path):
    """Copy source code to Kaggle dataset."""
    
    # Copy main source directory
    src_source = source_dir / "src"
    if src_source.exists():
        shutil.copytree(src_source, target_dir / "src", dirs_exist_ok=True)
    
    # Copy essential files
    essential_files = [
        "requirements.txt",
        "setup.py", 
        "README.md",
        "README_RNA_FOLDING.md"
    ]
    
    for file_name in essential_files:
        file_path = source_dir / file_name
        if file_path.exists():
            shutil.copy2(file_path, target_dir / file_name)

def prepare_rna_dataset(source_data_dir: Path, target_data_dir: Path):
    """Prepare RNA dataset files for Kaggle."""
    
    if not source_data_dir.exists():
        raise FileNotFoundError(f"Source data directory not found: {source_data_dir}")
    
    # Copy dataset files
    dataset_files = [
        "train_sequences.csv",
        "train_labels.csv", 
        "validation_sequences.csv",
        "validation_labels.csv",
        "test_sequences.csv"
    ]
    
    copied_files = []
    for file_name in dataset_files:
        source_file = source_data_dir / file_name
        if source_file.exists():
            target_file = target_data_dir / file_name
            shutil.copy2(source_file, target_file)
            copied_files.append(file_name)
            print(f"Copied: {file_name}")
        else:
            print(f"Warning: {file_name} not found in source")
    
    # Verify and analyze dataset
    analyze_dataset(target_data_dir, copied_files)
    
    return copied_files

def analyze_dataset(data_dir: Path, files: List[str]):
    """Analyze dataset and create summary."""
    
    analysis = {
        "files": files,
        "statistics": {}
    }
    
    for file_name in files:
        file_path = data_dir / file_name
        if file_path.exists() and file_name.endswith('.csv'):
            try:
                df = pd.read_csv(file_path)
                analysis["statistics"][file_name] = {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "size_mb": file_path.stat().st_size / (1024 * 1024)
                }
                
                # Special analysis for sequences
                if "sequence" in df.columns:
                    sequences = df["sequence"].dropna()
                    analysis["statistics"][file_name].update({
                        "avg_sequence_length": sequences.str.len().mean(),
                        "max_sequence_length": sequences.str.len().max(),
                        "min_sequence_length": sequences.str.len().min(),
                        "unique_sequences": len(sequences.unique())
                    })
                
                # Special analysis for coordinates
                coord_cols = [col for col in df.columns if col.startswith(('x_', 'y_', 'z_'))]
                if coord_cols:
                    analysis["statistics"][file_name]["coordinate_columns"] = coord_cols
                    analysis["statistics"][file_name]["nan_coordinates"] = df[coord_cols].isna().sum().sum()
                
            except Exception as e:
                analysis["statistics"][file_name] = {"error": str(e)}
    
    # Save analysis
    with open(data_dir / "dataset_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"Dataset analysis saved to: {data_dir / 'dataset_analysis.json'}")
    return analysis

def create_kaggle_metadata(output_dir: Path, dataset_info: Dict):
    """Create Kaggle dataset metadata."""
    
    metadata = {
        "title": "Stanford RNA 3D Folding Dataset with Physics-Enhanced Transformer",
        "id": "your-username/stanford-rna-3d-folding",
        "licenses": [{"name": "MIT"}],
        "keywords": ["rna", "protein-folding", "bioinformatics", "deep-learning", "transformer"],
        "collaborators": [],
        "data": []
    }
    
    # Add data files to metadata
    for file_name in dataset_info.get("files", []):
        file_path = output_dir / "data" / file_name
        if file_path.exists():
            metadata["data"].append({
                "description": f"RNA dataset file: {file_name}",
                "name": file_name,
                "totalBytes": file_path.stat().st_size,
                "columns": []
            })
    
    # Save metadata
    with open(output_dir / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

def create_kaggle_readme(output_dir: Path, analysis: Dict):
    """Create README for Kaggle dataset."""
    
    readme_content = f"""# Stanford RNA 3D Folding Dataset with Physics-Enhanced Transformer

## Overview
This dataset contains the Stanford RNA 3D folding competition data along with a complete physics-enhanced transformer implementation for RNA structure prediction.

## Dataset Statistics
"""
    
    for file_name, stats in analysis.get("statistics", {}).items():
        if isinstance(stats, dict) and "rows" in stats:
            readme_content += f"""
### {file_name}
- Rows: {stats['rows']:,}
- Columns: {len(stats['columns'])}
- Size: {stats['size_mb']:.2f} MB
"""
            if "avg_sequence_length" in stats:
                readme_content += f"- Average sequence length: {stats['avg_sequence_length']:.1f}\n"
                readme_content += f"- Max sequence length: {stats['max_sequence_length']}\n"
            
            if "coordinate_columns" in stats:
                readme_content += f"- Coordinate columns: {len(stats['coordinate_columns'])}\n"
                readme_content += f"- NaN coordinates: {stats['nan_coordinates']:,}\n"

    readme_content += """
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
"""
    
    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)

def main():
    """Main function to prepare Kaggle dataset."""
    
    # Paths
    project_root = Path(__file__).resolve().parents[1]
    source_data_dir = project_root / "datasets" / "stanford-rna-3d-folding"
    output_dir = Path("kaggle_dataset_package")
    
    print(f"Project root: {project_root}")
    print(f"Source data: {source_data_dir}")
    print(f"Output: {output_dir}")
    
    # Create dataset structure
    structure = create_kaggle_dataset_structure(project_root, output_dir)
    
    # Copy source code
    print("Copying source code...")
    copy_source_code(project_root, output_dir)
    
    # Copy Kaggle-specific files
    kaggle_dir = project_root / "kaggle"
    if kaggle_dir.exists():
        shutil.copytree(kaggle_dir / "configs", structure["configs"], dirs_exist_ok=True)
        shutil.copytree(kaggle_dir, structure["scripts"], dirs_exist_ok=True)
    
    # Prepare dataset
    print("Preparing RNA dataset...")
    try:
        copied_files = prepare_rna_dataset(source_data_dir, structure["data"])
        analysis = analyze_dataset(structure["data"], copied_files)
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Creating placeholder dataset info...")
        copied_files = []
        analysis = {"files": [], "statistics": {}}
    
    # Create Kaggle metadata
    print("Creating Kaggle metadata...")
    dataset_info = {"files": copied_files}
    create_kaggle_metadata(output_dir, dataset_info)
    create_kaggle_readme(output_dir, analysis)
    
    # Create zip file for upload
    zip_path = output_dir.parent / f"{output_dir.name}.zip"
    print(f"Creating zip file: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in output_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(output_dir)
                zipf.write(file_path, arcname)
    
    print(f"\n=== Kaggle Dataset Package Created ===")
    print(f"Directory: {output_dir}")
    print(f"Zip file: {zip_path}")
    print(f"Size: {zip_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"\nNext steps:")
    print(f"1. Upload {zip_path} to Kaggle as a private dataset")
    print(f"2. Name it 'stanford-rna-3d-folding'")
    print(f"3. Use the training notebook with this dataset")

if __name__ == "__main__":
    main()
