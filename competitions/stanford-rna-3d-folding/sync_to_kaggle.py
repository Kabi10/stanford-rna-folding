"""
Script to sync notebooks with Kaggle.
"""

import os
import json
import subprocess
from pathlib import Path

def create_kernel_metadata(notebook_path, competition="stanford-rna-3d-folding"):
    """Create metadata for Kaggle kernel."""
    notebook_name = Path(notebook_path).stem
    kernel_title = notebook_name.replace('_', ' ').title()
    
    metadata = {
        "id": f"kabitharma/{notebook_name}",  # Replace with your Kaggle username
        "title": kernel_title,
        "code_file": os.path.basename(notebook_path),
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "competition_sources": [competition],
        "kernel_sources": []
    }
    
    return metadata

def push_to_kaggle(notebook_path):
    """Push notebook to Kaggle."""
    # Create temporary directory
    temp_dir = Path(notebook_path).parent / "kaggle_temp"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Copy notebook
        import shutil
        notebook_name = Path(notebook_path).name
        temp_notebook = temp_dir / notebook_name
        shutil.copy2(notebook_path, temp_notebook)
        
        # Create metadata
        metadata = create_kernel_metadata(notebook_path)
        metadata_path = temp_dir / "kernel-metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Push to Kaggle
        result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", str(temp_dir)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Successfully pushed to Kaggle!")
            print(result.stdout)
        else:
            print("❌ Failed to push to Kaggle")
            print(f"Error: {result.stderr}")
            
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    notebook_dir = Path(__file__).parent / "notebooks"
    
    # Push all notebooks
    for notebook in notebook_dir.glob("*.ipynb"):
        print(f"\nPushing {notebook.name} to Kaggle...")
        push_to_kaggle(notebook) 