#!/usr/bin/env python
"""
Script to create a new competition workspace structure.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

def create_competition_workspace(competition_name):
    """
    Create a new competition workspace.
    
    Parameters:
    -----------
    competition_name : str
        Name of the competition (will be used as the folder name)
    """
    # Define paths
    base_dir = Path(__file__).parent.parent
    competition_dir = os.path.join(base_dir, 'competitions', competition_name)
    dataset_dir = os.path.join(base_dir, 'datasets', competition_name)
    notebook_dir = os.path.join(base_dir, 'notebooks', competition_name)
    submission_dir = os.path.join(base_dir, 'submissions', competition_name)
    model_dir = os.path.join(base_dir, 'models', competition_name)
    
    # Create directories
    for directory in [competition_dir, dataset_dir, notebook_dir, submission_dir, model_dir]:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Copy notebook template
    template_source = os.path.join(base_dir, 'notebooks', 'templates', 'competition_template.ipynb')
    template_dest = os.path.join(notebook_dir, f"{competition_name}_notebook.ipynb")
    
    if os.path.exists(template_source):
        shutil.copy(template_source, template_dest)
        print(f"Created notebook: {template_dest}")
    else:
        print("Warning: Notebook template not found. Please create notebooks manually.")
    
    # Create empty README for the competition
    readme_path = os.path.join(competition_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(f"# {competition_name}\n\n")
        f.write("## Description\n\n")
        f.write("Add competition description here.\n\n")
        f.write("## Data\n\n")
        f.write("Describe the data here.\n\n")
        f.write("## Approach\n\n")
        f.write("Describe your approach here.\n\n")
        f.write("## Results\n\n")
        f.write("Document your results here.\n")
    
    print(f"Created README: {readme_path}")
    
    # Update config.py
    config_path = os.path.join(base_dir, 'utils', 'config.py')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Replace competition name in config
        config_content = config_content.replace('CURRENT_COMPETITION = "your_competition_name"', 
                                               f'CURRENT_COMPETITION = "{competition_name}"')
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"Updated config.py with competition name: {competition_name}")
    
    print(f"\nCompetition workspace for '{competition_name}' created successfully!")
    print(f"\nNext steps:")
    print(f"1. Download competition data to the '{dataset_dir}' directory")
    print(f"2. Open the notebook at '{template_dest}'")
    print(f"3. Start exploring the data and building models")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a new competition workspace')
    parser.add_argument('competition_name', type=str, help='Name of the competition')
    
    args = parser.parse_args()
    
    create_competition_workspace(args.competition_name) 