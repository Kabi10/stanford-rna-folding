#!/usr/bin/env python
"""
Script to push local code to Kaggle as a notebook/kernel.
This script helps automate the process of updating your work on Kaggle.
"""

import os
import sys
import json
import argparse
import tempfile
import shutil
from pathlib import Path
import re
import datetime

def get_kaggle_username():
    """
    Get the Kaggle username from the kaggle.json credentials file.
    
    Returns:
    --------
    str or None
        Kaggle username if found, None otherwise
    """
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    
    if not os.path.exists(kaggle_json_path):
        print("Error: Kaggle credentials not found. Please set up your Kaggle API credentials first.")
        print("   See KAGGLE_SETUP.md for instructions.")
        return None
    
    try:
        with open(kaggle_json_path, 'r') as f:
            credentials = json.load(f)
            return credentials.get('username')
    except Exception as e:
        print(f"Error reading Kaggle credentials: {e}")
        return None

def create_metadata(username, title, code_file, competition=None, dataset=None, is_public=False):
    """
    Create kernel metadata dictionary.
    """
    # Create a slug-friendly version of the title
    slug = re.sub(r'[^a-zA-Z0-9]+', '-', title.lower()).strip('-')
    kernel_id = f"{username}/{slug}"
    
    # Create metadata dictionary
    metadata = {
        "id": kernel_id,
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": str(not is_public).lower(),
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": []
    }
    
    # Add competition source if provided
    if competition:
        metadata["competition_sources"] = [competition]
    
    # Add dataset source if provided
    if dataset:
        metadata["dataset_sources"] = [dataset]
    
    return metadata

def prepare_kaggle_push_dir(code_file, title=None, competition=None, dataset=None, is_private=True, config_file=None, resources=None):
    """
    Prepares a temporary directory with all necessary files for a Kaggle kernel push.
    
    Parameters:
    -----------
    code_file : str
        Path to the code file
    title : str, default=None
        Title of the kernel (if None, derives from filename)
    competition : str, default=None
        Competition to associate with the kernel
    dataset : str, default=None
        Dataset to associate with the kernel
    is_private : bool, default=True
        Whether the kernel should be private
    config_file : str, default=None
        Path to a configuration file to include with the kernel
    resources : str or list, default=None
        String containing space-separated resource paths, or a list of paths.
        
    Returns:
    --------
    str or None
        Path to the prepared temporary directory if successful, None otherwise.
    """
    # Check if file exists
    if not os.path.isfile(code_file):
        print(f"Error: File not found: {code_file}", file=sys.stderr)
        return None
    
    # Check if config file exists if provided
    if config_file and not os.path.isfile(config_file):
        print(f"Error: Config file not found: {config_file}", file=sys.stderr)
        return None
    
    # Get Kaggle username
    username = get_kaggle_username()
    if not username:
        return None # Error message already printed by get_kaggle_username
    
    # Auto-generate title if not provided
    if not title:
        title = os.path.splitext(os.path.basename(code_file))[0].replace('_', ' ').title()
    
    # Create a persistent temporary directory (caller is responsible for cleanup)
    try:
        # Use mkdtemp to create a directory that persists after the function returns
        temp_dir = tempfile.mkdtemp(prefix="kaggle_push_")
    except Exception as e:
        print(f"Error creating temporary directory: {e}", file=sys.stderr)
        return None

    try:
        # Copy the code file to the temp directory
        dest_file = os.path.join(temp_dir, os.path.basename(code_file))
        shutil.copy2(code_file, dest_file)
        
        # Copy config file if provided
        if config_file:
            # Copy to root of temp_dir
            config_dest = os.path.join(temp_dir, os.path.basename(config_file))
            shutil.copy2(config_file, config_dest)
            print(f"Prepared config file: {os.path.basename(config_file)}")
            
            # Also copy into a 'configs' subdir within temp_dir
            configs_dir = os.path.join(temp_dir, "configs")
            os.makedirs(configs_dir, exist_ok=True)
            config_in_configs_dir = os.path.join(configs_dir, os.path.basename(config_file))
            shutil.copy2(config_file, config_in_configs_dir)
        
        # Prepare list of resources
        resource_list = []
        if isinstance(resources, str):
            resource_list = resources.split() # Split the string by spaces
        elif isinstance(resources, list):
             resource_list = resources # Use the list directly

        # Copy additional resource files/directories if provided
        if resource_list:
            for resource in resource_list:
                resource_path = Path(resource)
                if not resource_path.exists():
                    print(f"Warning: Resource not found, skipping: {resource}")
                    continue
                
                dest_resource_path = Path(temp_dir) / resource_path.name
                if resource_path.is_file():
                    shutil.copy2(resource_path, dest_resource_path)
                    print(f"Prepared resource file: {resource_path.name}")
                elif resource_path.is_dir():
                    shutil.copytree(resource_path, dest_resource_path, dirs_exist_ok=True)
                    print(f"Prepared resource directory: {resource_path.name}")
        
        # Create metadata
        metadata = create_metadata(
            username=username,
            title=title,
            code_file=dest_file,
            competition=competition,
            dataset=dataset,
            is_public=not is_private
        )
        
        # Write metadata to file
        metadata_path = os.path.join(temp_dir, "kernel-metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Print kernel info for user
        print(f"Prepared files for Kaggle push:")
        print(f"  Kernel ID: {metadata['id']}")
        print(f"  Title: {title}")
        print(f"  Privacy: {'Private' if is_private else 'Public'}")
        if competition: print(f"  Competition: {competition}")
        if dataset: print(f"  Dataset: {dataset}")
        
        # Debug: print metadata content
        print("\nMetadata file content (kernel-metadata.json):")
        with open(metadata_path, 'r') as f:
            print(f.read())

        # Return the path to the prepared directory
        return temp_dir

    except Exception as e:
        print(f"Error preparing files in temporary directory {temp_dir}: {e}", file=sys.stderr)
        # Clean up the directory if preparation failed
        shutil.rmtree(temp_dir)
        return None

def main():
    parser = argparse.ArgumentParser(description='Prepare files for Kaggle push') # Updated description
    
    # Subparsers are no longer needed, just use the main parser for 'prepare'
    parser.add_argument('file', type=str, help='Path to the code file')
    parser.add_argument('--title', type=str, help='Title for the Kaggle kernel')
    parser.add_argument('--competition', type=str, help='Competition to associate with')
    parser.add_argument('--dataset', type=str, help='Dataset to associate with')
    parser.add_argument('--public', action='store_true', help='Make the kernel public')
    parser.add_argument('--config', type=str, help='Path to a configuration file to include')
    parser.add_argument('--resources', type=str, help='Space-separated list of additional resources to include')
    
    args = parser.parse_args()
    
    # Call the preparation function
    prepared_dir = prepare_kaggle_push_dir(
        code_file=args.file,
        title=args.title,
        competition=args.competition,
        dataset=args.dataset,
        is_private=not args.public,
        config_file=args.config,
        resources=args.resources
    )
    
    if prepared_dir:
        # Print the path of the prepared directory to stdout for the calling script
        print(prepared_dir)
        sys.exit(0) # Exit with success code
    else:
        sys.exit(1) # Exit with failure code

if __name__ == "__main__":
    main()