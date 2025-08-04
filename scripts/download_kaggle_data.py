#!/usr/bin/env python
"""
Script to download Kaggle competition data.
Requires kaggle API credentials to be set up.
"""

import os
import argparse
import subprocess
from pathlib import Path

def download_kaggle_competition(competition_name, destination=None):
    """
    Download Kaggle competition data.
    
    Parameters:
    -----------
    competition_name : str
        Name of the competition on Kaggle
    destination : str, optional
        Directory to download data to. If None, uses datasets/<competition_name>
    """
    # Ensure Kaggle API credentials are set up
    try:
        import kaggle
    except ImportError:
        print("Kaggle API package not installed. Installing...")
        subprocess.check_call(["pip", "install", "kaggle"])
        
    # Check for Kaggle API credentials
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if not os.path.exists(kaggle_json):
        print("Kaggle API credentials not found.")
        print("Please follow these steps to set up your Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll down to 'API' section and click 'Create New API Token'")
        print("3. This will download a kaggle.json file")
        print(f"4. Place the file in {kaggle_dir} directory (create it if it doesn't exist)")
        print("5. On Unix systems, run: chmod 600 ~/.kaggle/kaggle.json")
        return
    
    # Set up destination path
    base_dir = Path(__file__).parent.parent
    
    if destination is None:
        destination = os.path.join(base_dir, "datasets", competition_name)
    
    # Create destination directory if it doesn't exist
    os.makedirs(destination, exist_ok=True)
    
    print(f"Downloading {competition_name} data to {destination}...")
    
    # Use Kaggle API to download competition data
    try:
        cmd = ["kaggle", "competitions", "download", "-c", competition_name, "-p", destination]
        subprocess.check_call(cmd)
        
        # Unzip the data
        zip_file = os.path.join(destination, f"{competition_name}.zip")
        if os.path.exists(zip_file):
            import zipfile
            print(f"Extracting {zip_file}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(destination)
            
            # Remove the zip file after extraction
            os.remove(zip_file)
            print("Extraction complete. Removed zip file.")
        
        print(f"Successfully downloaded {competition_name} data to {destination}")
        
        # Update config.py with competition name
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
        
    except subprocess.CalledProcessError as e:
        print(f"Error downloading competition data: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def download_kaggle_dataset(dataset_name, destination=None):
    """
    Download Kaggle dataset.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset on Kaggle (format: 'owner/dataset-name')
    destination : str, optional
        Directory to download data to. If None, uses datasets/<dataset_name>
    """
    # Ensure Kaggle API credentials are set up
    try:
        import kaggle
    except ImportError:
        print("Kaggle API package not installed. Installing...")
        subprocess.check_call(["pip", "install", "kaggle"])
    
    # Check for Kaggle API credentials
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if not os.path.exists(kaggle_json):
        print("Kaggle API credentials not found.")
        print("Please follow these steps to set up your Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll down to 'API' section and click 'Create New API Token'")
        print("3. This will download a kaggle.json file")
        print(f"4. Place the file in {kaggle_dir} directory (create it if it doesn't exist)")
        print("5. On Unix systems, run: chmod 600 ~/.kaggle/kaggle.json")
        return
    
    # Set up destination path
    base_dir = Path(__file__).parent.parent
    
    if destination is None:
        # Extract dataset name without owner
        if '/' in dataset_name:
            dataset_folder = dataset_name.split('/')[1]
        else:
            dataset_folder = dataset_name
        
        destination = os.path.join(base_dir, "datasets", dataset_folder)
    
    # Create destination directory if it doesn't exist
    os.makedirs(destination, exist_ok=True)
    
    print(f"Downloading {dataset_name} to {destination}...")
    
    # Use Kaggle API to download dataset
    try:
        cmd = ["kaggle", "datasets", "download", "--unzip", "-d", dataset_name, "-p", destination]
        subprocess.check_call(cmd)
        
        print(f"Successfully downloaded {dataset_name} to {destination}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download Kaggle competition data or datasets')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Competition subparser
    competition_parser = subparsers.add_parser('competition', help='Download competition data')
    competition_parser.add_argument('competition_name', type=str, help='Name of the competition on Kaggle')
    competition_parser.add_argument('--dest', type=str, help='Destination directory', default=None)
    
    # Dataset subparser
    dataset_parser = subparsers.add_parser('dataset', help='Download dataset')
    dataset_parser.add_argument('dataset_name', type=str, help='Name of the dataset on Kaggle (format: owner/dataset-name)')
    dataset_parser.add_argument('--dest', type=str, help='Destination directory', default=None)
    
    args = parser.parse_args()
    
    if args.command == 'competition':
        download_kaggle_competition(args.competition_name, args.dest)
    elif args.command == 'dataset':
        download_kaggle_dataset(args.dataset_name, args.dest)
    else:
        parser.print_help() 