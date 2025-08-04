#!/usr/bin/env python
"""
Script to test Kaggle API connection and credentials.
This helps verify that your Kaggle API setup is working correctly.
"""

import os
import json
import argparse
import subprocess
from pathlib import Path

def check_kaggle_credentials():
    """
    Check if Kaggle API credentials are properly set up.
    """
    # Check if kaggle package is installed
    try:
        import kaggle
        print("✅ Kaggle package is installed")
    except ImportError:
        print("❌ Kaggle package is not installed")
        print("   Installing Kaggle package...")
        try:
            subprocess.check_call(["pip", "install", "kaggle"])
            print("✅ Kaggle package installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install Kaggle package")
            print("   Please install manually: pip install kaggle")
            return False
    
    # Check for kaggle.json
    home = Path.home()
    kaggle_dir = home / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_dir.exists():
        print(f"❌ Kaggle directory not found: {kaggle_dir}")
        return False
    
    if not kaggle_json.exists():
        print(f"❌ Kaggle API credentials file not found: {kaggle_json}")
        return False
    
    print(f"✅ Found Kaggle credentials file: {kaggle_json}")
    
    # Check if credentials are valid
    try:
        with open(kaggle_json, 'r') as f:
            credentials = json.load(f)
            
        if 'username' not in credentials or 'key' not in credentials:
            print("❌ Kaggle credentials file is invalid (missing username or key)")
            return False
        
        print(f"✅ Credentials format looks valid for username: {credentials['username']}")
        
        # Check file permissions on Unix systems
        if os.name != 'nt':  # not Windows
            permissions = oct(os.stat(kaggle_json).st_mode)[-3:]
            if permissions != '600':
                print(f"⚠️ Warning: Kaggle credentials file has permissions {permissions}, should be 600")
                print("   You can fix this with: chmod 600 ~/.kaggle/kaggle.json")
    
    except json.JSONDecodeError:
        print("❌ Kaggle credentials file is not valid JSON")
        return False
    except Exception as e:
        print(f"❌ Error reading Kaggle credentials: {str(e)}")
        return False
    
    return True

def test_kaggle_api():
    """
    Test connection to Kaggle API by listing competitions.
    """
    print("\nTesting Kaggle API connection...")
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "list", "-p", "5"], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Successfully connected to Kaggle API")
            print("\nAvailable competitions:")
            print(result.stdout)
            return True
        else:
            print("❌ Failed to connect to Kaggle API")
            print(f"Error: {result.stderr}")
            return False
    
    except Exception as e:
        print(f"❌ Error testing Kaggle API: {str(e)}")
        return False

def list_available_datasets(search_query=None):
    """
    List available datasets on Kaggle.
    
    Parameters:
    -----------
    search_query : str, optional
        Search query to filter datasets
    """
    print("\nListing Kaggle datasets...")
    
    cmd = ["kaggle", "datasets", "list", "-p", "10"]
    if search_query:
        cmd.extend(["--search", search_query])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Available datasets{f' for query \"{search_query}\"' if search_query else ''}:")
            print(result.stdout)
            return True
        else:
            print("❌ Failed to list datasets")
            print(f"Error: {result.stderr}")
            return False
    
    except Exception as e:
        print(f"❌ Error listing datasets: {str(e)}")
        return False

def list_my_datasets():
    """
    List the user's datasets on Kaggle.
    """
    print("\nListing your Kaggle datasets...")
    
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "list", "--mine"], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            print("Your datasets:")
            if result.stdout.strip():
                print(result.stdout)
            else:
                print("You don't have any datasets on Kaggle yet.")
            return True
        else:
            print("❌ Failed to list your datasets")
            print(f"Error: {result.stderr}")
            return False
    
    except Exception as e:
        print(f"❌ Error listing your datasets: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Kaggle API connection and credentials')
    parser.add_argument('--search', type=str, help='Search for datasets matching query', default=None)
    parser.add_argument('--my-datasets', action='store_true', help='List your datasets')
    
    args = parser.parse_args()
    
    print("Kaggle API Connection Test")
    print("=========================\n")
    
    # Check credentials
    creds_ok = check_kaggle_credentials()
    if not creds_ok:
        print("\n❌ Kaggle credentials check failed")
        print("Please review the setup instructions in KAGGLE_SETUP.md")
        exit(1)
    
    # Test API connection
    api_ok = test_kaggle_api()
    
    if args.search:
        list_available_datasets(args.search)
    
    if args.my_datasets:
        list_my_datasets()
    
    if creds_ok and api_ok:
        print("\n✅ All checks passed! Your Kaggle API connection is working correctly.")
        print("You can now use the Kaggle API to download competitions and datasets.")
    else:
        print("\n❌ Some checks failed. Please review the issues above.") 