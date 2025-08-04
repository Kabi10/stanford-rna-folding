#!/usr/bin/env python
"""
Script to initialize the Kaggle workspace environment.
This will set up a virtual environment, install dependencies,
and check Kaggle API connection.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def create_virtual_environment(venv_name="venv"):
    """
    Create a Python virtual environment.
    
    Parameters:
    -----------
    venv_name : str, default="venv"
        Name of the virtual environment
    
    Returns:
    --------
    bool
        Whether the virtual environment was created successfully
    """
    print(f"Creating virtual environment '{venv_name}'...")
    
    try:
        # Create the virtual environment
        subprocess.check_call([sys.executable, "-m", "venv", venv_name])
        
        # Get the path to the activate script
        if os.name == 'nt':  # Windows
            activate_script = os.path.join(venv_name, "Scripts", "activate")
            activate_cmd = f"{activate_script}"
        else:  # Unix/MacOS
            activate_script = os.path.join(venv_name, "bin", "activate")
            activate_cmd = f"source {activate_script}"
        
        print(f"✅ Virtual environment created successfully")
        print(f"\nTo activate the virtual environment, run:")
        print(f"  {activate_cmd}")
        
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False
    except Exception as e:
        print(f"❌ Error creating virtual environment: {e}")
        return False

def install_requirements():
    """
    Install requirements from requirements.txt.
    
    Returns:
    --------
    bool
        Whether requirements were installed successfully
    """
    print("\nInstalling dependencies from requirements.txt...")
    
    try:
        # Check if we're in a virtual environment
        in_venv = sys.prefix != sys.base_prefix
        if not in_venv:
            print("⚠️ Warning: Not running in a virtual environment")
            response = input("Continue with installation in the base Python environment? (y/n): ")
            if response.lower() != 'y':
                print("Installation aborted. Please activate your virtual environment first.")
                return False
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✅ Dependencies installed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def test_kaggle_connection():
    """
    Test the Kaggle API connection.
    
    Returns:
    --------
    bool
        Whether the connection test was successful
    """
    print("\nTesting Kaggle API connection...")
    
    try:
        # Run the test script
        result = subprocess.run(
            [sys.executable, "scripts/test_kaggle_connection.py"],
            capture_output=False  # Allow output to be displayed directly
        )
        
        return result.returncode == 0
    
    except Exception as e:
        print(f"❌ Error testing Kaggle API connection: {e}")
        return False

def create_example_competition(competition_name):
    """
    Create an example competition workspace.
    
    Parameters:
    -----------
    competition_name : str
        Name of the competition
    
    Returns:
    --------
    bool
        Whether the competition workspace was created successfully
    """
    print(f"\nCreating example competition workspace for '{competition_name}'...")
    
    try:
        # Run the create competition script
        result = subprocess.run(
            [sys.executable, "scripts/create_competition.py", competition_name],
            capture_output=False  # Allow output to be displayed directly
        )
        
        return result.returncode == 0
    
    except Exception as e:
        print(f"❌ Error creating competition workspace: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Initialize Kaggle workspace environment')
    parser.add_argument('--venv', type=str, default="venv", help='Name of the virtual environment to create')
    parser.add_argument('--no-venv', action='store_true', help='Skip virtual environment creation')
    parser.add_argument('--no-deps', action='store_true', help='Skip dependency installation')
    parser.add_argument('--no-test', action='store_true', help='Skip Kaggle API connection test')
    parser.add_argument('--example', type=str, help='Create example competition workspace with this name')
    
    args = parser.parse_args()
    
    print("Kaggle Workspace Initialization")
    print("==============================\n")
    
    # Create virtual environment
    if not args.no_venv:
        venv_created = create_virtual_environment(args.venv)
        if not venv_created:
            print("\n⚠️ Virtual environment creation failed. Continuing with other steps...")
    
    # Install dependencies
    if not args.no_deps:
        deps_installed = install_requirements()
        if not deps_installed:
            print("\n⚠️ Dependency installation failed. Please check the errors and try again.")
    
    # Test Kaggle API connection
    if not args.no_test:
        connection_ok = test_kaggle_connection()
        if not connection_ok:
            print("\n⚠️ Kaggle API connection test failed. Please check your API credentials.")
    
    # Create example competition workspace
    if args.example:
        competition_created = create_example_competition(args.example)
        if not competition_created:
            print(f"\n⚠️ Failed to create example competition workspace for '{args.example}'.")
    
    print("\nInitialization complete!")
    print("\nNext steps:")
    if not args.no_venv:
        print(f"1. Activate your virtual environment")
        if os.name == 'nt':  # Windows
            print(f"   {args.venv}\\Scripts\\activate")
        else:  # Unix/MacOS
            print(f"   source {args.venv}/bin/activate")
    
    print("2. Verify Kaggle API credentials (if not already done)")
    print("   python scripts/test_kaggle_connection.py")
    
    print("3. Create a competition workspace")
    print("   python scripts/create_competition.py <competition_name>")
    
    print("4. Download competition data")
    print("   python scripts/download_kaggle_data.py competition <competition_name>")
    
    print("\nHappy data science! 🚀")

if __name__ == "__main__":
    main() 