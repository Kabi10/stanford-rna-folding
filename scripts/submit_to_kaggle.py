#!/usr/bin/env python
"""
Script to submit predictions to Kaggle for the Stanford RNA 3D Folding competition.
"""

import argparse
import os
import subprocess
from pathlib import Path
import json
import datetime


def check_kaggle_api():
    """
    Check if the Kaggle API is configured correctly.
    
    Returns:
        bool: True if Kaggle API is configured, False otherwise
    """
    try:
        # Check if kaggle command is available
        result = subprocess.run(
            ["kaggle", "competitions", "list"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print("Error running Kaggle API:")
            print(result.stderr)
            return False
            
        return True
    except FileNotFoundError:
        print("Kaggle API not found. Please install it with 'pip install kaggle'")
        return False


def submit_to_kaggle(
    submission_file: str,
    competition: str = "stanford-ribonanza-rna-folding",
    message: str = None,
):
    """
    Submit a file to a Kaggle competition.
    
    Args:
        submission_file: Path to the submission file
        competition: Name of the competition
        message: Message for the submission
        
    Returns:
        bool: True if submission was successful, False otherwise
    """
    # Check if submission file exists
    if not os.path.isfile(submission_file):
        print(f"Submission file not found: {submission_file}")
        return False
    
    # Generate a default message if none provided
    if message is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"Submission from CLI tool at {timestamp}"
    
    # Build the command
    cmd = [
        "kaggle", "competitions", "submit", 
        "-c", competition,
        "-f", submission_file,
        "-m", message
    ]
    
    # Execute the command
    try:
        print("Submitting to Kaggle...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            print("Submission successful!")
            print(result.stdout)
            
            # Check if we can extract the submission URL
            for line in result.stdout.splitlines():
                if "https://www.kaggle.com/competitions/" in line:
                    print(f"View your submission at: {line.strip()}")
                    break
            return True
        else:
            print("Submission failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Error submitting to Kaggle: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Submit predictions to Kaggle for the Stanford RNA 3D Folding competition."
    )
    
    parser.add_argument(
        "--file", 
        type=str, 
        required=True,
        help="Path to the submission file."
    )
    
    parser.add_argument(
        "--competition", 
        type=str, 
        default="stanford-ribonanza-rna-folding",
        help="Kaggle competition name."
    )
    
    parser.add_argument(
        "--message", 
        type=str, 
        default=None,
        help="Message for the submission."
    )
    
    parser.add_argument(
        "--model-checkpoint", 
        type=str, 
        default=None,
        help="Path to the model checkpoint used for this submission (for record keeping)."
    )
    
    parser.add_argument(
        "--record-file", 
        type=str, 
        default="submissions/submission_records.json",
        help="Path to JSON file for recording submission details."
    )
    
    args = parser.parse_args()
    
    # Check if Kaggle API is configured
    if not check_kaggle_api():
        print("Please configure the Kaggle API first:")
        print("1. Go to https://www.kaggle.com/<username>/account")
        print("2. Create a new API token (will download kaggle.json)")
        print("3. Move the file to ~/.kaggle/kaggle.json")
        print("4. Run 'chmod 600 ~/.kaggle/kaggle.json'")
        return False
    
    # Create submissions record file directory if it doesn't exist
    os.makedirs(os.path.dirname(args.record_file), exist_ok=True)
    
    # Load existing records if available
    records = []
    if os.path.isfile(args.record_file):
        try:
            with open(args.record_file, "r") as f:
                records = json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading records file {args.record_file}. Starting with empty records.")
    
    # Submit to Kaggle
    success = submit_to_kaggle(args.file, args.competition, args.message)
    
    if success:
        # Record submission details
        submission_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "file": args.file,
            "competition": args.competition,
            "message": args.message,
            "model_checkpoint": args.model_checkpoint,
        }
        
        records.append(submission_record)
        
        # Save updated records
        with open(args.record_file, "w") as f:
            json.dump(records, f, indent=2)
            
        print(f"Submission record saved to {args.record_file}")
        return True
    else:
        print("Submission failed.")
        return False


if __name__ == "__main__":
    main() 