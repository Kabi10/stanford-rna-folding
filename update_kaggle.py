import json
import os
import subprocess
import sys

# Define metadata
metadata = {
    "id": "kabitharma/simple-rna-test-cpu",
    "title": "Simple RNA Test CPU",
    "code_file": "rna_folding_kaggle.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "false",
    "enable_internet": "false",
    "competition_sources": ["stanford-rna-3d-folding"]
}

# Ensure minimal_push directory exists
os.makedirs("minimal_push", exist_ok=True)

# Copy main file if it doesn't exist in the minimal_push directory
if not os.path.exists("minimal_push/rna_folding_kaggle.py"):
    if os.path.exists("rna_folding_kaggle.py"):
        with open("rna_folding_kaggle.py", "r") as src, open("minimal_push/rna_folding_kaggle.py", "w") as dst:
            dst.write(src.read())
        print("Copied rna_folding_kaggle.py to minimal_push directory")
    else:
        print("Error: rna_folding_kaggle.py not found", file=sys.stderr)
        sys.exit(1)

# Write metadata file
with open("minimal_push/kernel-metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("Created kernel-metadata.json")

# Change to minimal_push directory and run kaggle command
os.chdir("minimal_push")
print("Changed to minimal_push directory")
print("Current directory:", os.getcwd())

# Print content of kernel-metadata.json
with open("kernel-metadata.json", "r") as f:
    print("kernel-metadata.json content:")
    print(f.read())

# Run kaggle push command
try:
    result = subprocess.run(["kaggle", "kernels", "push"], check=True, capture_output=True, text=True)
    print("Kaggle push output:")
    print(result.stdout)
    if result.stderr:
        print("Stderr:", result.stderr)
except subprocess.CalledProcessError as e:
    print(f"Error pushing to Kaggle (exit code {e.returncode}):", file=sys.stderr)
    print(e.stdout)
    print(e.stderr, file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Exception: {e}", file=sys.stderr)
    sys.exit(1)

print("Successfully pushed to Kaggle") 
