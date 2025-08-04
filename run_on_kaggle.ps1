#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Pushes the RNA folding model to Kaggle and runs it on Kaggle's GPU
.DESCRIPTION
    This script activates the Python virtual environment, verifies the Kaggle CLI is available,
    and pushes the RNA folding model script to Kaggle to run on their GPU resources.
.EXAMPLE
    ./run_on_kaggle.ps1
    
    # Run with a specific config file
    ./run_on_kaggle.ps1 -ConfigFile "configs/biophysics_config.yaml"
#>
param (
    [string]$ConfigFile = "configs/biophysics_config.yaml",
    [switch]$PublicKernel = $false
)

# Step 1: Activate the virtual environment
Write-Host "Activating Python virtual environment..." -ForegroundColor Cyan
try {
    # Activate the virtual environment
    & .\venv\Scripts\Activate.ps1
    
    # Verify activation succeeded
    if (-not $env:VIRTUAL_ENV) {
        throw "Virtual environment activation failed!"
    }
    
    Write-Host "Virtual environment activated successfully!" -ForegroundColor Green
} catch {
    Write-Host "Failed to activate virtual environment: $_" -ForegroundColor Red
    Write-Host "Make sure your virtual environment is properly set up in the './venv' directory." -ForegroundColor Yellow
    exit 1
}

# Step 2: Verify Kaggle CLI is available
Write-Host "Verifying Kaggle CLI is available..." -ForegroundColor Cyan
try {
    $kaggleVersion = kaggle --version
    if (-not $?) {
        throw "Kaggle CLI not found!"
    }
    Write-Host "Kaggle CLI is available!" -ForegroundColor Green
} catch {
    Write-Host "Kaggle CLI not found or not working properly." -ForegroundColor Red
    Write-Host "Attempting to install Kaggle CLI..." -ForegroundColor Yellow
    
    pip install kaggle
    
    # Try again
    try {
        $kaggleVersion = kaggle --version
        if (-not $?) {
            throw "Kaggle CLI installation failed!"
        }
        Write-Host "Kaggle CLI installed successfully!" -ForegroundColor Green
    } catch {
        Write-Host "Failed to install Kaggle CLI. Please install it manually with 'pip install kaggle'." -ForegroundColor Red
        exit 1
    }
}

# Step 3: Verify Kaggle credentials are set up
Write-Host "Checking Kaggle credentials..." -ForegroundColor Cyan
$credentialsPath = "$HOME/.kaggle/kaggle.json"
if (-not (Test-Path $credentialsPath)) {
    Write-Host "Kaggle credentials not found at $credentialsPath" -ForegroundColor Red
    Write-Host "Please set up your Kaggle API credentials. You can download them from:" -ForegroundColor Yellow
    Write-Host "https://www.kaggle.com/account -> 'Create New API Token'" -ForegroundColor Yellow
    exit 1
}
Write-Host "Kaggle credentials found!" -ForegroundColor Green

# Step 4: Check that the specified config file exists
Write-Host "Checking config file: $ConfigFile..." -ForegroundColor Cyan
if (-not (Test-Path $ConfigFile)) {
    Write-Host "Config file not found: $ConfigFile" -ForegroundColor Red
    Write-Host "Please specify a valid config file path." -ForegroundColor Yellow
    exit 1
}
Write-Host "Config file found: $ConfigFile" -ForegroundColor Green

# Step 5: Prepare resources to include
$resources = @(
    "src",
    "configs"
)

# Resource validation
$validResources = @()
foreach ($resource in $resources) {
    if (Test-Path $resource) {
        $validResources += $resource
        Write-Host "Resource found: $resource" -ForegroundColor Green
    } else {
        Write-Host "Resource not found, skipping: $resource" -ForegroundColor Yellow
    }
}

# Generate a unique title for the kernel using the config name and current timestamp
$configName = [System.IO.Path]::GetFileNameWithoutExtension($ConfigFile)
$timestamp = Get-Date -Format "yyyyMMdd_HHmm"
$title = "RNA Fold $configName $timestamp"

# Prepare the public flag if needed
$publicFlag = if ($PublicKernel) { "--public" } else { "" }

# Step 6: Prepare push directory using Python script
Write-Host "Preparing files for Kaggle push..." -ForegroundColor Cyan

# Create a temporary directory
$tempDir = New-TemporaryFile | ForEach-Object { Remove-Item $_; New-Item -ItemType Directory -Path "$($_.FullName)_dir" }
Write-Host "Created temporary directory: $tempDir" -ForegroundColor Green

# Copy the main script
Copy-Item -Path "rna_folding_kaggle.py" -Destination $tempDir
Write-Host "Copied rna_folding_kaggle.py to temp directory" -ForegroundColor Green

# Copy the config file
Copy-Item -Path $ConfigFile -Destination $tempDir
$configFileName = Split-Path $ConfigFile -Leaf
Write-Host "Copied $configFileName to temp directory" -ForegroundColor Green

# Potentially copy additional resources
foreach ($resource in $validResources) {
    if (Test-Path $resource -PathType Container) {
        # Create the destination directory if it doesn't exist
        $destResourceDir = Join-Path $tempDir $resource
        New-Item -ItemType Directory -Path $destResourceDir -Force | Out-Null
        
        # Copy the directory content
        Copy-Item -Path "$resource\*" -Destination $destResourceDir -Recurse -Force
        Write-Host "Copied resource directory $resource to temp directory" -ForegroundColor Green
    } else {
        # Copy the file
        Copy-Item -Path $resource -Destination $tempDir
        Write-Host "Copied resource file $resource to temp directory" -ForegroundColor Green
    }
}

# Create the metadata file using JSON directly
$username = python -c "import os, json; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])" -ErrorAction SilentlyContinue
if (-not $username) {
    Write-Host "Unable to get Kaggle username from credentials. Using 'kabitharma' as fallback." -ForegroundColor Yellow
    $username = "kabitharma"
}

# Create slug from title
$slug = $title -replace '[^a-zA-Z0-9]+', '-' -replace '^-|-$', '' -replace '-+', '-'
$slug = $slug.ToLower()

$metadataPath = Join-Path $tempDir "kernel-metadata.json"
$metadata = @{
    id = "$username/$slug"
    title = $title
    code_file = "rna_folding_kaggle.py"
    language = "python"
    kernel_type = "script"
    is_private = "true"
    enable_gpu = "true"
    enable_internet = "false"
    competition_sources = @("stanford-rna-3d-folding")
}

# Write metadata to file as proper JSON
$metadataJson = $metadata | ConvertTo-Json
Set-Content -Path $metadataPath -Value $metadataJson
Write-Host "Created kernel-metadata.json with proper slug: $slug" -ForegroundColor Green

# Debug: Output the metadata content
Write-Host "Metadata JSON content:" -ForegroundColor Cyan
Get-Content -Path $metadataPath | Write-Host -ForegroundColor Gray

# Step 7: Push to Kaggle using Kaggle CLI directly
Write-Host "Pushing to Kaggle from temporary directory..." -ForegroundColor Cyan

try {
    Push-Location $tempDir
    Write-Host "Executing: kaggle kernels push -p ." -ForegroundColor Yellow
    
    # Execute Kaggle push directly - run it using cmd /c to ensure full output capture
    $kaggleOutput = cmd /c kaggle kernels push -p . 2>&1
    $kaggleExitCode = $LASTEXITCODE
    
    # Print Kaggle output
    Write-Host "Kaggle CLI Output:" -ForegroundColor Cyan
    Write-Host $kaggleOutput -ForegroundColor Gray
    
    if ($kaggleExitCode -ne 0) {
        throw "Kaggle CLI push command failed (Exit Code: $kaggleExitCode). See output above for details."
    }
    
    Write-Host "`nSuccessfully pushed to Kaggle!" -ForegroundColor Green

} catch {
    Write-Host "Error during Kaggle push or cleanup: $_" -ForegroundColor Red
    # Ensure we pop location even if push fails
    if ($PWD.Path -eq $tempDir) {
        Pop-Location
    }
    # Attempt cleanup
    Write-Host "Attempting cleanup of temporary directory: $tempDir" -ForegroundColor Yellow
    if (Test-Path $tempDir) {
        Remove-Item -Recurse -Force $tempDir -ErrorAction Continue
    }
    exit 1
} finally {
    # Ensure we always pop location and clean up temp dir on success
    if ($?) { # Check if the try block succeeded
        if ($PWD.Path -eq $tempDir) {
            Pop-Location
        }
        Write-Host "Cleaning up temporary directory: $tempDir" -ForegroundColor Cyan
        if (Test-Path $tempDir) {
            Remove-Item -Recurse -Force $tempDir -ErrorAction Continue
        }
    }
}

# Step 8: Display completion message (only runs if push succeeded)
Write-Host "`nComplete! Your model should now be running on Kaggle with the $configName configuration." -ForegroundColor Green
Write-Host "You can monitor your model's execution in the Kaggle web interface." -ForegroundColor Cyan
Write-Host "Remember to pull the results when it's finished using:" -ForegroundColor Cyan
if ($username) {
    Write-Host "kaggle kernels output $username/$slug -p ./kaggle_outputs" -ForegroundColor Yellow
} else {
    Write-Host "(Could not determine username automatically for pull command)" -ForegroundColor Yellow
} 