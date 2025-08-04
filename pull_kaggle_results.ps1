#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Pulls results from a Kaggle kernel run
.DESCRIPTION
    This script activates the Python virtual environment and pulls the results
    from a Kaggle kernel run of the RNA folding model.
.PARAMETER KernelId
    The ID of the Kaggle kernel to pull results from (e.g., "username/kernel-name")
.PARAMETER OutputDir
    The directory to save the results to (default: "./kaggle_outputs")
.EXAMPLE
    ./pull_kaggle_results.ps1 -KernelId "username/stanford-rna-3d-folding-run-2025-04-01-1200"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$KernelId,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputDir = "./kaggle_outputs"
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
    
    Write-Host "✅ Virtual environment activated successfully!" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to activate virtual environment: $_" -ForegroundColor Red
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
    Write-Host "✅ Kaggle CLI is available!" -ForegroundColor Green
} catch {
    Write-Host "❌ Kaggle CLI not found or not working properly." -ForegroundColor Red
    Write-Host "Please make sure Kaggle CLI is installed and working." -ForegroundColor Yellow
    exit 1
}

# Step 3: Create output directory if it doesn't exist
if (-not (Test-Path $OutputDir)) {
    Write-Host "Creating output directory $OutputDir..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
    Write-Host "✅ Output directory created!" -ForegroundColor Green
}

# Step 4: Pull the results from Kaggle
Write-Host "Pulling results from Kaggle kernel $KernelId..." -ForegroundColor Cyan
kaggle kernels output $KernelId -p $OutputDir

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to pull results from Kaggle." -ForegroundColor Red
    exit 1
}

Write-Host "`n✨ Complete! Results have been downloaded to $OutputDir" -ForegroundColor Green

# Step 5: List the downloaded files
Write-Host "Files downloaded:" -ForegroundColor Cyan
Get-ChildItem -Path $OutputDir -Recurse | Where-Object { -not $_.PSIsContainer } | ForEach-Object {
    $relativePath = $_.FullName.Substring((Resolve-Path $OutputDir).Path.Length + 1)
    Write-Host "- $relativePath" -ForegroundColor Yellow
} 