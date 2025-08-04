#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Checks the status of your Kaggle kernels
.DESCRIPTION
    This script activates the Python virtual environment and lists all your
    Kaggle kernels with their statuses.
.EXAMPLE
    ./check_kaggle_kernels.ps1
#>

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

# Step 2: List all your Kaggle kernels
Write-Host "Listing your Kaggle kernels..." -ForegroundColor Cyan
kaggle kernels list --mine

Write-Host "`nTo check a specific kernel's status, run:" -ForegroundColor Cyan
Write-Host "kaggle kernels status username/kernel-slug" -ForegroundColor Yellow

Write-Host "`nTo pull results when a kernel is complete, run:" -ForegroundColor Cyan
Write-Host "./pull_kaggle_results.ps1 -KernelId username/kernel-slug" -ForegroundColor Yellow 