#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Checks the status of a specific Kaggle kernel
.DESCRIPTION
    This script activates the Python virtual environment and checks the status
    of a specific Kaggle kernel.
.PARAMETER KernelId
    The ID of the Kaggle kernel to check (e.g., "username/kernel-name")
.EXAMPLE
    ./check_kernel_status.ps1 -KernelId "username/stanford-rna-3d-folding-run-2025-04-01-1200"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$KernelId
)

# Step 1: Activate the virtual environment
Write-Host "Activating Python virtual environment..."
try {
    # Activate the virtual environment
    & .\venv\Scripts\Activate.ps1
    
    # Verify activation succeeded
    if (-not $env:VIRTUAL_ENV) {
        throw "Virtual environment activation failed!"
    }
    
    Write-Host "Virtual environment activated successfully!"
} catch {
    Write-Host "Failed to activate virtual environment: $_"
    Write-Host "Make sure your virtual environment is properly set up in the './venv' directory."
    exit 1
}

# Step 2: Check the kernel status
Write-Host "Checking status of Kaggle kernel '$KernelId'..."
$statusOutput = kaggle kernels status $KernelId

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to check kernel status for '$KernelId'."
    # Print the status output in case of error
    Write-Host "Output from Kaggle CLI:"
    Write-Host $statusOutput
    exit 1
}

# Step 3: Display the status
Write-Host "Status: $statusOutput"

# Step 4: Parse the status to determine if the kernel is complete
# Kaggle status strings can vary, common ones include: complete, running, queued, error, failed
$isComplete = $statusOutput -match "complete"
$hasFailed = $statusOutput -match "error|failed"
$isRunning = $statusOutput -match "running|queued"

# Step 5: Suggest next steps based on status
if ($isComplete) {
    Write-Host "Kernel has completed execution!"
    Write-Host "To pull the results, run:"
    Write-Host "./pull_kaggle_results.ps1 -KernelId '$KernelId'"
} elseif ($hasFailed) {
    Write-Host "Kernel execution failed or encountered an error."
    Write-Host "Check the Kaggle website for logs and details."
} elseif ($isRunning) {
    Write-Host "Kernel is still running or queued."
    Write-Host "Check again later or monitor its progress on the Kaggle website."
} else {
    # Handle cases where status is not recognized
    Write-Host "Kernel status is unclear or not one of the expected states."
    Write-Host "Check the Kaggle website for more details."
}

# Provide URL to the kernel
$kernelUrl = "https://www.kaggle.com/code/$KernelId"
Write-Host "View the kernel on Kaggle: $kernelUrl"
