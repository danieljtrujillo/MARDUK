# MARDUK — One-shot Windows setup for Docker + WSL2
# Self-elevates to Administrator if not already elevated.

# --- Self-Elevation ---
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)
if (-not $isAdmin) {
    Write-Host "Requesting Administrator privileges..." -ForegroundColor Yellow
    $scriptPath = $MyInvocation.MyCommand.Definition
    Start-Process powershell.exe -Verb RunAs -ArgumentList "-ExecutionPolicy Bypass -File `"$scriptPath`""
    exit
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " MARDUK: Setting up Docker + WSL2" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

$needsReboot = $false

# 1. Enable WSL feature
Write-Host "`n[1/4] Enabling Windows Subsystem for Linux..." -ForegroundColor Yellow
try {
    $wsl = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
    if ($wsl.State -ne 'Enabled') {
        Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -NoRestart -All | Out-Null
        $needsReboot = $true
        Write-Host "  WSL feature enabled." -ForegroundColor Green
    } else {
        Write-Host "  WSL feature already enabled." -ForegroundColor Green
    }
} catch {
    Write-Host "  ERROR: Failed to enable WSL — $_" -ForegroundColor Red
}

# 2. Enable Virtual Machine Platform
Write-Host "`n[2/4] Enabling Virtual Machine Platform..." -ForegroundColor Yellow
try {
    $vmp = Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
    if ($vmp.State -ne 'Enabled') {
        Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -NoRestart -All | Out-Null
        $needsReboot = $true
        Write-Host "  VM Platform enabled." -ForegroundColor Green
    } else {
        Write-Host "  VM Platform already enabled." -ForegroundColor Green
    }
} catch {
    Write-Host "  ERROR: Failed to enable VM Platform — $_" -ForegroundColor Red
}

# 3. Set WSL default version to 2
Write-Host "`n[3/4] Setting WSL default version to 2..." -ForegroundColor Yellow
wsl --set-default-version 2 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  WSL2 set as default." -ForegroundColor Green
} else {
    Write-Host "  WARNING: Could not set WSL2 default (may need reboot first)." -ForegroundColor Yellow
}

# 4. Install WSL kernel update
Write-Host "`n[4/4] Installing/updating WSL kernel..." -ForegroundColor Yellow
wsl --update 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  WSL kernel updated." -ForegroundColor Green
} else {
    Write-Host "  WARNING: WSL update returned exit code $LASTEXITCODE." -ForegroundColor Yellow
}

# 5. Add Docker to system PATH if not present
$dockerBin = "C:\Program Files\Docker\Docker\resources\bin"
$currentPath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
if ($currentPath -notlike "*$dockerBin*") {
    Write-Host "`nAdding Docker to system PATH..." -ForegroundColor Yellow
    [System.Environment]::SetEnvironmentVariable("Path", "$currentPath;$dockerBin", "Machine")
    Write-Host "  Docker added to PATH." -ForegroundColor Green
} else {
    Write-Host "`nDocker already in system PATH." -ForegroundColor Green
}

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host " Setup complete!" -ForegroundColor Green
if ($needsReboot) {
    Write-Host " A REBOOT is required for WSL2 to activate." -ForegroundColor Red
} else {
    Write-Host " No reboot needed — features were already enabled." -ForegroundColor Green
}
Write-Host " After reboot (if needed):" -ForegroundColor Yellow
Write-Host "   1. Docker Desktop will auto-start" -ForegroundColor White
Write-Host "   2. Accept the Docker license agreement" -ForegroundColor White
Write-Host "   3. Run: docker login" -ForegroundColor White
Write-Host "   4. Run: python deploy_runpod.py --help" -ForegroundColor White
Write-Host "============================================" -ForegroundColor Cyan

if ($needsReboot) {
    $restart = Read-Host "`nReboot now? (y/n)"
    if ($restart -eq 'y') {
        Restart-Computer -Force
    } else {
        Write-Host "Remember to reboot before using Docker!" -ForegroundColor Yellow
    }
}

# Keep the elevated window open so the user can read output
Read-Host "`nPress Enter to close"
