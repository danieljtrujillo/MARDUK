<#
.SYNOPSIS
    MARDUK — Build, Push & Deploy to RunPod (B200)

.DESCRIPTION
    One-shot script that:
      1. Commits latest code to GitHub
      2. Builds the Docker image
      3. Pushes to Docker Hub
      4. Launches a B200 pod on RunPod via API

.EXAMPLE
    .\deploy.ps1                          # Interactive — prompts for everything
    .\deploy.ps1 -SkipBuild               # Skip Docker, just launch pod
    .\deploy.ps1 -BuildOnly               # Build & push, don't launch pod
    .\deploy.ps1 -DockerUser myuser -RunPodKey rp_xxxxx
#>
param(
    [string]$DockerUser,
    [string]$ImageName   = "marduk",
    [string]$ImageTag    = "latest",
    [string]$RunPodKey,
    [string]$GpuType     = "NVIDIA B200",
    [int]$GpuCount       = 1,
    [int]$VolumeGB       = 50,
    [int]$Epochs         = 25,
    [int]$BatchSize      = 64,
    [switch]$SkipBuild,
    [switch]$BuildOnly,
    [switch]$SkipGitPush
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# ── Helpers ─────────────────────────────────────────────────────────────────
function Write-Step  { param([string]$n, [string]$msg) Write-Host "`n[$n] $msg" -ForegroundColor Cyan }
function Write-Ok    { param([string]$msg) Write-Host "  ✓ $msg" -ForegroundColor Green }
function Write-Warn  { param([string]$msg) Write-Host "  ⚠ $msg" -ForegroundColor Yellow }
function Write-Fail  { param([string]$msg) Write-Host "  ✗ $msg" -ForegroundColor Red }
function Bail        { param([string]$msg) Write-Fail $msg; exit 1 }

# ── Ensure Docker is on PATH ───────────────────────────────────────────────
$dockerBin = "C:\Program Files\Docker\Docker\resources\bin"
if ($env:Path -notlike "*$dockerBin*") { $env:Path = "$dockerBin;$env:Path" }

# ── Banner ──────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════╗" -ForegroundColor DarkYellow
Write-Host "  ║   MARDUK — RunPod B200 Deployment Script     ║" -ForegroundColor DarkYellow
Write-Host "  ╚══════════════════════════════════════════════╝" -ForegroundColor DarkYellow
Write-Host ""

# ═════════════════════════════════════════════════════════════════════════════
# STEP 0: Gather credentials
# ═════════════════════════════════════════════════════════════════════════════
Write-Step "0" "Checking credentials..."

# Docker Hub
if (-not $SkipBuild) {
    if (-not $DockerUser) {
        $DockerUser = Read-Host "  Docker Hub username"
        if (-not $DockerUser) { Bail "Docker Hub username required for build/push" }
    }
    $imageLocal  = "${ImageName}:${ImageTag}"
    $imageRemote = "${DockerUser}/${ImageName}:${ImageTag}"
    Write-Ok "Image: $imageRemote"
}

# RunPod
if (-not $BuildOnly) {
    if (-not $RunPodKey) {
        # Check env var
        if ($env:RUNPOD_API_KEY) {
            $RunPodKey = $env:RUNPOD_API_KEY
            Write-Ok "RunPod key from env var"
        } else {
            $RunPodKey = Read-Host "  RunPod API key (https://runpod.io/console/user/settings)"
            if (-not $RunPodKey) { Bail "RunPod API key required to launch pod" }
        }
    }
    Write-Ok "RunPod key: $($RunPodKey.Substring(0,6))..."
}

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: Git commit & push
# ═════════════════════════════════════════════════════════════════════════════
if (-not $SkipGitPush) {
    Write-Step "1" "Pushing latest code to GitHub..."
    $dirty = git status --porcelain 2>$null
    if ($dirty) {
        git add -A
        # Release any stale lock on COMMIT_EDITMSG
        $lockFile = Join-Path (git rev-parse --git-dir) "COMMIT_EDITMSG.lock"
        if (Test-Path $lockFile) { Remove-Item $lockFile -Force -ErrorAction SilentlyContinue }
        $commitMsg = "deploy: pre-RunPod snapshot $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
        $commitResult = git commit -m $commitMsg 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "git commit failed (non-fatal): $commitResult"
        } else {
            Write-Ok "Committed changes"
        }
    } else {
        Write-Ok "Working tree clean"
    }
    $pushResult = git push origin main 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "git push failed (non-fatal): $pushResult"
    } else {
        Write-Ok "Pushed to origin/main"
    }
} else {
    Write-Step "1" "Skipping git push (--SkipGitPush)"
}

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: Docker Build
# ═════════════════════════════════════════════════════════════════════════════
if (-not $SkipBuild) {
    Write-Step "2" "Building Docker image..."

    # Verify daemon
    $daemonOk = docker info 2>&1 | Select-String "Server Version"
    if (-not $daemonOk) {
        Write-Warn "Docker daemon not running — starting Docker Desktop..."
        Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        $tries = 0
        while ($tries -lt 24) {
            Start-Sleep 5; $tries++
            $check = docker info 2>&1 | Select-String "Server Version"
            if ($check) { break }
            Write-Host "    waiting... ($($tries * 5)s)" -ForegroundColor DarkGray
        }
        if ($tries -ge 24) { Bail "Docker daemon did not start within 2 minutes" }
    }
    Write-Ok "Docker daemon ready"

    # Ensure logged in
    $loginCheck = docker info 2>&1 | Select-String "Username"
    if (-not $loginCheck) {
        Write-Warn "Not logged in to Docker Hub — logging in..."
        docker login
        if ($LASTEXITCODE -ne 0) { Bail "Docker login failed" }
    }
    Write-Ok "Docker Hub authenticated"

    # Build
    Write-Host "    Building $imageLocal ..." -ForegroundColor DarkGray
    docker build -t $imageLocal .
    if ($LASTEXITCODE -ne 0) { Bail "Docker build failed" }
    Write-Ok "Built $imageLocal"

    # Tag & Push
    Write-Step "3" "Pushing to Docker Hub..."
    docker tag $imageLocal $imageRemote
    docker push $imageRemote
    if ($LASTEXITCODE -ne 0) { Bail "Docker push failed" }
    Write-Ok "Pushed $imageRemote"
} else {
    Write-Step "2-3" "Skipping Docker build/push (--SkipBuild)"
    if ($DockerUser) {
        $imageRemote = "${DockerUser}/${ImageName}:${ImageTag}"
    } else {
        $imageRemote = $null
    }
}

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: Launch RunPod Pod
# ═════════════════════════════════════════════════════════════════════════════
if (-not $BuildOnly) {
    Write-Step "4" "Launching RunPod pod..."

    # Build the GraphQL mutation to create a pod
    # If we have a Docker image, use a custom template; otherwise use stock PyTorch
    if ($imageRemote) {
        $dockerImage = $imageRemote
        $startCmd = "cd /workspace/marduk && bash runpod_start.sh --batch-size $BatchSize --epochs $Epochs"
        Write-Ok "Using custom image: $dockerImage"
    } else {
        # No docker image — use RunPod stock pytorch and clone from GitHub
        $dockerImage = "runpod/pytorch:1.0.3-cu1300-torch280-ubuntu2404"
        $startCmd = @"
cd /workspace && git clone https://github.com/danieljtrujillo/MARDUK.git marduk && cd marduk && pip install -r requirements.txt && pip install 'mamba-ssm>=2.2.2' && bash runpod_start.sh --batch-size $BatchSize --epochs $Epochs
"@
        Write-Ok "Using stock RunPod image + git clone"
    }

    $gqlMutation = @"
mutation {
  podFindAndDeployOnDemand(input: {
    name: "marduk-b200"
    imageName: "$dockerImage"
    gpuTypeId: "$GpuType"
    gpuCount: $GpuCount
    volumeInGb: $VolumeGB
    containerDiskInGb: 40
    ports: "8888/http,22/tcp"
    volumeMountPath: "/workspace"
    startSsh: true
    dockerArgs: "$($startCmd -replace '"', '\"')"
  }) {
    id
    imageName
    machine {
      gpuDisplayName
    }
  }
}
"@

    $headers = @{
        "Content-Type"  = "application/json"
        "Authorization" = "Bearer $RunPodKey"
    }
    $body = @{ query = $gqlMutation } | ConvertTo-Json -Depth 5

    Write-Host "    Calling RunPod API..." -ForegroundColor DarkGray
    try {
        $resp = Invoke-RestMethod -Uri "https://api.runpod.io/graphql" `
                                  -Method POST `
                                  -Headers $headers `
                                  -Body $body

        if ($resp.errors) {
            Write-Fail "RunPod API error:"
            $resp.errors | ForEach-Object { Write-Host "    $($_.message)" -ForegroundColor Red }
            exit 1
        }

        $pod = $resp.data.podFindAndDeployOnDemand
        $podId = $pod.id
        $gpuName = $pod.machine.gpuDisplayName

        Write-Host ""
        Write-Host "  ╔══════════════════════════════════════════════╗" -ForegroundColor Green
        Write-Host "  ║   Pod launched successfully!                  ║" -ForegroundColor Green
        Write-Host "  ╚══════════════════════════════════════════════╝" -ForegroundColor Green
        Write-Host ""
        Write-Host "    Pod ID:    $podId" -ForegroundColor White
        Write-Host "    GPU:       $gpuName" -ForegroundColor White
        Write-Host "    Image:     $($pod.imageName)" -ForegroundColor White
        Write-Host "    Dashboard: https://www.runpod.io/console/pods/$podId" -ForegroundColor Yellow
        Write-Host ""

        if (-not $imageRemote) {
            Write-Host "    The pod will auto-clone from GitHub and start training." -ForegroundColor DarkGray
            Write-Host "    Monitor at: https://www.runpod.io/console/pods/$podId" -ForegroundColor DarkGray
        } else {
            Write-Host "    Training starts automatically via dockerArgs." -ForegroundColor DarkGray
        }
    }
    catch {
        Bail "RunPod API call failed: $_"
    }
} else {
    Write-Step "4" "Skipping pod launch (--BuildOnly)"
}

# ═════════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "  Done." -ForegroundColor Green
Write-Host ""
