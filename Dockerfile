# ---------------------------------------------------------------------------
# MARDUK — RunPod Deployment Image
# Base: PyTorch 2.8 + CUDA 13.0 (Blackwell B200 compatible)
# ---------------------------------------------------------------------------
FROM runpod/pytorch:1.0.3-cu1300-torch280-ubuntu2404

LABEL maintainer="danieljtrujillo"
LABEL description="MARDUK: Akkadian→English Neural MT on RunPod"

# Install to /app/marduk — NOT /workspace, which RunPod overwrites with a volume mount.
WORKDIR /app/marduk

# System deps (sentencepiece needs cmake, mamba-ssm needs ninja + nvcc from devel image)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Python deps — install before copying source for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# mamba-ssm builds from source (needs CUDA toolkit from devel image)
RUN pip install --no-cache-dir "mamba-ssm>=2.2.2"

# Copy project
COPY . .

# Make scripts executable
RUN chmod +x scripts/*.sh runpod_start.sh

# Auto-start: run the pipeline, then keep alive for inspection via web terminal.
# Outputs are symlinked to /workspace (persistent volume) so they survive restarts.
CMD ["bash", "-c", "mkdir -p /workspace/outputs && ln -sfn /workspace/outputs /app/marduk/outputs && cd /app/marduk && bash runpod_start.sh 2>&1; echo 'Pipeline finished. Container staying alive for inspection.'; sleep infinity"]
