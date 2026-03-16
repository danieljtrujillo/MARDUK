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

# Pre-download ByT5-base weights so training doesn't wait for downloads
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    AutoTokenizer.from_pretrained('google/byt5-base'); \
    AutoModelForSeq2SeqLM.from_pretrained('google/byt5-base')"

# Copy project
COPY . .

# Make scripts executable
RUN chmod +x scripts/*.sh runpod_start.sh

# Expose port 8888 for the web dashboard (RunPod HTTP service)
EXPOSE 8888

# Entrypoint: symlink persistent dirs to /workspace volume, then launch web dashboard.
# All training outputs, processed data, and checkpoints persist across pod restarts.
COPY entrypoint.sh /app/marduk/entrypoint.sh
RUN chmod +x /app/marduk/entrypoint.sh
CMD ["/app/marduk/entrypoint.sh"]
