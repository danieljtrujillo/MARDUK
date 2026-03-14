"""MARDUK — RunPod deployment helper.

Usage:
    # Build & push Docker image, then launch a B200 pod:
    python deploy_runpod.py --docker-user YOUR_DOCKERHUB_USERNAME --api-key YOUR_RUNPOD_KEY --gpu B200

    # Skip Docker build (image already pushed):
    python deploy_runpod.py --api-key YOUR_RUNPOD_KEY --gpu B200 --skip-build

    # Just build & push, don't launch a pod:
    python deploy_runpod.py --docker-user YOUR_DOCKERHUB_USERNAME --build-only
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

# Persistent config file to store defaults like docker-user
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".deploy_config")


def _load_config() -> dict[str, str]:
    cfg: dict[str, str] = {}
    if os.path.isfile(_CONFIG_PATH):
        with open(_CONFIG_PATH) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    cfg[k.strip()] = v.strip()
    return cfg


def _save_config(cfg: dict[str, str]) -> None:
    with open(_CONFIG_PATH, "w") as f:
        for k, v in cfg.items():
            f.write(f"{k}={v}\n")

def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print(f"\n> {cmd}")
    return subprocess.run(cmd, shell=True, check=check)


def parse_args() -> argparse.Namespace:
    cfg = _load_config()
    p = argparse.ArgumentParser(description="Deploy MARDUK to RunPod")
    p.add_argument("--docker-user", default=cfg.get("docker-user"),
                   help="Docker Hub username (saved after first use)")
    p.add_argument("--image-name", default="marduk", help="Docker image name")
    p.add_argument("--image-tag", default="latest", help="Docker image tag")
    p.add_argument("--api-key", help="RunPod API key")
    p.add_argument("--gpu", default="NVIDIA B200",
                   help="GPU type (default: NVIDIA B200)")
    p.add_argument("--gpu-count", type=int, default=1)
    p.add_argument("--volume-size", type=int, default=50,
                   help="Persistent volume in GB")
    p.add_argument("--skip-build", action="store_true",
                   help="Skip Docker build/push")
    p.add_argument("--build-only", action="store_true",
                   help="Only build & push, don't launch pod")
    return p.parse_args()


def docker_build_push(user: str, name: str, tag: str) -> str:
    """Build the Docker image and push to Docker Hub. Returns full image ref."""
    image_local = f"{name}:{tag}"
    image_remote = f"{user}/{name}:{tag}"

    run(f"docker build -t {image_local} .")
    run(f"docker tag {image_local} {image_remote}")
    run(f"docker push {image_remote}")

    print(f"\nImage pushed: {image_remote}")
    return image_remote


def launch_pod(api_key: str, image: str, gpu_type: str, gpu_count: int,
               volume_size: int) -> None:
    """Launch a RunPod GPU pod via the Python SDK."""
    import runpod
    runpod.api_key = api_key

    print(f"\nLaunching RunPod pod...")
    print(f"  Image:  {image}")
    print(f"  GPU:    {gpu_count}x {gpu_type}")
    print(f"  Volume: {volume_size} GB")

    pod = runpod.create_pod(
        name="marduk-train",
        image_name=image,
        gpu_type_id=gpu_type,
        gpu_count=gpu_count,
        volume_in_gb=volume_size,
        ports="8888/http,22/tcp",
        docker_args="",
        # Mount volume at /workspace so data persists across restarts
        volume_mount_path="/workspace",
    )

    pod_id = pod.get("id", "unknown")
    print(f"\n  Pod created!  ID: {pod_id}")
    print(f"  Dashboard: https://www.runpod.io/console/pods/{pod_id}")
    print(f"\n  Pipeline auto-starts. To manually re-run:")
    print(f"    cd /app/marduk && bash runpod_start.sh")


def main() -> None:
    args = parse_args()
    image_ref = f"{args.docker_user}/{args.image_name}:{args.image_tag}" if args.docker_user else None

    # Persist docker-user so it's remembered for next time
    if args.docker_user:
        cfg = _load_config()
        if cfg.get("docker-user") != args.docker_user:
            cfg["docker-user"] = args.docker_user
            _save_config(cfg)
            print(f"Saved docker-user='{args.docker_user}' to .deploy_config")

    # --- Docker build & push ---
    if not args.skip_build:
        if not args.docker_user:
            print("ERROR: --docker-user required for Docker build/push")
            print("  Provide it once and it will be remembered:")
            print("    python deploy_runpod.py --docker-user YOUR_USERNAME ...")
            sys.exit(1)
        # Ensure logged in
        run("docker info > NUL 2>&1", check=False)
        image_ref = docker_build_push(args.docker_user, args.image_name, args.image_tag)

    if args.build_only:
        print("\n--build-only: skipping pod launch.")
        return

    # --- Launch RunPod pod ---
    if not args.api_key:
        print("ERROR: --api-key required to launch a RunPod pod")
        sys.exit(1)
    if not image_ref:
        if not args.docker_user:
            print("ERROR: --docker-user required (need to know image name)")
            sys.exit(1)
        image_ref = f"{args.docker_user}/{args.image_name}:{args.image_tag}"

    launch_pod(args.api_key, image_ref, args.gpu, args.gpu_count, args.volume_size)


if __name__ == "__main__":
    main()
