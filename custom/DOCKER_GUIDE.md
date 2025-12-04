# Docker Setup Guide for SAM 3D Objects

## Overview

This Dockerfile creates a container with CUDA 12.1, Python 3.11, and the base dependencies for SAM 3D Objects. The code will be mounted as a volume from Windows, so you don't need to copy it into the image.

## Prerequisites

- Docker Desktop with WSL2 backend (Windows)
- NVIDIA GPU with CUDA 12.1 support
- At least 32 GB VRAM (as per SAM 3D Objects requirements)

## Building the Docker Image

Build from the project root directory so the environment file can be copied:

```bash
cd D:\AI\sam-3d-objects
docker build -f custom/Dockerfile -t sam3d-objects:1.0.0 .
```

The Dockerfile will copy `environments/default.yml` to create the conda environment with all CUDA and system dependencies.

## Running the Container

### Basic Run

```bash
docker run --gpus all -it --rm \
  -v D:\AI\sam-3d-objects:/workspace \
  sam3d-objects:latest
```

### With Additional Volume Mounts (for checkpoints, outputs, etc.)

```bash
docker run --gpus all -it --rm \
  -v D:\AI\sam-3d-objects:/workspace \
  -v D:\AI\sam-3d-objects\checkpoints:/workspace/checkpoints \
  -v D:\AI\sam-3d-objects\outputs:/workspace/outputs \
  sam3d-objects:latest
```

## Dependencies

All dependencies are pre-installed in the image during build:
- Conda environment with CUDA/system dependencies from `environments/default.yml`
- PyTorch 2.5.1 with CUDA 12.1 support
- All pip dependencies from `requirements*.txt` files
- Hydra patch applied

The code will be mounted as a volume, so any code changes are immediately available without rebuilding the image.

## Notes

- The container uses CUDA 12.1 base image
- Python 3.11 is installed via conda
- The conda environment `sam3d-objects` is automatically activated on container start
- Code is mounted from Windows, so changes are reflected immediately
- PyTorch with CUDA 12.1 support is pre-installed

## Troubleshooting

### GPU Not Detected
Make sure Docker Desktop has GPU support enabled and you're using `--gpus all` flag.

### CUDA Version Mismatch
Ensure your host NVIDIA driver supports CUDA 12.1. Check with `nvidia-smi`.

### Permission Issues
If you encounter permission issues with mounted volumes, you may need to adjust file permissions or use Docker's user mapping.

