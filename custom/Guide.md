## Project Summary: SAM 3D Objects

### What is it?
SAM 3D Objects is a foundation model from Meta that reconstructs 3D geometry, texture, and layout from a single image. It handles occlusion and clutter and supports single and multi-object reconstruction.

### Model Architecture
- DiT (Diffusion Transformer) with flow matching
- Sparse Transformer blocks for efficient processing
- Multiple 3D representations:
  - Gaussian Splatting (primary output)
  - Mesh (FlexiCubes)
  - Radiance Fields
- Condition embedders for image conditioning
- Structured latent VAE for encoding/decoding

### System Requirements

Operating System:
- Linux 64-bit (required)
- The conda environment file targets `linux-64`

GPU Requirements:
- NVIDIA GPU with at least 32 GB VRAM
- CUDA 12.1 support required
- The project uses CUDA 12.1 toolkits and libraries

### Installation Steps

1. Prerequisites:
   - Linux 64-bit system
   - NVIDIA GPU with ≥32 GB VRAM
   - CUDA 12.1 installed
   - Mamba or Conda package manager

2. Setup Python Environment:
```bash
# Create sam3d-objects environment
mamba env create -f environments/default.yml
mamba activate sam3d-objects

# Set up PyTorch/CUDA dependencies
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"

# Install sam3d-objects and core dependencies
pip install -e '.[dev]'
pip install -e '.[p3d]'  # pytorch3d dependency

# For inference
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'

# Apply patches
./patching/hydra
```

3. Download Checkpoints:
   - Request access to the [HuggingFace repo](https://huggingface.co/facebook/sam-3d-objects)
   - Authenticate: `hf auth login`
   - Download checkpoints:
```bash
pip install 'huggingface-hub[cli]<1.0'
TAG=hf
hf download --repo-type model --local-dir checkpoints/${TAG}-download --max-workers 1 facebook/sam-3d-objects
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download
```

4. Quick Start:
```bash
python demo.py
```

### Key Dependencies
- PyTorch with CUDA 12.1
- PyTorch3D
- Kaolin
- GSplat (Gaussian Splatting)
- Flash Attention
- Various 3D processing libraries (Open3D, PyMeshFix, etc.)

### Important Notes
- Linux-only: Windows is not supported
- High VRAM requirement: 32 GB minimum
- Checkpoint access: Requires HuggingFace approval
- CUDA 12.1: Must match the specified CUDA version

The project is designed for research and production use in 3D reconstruction from images, with a focus on robustness in real-world scenarios.