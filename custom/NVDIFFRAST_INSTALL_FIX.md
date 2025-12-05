# Fixing nvdiffrast CUDA Compilation Issue

## Problem

nvdiffrast is installed but fails to compile/load its CUDA extensions with errors like:
- `cuda_runtime.h: No such file or directory`
- `cannot open shared object file: No such file or directory`

This happens because nvdiffrast requires CUDA development headers to compile its extensions.

## Solution

### Step 1: Check Current CUDA Installation

```bash
# Check CUDA version
nvcc --version

# Check if CUDA is in PATH
echo $CUDA_HOME
echo $LD_LIBRARY_PATH

# Check PyTorch CUDA version
python -c "import torch; print(f'PyTorch CUDA version: {torch.version.cuda}')"
```

### Step 2: Install CUDA Toolkit (if not installed)

On AWS Ubuntu, install CUDA toolkit:

```bash
# For CUDA 12.1 (common for PyTorch 2.x)
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# Or use package manager (Ubuntu)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-1
```

### Step 3: Set CUDA Environment Variables

Add to `~/.bashrc` or run in current session:

```bash
# For CUDA 12.1
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# For CUDA 12.9 (if using that version)
# export CUDA_HOME=/usr/local/cuda-12.9
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Reload environment
source ~/.bashrc
```

### Step 4: Install CUDA Development Headers (Alternative)

If you don't want to install full CUDA toolkit, install just the headers:

```bash
# For CUDA 12.1
sudo apt-get install -y cuda-cudart-dev-12-1

# Or install via pip (may work for some cases)
pip install "nvidia-cuda-runtime-cu12==12.1.*"
```

### Step 5: Reinstall nvdiffrast from Source

```bash
# Uninstall existing nvdiffrast
pip uninstall nvdiffrast -y

# Clear torch extension cache (important!)
rm -rf ~/.cache/torch_extensions

# Clone and install from source
cd /tmp
git clone https://github.com/NVlabs/nvdiffrast.git
cd nvdiffrast
pip install .

# Or install directly
pip install git+https://github.com/NVlabs/nvdiffrast.git
```

### Step 6: Verify Installation

```bash
python -c "import nvdiffrast.torch as dr; ctx = dr.RasterizeCudaContext(); print('nvdiffrast working!')"
```

## Quick Fix Script for AWS

Run this on your AWS instance:

```bash
#!/bin/bash
# Quick fix for nvdiffrast on AWS

# 1. Check CUDA
echo "Checking CUDA..."
nvcc --version || echo "CUDA not found in PATH"

# 2. Find CUDA installation
CUDA_PATH=$(find /usr/local -name "cuda-*" -type d 2>/dev/null | head -1)
if [ -z "$CUDA_PATH" ]; then
    CUDA_PATH=$(find /usr -name "cuda" -type d 2>/dev/null | head -1)
fi

if [ -n "$CUDA_PATH" ]; then
    echo "Found CUDA at: $CUDA_PATH"
    export CUDA_HOME=$CUDA_PATH
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    echo "export CUDA_HOME=$CUDA_PATH" >> ~/.bashrc
    echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
else
    echo "CUDA not found. Installing CUDA toolkit..."
    # Install CUDA 12.1
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-1
    export CUDA_HOME=/usr/local/cuda-12.1
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
fi

# 3. Clear cache and reinstall
echo "Clearing cache and reinstalling nvdiffrast..."
rm -rf ~/.cache/torch_extensions
pip uninstall nvdiffrast -y
pip install git+https://github.com/NVlabs/nvdiffrast.git

# 4. Test
echo "Testing nvdiffrast..."
python -c "import nvdiffrast.torch as dr; ctx = dr.RasterizeCudaContext(); print('✓ nvdiffrast working!')" || echo "✗ nvdiffrast still not working"
```

## Alternative: Use Pre-compiled Extensions

If compilation still fails, you can try to use a pre-compiled version or disable texture baking:

1. **Disable texture baking** (reduces quality but works without nvdiffrast):
   - Set `with_texture_baking=False` in the code
   - This will skip the texture baking step that requires nvdiffrast

2. **Use PyTorch3D rendering** (if available):
   - The code already supports `rendering_engine="pytorch3d"` as an alternative
   - But texture baking still requires nvdiffrast

## Notes

- nvdiffrast requires CUDA toolkit (not just runtime) for compilation
- The torch extension cache (`~/.cache/torch_extensions`) should be cleared before reinstalling
- Make sure CUDA version matches PyTorch's CUDA version
- On AWS, you may need to install CUDA toolkit via package manager or download from NVIDIA

## Verification

After installation, test with:

```python
import nvdiffrast.torch as dr
import torch

# Test RastContext creation
ctx = dr.RasterizeCudaContext()
print("nvdiffrast is working!")
```

If this works, your USD export should now work with full quality (hole filling and texture baking).

