#!/bin/bash
# Quick fix script for nvdiffrast on AWS
# Run this script on your AWS instance to fix nvdiffrast compilation issues

set -e  # Exit on error

echo "=========================================="
echo "nvdiffrast Fix Script for AWS"
echo "=========================================="

# 1. Check CUDA
echo ""
echo "Step 1: Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    echo "✓ nvcc found:"
    nvcc --version
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "  CUDA version: $CUDA_VERSION"
else
    echo "✗ nvcc not found in PATH"
    CUDA_VERSION=""
fi

# 2. Find CUDA installation
echo ""
echo "Step 2: Locating CUDA installation..."
CUDA_PATH=""
for path in /usr/local/cuda-12.9 /usr/local/cuda-12.1 /usr/local/cuda-12.0 /usr/local/cuda; do
    if [ -d "$path" ] && [ -f "$path/bin/nvcc" ]; then
        CUDA_PATH="$path"
        echo "✓ Found CUDA at: $CUDA_PATH"
        break
    fi
done

if [ -z "$CUDA_PATH" ]; then
    # Try to find in /usr
    CUDA_PATH=$(find /usr -name "nvcc" -type f 2>/dev/null | head -1 | xargs dirname | xargs dirname)
    if [ -n "$CUDA_PATH" ]; then
        echo "✓ Found CUDA at: $CUDA_PATH"
    else
        echo "✗ CUDA toolkit not found"
        echo ""
        echo "Installing CUDA toolkit 12.1..."
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
        sudo dpkg -i cuda-keyring_1.0-1_all.deb
        sudo apt-get update
        sudo apt-get -y install cuda-toolkit-12-1
        CUDA_PATH="/usr/local/cuda-12.1"
        echo "✓ CUDA toolkit installed at: $CUDA_PATH"
    fi
fi

# 3. Set environment variables
echo ""
echo "Step 3: Setting CUDA environment variables..."
export CUDA_HOME="$CUDA_PATH"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Add to .bashrc if not already there
if ! grep -q "CUDA_HOME=$CUDA_PATH" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# CUDA environment variables" >> ~/.bashrc
    echo "export CUDA_HOME=$CUDA_PATH" >> ~/.bashrc
    echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "✓ Added to ~/.bashrc"
else
    echo "✓ Already in ~/.bashrc"
fi

echo "  CUDA_HOME=$CUDA_HOME"
echo "  PATH includes: $CUDA_HOME/bin"
echo "  LD_LIBRARY_PATH includes: $CUDA_HOME/lib64"

# 4. Verify CUDA headers
echo ""
echo "Step 4: Checking for CUDA headers..."
if [ -f "$CUDA_PATH/include/cuda_runtime.h" ]; then
    echo "✓ CUDA headers found: $CUDA_PATH/include/cuda_runtime.h"
else
    echo "✗ CUDA headers not found"
    echo "  Installing CUDA development headers..."
    sudo apt-get install -y cuda-cudart-dev-12-1 || sudo apt-get install -y cuda-cudart-dev
    if [ -f "$CUDA_PATH/include/cuda_runtime.h" ]; then
        echo "✓ CUDA headers installed"
    else
        echo "⚠ Warning: CUDA headers still not found. You may need to install full CUDA toolkit."
    fi
fi

# 5. Clear cache and reinstall nvdiffrast
echo ""
echo "Step 5: Clearing cache and reinstalling nvdiffrast..."
rm -rf ~/.cache/torch_extensions
echo "✓ Cleared torch extension cache"

echo "  Uninstalling nvdiffrast..."
pip uninstall nvdiffrast -y || true

echo "  Installing nvdiffrast from source..."
pip install git+https://github.com/NVlabs/nvdiffrast.git

# 6. Test installation
echo ""
echo "Step 6: Testing nvdiffrast installation..."
python << EOF
import sys
try:
    import nvdiffrast.torch as dr
    import torch
    ctx = dr.RasterizeCudaContext()
    print("✓ nvdiffrast is working!")
    sys.exit(0)
except Exception as e:
    print(f"✗ nvdiffrast test failed: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure CUDA_HOME is set correctly")
    print("2. Check that CUDA headers are installed")
    print("3. Try: pip uninstall nvdiffrast && pip install git+https://github.com/NVlabs/nvdiffrast.git")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ nvdiffrast is now working!"
    echo "=========================================="
    echo ""
    echo "You can now run demo.py and USD export should work with full quality."
else
    echo ""
    echo "=========================================="
    echo "✗ nvdiffrast installation failed"
    echo "=========================================="
    echo ""
    echo "Please check the error messages above and refer to:"
    echo "  custom/NVDIFFRAST_INSTALL_FIX.md"
    echo ""
    exit 1
fi

