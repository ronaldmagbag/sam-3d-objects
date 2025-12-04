# SAM 3D Objects Dockerfile
# Base image with CUDA 12.1 support
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/opt/conda/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda (mamba-compatible)
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean --all -y && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Accept conda Terms of Service and install mamba
RUN /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    /opt/conda/bin/conda install -y -c conda-forge mamba

# Set working directory (code will be mounted as volume)
WORKDIR /workspace

# Copy environment file needed for conda environment creation
COPY environments/default.yml /tmp/default.yml

# Create conda environment from default.yml with all CUDA/system dependencies
SHELL ["/bin/bash", "-c"]
RUN source /opt/conda/etc/profile.d/conda.sh && \
    /opt/conda/bin/mamba env create -f /tmp/default.yml && \
    /opt/conda/bin/mamba clean -afy && \
    rm /tmp/default.yml

# Set up pip indexes for PyTorch and NVIDIA packages
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

# Install PyTorch with CUDA 12.1 support
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate sam3d-objects && \
    pip install --no-cache-dir torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
# Copy dependency files needed for installation
# Build context should be project root: docker build -f custom/Dockerfile .
COPY pyproject.toml /workspace/
COPY requirements*.txt /workspace/
COPY patching/ /workspace/patching/
COPY sam3d_objects/ /workspace/sam3d_objects/

# Install sam3d-objects and all pip dependencies
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate sam3d-objects && \
    cd /workspace && \
    pip install --no-cache-dir -e '.[dev]' && \
    pip install --no-cache-dir -e '.[p3d]' && \
    pip install --no-cache-dir -e '.[inference]' && \
    python /workspace/patching/hydra

# Create entrypoint script
RUN echo '#!/bin/bash\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate sam3d-objects\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["/bin/bash"]

