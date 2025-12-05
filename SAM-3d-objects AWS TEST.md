SAM-3d-objects AWS TEST

ssh -i eks.pem ubuntu@ec2-98-87-168-8.compute-1.amazonaws.com

git clone --recursive https://github.com/facebookresearch/sam-3d-objects.git
cd sam-3d-objects

# 24.04
sudo apt-get update && sudo apt install -y python3-pip

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda --version

conda env create -f environments/default.yml
conda activate sam3d-objects

export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

pip install -e '.[dev]'
pip install -e '.[p3d]'
pip install -e '.[inference]'
./patching/hydra


pip install 'huggingface-hub[cli]<1.0'

TAG=hf
hf download \
  --repo-type model \
  --local-dir checkpoints/${TAG}-download \
  --max-workers 1 \
  facebook/sam-3d-objects
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download



# Make sure you're in the directory where eks.pem is located
cd D:\AI\sam-3d-objects

# Download the splat.ply file
scp -i eks.pem ubuntu@ec2-54-87-137-237.compute-1.amazonaws.com:~/sam-3d-objects/splat.ply ./