import sys

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

# run model
# Skip mesh decoding to avoid OOM error - only decode gaussian splatting
# If you need mesh, try a larger GPU instance (g5.4xlarge with 48GB) or reduce input resolution
output = inference(image, mask, seed=42)
# output = inference(image, mask, seed=42, decode_formats=["gaussian"])

# export gaussian splat
output["gs"].save_ply(f"splat.ply")
print("Your reconstruction has been saved to splat.ply")
