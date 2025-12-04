import sys
import os
import argparse

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_masks

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process 3D objects from image and masks")
parser.add_argument(
    "image_folder",
    type=str,
    help="Path to folder containing image.png and mask files (0.png, 1.png, ...)",
    default="notebook/images/shutterstock_stylish_kidsroom_1640806567",
    nargs="?",
)
args = parser.parse_args()

# Image and mask folder path
image_folder = args.image_folder
image_path = os.path.join(image_folder, "image.png")

# Validate image folder exists
if not os.path.exists(image_folder):
    print(f"Error: Image folder does not exist: {image_folder}")
    sys.exit(1)

# Validate image.png exists
if not os.path.exists(image_path):
    print(f"Error: image.png not found in folder: {image_folder}")
    sys.exit(1)

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
print(f"Loading image from: {image_path}")
image = load_image(image_path)

# Find all masks in the folder (0.png, 1.png, 2.png, ...)
print(f"Scanning for masks in: {image_folder}")
masks = load_masks(image_folder, indices_list=None, extension=".png")
num_masks = len(masks)
print(f"Found {num_masks} masks to process")

# Create output directory if it doesn't exist
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Process each mask one by one
for idx, mask in enumerate(masks):
    print(f"\n{'='*60}")
    print(f"Processing mask {idx}/{num_masks-1} (mask_{idx}.png)")
    print(f"{'='*60}")
    
    try:
        # run model
        # Skip mesh decoding to avoid OOM error - only decode gaussian splatting
        # If you need mesh, try a larger GPU instance (g5.4xlarge with 48GB) or reduce input resolution
        output = inference(image, mask, seed=42)
        # output = inference(image, mask, seed=42, decode_formats=["gaussian"])

        # export gaussian splat with unique filename
        output_filename = os.path.join(output_dir, f"splat_mask_{idx}.ply")
        output["gs"].save_ply(output_filename)
        print(f"✓ Saved: {output_filename}")
        
    except Exception as e:
        print(f"✗ Error processing mask {idx}: {str(e)}")
        continue

print(f"\n{'='*60}")
print(f"Processing complete! Generated {num_masks} splat files in '{output_dir}' directory")
print(f"{'='*60}")
