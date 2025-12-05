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
    help="Path to folder containing image.png/jpg and masks/ subfolder with mask files (0.png, 1.png, ...)",
    default="notebook/images/shutterstock_stylish_kidsroom_1640806567",
    nargs="?",
)
args = parser.parse_args()

# Image and mask folder path
image_folder = args.image_folder

# Validate image folder exists
if not os.path.exists(image_folder):
    print(f"Error: Image folder does not exist: {image_folder}")
    sys.exit(1)

# Find image file (supports both PNG and JPG)
image_path = None
for ext in [".png", ".jpg", ".PNG", ".JPG", ".jpeg", ".JPEG"]:
    candidate = os.path.join(image_folder, f"image{ext}")
    if os.path.exists(candidate):
        image_path = candidate
        break

if image_path is None:
    print(f"Error: image.png or image.jpg not found in folder: {image_folder}")
    sys.exit(1)

print(f"Found image: {image_path}")

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
print(f"Loading image from: {image_path}")
image = load_image(image_path)

# Find all masks in the masks/ subfolder (0.png, 1.png, 2.png, ...)
masks_folder = os.path.join(image_folder, "masks")
if not os.path.exists(masks_folder):
    print(f"Error: Masks folder does not exist: {masks_folder}")
    print("Please run sam3_mask_generator.py first to generate masks, or create the masks/ folder manually")
    sys.exit(1)

print(f"Scanning for masks in: {masks_folder}")
masks = load_masks(masks_folder, indices_list=None, extension=".png")
num_masks = len(masks)
print(f"Found {num_masks} masks to process")

# Create output directories in image folder (ply/ and usd/ subfolders)
ply_dir = os.path.join(image_folder, "ply")
usd_dir = os.path.join(image_folder, "usd")
os.makedirs(ply_dir, exist_ok=True)
os.makedirs(usd_dir, exist_ok=True)
print(f"PLY output directory: {ply_dir}")
print(f"USD output directory: {usd_dir}")

# USD export settings
usd_scale_factor = 100.0  # Adjust the scale factor to match your scene units (default 100)
embed_textures = True

# Process each mask one by one
for idx, mask in enumerate(masks):
    print(f"\n{'='*60}")
    print(f"Processing mask {idx}/{num_masks-1} (mask_{idx}.png)")
    print(f"{'='*60}")
    
    try:
        # Set USD export path for this mask
        usd_path = os.path.join(usd_dir, f"reconstruction_mask_{idx}.usd")
        
        # run model with USD export support
        # USD export requires mesh decoding, so we need to include "mesh" in decode_formats
        # Note: This may cause OOM errors on smaller GPUs - try g5.4xlarge (48GB) if needed
        output = inference(
            image, 
            mask, 
            seed=42,
            export_usd_path=usd_path,
            usd_scale_factor=usd_scale_factor,
            embed_textures=embed_textures,
            decode_formats=["gaussian", "mesh"]  # Include mesh for USD export
        )

        # export gaussian splat with unique filename
        ply_filename = os.path.join(ply_dir, f"splat_mask_{idx}.ply")
        output["gs"].save_ply(ply_filename)
        print(f"✓ Saved PLY: {ply_filename}")
        
        # Check if USD export was successful
        if output.get("usd_path"):
            print(f"✓ Saved USD: {output['usd_path']}")
            if output.get("usdz_path"):
                print(f"✓ Saved USDZ: {output['usdz_path']}")
        elif usd_path is not None:
            # Diagnose why USD export failed
            if "mesh" not in output:
                print(f"⚠ USD export failed for mask {idx}: Mesh data not available (mesh decoding may have failed or was skipped)")
                print(f"  Hint: USD export requires mesh decoding. If you got OOM errors, try a larger GPU (g5.4xlarge with 48GB)")
            else:
                print(f"⚠ USD export requested but failed for mask {idx}; check logs above for error details.")
        
    except Exception as e:
        print(f"✗ Error processing mask {idx}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'='*60}")
print(f"Processing complete! Generated {num_masks} PLY files in '{ply_dir}' directory")
print(f"USD files saved in '{usd_dir}' directory (if export was successful)")
print(f"{'='*60}")
