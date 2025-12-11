import sys
import os
import argparse
from pathlib import Path
import numpy as np

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process entire image as 3D object (no mask needed)")
parser.add_argument(
    "image_path",
    type=str,
    help="Path to input image file (PNG, JPG, etc.)",
    nargs="?",
    default=None,
)
parser.add_argument(
    "--output-dir",
    type=str,
    help="Output directory for PLY and USD files (default: same directory as image)",
    default=None,
)
parser.add_argument(
    "--usd-scale",
    type=float,
    help="USD scale factor (default: 100.0)",
    default=100.0,
)
parser.add_argument(
    "--no-mesh",
    action="store_true",
    help="Skip mesh decoding (faster, but no USD export)",
)
args = parser.parse_args()

# Validate image path
if args.image_path is None:
    print("Error: Please provide an image path")
    parser.print_help()
    sys.exit(1)

image_path = args.image_path

if not os.path.exists(image_path):
    print(f"Error: Image file does not exist: {image_path}")
    sys.exit(1)

# Determine output directory
if args.output_dir is None:
    # Use same directory as image
    output_dir = os.path.dirname(os.path.abspath(image_path))
    if output_dir == "":
        output_dir = "."
else:
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

# Create output subdirectories
ply_dir = os.path.join(output_dir, "ply")
usd_dir = os.path.join(output_dir, "usd")
os.makedirs(ply_dir, exist_ok=True)
os.makedirs(usd_dir, exist_ok=True)

# Get image filename without extension for output naming
image_name = Path(image_path).stem

print(f"{'='*60}")
print(f"Processing entire image as 3D object")
print(f"{'='*60}")
print(f"Input image: {image_path}")
print(f"Output directory: {output_dir}")
print(f"{'='*60}\n")

# Load image
print(f"Loading image: {os.path.basename(image_path)}")
image = load_image(image_path)
print(f"  Image shape: {image.shape}")

# Create full-image mask (all pixels = True)
# The mask should be a 2D boolean array with same height and width as image
height, width = image.shape[:2]
full_mask = np.ones((height, width), dtype=bool)
print(f"  Created full-image mask: {full_mask.shape} (all pixels enabled)")

# Load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
print(f"\nLoading model from: {config_path}")
inference = Inference(config_path, compile=False)
print("Model loaded successfully\n")

# Set output filenames
ply_filename = os.path.join(ply_dir, f"{image_name}_full.ply")
usd_filename = os.path.join(usd_dir, f"{image_name}_full.usd")
usdz_filename = os.path.join(usd_dir, f"{image_name}_full.usdz")

# Check if output files already exist
if os.path.exists(ply_filename) and (os.path.exists(usd_filename) or os.path.exists(usdz_filename)):
    print(f"⚠ Output files already exist:")
    if os.path.exists(ply_filename):
        print(f"  - {ply_filename}")
    if os.path.exists(usd_filename):
        print(f"  - {usd_filename}")
    if os.path.exists(usdz_filename):
        print(f"  - {usdz_filename}")
    print("\nSkipping processing. Delete files to reprocess.")
    sys.exit(0)

# USD export settings
usd_scale_factor = args.usd_scale
embed_textures = True

# Determine decode formats
if args.no_mesh:
    decode_formats = ["gaussian"]  # Skip mesh for faster processing
    usd_path = None  # Can't export USD without mesh
    print("⚠ Mesh decoding disabled - USD export will be skipped")
else:
    decode_formats = ["gaussian", "mesh"]  # Include mesh for USD export
    usd_path = usd_filename

print(f"\nProcessing entire image...")
print(f"  Decode formats: {decode_formats}")
if usd_path:
    print(f"  USD export: {usd_path}")
print()

try:
    # Run inference with full-image mask
    output = inference(
        image,
        full_mask,
        seed=42,
        export_usd_path=usd_path,
        usd_scale_factor=usd_scale_factor,
        embed_textures=embed_textures,
        decode_formats=decode_formats
    )

    # Export gaussian splat
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
            print(f"⚠ USD export failed: Mesh data not available (mesh decoding may have failed or was skipped)")
            print(f"  Hint: USD export requires mesh decoding. If you got OOM errors, try a larger GPU (g5.4xlarge with 48GB)")
        else:
            print(f"⚠ USD export requested but failed; check logs above for error details.")

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  PLY file: {ply_filename}")
    if output.get("usd_path"):
        print(f"  USD file: {output['usd_path']}")
    if output.get("usdz_path"):
        print(f"  USDZ file: {output['usdz_path']}")
    print(f"{'='*60}")

except Exception as e:
    print(f"\n✗ Error processing image: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

