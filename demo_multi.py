import sys
import os
import argparse
from pathlib import Path

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_masks, make_scene

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process multiple objects from image and combine into single PLY")
parser.add_argument(
    "image_path",
    type=str,
    help="Path to input image file (PNG, JPG, etc.)",
    nargs="?",
    default=None,
)
parser.add_argument(
    "--masks-dir",
    type=str,
    help="Path to masks directory (default: masks/ subfolder in same directory as image)",
    default=None,
)
parser.add_argument(
    "--output",
    type=str,
    help="Output PLY file path (default: {image_name}_multi.ply in same directory as image)",
    default=None,
)
parser.add_argument(
    "--skip-existing",
    action="store_true",
    help="Skip processing if output file already exists",
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

# Determine masks directory
if args.masks_dir is None:
    # Use masks/ subfolder in same directory as image
    image_dir = os.path.dirname(os.path.abspath(image_path))
    masks_dir = os.path.join(image_dir, "masks")
else:
    masks_dir = args.masks_dir

if not os.path.exists(masks_dir):
    print(f"Error: Masks directory does not exist: {masks_dir}")
    print("Expected structure: image.png and masks/ subfolder with mask files (0.png, 1.png, ...)")
    sys.exit(1)

# Determine output file path
if args.output is None:
    # Use same directory as image
    output_dir = os.path.dirname(os.path.abspath(image_path))
    if output_dir == "":
        output_dir = "."
    image_name = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{image_name}_multi.ply")
else:
    output_path = args.output
    # Create output directory if needed
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

# Check if output already exists
if args.skip_existing and os.path.exists(output_path):
    print(f"Output file already exists: {output_path}")
    print("Skipping processing. Use without --skip-existing to reprocess.")
    sys.exit(0)

print(f"{'='*60}")
print(f"Processing multiple objects and combining into single PLY")
print(f"{'='*60}")
print(f"Input image: {image_path}")
print(f"Masks directory: {masks_dir}")
print(f"Output PLY: {output_path}")
print(f"{'='*60}\n")

# Load image
print(f"Loading image: {os.path.basename(image_path)}")
image = load_image(image_path)
print(f"  Image shape: {image.shape}")

# Load all masks
print(f"\nLoading masks from: {masks_dir}")
masks = load_masks(masks_dir, indices_list=None, extension=".png")
num_masks = len(masks)
if num_masks == 0:
    print(f"Error: No masks found in {masks_dir}")
    print("Expected mask files: 0.png, 1.png, 2.png, ...")
    sys.exit(1)

print(f"  Found {num_masks} mask(s) to process")

# Load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
print(f"\nLoading model from: {config_path}")
inference = Inference(config_path, compile=False)
print("Model loaded successfully\n")

# Process each mask
print(f"Processing {num_masks} object(s)...")
outputs = []
processed_count = 0
failed_count = 0

for idx, mask in enumerate(masks):
    print(f"  Processing mask {idx}/{num_masks-1} (mask_{idx}.png)...", end=" ", flush=True)
    
    try:
        # Run inference for this mask
        output = inference(image, mask, seed=42, decode_formats=["gaussian"])
        outputs.append(output)
        processed_count += 1
        print("✓")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        failed_count += 1
        import traceback
        traceback.print_exc()
        continue

if processed_count == 0:
    print(f"\nError: Failed to process any masks")
    sys.exit(1)

print(f"\n  Processed: {processed_count}/{num_masks} masks")
if failed_count > 0:
    print(f"  Failed: {failed_count} masks")

# Combine all objects into a single scene
print(f"\nCombining {processed_count} object(s) into single scene...")
try:
    scene_gs = make_scene(*outputs)
    print("  Scene combined successfully")
except Exception as e:
    print(f"✗ Error combining scene: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Save combined PLY file
print(f"\nSaving combined PLY file...")
try:
    scene_gs.save_ply(output_path)
    print(f"✓ Saved: {output_path}")
except Exception as e:
    print(f"✗ Error saving PLY file: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n{'='*60}")
print(f"Processing complete!")
print(f"  Input masks: {num_masks}")
print(f"  Processed: {processed_count}")
print(f"  Failed: {failed_count}")
print(f"  Output PLY: {output_path}")
print(f"{'='*60}")

