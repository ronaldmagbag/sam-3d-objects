import sys
import os
import argparse
from pathlib import Path

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_masks

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process 3D objects from image and masks")
parser.add_argument(
    "parent_folder",
    type=str,
    help="Path to parent folder containing image subfolders, or a single image folder. Each subfolder should contain image.png/jpg and masks/ subfolder with mask files (0.png, 1.png, ...)",
    default="notebook/images/shutterstock_stylish_kidsroom_1640806567",
    nargs="?",
)
args = parser.parse_args()

# Parent folder path
parent_folder = args.parent_folder

# Validate parent folder exists
if not os.path.exists(parent_folder):
    print(f"Error: Folder does not exist: {parent_folder}")
    sys.exit(1)

def process_image_folder(image_folder, inference):
    """Process a single image folder with its masks."""
    # Find image file (supports both PNG and JPG)
    image_path = None
    for ext in [".png", ".jpg", ".PNG", ".JPG", ".jpeg", ".JPEG"]:
        candidate = os.path.join(image_folder, f"image{ext}")
        if os.path.exists(candidate):
            image_path = candidate
            break

    if image_path is None:
        print(f"  ⚠ Skipping {image_folder}: image.png or image.jpg not found")
        return

    print(f"  Found image: {os.path.basename(image_path)}")

    # Load image (RGBA only, mask is embedded in the alpha channel)
    image = load_image(image_path)

    # Find all masks in the masks/ subfolder (0.png, 1.png, 2.png, ...)
    masks_folder = os.path.join(image_folder, "masks")
    if not os.path.exists(masks_folder):
        print(f"  ⚠ Skipping {image_folder}: Masks folder does not exist")
        return

    masks = load_masks(masks_folder, indices_list=None, extension=".png")
    num_masks = len(masks)
    if num_masks == 0:
        print(f"  ⚠ Skipping {image_folder}: No masks found")
        return

    print(f"  Found {num_masks} masks to process")

    # Create output directories in image folder (ply/ and usd/ subfolders)
    ply_dir = os.path.join(image_folder, "ply")
    usd_dir = os.path.join(image_folder, "usd")
    os.makedirs(ply_dir, exist_ok=True)
    os.makedirs(usd_dir, exist_ok=True)

    # USD export settings
    usd_scale_factor = 100.0  # Adjust the scale factor to match your scene units (default 100)
    embed_textures = True

    # Process each mask one by one
    processed_count = 0
    skipped_count = 0
    for idx, mask in enumerate(masks):
        # Check if output files already exist
        ply_filename = os.path.join(ply_dir, f"splat_mask_{idx}.ply")
        usd_filename = os.path.join(usd_dir, f"reconstruction_mask_{idx}.usd")
        usdz_filename = os.path.join(usd_dir, f"reconstruction_mask_{idx}.usdz")
        
        # Skip if both PLY and USD exist
        if os.path.exists(ply_filename) and (os.path.exists(usd_filename) or os.path.exists(usdz_filename)):
            print(f"  ⊘ Skipping mask {idx}: Output files already exist")
            skipped_count += 1
            continue
        
        print(f"  Processing mask {idx}/{num_masks-1} (mask_{idx}.png)")
        
        try:
            # Set USD export path for this mask
            usd_path = usd_filename
            
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
            output["gs"].save_ply(ply_filename)
            print(f"    ✓ Saved PLY: {os.path.basename(ply_filename)}")
            
            # Check if USD export was successful
            if output.get("usd_path"):
                print(f"    ✓ Saved USD: {os.path.basename(output['usd_path'])}")
                if output.get("usdz_path"):
                    print(f"    ✓ Saved USDZ: {os.path.basename(output['usdz_path'])}")
            elif usd_path is not None:
                # Diagnose why USD export failed
                if "mesh" not in output:
                    print(f"    ⚠ USD export failed for mask {idx}: Mesh data not available (mesh decoding may have failed or was skipped)")
                    print(f"      Hint: USD export requires mesh decoding. If you got OOM errors, try a larger GPU (g5.4xlarge with 48GB)")
                else:
                    print(f"    ⚠ USD export requested but failed for mask {idx}; check logs above for error details.")
            
            processed_count += 1
            
        except Exception as e:
            print(f"    ✗ Error processing mask {idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"  ✓ Completed: {processed_count} processed, {skipped_count} skipped")
    return processed_count, skipped_count


# Determine if parent_folder is a single image folder or contains subfolders
parent_path = Path(parent_folder)

# Check if it's a single image folder (has image.png/jpg directly)
is_single_folder = False
for ext in [".png", ".jpg", ".PNG", ".JPG", ".jpeg", ".JPEG"]:
    if (parent_path / f"image{ext}").exists():
        is_single_folder = True
        break

# Load model once (shared across all image folders)
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
print(f"Loading model from: {config_path}")
inference = Inference(config_path, compile=False)
print("Model loaded successfully\n")

total_processed = 0
total_skipped = 0

if is_single_folder:
    # Process as a single image folder
    print(f"{'='*60}")
    print(f"Processing single image folder: {parent_folder}")
    print(f"{'='*60}")
    processed, skipped = process_image_folder(parent_folder, inference)
    if processed is not None:
        total_processed += processed
        total_skipped += skipped
else:
    # Process as parent folder with subfolders
    print(f"{'='*60}")
    print(f"Processing parent folder: {parent_folder}")
    print(f"Looking for image subfolders...")
    print(f"{'='*60}\n")
    
    # Find all subfolders that might contain images
    image_folders = []
    for item in parent_path.iterdir():
        if item.is_dir():
            # Check if this subfolder has an image file
            for ext in [".png", ".jpg", ".PNG", ".JPG", ".jpeg", ".JPEG"]:
                if (item / f"image{ext}").exists():
                    image_folders.append(item)
                    break
    
    if not image_folders:
        print(f"Error: No image subfolders found in {parent_folder}")
        print("Expected structure: parent_folder/subfolder1/image.png, parent_folder/subfolder2/image.png, ...")
        sys.exit(1)
    
    print(f"Found {len(image_folders)} image subfolder(s)\n")
    
    # Process each image folder
    for idx, image_folder in enumerate(sorted(image_folders)):
        print(f"\n{'='*60}")
        print(f"Processing folder {idx+1}/{len(image_folders)}: {image_folder.name}")
        print(f"{'='*60}")
        
        processed, skipped = process_image_folder(str(image_folder), inference)
        if processed is not None:
            total_processed += processed
            total_skipped += skipped

print(f"\n{'='*60}")
print(f"All processing complete!")
print(f"  Total processed: {total_processed} masks")
print(f"  Total skipped: {total_skipped} masks (already existed)")
print(f"{'='*60}")
