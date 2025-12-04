# PLY Color Fix for MeshLab

## Problem

When opening `splat.ply` files in MeshLab, the point cloud appears as a single color instead of showing the original image colors.

## Root Cause

The issue was **not** with:
- ❌ MeshLab (it's working correctly)
- ❌ The inference pipeline (colors are generated correctly)
- ❌ The demo.py save code (it calls the right method)

The issue was with **how colors are saved in the PLY file**:

1. **Gaussian Splatting uses Spherical Harmonics (SH) coefficients** to store colors, not direct RGB values
2. The PLY file was saving SH coefficients (`f_dc_0`, `f_dc_1`, `f_dc_2`) instead of RGB colors
3. **MeshLab expects standard RGB vertex colors** (`red`, `green`, `blue` attributes)
4. MeshLab doesn't know how to interpret SH coefficients, so it displays a default single color

## Solution

Modified `save_ply()` method in `sam3d_objects/model/backbone/tdfy_dit/representations/gaussian/gaussian_model.py` to:

1. **Convert SH coefficients to RGB colors** using the standard Gaussian Splatting conversion:
   - For `sh_degree=0`: `RGB = 0.5 + 0.282095 * f_dc`
   - This matches how the renderer converts SH to RGB (see `gaussian_render.py`)

2. **Add RGB color attributes** to the PLY file:
   - Added `red`, `green`, `blue` attributes (uint8, 0-255 range)
   - These are standard PLY format attributes that MeshLab can read

3. **Preserve original SH coefficients** - The original `f_dc_*` attributes are still saved for compatibility with Gaussian Splatting viewers

## Changes Made

**File:** `sam3d_objects/model/backbone/tdfy_dit/representations/gaussian/gaussian_model.py`

- Added SH to RGB conversion before saving
- Added RGB color attributes to PLY file structure
- Colors are now properly visible in MeshLab

## Testing

After regenerating the PLY file:

1. Run `demo.py` again on your AWS instance
2. Download the new `splat.ply` file
3. Open it in MeshLab - colors should now match the original image!

## Technical Details

- **SH Coefficients**: Spherical harmonics are used for view-dependent color representation
- **Conversion Formula**: `RGB = clamp(0.5 + 0.282095 * f_dc, 0.0, 1.0)`
- **PLY Format**: Standard vertex colors use `red`, `green`, `blue` as uint8 attributes
- **Compatibility**: The fix maintains backward compatibility - original SH coefficients are still saved

## Alternative Viewers

If you still have issues with MeshLab, you can also use:
- **CloudCompare** - Good PLY viewer with color support
- **Open3D** - Python library for 3D visualization
- **Gaussian Splatting viewers** - Specialized viewers that understand SH coefficients (e.g., SIBR viewer)

