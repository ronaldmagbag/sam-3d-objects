# nvdiffrast Missing Error Fix

## Problem

When trying to export USD files, the code fails with:
```
ModuleNotFoundError: No module named 'nvdiffrast'
```

This happens because USD export requires mesh postprocessing, which uses `nvdiffrast` for hole filling.

## Root Cause

1. USD export requires mesh data
2. Mesh postprocessing (`with_mesh_postprocess=True`) is enabled for USD export
3. Hole filling in mesh postprocessing requires `nvdiffrast`
4. `nvdiffrast` is not installed on the system

## Solution Applied

Updated `postprocessing_utils.py` to:
1. **Check for nvdiffrast availability** before attempting hole filling
2. **Skip hole filling gracefully** if nvdiffrast is not available
3. **Continue with mesh processing** without hole filling (mesh will still be usable for USD export)

## Installation Option (Optional)

If you want hole filling functionality, install nvdiffrast:

```bash
git clone https://github.com/NVlabs/nvdiffrast.git
cd nvdiffrast
pip install .
```

**Note:** nvdiffrast requires CUDA and may need to be compiled from source.

## What Changed

**File:** `sam3d_objects/model/backbone/tdfy_dit/utils/postprocessing_utils.py`

- Added nvdiffrast availability check in `postprocess_mesh()`
- Skips hole filling if nvdiffrast is not available
- Shows warning message but continues processing
- Mesh will still be generated and can be exported to USD (just without hole filling)

## Impact

- ✅ USD export will now work without nvdiffrast
- ✅ Mesh quality may be slightly lower (no hole filling)
- ✅ No crashes - graceful degradation
- ⚠️ For best quality, install nvdiffrast

## Testing

After updating the code, USD export should work. The mesh may have some holes, but it will be exportable to USD format.

