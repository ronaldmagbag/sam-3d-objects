# USD Exporter Dependencies

## Overview

The USD exporter feature allows exporting 3D meshes to USD (Universal Scene Description) format, which is widely used in 3D graphics pipelines and AR/VR applications.

## Required Dependencies

### Primary Dependency: `usd-core`

The USD exporter requires the `usd-core` Python package, which provides the `pxr` module (Pixar's USD Python bindings).

**Installation:**
```bash
pip install usd-core
```

**Note:** `usd-core` is a large package (~25-40MB) and may take some time to install. It's available for:
- Linux (x86_64)
- macOS (universal)
- Windows (AMD64)

### Python Version Compatibility

The `usd-core` package is available for Python 3.8-3.12. Check the [usd-core PyPI page](https://pypi.org/project/usd-core/) for the latest version and compatibility.

## Imported Modules

The USD exporter uses the following modules from `pxr`:

### For USD Export (`to_usd` function):
- `Usd` - Core USD stage and primitives
- `UsdGeom` - Geometry primitives (meshes, transforms)
- `UsdShade` - Material and shader definitions
- `Sdf` - Scene Description Foundation (data types, paths)
- `Gf` - Graphics Foundation (vectors, matrices)

### For USDZ Export (`to_usdz` function):
- `UsdUtils` - USD utilities (packaging, archiving)
- `Sdf` - Scene Description Foundation

## Error Handling

The code includes error handling that will raise an `ImportError` if `usd-core` is not installed:

```python
ImportError: USD export requires the 'pxr' package (e.g. `pip install usd-core`).
```

If USD export fails, the code will:
1. Log the error
2. Set `outputs["usd_path"] = None`
3. Continue processing (won't crash the entire pipeline)

## Features

### USD Export Features:
- ✅ Mesh geometry (vertices, faces)
- ✅ UV coordinates
- ✅ Vertex colors
- ✅ Texture mapping (with embedded PNG textures)
- ✅ PBR materials (UsdPreviewSurface)
- ✅ Configurable scale factor
- ✅ Optional texture embedding

### USDZ Export:
- ✅ Automatic USDZ packaging (single-file archive)
- ✅ Includes referenced textures
- ✅ Portable format for AR/VR applications

## Usage in Code

### In `demo.py`:
```python
output = inference(
    image, 
    mask, 
    seed=42,
    export_usd_path=usd_path,           # Path to output USD file
    usd_scale_factor=usd_scale_factor,   # Scale factor (default: 100.0)
    embed_textures=embed_textures       # Embed textures (default: True)
)
```

### In `inference.py`:
```python
def __call__(
    self,
    image: Union[Image.Image, np.ndarray],
    mask: Optional[Union[None, Image.Image, np.ndarray]],
    seed: Optional[int] = None,
    pointmap=None,
    decode_formats=None,
    export_usd_path: Optional[str] = None,  # USD export path
    usd_scale_factor: float = 100.0,         # Scale factor
    embed_textures: bool = True,             # Embed textures
) -> dict:
```

## Output Files

When USD export is enabled, the following files are generated:

1. **`.usd` file** - Main USD file with mesh and materials
2. **`_albedo.png`** - Texture file (if `embed_textures=True`)
3. **`.usdz` file** - Packaged USDZ archive (optional, created automatically)

All files are saved in the `usd/` subfolder inside the image folder.

## Installation Instructions

### For New Installations:
```bash
# Install usd-core
pip install usd-core

# Verify installation
python -c "from pxr import Usd; print('USD installed successfully')"
```

### For Existing Installations:
```bash
# Add to requirements.txt or environment.yml
echo "usd-core" >> requirements.txt

# Or install directly
pip install usd-core
```

## Troubleshooting

### Import Error
If you get `ImportError: USD export requires the 'pxr' package`:
- Install: `pip install usd-core`
- Verify: `python -c "from pxr import Usd"`

### USD Export Fails Silently
- Check logs for error messages
- Verify `usd_path` is writable
- Ensure mesh data is available (requires mesh decoding, not just Gaussian splatting)

### Large File Sizes
- USD files can be large, especially with embedded textures
- Consider disabling texture embedding: `embed_textures=False`
- Use USDZ for better compression

## Platform-Specific Notes

### Linux
- Requires glibc 2.17+ (most modern distributions)
- May need additional system libraries (check usd-core documentation)

### macOS
- Universal binary supports both Intel and Apple Silicon
- No additional dependencies required

### Windows
- Requires Visual C++ Redistributables
- May need to install from wheel file directly

## Version Information

The code uses `usd-core` version 25.11 (as seen in uv.lock). Check for updates:
```bash
pip install --upgrade usd-core
```

## References

- [USD Documentation](https://openusd.org/)
- [usd-core PyPI](https://pypi.org/project/usd-core/)
- [Pixar USD GitHub](https://github.com/PixarAnimationStudios/USD)

