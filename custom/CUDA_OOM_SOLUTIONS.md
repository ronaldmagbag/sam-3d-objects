# CUDA Out-of-Memory Error Solutions

## Error Explanation

The error you're encountering is a **CUDA out-of-memory (OOM) error** during the mesh decoding phase:

```
RuntimeError: cuda failed with error 2 out of memory
```

**What's happening:**
- The error occurs in `decode_slat()` when trying to decode the mesh representation
- The mesh decoder uses sparse convolutions (spconv) which are very memory-intensive
- Even with 24GB GPU memory (g5.2xlarge), the mesh decoder can exceed available memory, especially for complex scenes

**Error Location:**
- File: `sam3d_objects/pipeline/inference_pipeline.py`, line 608
- Function: `decode_slat()` → `self.models["slat_decoder_mesh"](slat)`
- The sparse convolution operations in the mesh decoder require significant GPU memory

## Solutions

### ✅ Solution 1: Skip Mesh Decoding (RECOMMENDED)

**Skip mesh decoding and only use Gaussian Splatting**, which uses much less memory:

```python
# In demo.py or your inference code:
output = inference(image, mask, seed=42, decode_formats=["gaussian"])
```

**Benefits:**
- Uses significantly less GPU memory
- Gaussian splatting is often sufficient for visualization
- Still produces high-quality 3D reconstructions

**Note:** If you need mesh output, you'll need to use a larger GPU instance.

### ✅ Solution 2: Use Larger GPU Instance

If you absolutely need mesh output, upgrade to:
- **g5.4xlarge** (48GB GPU memory) - 2x the memory
- **g5.8xlarge** (96GB GPU memory) - 4x the memory

### ✅ Solution 3: Reduce Input Resolution

Reduce the input image resolution before processing:
- Downsample images to smaller sizes (e.g., 512x512 instead of 1024x1024)
- This reduces the sparse structure size and memory requirements

### ✅ Solution 4: Clear GPU Cache (Already Implemented)

The code has been updated to clear GPU cache before and after mesh decoding. This helps but may not be enough for very large scenes.

### ✅ Solution 5: Use Mixed Precision

Ensure you're using the default `dtype="bfloat16"` in the pipeline configuration, which reduces memory usage compared to float32.

## Modified Files

1. **`notebook/inference.py`**: Added `decode_formats` parameter to `__call__` method
2. **`sam3d_objects/pipeline/inference_pipeline.py`**: Added GPU cache clearing before/after mesh decoding
3. **`demo.py`**: Updated to skip mesh decoding by default

## Testing

After applying these changes, test with:

```bash
python demo.py
```

The inference should now complete successfully with only Gaussian splatting output, avoiding the OOM error.

## If You Still Need Mesh Output

If mesh output is required for your use case:

1. **Upgrade to g5.4xlarge or larger** (recommended)
2. **Process smaller scenes** or reduce input resolution
3. **Process in batches** if dealing with multiple objects
4. **Contact the SAM-3D-Objects team** for memory optimization suggestions

## Additional Notes

- The mesh decoder is optional - Gaussian splatting provides excellent visualization quality
- Mesh decoding is the most memory-intensive part of the pipeline
- The error occurs specifically in sparse convolution operations, which are hard to optimize further without architectural changes

