# CUDA Device-Side Assert Error - Fix Documentation

## Error Description

```
CUDA error: device-side assert triggered
```

This error occurs when token IDs passed to the model are **out of vocabulary bounds** (token_id >= vocab_size).

## Root Cause

The previous fix that forced `pad_token_id` to equal `eos_token_id` was causing issues:

```python
# OLD CODE (PROBLEMATIC)
if _global_tokenizer.pad_token_id is None or _global_tokenizer.pad_token_id != _global_tokenizer.eos_token_id:
    _global_tokenizer.pad_token_id = _global_tokenizer.eos_token_id  # May be out of bounds!
    _global_model.config.pad_token_id = _global_tokenizer.pad_token_id  # Causes CUDA error
```

**Problem**: Some tokenizers have `eos_token_id` >= `vocab_size`, which causes CUDA to trigger a device-side assertion when the model tries to use these invalid token IDs.

## Fix Applied

### 1. Token ID Validation (model_runner.py)

Added comprehensive validation to ensure all token IDs are within vocabulary bounds:

```python
# NEW CODE (FIXED)
vocab_size = _global_tokenizer.vocab_size

# Only set pad_token_id if None, don't force it to match eos
if _global_tokenizer.pad_token_id is None:
    if _global_tokenizer.eos_token_id is not None:
        # Validate eos_token_id is within bounds BEFORE using it
        if _global_tokenizer.eos_token_id < vocab_size:
            _global_tokenizer.pad_token_id = _global_tokenizer.eos_token_id
        else:
            # Fallback to last valid token ID
            _global_tokenizer.pad_token_id = vocab_size - 1
            logger.warning(f"eos_token_id out of bounds, using fallback")

# Validate ALL token IDs before generation
if eos_token_id is not None and eos_token_id >= vocab_size:
    raise RuntimeError(f"eos_token_id ({eos_token_id}) >= vocab_size ({vocab_size})")

if pad_token_id is not None and pad_token_id >= vocab_size:
    raise RuntimeError(f"pad_token_id ({pad_token_id}) >= vocab_size ({vocab_size})")
```

### 2. Removed Forced Token ID Matching

The code no longer forces `pad_token_id == eos_token_id`. Instead:
- Sets `pad_token_id = eos_token_id` ONLY if `pad_token_id` is `None` AND `eos_token_id` is valid
- Validates both IDs are within `vocab_size` before use
- Raises clear errors if validation fails

### 3. Generation Parameters

```python
# Do NOT pass pad_token_id to generation (let model use its defaults)
generation_kwargs = {
    "max_new_tokens": max_new_tokens,
    "eos_token_id": eos_token_id,  # Only if valid
    "repetition_penalty": 1.1,
    # NO pad_token_id here
}
```

### 4. Diagnostic Tool (Colab Notebook)

Added a new diagnostic cell (Cell 9) to check token configuration before starting the server:

```python
# Run this cell if you get CUDA errors
# It will check if token IDs are within vocab bounds
```

## How to Use the Fix

### Option 1: Just Run (Automatic Validation)

The fix will automatically:
1. Validate token IDs during model loading
2. Raise clear error messages if issues detected
3. Suggest alternative models or fallbacks

### Option 2: Run Diagnostic First (Recommended)

**In Colab Notebook:**
1. Run Cell 6 (Configure Environment)
2. **Run NEW Cell 9 (Diagnostic)** - checks token configuration
3. If diagnostic shows issues, change `MODEL_ID` in Cell 6
4. Run Cell 11 (Start Server)

The diagnostic will tell you:
- ✅ Token IDs are valid → safe to proceed
- ❌ Token IDs out of bounds → change model

## Alternative Models (if current model fails)

If `google/medgemma-1.5-4b-it` has token ID issues, try:

```python
# In Cell 6, change MODEL_ID to one of these:
os.environ['MODEL_ID'] = 'google/gemma-2b-it'      # Smaller, more stable
os.environ['MODEL_ID'] = 'google/gemma-7b-it'      # Larger, better quality
os.environ['MODEL_ID'] = 'google/medgemma-2b'      # Medical model (smaller)
```

**Note**: Models ending in `-it` are instruction-tuned and generally work better for clinical notes.

## Expected Behavior After Fix

### Before Fix:
```
CUDA error: device-side assert triggered
latency_ms=33.33  # Fails immediately
```

### After Fix (if token IDs are valid):
```
Model loaded successfully
Generation complete in 60-90s
Output: [proper SOAP note]
```

### After Fix (if token IDs are invalid):
```
❌ Error: eos_token_id (256000) >= vocab_size (256000)
   This indicates a tokenizer/model mismatch.
   Try: google/gemma-2b-it or google/gemma-7b-it
```

## Technical Details

### Why Token IDs Must Be Within Bounds

CUDA kernels use token IDs as array indices:
```
embedding_vector = embedding_table[token_id]
```

If `token_id >= vocab_size`, this causes an **out-of-bounds memory access**, triggering the device-side assert.

### Why Previous Fix Caused Issues

The Gemini suggestion to align `pad_token_id` with `eos_token_id` was valid for **preventing pad token generation**, but it didn't account for models where `eos_token_id` itself is invalid.

### Current Approach

1. **Validate first**: Check if token IDs are within bounds
2. **Set safely**: Only use valid token IDs
3. **Fail fast**: Raise clear errors if validation fails
4. **Fallback**: Use `vocab_size - 1` as last resort

## Verification Steps

1. **Check logs** for token validation messages:
   ```
   Token IDs validated: eos=1, pad=1, vocab_size=256000
   ✅ eos_token_id is valid (within vocab bounds)
   ```

2. **Run diagnostic cell** (Cell 9) to check before starting server

3. **Monitor latency**:
   - Immediate failure (< 1 second): Token ID issue
   - Long wait (60-90s): Normal generation
   - Timeout (> 5 min): Model may be stuck (reduce max_tokens)

## Files Modified

- `app/services/model_runner.py`: Added token ID validation, removed forced matching
- `medgemma_colab_setup.ipynb`: Added diagnostic cell (Cell 9)
- `CUDA_ERROR_FIX.md`: This documentation

## Related Issues

- Pad token generation issue (fixed separately)
- Attention mask handling (fixed)
- Sampling configuration (fixed)

All issues are now addressed with proper validation and fallbacks.
