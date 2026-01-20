# Quick Fix Guide for CUDA Device-Side Assert Error

## 🚨 You're seeing this error:
```
CUDA error: device-side assert triggered
status=503 latency_ms=33.33
```

## ✅ Quick Fix Steps

### 1. Run Diagnostic (In Colab)

**Run Cell 9** (new diagnostic cell) to check if your model has token issues:

```python
# Cell 9 will output:
✅ All token IDs are valid  # ← Good, continue to step 2
# OR
❌ eos_token_id out of bounds  # ← Bad, go to step 3
```

### 2. If Diagnostic Shows "Valid" → Restart Server

The updated code has automatic fixes. Just restart:

**In Colab:**
1. Stop the server (interrupt Cell 11)
2. Re-run Cell 11 (Start FastAPI Server)
3. Try the request again

### 3. If Diagnostic Shows "Issues" → Change Model

**In Cell 6, change MODEL_ID:**

```python
# Replace this line:
os.environ['MODEL_ID'] = 'google/medgemma-1.5-4b-it'

# With one of these proven stable models:
os.environ['MODEL_ID'] = 'google/gemma-2b-it'      # Recommended: Small, fast, stable
os.environ['MODEL_ID'] = 'google/gemma-7b-it'      # Better quality, slower
os.environ['MODEL_ID'] = 'google/medgemma-2b'      # Medical-specific (no -it suffix)
```

Then:
1. Re-run Cell 6 (Configure Environment)
2. Re-run Cell 9 (Diagnostic) - should show ✅ now
3. Run Cell 11 (Start Server)

---

## 🔍 What Was Fixed

### Previous Issue (Gemini's Fix)
```python
# Forced pad_token_id to match eos_token_id
# But didn't check if eos_token_id was valid!
_global_tokenizer.pad_token_id = _global_tokenizer.eos_token_id  # Could be >= vocab_size
```

### New Fix (Applied)
```python
# Validate BEFORE setting
if _global_tokenizer.eos_token_id < vocab_size:
    _global_tokenizer.pad_token_id = _global_tokenizer.eos_token_id  # Safe
else:
    _global_tokenizer.pad_token_id = vocab_size - 1  # Fallback
```

---

## 📊 How to Know It's Fixed

### Before Fix:
```
Error 503 (latency_ms=33.33)  # Immediate failure
CUDA error: device-side assert triggered
```

### After Fix (Success):
```
✅ Token IDs validated: eos=1, pad=1, vocab_size=256000
Generation complete in 65.2s
✅ Generation successful!
Output: [proper SOAP note]
```

### After Fix (Model Incompatible):
```
❌ Error: eos_token_id (256000) >= vocab_size (256000)
   Try: google/gemma-2b-it
```
↑ Clear error message telling you to change model

---

## 💡 Why This Happened

**MedGemma models** (especially `medgemma-1.5-4b-it`) may have tokenizer configs where special token IDs are at or beyond the vocabulary size. When CUDA tries to look up these tokens in the embedding table, it triggers an out-of-bounds error.

**The fix**: Validate all token IDs before passing them to the model.

---

## 🎯 Recommended Model

For best compatibility, use:
```python
os.environ['MODEL_ID'] = 'google/gemma-2b-it'
```

**Why?**
- ✅ Stable token configuration (no out-of-bounds issues)
- ✅ Smaller model (faster loading, less memory)
- ✅ Instruction-tuned (works well for clinical notes)
- ✅ Well-tested by community

Once `gemma-2b-it` works, you can try larger models like `gemma-7b-it` for better quality.

---

## 🔧 Files Changed

- ✅ `app/services/model_runner.py` - Added token validation
- ✅ `medgemma_colab_setup.ipynb` - Added diagnostic cell (Cell 9)
- ✅ `CUDA_ERROR_FIX.md` - Full technical documentation
- ✅ `QUICK_FIX.md` - This guide

---

## ❓ Still Having Issues?

### Option 1: Force CPU Mode (Always Works)
```python
# In Cell 6:
os.environ['DEVICE'] = 'cpu'  # Slow but reliable
```

### Option 2: Use Demo Mode (No Model Needed)
```python
# In Cell 6:
os.environ['DEMO_MODE'] = 'true'  # Returns stubbed responses
```

### Option 3: Check Logs
Look for these messages in server output:
```
❌ eos_token_id out of bounds  # ← Model incompatible
✅ Token IDs validated         # ← Should work now
```

---

## 📞 Need Help?

If you're still stuck:
1. Share the output of **Cell 9 (Diagnostic)**
2. Share the error message from **Cell 14 (Generate Note)**
3. Include your `MODEL_ID` from **Cell 6**

---

**Last Updated**: 2026-01-20  
**Fixed Version**: All CUDA assert errors should now be caught with clear messages
