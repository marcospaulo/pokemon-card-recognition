# Preprocessing Inconsistency Found

**Date:** 2026-01-12
**Issue:** Hailo calibration scripts missing ImageNet normalization

---

## Problem

The Hailo compilation calibration data preparation scripts normalize images to [0,1] but DO NOT apply ImageNet normalization (mean/std adjustment). This is inconsistent with:
1. How the model was trained
2. How ONNX inference works (test_inference_pi.py)
3. How reference embeddings were generated

---

## Affected Files

### ❌ Missing ImageNet Normalization

**File:** `scripts/prepare_calibration_npy.py`
```python
# Current (line 20):
img_array = np.array(img, dtype=np.float32) / 255.0
# Missing: ImageNet mean/std normalization
```

**File:** `scripts/hailo_compile.py`
```python
# Current (line 32):
img_array = np.array(img, dtype=np.float32) / 255.0
# Comment says: "Just normalize to [0, 1] range"
# Missing: ImageNet mean/std normalization
```

### ✅ Correct Preprocessing (for reference)

**File:** `scripts/test_inference_pi.py` (lines 29-34)
```python
img_array = np.array(img).astype(np.float32) / 255.0

# ImageNet normalization
mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
img_array = (img_array - mean) / std
```

---

## Why This Matters

### For ONNX Inference
- ✅ Works correctly because `test_inference_pi.py` has proper preprocessing
- User confirmed: "the Raspberry Pi said that it used the ONNX Model on CPU to extract embeddings then it compared against the reference database and that worked for recognition"

### For Hailo Inference
- ⚠️ **Potentially incorrect** - Calibration data doesn't match training distribution
- Hailo HEF model was compiled with [0,1] normalized calibration data
- But model expects ImageNet-normalized inputs
- This could cause accuracy degradation when using Hailo accelerator

---

## Questions to Answer

1. **Does Hailo compilation automatically apply ImageNet normalization?**
   - Check Hailo SDK documentation
   - Check if model ONNX graph includes normalization layer

2. **Was the Hailo HEF model tested with actual inference?**
   - If yes and accuracy is good → Hailo might handle normalization internally
   - If no → Need to recompile with correct calibration data

3. **Should calibration data match inference preprocessing exactly?**
   - Generally YES for quantization accuracy
   - But some compilers apply preprocessing automatically

---

## Recommended Fix (If Needed)

### Option 1: Fix Calibration Scripts
Update `prepare_calibration_npy.py` and `hailo_compile.py`:

```python
def preprocess_image(image_path, size=(224, 224)):
    """Preprocess image matching training/inference exactly"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size, Image.BILINEAR)

    # Normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Apply ImageNet normalization (CRITICAL)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img_array = (img_array - mean) / std

    # Keep HWC format for Hailo (no transpose)
    return img_array
```

### Option 2: Verify Current Hailo Model
1. Test Hailo inference on known cards
2. Compare embeddings with ONNX inference
3. If embeddings match → no fix needed (Hailo handles it)
4. If embeddings differ → recompile with fixed calibration

---

## Action Items

- [ ] Check if Hailo HEF has been tested with actual inference
- [ ] Compare Hailo embeddings vs ONNX embeddings on same image
- [ ] Review Hailo SDK docs on preprocessing expectations
- [ ] Decide if recompilation is needed
- [ ] If yes: Update calibration scripts + recompile HEF

---

## Test Methodology

To verify if this is an issue:

```python
import numpy as np
import onnxruntime as ort
from hailo_platform import HEF, InferVStreams

# 1. Load same test image
image = load_test_card("base1-4.png")

# 2. Get ONNX embedding (with correct preprocessing)
onnx_embedding = onnx_model.run(preprocess_with_imagenet(image))

# 3. Get Hailo embedding (with current preprocessing)
hailo_embedding = hailo_model.infer(preprocess_without_imagenet(image))

# 4. Compare
cosine_sim = np.dot(onnx_embedding, hailo_embedding) / (
    np.linalg.norm(onnx_embedding) * np.linalg.norm(hailo_embedding)
)

print(f"Cosine similarity: {cosine_sim:.4f}")
# Expected: >0.99 if consistent
# If <0.90: Major preprocessing mismatch
```

---

## Status

**Current Priority:** 🟡 **MEDIUM**
- ONNX inference works correctly (user confirmed)
- Hailo inference status unknown (needs testing)
- Won't block YOLO distillation work
- Should verify before production Hailo deployment

---

## Related Files

- `test_student_model.py` - Correct preprocessing reference
- `STUDENT_MODEL_TEST_REPORT.md` - Validation results
- `scripts/generate_reference_embeddings.py` - How reference DB was built (correct)
