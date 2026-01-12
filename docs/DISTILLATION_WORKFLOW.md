# YOLO Model Distillation for IMX500

Complete workflow to compress YOLO11n-OBB from 2.6M → ~1M parameters for IMX500 deployment.

---

## Overview

**Problem:** YOLO11n-OBB is too large for IMX500 constraints
- Current: 2.6M parameters, 10.3MB ONNX
- IMX500 Limit: ~1.3M parameters, <8MB total (model + activations)

**Solution:** Knowledge distillation to smaller student model
- Teacher: Trained YOLO11n-OBB (2.6M params, 96-98% confidence)
- Student: YOLOv8n-obb (target ~1M params)
- Result: <1.22MB INT8 model for IMX500

---

## Prerequisites

### Local Machine (macOS)
- Python 3.8+
- PyTorch + Ultralytics
- Training dataset (10k Pokemon card images)

### EC2 Compilation Server
- **Instance:** ubuntu@18.118.102.134
- **Has:** Hailo SDK + Sony IMX500 SDK
- **Access:** SSH key configured

---

## Workflow Steps

### Step 1: Prepare Training Dataset

**Option A: Download from Ultralytics Hub**

The model was trained on this dataset:
```bash
# Dataset: https://hub.ultralytics.com/datasets/8awcqoIQP0jIXIMDOCsC
# 10,000 synthetic images with Pokemon cards

# Download manually from Ultralytics Hub or set API key
export ULTRALYTICS_API_KEY=your_key_here
```

**Option B: Use Existing Dataset**

If you already have the dataset locally:
```bash
python scripts/prepare_training_dataset.py
```

This will locate existing datasets or guide you on setup.

**Expected Dataset Structure:**
```
pokemon_cards_obb/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

---

### Step 2: Run Distillation Training

**Local Machine (macOS):**

```bash
cd /Users/marcos/dev/raspberry-pi/pokemon-card-recognition

# Run distillation
python scripts/distill_yolo_for_imx500.py pokemon_cards_obb.yaml
```

**Training Configuration:**
- Teacher: `/Users/marcos/dev/raspberry-pi/pokemon_card_detector/weights/hub/dQfecRsRsXbAKXOXHLHJ/best.pt`
- Student: YOLOv8n-obb (starts from pretrained base)
- Epochs: 100 (with early stopping)
- Batch: 16
- Image size: 416×416

**Expected Training Time:**
- With GPU: ~2-3 hours
- Without GPU: ~8-10 hours

**Output:**
```
models/detection/distilled/distilled_pokemon_obb/
├── weights/
│   ├── best.pt           # Best checkpoint
│   └── last.pt           # Last checkpoint
└── results.png           # Training curves
```

**Verify Model Size:**
```python
from ultralytics import YOLO

model = YOLO('models/detection/distilled/distilled_pokemon_obb/weights/best.pt')
params = sum(p.numel() for p in model.model.parameters())
print(f"Parameters: {params:,}")
# Target: ~1M parameters
```

---

### Step 3: Export to ONNX

**Local Machine:**

```bash
python scripts/export_yolo_to_imx500.py \
  models/detection/distilled/distilled_pokemon_obb/weights/best.pt
```

**Output:**
```
models/detection/pokemon_yolo_distilled_imx500.onnx
```

**Verify ONNX Size:**
```bash
ls -lh models/detection/pokemon_yolo_distilled_imx500.onnx
# Target: <6MB (leaves 2MB for activations in 8MB budget)
```

---

### Step 4: Transfer to EC2

**Transfer ONNX to Compilation Server:**

```bash
# Upload distilled ONNX
scp models/detection/pokemon_yolo_distilled_imx500.onnx \
  ubuntu@18.118.102.134:~/models/

# Also upload calibration images (1,024 images for quantization)
# Option 1: Download from S3 on EC2
# Option 2: Upload local calibration set
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/calibration/ \
  ./calibration/

scp -r calibration/ ubuntu@18.118.102.134:~/calibration/
```

---

### Step 5: Compile for IMX500 (On EC2)

**SSH to EC2:**
```bash
ssh ubuntu@18.118.102.134
```

**Compile to RPK:**
```bash
cd ~/models

# Convert ONNX to IMX500 RPK format
imx500-convert-model \
  --model pokemon_yolo_distilled_imx500.onnx \
  --input-shape 1,3,416,416 \
  --output pokemon_yolo_distilled.rpk \
  --calibration ~/calibration/ \
  --quantization int8
```

**Expected Output:**
```
pokemon_yolo_distilled.rpk
```

**Verify Size:**
```bash
ls -lh pokemon_yolo_distilled.rpk
# Should be <2MB after INT8 quantization
```

---

### Step 6: Compile for Hailo (Optional, On EC2)

If you also want to run detection on Hailo-8 instead:

```bash
# Convert ONNX to Hailo HEF
hailo parser onnx pokemon_yolo_distilled_imx500.onnx \
  --output pokemon_yolo_distilled.har

hailo optimize pokemon_yolo_distilled.har \
  --calib-path ~/calibration/ \
  --output pokemon_yolo_distilled_optimized.har

hailo compiler pokemon_yolo_distilled_optimized.har \
  --output pokemon_yolo_distilled.hef
```

---

### Step 7: Download Compiled Models

**From EC2 back to local machine:**

```bash
# Download IMX500 RPK
scp ubuntu@18.118.102.134:~/models/pokemon_yolo_distilled.rpk \
  models/detection/

# Download Hailo HEF (if compiled)
scp ubuntu@18.118.102.134:~/models/pokemon_yolo_distilled.hef \
  models/detection/
```

---

### Step 8: Test on Raspberry Pi

**Transfer to Pi:**

```bash
# For IMX500 camera
scp models/detection/pokemon_yolo_distilled.rpk \
  pi@raspberrypi:/home/pi/models/

# For Hailo-8 NPU (if compiled)
scp models/detection/pokemon_yolo_distilled.hef \
  pi@raspberrypi:/home/pi/models/
```

**Test IMX500 Detection:**

```python
#!/usr/bin/env python3
"""Test distilled YOLO on IMX500"""

from picamera2 import Picamera2
import imx500

# Initialize IMX500
picam2 = Picamera2()
model = imx500.load_model('/home/pi/models/pokemon_yolo_distilled.rpk')

# Configure camera
config = picam2.create_still_configuration()
picam2.configure(config)
picam2.start()

# Run detection
detections = model.detect(picam2.capture_array())

print(f"Detected {len(detections)} cards")
for det in detections:
    print(f"  Confidence: {det['confidence']:.2%}")
    print(f"  Box: {det['box']}")
```

---

## Verification Checklist

After distillation, verify:

- [ ] Model has ~1M parameters (not 2.6M)
- [ ] ONNX export is <6MB
- [ ] Compiles successfully to RPK on EC2
- [ ] RPK is <2MB after INT8 quantization
- [ ] Detection accuracy is 85-95% (acceptable vs. 96-98% teacher)
- [ ] Inference runs on IMX500 without memory errors
- [ ] Detection speed is <100ms on IMX500

---

## Troubleshooting

### Model Still Too Large

If distilled model exceeds 1.3M parameters:

**Option 1: Use Even Smaller Base**
```python
# Try YOLOv8s instead of YOLOv8n
# Reduce depth/width multipliers
```

**Option 2: Prune After Training**
```python
# Apply structured pruning to remove channels
from torch.nn.utils import prune
# ... pruning code
```

### Low Detection Accuracy

If accuracy drops below 85%:

**Option 1: Longer Training**
```bash
# Increase epochs to 200
python scripts/distill_yolo_for_imx500.py pokemon_cards_obb.yaml
# Edit script to set epochs=200
```

**Option 2: Adjust Temperature**
```python
# In distillation script, increase temperature
# Higher temperature = softer teacher labels
temperature = 0.7  # from 0.5
```

### Compilation Fails on EC2

**IMX500 SDK Issues:**
```bash
# Check SDK version
imx500-convert-model --version

# Update if needed
sudo apt update
sudo apt install imx500-sdk
```

**Hailo SDK Issues:**
```bash
# Check Hailo installation
hailortcli fw-control identify

# Reinstall if needed
wget https://hailo.ai/downloads/hailo-sdk-v3.27.0.tar.gz
```

---

## Expected Results

### Model Comparison

| Model | Parameters | Size | Accuracy | Device |
|-------|-----------|------|----------|--------|
| Teacher (YOLO11n-OBB) | 2.6M | 10.3MB | 96-98% | CPU/GPU |
| Student (Distilled) | ~1M | ~4MB | 90-95% | CPU/GPU |
| Student (INT8) | ~1M | ~1.2MB | 88-93% | IMX500 |

### Performance

**IMX500 On-Sensor:**
- Inference: <100ms (including preprocessing)
- Power: ~0.5W additional
- Latency: Minimal (no frame transfer)

**Hailo-8 NPU:**
- Inference: ~30ms
- Power: ~2W additional
- Latency: Low (MIPI interface)

---

## Pipeline Integration

### Full System Architecture

```
Option 1: IMX500 Detection + Hailo Recognition
┌─────────┐
│ IMX500  │ Detect cards (distilled YOLO, <100ms)
└────┬────┘
     ↓ Crop regions
┌─────────┐
│ Hailo-8 │ Generate embeddings (EfficientNet, ~30ms)
└────┬────┘
     ↓ 768-dim vectors
┌─────────┐
│ uSearch │ Find matches (cosine similarity, <5ms)
└─────────┘

Total: ~135ms per card
```

```
Option 2: All on Hailo-8
┌─────────┐
│ Hailo-8 │ Detect cards (distilled YOLO, ~30ms)
└────┬────┘
     ↓ Crop regions
┌─────────┐
│ Hailo-8 │ Generate embeddings (EfficientNet, ~30ms)
└────┬────┘
     ↓ 768-dim vectors
┌─────────┐
│ uSearch │ Find matches (cosine similarity, <5ms)
└─────────┘

Total: ~65ms per card
```

**Recommendation:** Option 2 (all on Hailo) is simpler and faster. Only use IMX500 if you need ultra-low power or want to filter frames before sending to Hailo.

---

## Cost

- **Training:** ~$0 (local GPU) or ~$2 (cloud GPU for 3 hours)
- **Compilation:** $0 (using existing EC2 instance)
- **Storage:** +20MB on S3 (~$0.0005/month)

**Total:** Negligible additional cost

---

**Last Updated:** 2026-01-12
**Status:** Ready for execution
**EC2 Compiler:** ubuntu@18.118.102.134 (Hailo SDK + Sony IMX500 SDK)
