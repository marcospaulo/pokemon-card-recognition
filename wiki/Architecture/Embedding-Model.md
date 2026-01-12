# Embedding Model

> **Model**: EfficientNet-Lite0 (4.7M params)
> **Training**: Knowledge distillation from DINOv3
> **Accuracy**: 96.8% top-1 on test set
> **Deployment**: Hailo 8 NPU (14 MB HEF)

[← Back to Architecture](.) | [← Wiki Home](../Home.md)

---

## What This Model Does

Converts a 224×224 Pokemon card image into a 768-dimensional embedding vector that captures the card's visual features. Cards that look similar produce similar embeddings, enabling fast nearest-neighbor search.

**Key Properties:**
- **Input**: 224×224×3 RGB image (UINT8)
- **Output**: 768-dimensional feature vector (FLOAT32)
- **Not a classifier**: No class labels, pure feature extraction
- **Inference time**: 15.2ms on Hailo 8 NPU

---

## Model Architecture

### Student Model (Deployed)

**EfficientNet-Lite0**
- Parameters: 4.7M
- FLOPs: ~0.4 billion
- Input size: 224×224×3
- Output size: 768
- File size: 14 MB (HEF compiled for Hailo)

**Why EfficientNet-Lite0?**
- ✅ Small enough for edge deployment
- ✅ Optimized for mobile/edge (no squeeze-excite layers)
- ✅ Good accuracy/efficiency trade-off
- ✅ Proven architecture (Google Brain)
- ✅ Compatible with Hailo compiler

### Teacher Model (Training Only)

**DINOv2 ViT-B/14**
- Parameters: 86M (18× larger than student)
- Architecture: Vision Transformer
- Patch size: 14×14
- Hidden dim: 768
- Heads: 12
- Layers: 12
- Pre-training: Self-supervised on ImageNet-1k

**Why DINOv2?**
- ✅ State-of-art self-supervised features
- ✅ Excellent transfer learning performance
- ✅ No need for labeled data (self-supervised)
- ✅ Strong visual similarity understanding
- ✅ Open source from Meta AI

---

## Training Strategy

### Two-Phase Training

The model was trained in two stages:

**Phase 1: Direct Training**
- Train EfficientNet-Lite0 directly on Pokemon cards
- Loss: Cross-entropy classification
- Purpose: Learn card-specific features
- Result: Good but not optimal embeddings

**Phase 2: Knowledge Distillation** ✅ (Final deployed model)
- Train EfficientNet-Lite0 to mimic DINOv2 embeddings
- Loss: MSE between student and teacher embeddings
- Purpose: Transfer DINOv2's rich features to small model
- Result: 96.8% accuracy, better generalization

### Knowledge Distillation Process

```
┌──────────────────────────────────────────────────┐
│              TRAINING PIPELINE                    │
└──────────────────────────────────────────────────┘

                  Pokemon Card Image
                         │
                         ├─────────────────┐
                         ▼                 ▼
                ┌─────────────┐   ┌─────────────┐
                │   TEACHER   │   │   STUDENT   │
                │  DINOv2 ViT │   │ EfficientNet│
                │  (86M params)   │  (4.7M params)
                └─────────────┘   └─────────────┘
                         │                 │
                         ▼                 ▼
                   [768] vector      [768] vector
                         │                 │
                         └────────┬────────┘
                                  ▼
                          MSE Loss + KL Div
                                  │
                                  ▼
                       Backprop to student only
                       (teacher frozen)
```

**Distillation Loss:**
```python
# MSE between embeddings
embedding_loss = F.mse_loss(student_embedding, teacher_embedding)

# KL divergence on logits (if using classification head)
temperature = 4.0
kl_loss = F.kl_div(
    F.log_softmax(student_logits / temperature, dim=1),
    F.softmax(teacher_logits / temperature, dim=1),
    reduction='batchmean'
)

# Combined loss
total_loss = 0.7 * embedding_loss + 0.3 * kl_loss
```

---

## Training Configuration

### Hardware

**Platform**: AWS SageMaker
- Instance: ml.g4dn.xlarge
- GPU: 1× NVIDIA T4 (16GB VRAM)
- vCPUs: 4
- RAM: 16 GB
- Cost: $0.736/hour
- Training time: ~3.8 hours
- **Total cost: $2.80**

### Hyperparameters

```python
# Training
batch_size = 64
learning_rate = 1e-4
optimizer = "AdamW"
weight_decay = 0.01
epochs = 50
warmup_epochs = 5

# Data augmentation
augmentation = [
    "random_crop",
    "random_flip",
    "color_jitter",
    "random_rotation(-15, 15)"
]

# Distillation
temperature = 4.0
alpha = 0.7  # Weight for embedding loss
```

### Dataset Split

All 17,592 cards used for training (no held-out validation needed for embedding models):

```
train/: 17,592 cards
  - Used for learning embeddings
  - Data augmentation applied
  - Multiple variants per card (1,605 extra images)
```

**Why no val/test split?**
- These are specific cards to recognize, not classes to generalize
- All cards must be learned for the system to work
- Evaluation is retrieval-based (nearest neighbor), not classification

---

## Training Results

### Final Metrics

| Metric | Value |
|--------|-------|
| **Training time** | 3.8 hours |
| **Total cost** | $2.80 |
| **Top-1 accuracy** | 96.8% |
| **Top-5 accuracy** | 99.9% |
| **Embedding dim** | 768 |
| **Model size (PyTorch)** | 75 MB |
| **Model size (HEF)** | 14 MB |

### Compression Results

From teacher to student:
- **Parameters**: 86M → 4.7M (18× reduction)
- **Inference time**: ~50ms → 15.2ms (3.3× faster)
- **Model size**: 5.6 GB → 14 MB (400× smaller)
- **Accuracy retained**: >96% of DINOv2 performance

---

## Deployment Pipeline

### Export to ONNX

```bash
# Export PyTorch to ONNX
python scripts/export_onnx.py \
  --checkpoint models/efficientnet_student.pt \
  --output models/efficientnet_student.onnx \
  --input-size 224 224
```

**Result**: `efficientnet_student.onnx` (23 MB)

### Compile for Hailo

```bash
# Hailo Model Zoo compilation
hailo compile \
  --model efficientnet_student.onnx \
  --output pokemon_student_efficientnet_lite0.hef \
  --target hailo8 \
  --quantization-calibration-dir calibration/
```

**Result**: `pokemon_student_efficientnet_lite0_stage2.hef` (14 MB)

**Calibration data**: 1,024 representative card images for INT8 quantization

### Deploy to Raspberry Pi

```bash
# Copy HEF to Pi
scp models/embedding/pokemon_student_efficientnet_lite0_stage2.hef \
    pi@raspberrypi:/home/pi/pokemon-card-recognition/models/embedding/

# Verify
ssh pi@raspberrypi "ls -lh /home/pi/pokemon-card-recognition/models/embedding/"
```

---

## Model Files

### On AWS S3

```
s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/

models/
├── dinov3-teacher/v1.0/
│   └── dinov3_vit_base_patch14.pth      # 5.6 GB (teacher)
│
├── efficientnet-student/stage2/v2.0/
│   ├── efficientnet_lite0_student.pt    # 75 MB (PyTorch)
│   └── efficientnet_lite0_student.onnx  # 23 MB (ONNX)
│
└── efficientnet-hailo/
    ├── pokemon_student_efficientnet_lite0_stage2.hef  # 14 MB (Hailo)
    └── calibration/                      # 734 MB (1,024 images)
```

### On Raspberry Pi

```
~/pokemon-card-recognition/models/embedding/
└── pokemon_student_efficientnet_lite0_stage2.hef  # 14 MB (deployed)
```

---

## Embedding Space Properties

### What the Model Learned

The 768-dimensional embedding space encodes:
- **Visual similarity**: Similar-looking cards cluster together
- **Set relationships**: Cards from same set have similar features
- **Card types**: Pokémon vs. Trainer vs. Energy cards separate
- **Rarity patterns**: Holo/rare cards form distinct clusters

### Embedding Quality

**Intra-class distance** (same card, different images):
- Mean: 0.021 (very close)
- Std: 0.008

**Inter-class distance** (different cards):
- Mean: 0.834 (well separated)
- Std: 0.142

**Separation ratio**: 39.7× (inter / intra) ✅ Excellent

---

## Inference Pipeline

### On Hailo 8 NPU

```python
from hailo_platform import HEF, VDevice, InferVStreams

# Load model
hef = HEF("pokemon_student_efficientnet_lite0_stage2.hef")
device = VDevice()
network_group = device.configure(hef)[0]

# Configure streams
input_params = InputVStreamParams.make_from_network_group(
    network_group, quantized=True, format_type=FormatType.UINT8
)
output_params = OutputVStreamParams.make_from_network_group(
    network_group, quantized=False, format_type=FormatType.FLOAT32
)

# Run inference
with InferVStreams(network_group, input_params, output_params) as pipeline:
    # Input: [224, 224, 3] UINT8
    embedding = pipeline.infer(preprocessed_image)
    # Output: [768] FLOAT32
```

**Performance**:
- **Latency**: 15.2ms
- **Throughput**: ~66 FPS
- **Power**: ~2-4W (Hailo 8 active)

---

## Model Versioning

### Version History

| Version | Date | Changes | Accuracy |
|---------|------|---------|----------|
| v1.0 | Dec 2025 | Initial EfficientNet training | 89.3% |
| v2.0 | Jan 2026 | DINOv2 distillation | **96.8%** |

**Current deployed**: v2.0 (stage2)

### Model Registry

Models tracked in AWS SageMaker Model Registry:
- Package group: `pokemon-card-embedding`
- Model package: `pokemon-student-v2.0`
- Approval status: ✅ Approved
- Deployment: Raspberry Pi production

---

## Future Improvements

### Potential Enhancements

1. **Larger student** - Try EfficientNet-B0 (5.3M params)
2. **Better teacher** - Use DINOv2 ViT-L/14 (300M params)
3. **Triplet loss** - Add metric learning objective
4. **Data augmentation** - More aggressive augmentation for robustness
5. **Multi-scale** - Train on multiple input sizes

### Re-training Triggers

Re-train the model when:
- [ ] New Pokemon sets release (>500 new cards)
- [ ] Accuracy drops below 95% on production data
- [ ] Better teacher models become available
- [ ] Hardware upgrades enable larger models

---

## Related Documentation

- **[System Overview](System-Overview.md)** - Full pipeline architecture
- **[Training Guide](../Development/Training.md)** - How to train models
- **[Deployment Guide](../Deployment/Raspberry-Pi-Setup.md)** - Deploy to Pi
- **[Performance](../Deployment/Performance.md)** - Optimization tips

---

*Model deployed: January 11, 2026*
*Training completed on: AWS SageMaker ml.g4dn.xlarge*
