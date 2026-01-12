---
skill_name: pokemon-card-recognition:model-details
description: Detailed model architectures, parameters, compression ratios, and training configurations
tags: [models, dinov3, efficientnet, hailo, knowledge-distillation]
---

# Model Details & Training Configuration

## Model Architecture Overview

```
Teacher (DINOv3 ViT-B/14)
    ↓ Knowledge Distillation
Student Stage 1 (EfficientNet-Lite0)
    ↓ Hard Negative Mining
Student Stage 2 (EfficientNet-Lite0)
    ↓ INT8 Quantization
Hailo Edge Model (HEF)
```

---

## Teacher Model: DINOv3 ViT-B/14

### Architecture

```yaml
Model: facebook/dinov3-base
Architecture: Vision Transformer (ViT-B/14)
Patch Size: 14x14
Image Size: 518x518
Parameters: 304M
Embedding Dimension: 768
Layers: 12 transformer blocks
Attention Heads: 12
Hidden Size: 768
MLP Ratio: 4
```

### Training Configuration

```python
Training Job: pokemon-card-dinov3-teacher-2026-01-10-13-31-34-937

Instance: ml.p4d.24xlarge (8xA100 80GB)
Duration: 12 minutes
Batch Size: 128 (16 per GPU)
Learning Rate: 5e-5
Optimizer: AdamW
Weight Decay: 0.05
Warmup Steps: 100
Max Steps: 2000

Loss Function: Contrastive loss (InfoNCE)
Temperature: 0.07
Augmentation: AutoAugment + RandomResizedCrop
```

### Model Performance

```yaml
Training Cost: $4.00
Training Time: 12 minutes
Model Size: 5.6 GB (uncompressed)
Format: SageMaker model.tar.gz
Embedding Quality: Baseline (100%)
Inference Speed (A100): ~20ms per image
```

### Model Registry

```yaml
Model Package ARN: arn:aws:sagemaker:us-east-2:943271038849:model-package/pokemon-card-recognition-models/4
Approval Status: Approved
Model Type: Teacher (embedding generation)
Use Case: High-accuracy embedding generation
Deployment: Cloud inference (expensive)
```

### S3 Location

```
s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/dinov3-teacher/v1.0/model.tar.gz
```

---

## Student Model Stage 1: EfficientNet-Lite0

### Architecture

```yaml
Model: EfficientNet-Lite0 (custom embedding head)
Parameters: 4.7M
Embedding Dimension: 768 (match teacher)
Input Size: 224x224
Depth Multiplier: 1.0
Width Multiplier: 1.0
Dropout: 0.2
```

### Training Configuration

```python
Training Job: pytorch-training-2026-01-10-xx-xx-xx-xxx (Stage 1)

Instance: ml.p4d.24xlarge (8xA100 80GB)
Duration: 15 minutes
Batch Size: 256 (32 per GPU)
Learning Rate: 1e-3
Optimizer: AdamW
Weight Decay: 1e-4
Scheduler: CosineAnnealingLR
Temperature: 0.5

Loss Function: Knowledge Distillation (KL divergence + MSE)
Teacher Model: DINOv3 (frozen)
Alpha (KL): 0.7
Alpha (MSE): 0.3
```

### Model Performance

```yaml
Training Cost: $4.00
Training Time: 15 minutes
Compression Ratio: 64.7x (304M → 4.7M parameters)
Embedding Quality: ~92% of teacher
Status: Transitional (not preserved)
```

### Important Note

**Stage 1 is marked as "transitional"** - checkpoints were not preserved to production storage. This was an intermediate training phase used to bootstrap the student model. Only the final Stage 2 model (with hard negative mining) was saved.

---

## Student Model Stage 2: EfficientNet-Lite0

### Architecture

```yaml
Model: EfficientNet-Lite0 (Stage 2 - fine-tuned)
Parameters: 4.7M
Embedding Dimension: 768
Input Size: 224x224
Base: Stage 1 checkpoint
Additional Training: Hard negative mining
```

### Training Configuration

```python
Training Job: pytorch-training-2026-01-11-23-31-10-757

Instance: ml.p4d.24xlarge (8xA100 80GB)
Duration: 10 minutes
Batch Size: 256 (32 per GPU)
Learning Rate: 5e-4 (reduced from Stage 1)
Optimizer: AdamW
Weight Decay: 1e-4
Scheduler: CosineAnnealingLR

Loss Function: Hard Negative Contrastive Loss
Mining Strategy: Semi-hard negatives (distance-based)
Margin: 0.2
Positives per Anchor: 2
Negatives per Anchor: 8
```

### Model Performance

```yaml
Training Cost: $3.00
Training Time: 10 minutes
Model Size (PyTorch): 74.7 MB
Model Size (ONNX): 22.8 MB
Compression Ratio: 64.7x from teacher
Embedding Quality: ~95% of teacher (improved from Stage 1)
Inference Speed (CPU): ~80ms per image
Inference Speed (A100): ~5ms per image
```

### Model Registry

```yaml
Model Package ARN: arn:aws:sagemaker:us-east-2:943271038849:model-package/pokemon-card-recognition-models/5
Approval Status: Approved
Model Type: Student (compressed)
Use Case: Production inference (cloud or local)
Deployment: Cloud endpoint or local PyTorch/ONNX
```

### S3 Locations

**PyTorch Checkpoint:**
```
s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-student/stage2/v2.0/student_stage2_final.pt
```

**ONNX Export:**
```
s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-student/stage2/v2.0/student_stage2_final.onnx
```

### Export Commands

**PyTorch → ONNX:**
```python
import torch

model = torch.load('student_stage2_final.pt')
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    'student_stage2_final.onnx',
    opset_version=13,
    input_names=['image'],
    output_names=['embedding'],
    dynamic_axes={'image': {0: 'batch_size'}}
)
```

---

## Hailo Edge Model: INT8 Quantized

### Architecture

```yaml
Base Model: Student Stage 2 (EfficientNet-Lite0)
Target NPU: Hailo-8L
Quantization: INT8 (from FP32)
Format: HEF (Hailo Executable Format)
Parameters: 4.7M (quantized)
Model Size: 13.8 MB
```

### Compilation Configuration

```python
Compilation Job: hailo-compilation-2026-01-11-xx-xx-xx-xxx

Instance: ml.m5.2xlarge (8 vCPU, 32GB RAM)
Duration: 60 minutes
Hailo SDK: v3.27.0
Model Optimization: Compression + quantization
Calibration Dataset: 1,024 card images

Quantization:
  - Precision: INT8
  - Calibration: MinMax algorithm
  - Batch Norm Folding: Enabled
  - Channel Quantization: Enabled
  - Per-Channel Scales: Enabled
```

### Calibration Process

**Calibration Dataset:**
```
Location: s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/calibration/
Images: 1,024 card images (stratified sample from 17,592)
Size: 734 MB
Selection: Representative sample across all 2,199 unique cards
```

**Calibration Command:**
```bash
hailo parser onnx student_stage2_final.onnx \
  --output-har student.har

hailo optimize student.har \
  --calib-path calibration/ \
  --output-har student_optimized.har

hailo compiler student_optimized.har \
  --output-hef pokemon_student_efficientnet_lite0_stage2.hef
```

### Model Performance

```yaml
Compilation Cost: $0.50
Compilation Time: 60 minutes
Model Size: 13.8 MB (INT8 compressed)
Quantization Accuracy Loss: <1% (vs. FP32 student)
Target Device: Raspberry Pi 5 + Hailo-8L
NPU Performance: 26.8 TOPS
Inference Speed: ~30ms per image (including preprocessing)
Power Consumption: ~2W additional (Hailo-8L)
```

### S3 Location

```
s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-hailo/v2.1/pokemon_student_efficientnet_lite0_stage2.hef
```

### Deployment to Raspberry Pi

**Transfer Model:**
```bash
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-hailo/v2.1/pokemon_student_efficientnet_lite0_stage2.hef ./

scp pokemon_student_efficientnet_lite0_stage2.hef pi@raspberrypi:/home/pi/models/
```

**Inference Code:**
```python
from hailo_platform import HailoDevice, InferenceContext
import numpy as np
import cv2

# Initialize Hailo device
device = HailoDevice()
model = device.create_infer_model('models/pokemon_student_efficientnet_lite0_stage2.hef')

# Preprocess image
image = cv2.imread('card.jpg')
image = cv2.resize(image, (224, 224))
image = image.astype(np.float32) / 255.0
image = np.transpose(image, (2, 0, 1))  # HWC → CHW
image = np.expand_dims(image, axis=0)   # Add batch dimension

# Run inference
with InferenceContext(model) as ctx:
    embedding = ctx.run(image)

# embedding shape: (1, 768)
```

---

## Compression Metrics

### Size Comparison

```yaml
Teacher (DINOv3):
  Parameters: 304M
  FP32 Size: 5.6 GB
  Precision: FP32

Student Stage 2 (PyTorch):
  Parameters: 4.7M
  FP32 Size: 74.7 MB
  Precision: FP32
  Compression: 64.7x

Student Stage 2 (ONNX):
  Parameters: 4.7M
  Size: 22.8 MB
  Precision: FP32
  Compression: 245x vs. Teacher

Hailo Edge (HEF):
  Parameters: 4.7M
  INT8 Size: 13.8 MB
  Precision: INT8
  Compression: 405x vs. Teacher
```

### Performance Comparison

```yaml
Metric                  | Teacher | Student Stage 2 | Hailo Edge
------------------------|---------|-----------------|------------
Parameters              | 304M    | 4.7M            | 4.7M (INT8)
Model Size              | 5.6 GB  | 22.8 MB (ONNX)  | 13.8 MB
Compression             | 1x      | 64.7x           | 405x
Embedding Quality       | 100%    | 95%             | 94%
Inference (A100)        | 20ms    | 5ms             | N/A
Inference (CPU)         | 500ms   | 80ms            | N/A
Inference (Hailo-8L)    | N/A     | N/A             | 30ms
Deployment Cost/year    | $881    | $881 (cloud)    | $104 (edge)
```

### Accuracy Retention

```yaml
DINOv2 Teacher: 100% (baseline)
Student Stage 1: 92% (initial distillation)
Student Stage 2: 95% (hard negative mining improvement)
Hailo INT8: 94% (minimal quantization loss)
```

**Conclusion:** 405x compression with only 6% accuracy loss

---

## Training Dataset

### Dataset Statistics

```yaml
Total Images: 17,592
Unique Cards: 2,199
Average Images per Card: 8
Image Resolution: Variable (resized to 518x518 for teacher, 224x224 for student)
Format: JPG/PNG
Total Size: 13 GB (raw images)
```

### Data Split

```yaml
Training: 14,074 images (80%)
Validation: 1,759 images (10%)
Test: 1,759 images (10%)
Stratified by Card ID: Yes
```

### S3 Locations

**Raw Images:**
```
s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/raw/card_images/
```

**Processed (Classification):**
```
s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/processed/classification/
```

**Calibration (Hailo):**
```
s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/calibration/
```

---

## Reference Database

### Production Inference Database

**Purpose:** Vector similarity search for card recognition

**Components:**
1. **Embeddings:** 17,592 x 768 FP32 embeddings (51.5 MB)
2. **uSearch Index:** ARM-optimized HNSW index (54.0 MB)
3. **Index Mapping:** Row → card_id lookup (652 KB)
4. **Metadata:** Card details (name, set, rarity, etc.) (543 KB)

**Total Size:** 106 MB

### Generation Process

```python
# Generate embeddings using Student Stage 2
import torch
import numpy as np

model = torch.load('student_stage2_final.pt')
model.eval()

embeddings = []
for image_path in card_images:
    image = preprocess(image_path)
    with torch.no_grad():
        embedding = model(image)
    embeddings.append(embedding.cpu().numpy())

embeddings = np.vstack(embeddings)  # Shape: (17592, 768)
np.save('embeddings.npy', embeddings)
```

```python
# Build uSearch index (ARM-optimized)
from usearch.index import Index

index = Index(
    ndim=768,
    metric='cos',  # Cosine similarity
    dtype='f32',
    connectivity=16,  # HNSW parameter
    expansion_add=128,
    expansion_search=64
)

for i, embedding in enumerate(embeddings):
    index.add(i, embedding)

index.save('usearch.index')
```

### S3 Location

```
s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/
├── embeddings.npy        # 51.5 MB
├── usearch.index         # 54.0 MB
├── index.json            # 652 KB
└── metadata.json         # 543 KB
```

### Usage in Inference

```python
import numpy as np
from usearch.index import Index
import json

# Load reference database
index = Index.restore('reference/usearch.index')
embeddings = np.load('reference/embeddings.npy')
with open('reference/index.json') as f:
    index_mapping = json.load(f)
with open('reference/metadata.json') as f:
    metadata = json.load(f)

# Query with new card image
query_embedding = model(new_image)  # Shape: (768,)
matches = index.search(query_embedding, k=5)

# Get top 5 matches
for i, (row_id, distance) in enumerate(zip(matches.keys, matches.distances)):
    card_id = index_mapping[str(row_id)]
    card_info = metadata[card_id]
    similarity = 1 - distance  # Convert distance to similarity
    print(f"{i+1}. {card_info['name']} (Set: {card_info['set']}) - Similarity: {similarity:.3f}")
```

---

## Model Lineage

### Parent-Child Relationships

```yaml
DINOv2 Teacher (Model #4):
  Parent: None (foundation model)
  Children:
    - Student Stage 1
  Purpose: High-accuracy teacher for knowledge distillation

Student Stage 1:
  Parent: DINOv2 Teacher
  Children:
    - Student Stage 2
  Purpose: Initial compression via distillation
  Status: Transitional (not preserved)

Student Stage 2 (Model #5):
  Parent: Student Stage 1
  Children:
    - Hailo Edge Model
  Purpose: Fine-tuned with hard negatives
  Status: Production

Hailo Edge Model:
  Parent: Student Stage 2
  Children: None (deployment model)
  Purpose: Edge inference on Raspberry Pi
  Status: Production
```

---

## Training Cost Breakdown

```yaml
Teacher Training: $4.00
  Instance: ml.p4d.24xlarge (8xA100)
  Duration: 12 minutes
  Cost: $32.77/hour → $6.55 for 12 min → $4.00 actual

Student Stage 1: $4.00
  Instance: ml.p4d.24xlarge (8xA100)
  Duration: 15 minutes
  Cost: $32.77/hour → $8.19 for 15 min → $4.00 actual

Student Stage 2: $3.00
  Instance: ml.p4d.24xlarge (8xA100)
  Duration: 10 minutes
  Cost: $32.77/hour → $5.46 for 10 min → $3.00 actual

Hailo Compilation: $0.50
  Instance: ml.m5.2xlarge (8 vCPU)
  Duration: 60 minutes
  Cost: $0.461/hour → $0.461 for 60 min → $0.50 actual

Total Training Cost: $11.50 USD
```

---

**Last Updated:** 2026-01-12
**Models in Production:** 2 (Teacher #4, Student Stage 2 #5)
**Models in Edge:** 1 (Hailo HEF v2.1)
