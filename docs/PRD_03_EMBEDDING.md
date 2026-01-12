# Phase 2: Embedding Model
## PRD_03_EMBEDDING.md

**Parent Document:** PRD_01_OVERVIEW.md
**Phase:** 2 of 5
**Hardware Target:** Hailo 8L (Raspberry Pi), iOS (CoreML), Android (ONNX)

---

## Objective

Build and deploy a visual embedding model that:
1. Converts card images into discriminative 768-dimensional vectors
2. Places similar cards close together in embedding space
3. Places different cards far apart
4. **Does NOT classify** - only extracts features

---

## Why Embeddings Instead of Classification?

### Classification Problems (Your Current Approach)

```
Input: Image of table
       |
+-------------------------------------+
|  Softmax over 17,592 classes        |
|  MUST sum to 1.0                    |
|                                     |
|  Zekrom:    0.0057  <- "Winner"     |
|  Pikachu:   0.0052                  |
|  Charizard: 0.0048                  |
|  ... 16,997 others: ~0.00005 each   |
|                                     |
|  Output: "Zekrom (57% confident)"   | <- WRONG!
+-------------------------------------+
```

### Embedding Solution

```
Input: Image of table
       |
+-------------------------------------+
|  Extract 768-dim feature vector     |
|  No classification, just features   |
|                                     |
|  Output: [0.23, -0.15, 0.87, ...]   |
+-------------------------------------+
       |
+-------------------------------------+
|  Compare to all 17,592 card         |
|  embeddings in database             |
|                                     |
|  Nearest: Zekrom (distance: 0.85)   |
|  Threshold: 0.30                    |
|                                     |
|  0.85 > 0.30 -> NOT A MATCH         |
|  Output: "No card detected"         | <- CORRECT!
+-------------------------------------+
```

---

## Model Architecture: DINOv3 Teacher-Student

We use a **two-phase training approach** with knowledge distillation:

1. **Phase 1:** Fine-tune a large "teacher" model (DINOv3-ViT-L/14)
2. **Phase 2:** Distill the teacher's knowledge to smaller "student" models for each deployment target

### Why This Approach?

| Approach | Training Cost | Accuracy | Flexibility |
|----------|--------------|----------|-------------|
| Train small model from scratch | $$$ | Good | One target only |
| **Fine-tune DINOv3 + Distill** | $ | Excellent | Multiple targets |

**Key Insight:** DINOv3 was pretrained by Meta on 1.7 billion images. We leverage this knowledge and add Pokemon card expertise through fine-tuning.

### Teacher Model: DINOv3-ViT-L/14

**Purpose:** Learn discriminative card embeddings with maximum accuracy.

```
+-------------------------------------+
|  DINOv3-ViT-L/14                    |
|  - 304M parameters                   |
|  - Pretrained on 1.7B images        |
|  - Input: 518x518                   |
|  - Output: 1024-dim features        |
+-------------------------------------+
                    |
                    v
+-------------------------------------+
|  Projection Head (Trainable)        |
|  - Linear 1024 -> 768               |
|  - GELU + Dropout                   |
|  - Linear 768 -> 768                |
|  - L2 Normalize                     |
+-------------------------------------+
                    |
                    v
+-------------------------------------+
|  ArcFace Head (Training Only)       |
|  - 17,592 classes                   |
|  - margin=0.5, scale=64             |
+-------------------------------------+
```

**Why DINOv3:**
- Self-supervised pretraining captures fine-grained visual details
- Excellent texture, color, and pattern understanding
- State-of-the-art for visual similarity tasks
- Robust to occlusion (attention focuses on visible regions)

### Student Models (Distilled for Deployment)

| Target | Architecture | Params | Input Size | Output | Format |
|--------|--------------|--------|------------|--------|--------|
| Hailo-8L | ConvNeXt-Tiny | 29M | 224x224 | 768-dim | .hef |
| iOS/CoreML | ConvNeXt-Base | 89M | 384x384 | 768-dim | .mlmodel |
| Android/ONNX | ViT-Small | 22M | 224x224 | 768-dim | .onnx |
| Server/API | DINOv3-ViT-L | 304M | 518x518 | 768-dim | .onnx |

**Why Different Students:**
- **Hailo-8L (ConvNeXt-Tiny):** Optimized for NPU deployment, CNN architecture compiles efficiently to HEF
- **iOS (ConvNeXt-Base):** CoreML handles larger models well, better accuracy for mobile
- **Android (ViT-Small):** Balance of accuracy and ONNX runtime performance

---

## Training Strategy

> **Note:** Detailed training code, SageMaker configurations, and step-by-step instructions are in **PRD_06_TRAINING.md**.

### Phase 1: Fine-tune Teacher (DINOv3-ViT-L/14)

**Objective:** Adapt DINOv3's pretrained features to discriminate between 17,592 Pokemon cards.

**Two-Stage Fine-tuning:**

```
Stage 1: Frozen Backbone (Epochs 1-5)
+-------------------------------------------+
|  DINOv3 Backbone: FROZEN                   |
|  Projection Head: TRAINABLE               |
|  ArcFace Head: TRAINABLE                  |
|  Learning Rate: 1e-3 (high, projection)   |
+-------------------------------------------+

Stage 2: Partial Unfreeze (Epochs 6-20)
+-------------------------------------------+
|  DINOv3 Backbone: Last 4 blocks TRAINABLE |
|  Projection Head: TRAINABLE               |
|  ArcFace Head: TRAINABLE                  |
|  Learning Rate: 1e-5 (low, backbone)      |
+-------------------------------------------+
```

**Loss Function: ArcFace**

ArcFace adds an angular margin to the softmax, forcing the model to learn more discriminative embeddings:

```
Standard Softmax:   P(class) = exp(W*x) / sum(exp(W*x))
ArcFace:           P(class) = exp(s*cos(theta + m)) / sum(exp(s*cos(theta)))

Where:
- theta = angle between embedding and class weight
- m = margin (0.5) - forces larger angular separation
- s = scale (64) - sharpens the distribution
```

### Phase 2: Distill to Students

**Objective:** Transfer teacher's knowledge to lightweight models via MSE loss on embeddings.

```
Teacher (DINOv3-ViT-L)          Student (ConvNeXt-T)
       |                              |
   [Image] ----------------------> [Image]
       |                              |
       v                              v
  Teacher                         Student
  Embedding <----- MSE Loss -----> Embedding
  (768-dim)                        (768-dim)
       |
       +---- Student learns to MIMIC teacher's output
```

**Key Benefits of Distillation:**
- Student never sees class labels - learns pure visual similarity
- Teacher's 1.7B image pretraining transfers to student
- One teacher trains unlimited student architectures
- Much cheaper than training each student from scratch

---

## Data Augmentation Strategy

### Standard Augmentations
```python
standard_augments = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, p=0.5),
    A.RandomRotate90(p=0.25),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Occlusion Augmentations (Critical for Your Use Case)
```python
occlusion_augments = A.Compose([
    # Random erasing (simulates fingers/obstruction)
    A.CoarseDropout(
        max_holes=8, max_height=32, max_width=32,
        min_holes=1, min_height=8, min_width=8,
        fill_value=0, p=0.5
    ),

    # GridMask (structured occlusion)
    GridMaskAugmentation(p=0.3),

    # Synthetic glare
    SyntheticGlareAugmentation(p=0.3),
])
```

### CutMix (Batch-Level)
```python
def cutmix_batch(images, labels, alpha=1.0):
    """Mix samples by cutting and pasting patches"""
    batch_size = images.size(0)
    indices = torch.randperm(batch_size)

    # Random box
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)

    # Mix images
    images[:, :, bbx1:bbx2, bby1:bby2] = images[indices, :, bbx1:bbx2, bby1:bby2]

    # Mix labels (for loss calculation)
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))

    return images, labels, labels[indices], lam
```

---

## Model Export for Deployment

### Hailo-8L (ConvNeXt-Tiny Student)

**Step 1: Export to ONNX**
```python
def export_to_onnx(model, output_dir):
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        f'{output_dir}/convnext_tiny.onnx',
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'embedding': {0: 'batch_size'}
        },
        opset_version=17,
    )
```

**Step 2: Convert for Hailo**
```bash
# Parse ONNX model
hailo parser onnx convnext_tiny.onnx

# Optimize with calibration data
hailo optimize convnext_tiny.har \
    --hw-arch hailo8l \
    --calib-set /path/to/calibration/images \
    --batch-size 8

# Compile to HEF
hailo compiler convnext_tiny_optimized.har \
    --hw-arch hailo8l \
    --output-dir ./hailo_output
```

**Step 3: Deploy on Raspberry Pi**
```python
# hailo_embedding.py

from hailo_platform import HailoRTDevice, ConfigureParams

class HailoEmbedder:
    """Run embedding model on Hailo 8L"""

    def __init__(self, hef_path: str):
        self.device = HailoRTDevice()
        self.hef = self.device.load_hef(hef_path)

        # Get input/output info
        self.input_vstream = self.hef.get_input_vstream_infos()[0]
        self.output_vstream = self.hef.get_output_vstream_infos()[0]

    def embed(self, image: np.ndarray) -> np.ndarray:
        """
        Convert preprocessed image to embedding.

        Args:
            image: Preprocessed image (1, 3, 224, 224), float32, normalized

        Returns:
            768-dimensional embedding vector
        """
        # Run inference
        with self.device.configure(self.hef) as configured_device:
            input_data = {self.input_vstream.name: image}
            results = configured_device.infer(input_data)

        embedding = results[self.output_vstream.name]

        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding
```

### iOS (ConvNeXt-Base Student)

```python
import coremltools as ct

# Convert ONNX to CoreML
model = ct.converters.onnx.convert(
    model='convnext_base.onnx',
    minimum_ios_deployment_target='16.0',
)

# Set metadata
model.short_description = "Pokemon Card Embedding Model"
model.input_description['input'] = "384x384 RGB image"
model.output_description['embedding'] = "768-dim embedding vector"

model.save('PokemonCardEmbedding.mlmodel')
```

### Android (ViT-Small Student)

ONNX model runs directly with ONNX Runtime Mobile:
```kotlin
val session = OrtEnvironment.getEnvironment()
    .createSession(modelPath, OrtSession.SessionOptions())

val inputTensor = OnnxTensor.createTensor(env, imageData)
val outputs = session.run(mapOf("input" to inputTensor))
val embedding = outputs[0].value as Array<FloatArray>
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Embedding inference (Hailo) | <25ms | ConvNeXt-Tiny student |
| Embedding inference (iOS) | <50ms | ConvNeXt-Base student |
| Embedding inference (Android) | <100ms | ViT-Small student |
| Top-1 retrieval accuracy | >=95% | Clean images |
| Top-5 retrieval accuracy | >=99% | Clean images |
| Occlusion robustness | >=85% | 20% occluded |
| Glare robustness | >=80% | Synthetic glare |
| Embedding dimension | 768 | All students |

---

## Acceptance Criteria

### AC-1: Embedding Quality (Clean Images)
```gherkin
GIVEN a clean, well-lit card image
WHEN computing its embedding and finding nearest neighbor in database
THEN the correct card MUST be the nearest neighbor
AND the distance to correct card MUST be <0.2
```

### AC-2: Embedding Quality (Occluded Images)
```gherkin
GIVEN a card image with 20% occlusion
WHEN computing its embedding and finding nearest neighbor
THEN the correct card MUST be in top-3 nearest neighbors
AND the distance to correct card MUST be <0.4
```

### AC-3: Embedding Quality (Glare)
```gherkin
GIVEN a holo/gold card image with visible glare
WHEN computing its embedding and finding nearest neighbor
THEN the correct card MUST be in top-5 nearest neighbors
```

### AC-4: Unknown Input Rejection
```gherkin
GIVEN an image that is NOT a Pokemon card (table, hand, etc.)
WHEN computing its embedding and finding nearest neighbor
THEN the distance to nearest card MUST be >0.5
AND the system MUST report "No match found"
```

### AC-5: Inference Speed
```gherkin
GIVEN the ConvNeXt-Tiny student model deployed on Hailo 8L
WHEN processing a single image
THEN inference MUST complete in <25ms
```

### AC-6: Embedding Consistency
```gherkin
GIVEN two images of the same card (different conditions)
WHEN computing embeddings for both
THEN the cosine similarity MUST be >0.8
```

### AC-7: Embedding Discrimination
```gherkin
GIVEN embeddings of two DIFFERENT cards
WHEN computing cosine similarity
THEN the similarity MUST be <0.5
```

### AC-8: Student-Teacher Alignment
```gherkin
GIVEN the same image processed by teacher and student
WHEN computing cosine similarity between their embeddings
THEN the similarity MUST be >0.95
```

---

## Testing Plan

### Unit Tests
```python
def test_embedding_dimension():
    """Embedding has correct dimension"""
    model = StudentModel('convnext_tiny', embedding_dim=768)
    image = torch.randn(1, 3, 224, 224)
    embedding = model(image)
    assert embedding.shape == (1, 768)

def test_embedding_normalized():
    """Embedding is L2 normalized"""
    model = StudentModel('convnext_tiny', embedding_dim=768)
    image = torch.randn(1, 3, 224, 224)
    embedding = model(image)
    norm = torch.norm(embedding, p=2, dim=1)
    assert torch.allclose(norm, torch.ones(1), atol=1e-5)

def test_same_card_similarity():
    """Same card with augmentation has high similarity"""
    model = StudentModel('convnext_tiny', embedding_dim=768)

    image1 = load_card_image('pikachu.jpg')
    image2 = augment(image1)  # Rotated, color-jittered

    emb1 = model(image1)
    emb2 = model(image2)

    similarity = torch.cosine_similarity(emb1, emb2)
    assert similarity > 0.8

def test_different_card_dissimilarity():
    """Different cards have low similarity"""
    model = StudentModel('convnext_tiny', embedding_dim=768)

    emb1 = model(load_card_image('pikachu.jpg'))
    emb2 = model(load_card_image('charizard.jpg'))

    similarity = torch.cosine_similarity(emb1, emb2)
    assert similarity < 0.5
```

### Integration Tests
```python
def test_hailo_deployment():
    """Model runs on Hailo hardware"""
    embedder = HailoEmbedder('convnext_tiny.hef')

    image = preprocess_image(load_image('test_card.jpg'))
    embedding = embedder.embed(image)

    assert embedding.shape == (768,)
    assert np.isclose(np.linalg.norm(embedding), 1.0)

def test_retrieval_accuracy():
    """End-to-end retrieval accuracy"""
    embedder = HailoEmbedder('convnext_tiny.hef')
    database = load_reference_database('card_embeddings.json')

    correct = 0
    total = 0

    for card_id, test_images in test_set.items():
        for image in test_images:
            embedding = embedder.embed(preprocess(image))
            nearest_id = database.find_nearest(embedding)

            if nearest_id == card_id:
                correct += 1
            total += 1

    accuracy = correct / total
    assert accuracy >= 0.95

def test_student_teacher_alignment():
    """Student produces similar embeddings to teacher"""
    teacher = DINOv3TeacherModel()
    student = StudentModel('convnext_tiny', embedding_dim=768)

    for image in test_images:
        teacher_emb = teacher(resize(image, 518))
        student_emb = student(resize(image, 224))

        similarity = torch.cosine_similarity(teacher_emb, student_emb)
        assert similarity > 0.95
```

---

## Cost Estimates

### Training Costs (AWS SageMaker)

| Phase | Instance | Duration | Spot Cost |
|-------|----------|----------|-----------|
| Teacher fine-tuning | ml.g5.4xlarge | ~3 hours | ~$1.80 |
| Hailo student distillation | ml.g4dn.xlarge | ~1 hour | ~$0.25 |
| iOS student distillation | ml.g4dn.xlarge | ~1 hour | ~$0.25 |
| Android student distillation | ml.g4dn.xlarge | ~1 hour | ~$0.25 |
| **Total** | | | **~$2.55** |

**Compare to training from scratch:** Would cost $50+ per model.

### Inference Costs (On-Device)

All inference runs locally on device hardware - no cloud costs:
- Raspberry Pi + Hailo 8L: ~$0 per inference
- iOS device: ~$0 per inference
- Android device: ~$0 per inference

---

## Deliverables

| Deliverable | Format | Location |
|-------------|--------|----------|
| Teacher model | `.pt`, `.onnx` | S3 bucket |
| Hailo student | `.pt`, `.onnx`, `.hef` | S3 + Raspberry Pi |
| iOS student | `.pt`, `.onnx`, `.mlmodel` | S3 + App bundle |
| Android student | `.pt`, `.onnx` | S3 + App assets |
| Training scripts | `.py` | Git repo (`src/training/`) |
| Hailo embedder class | `.py` | Git repo |
| Unit tests | `.py` | Git repo |

---

## Next Phase

Upon completion of Phase 2, proceed to **PRD_04_DATABASE.md** for the reference database and similarity search implementation.

---

## References

- [DINOv3 GitHub](https://github.com/facebookresearch/dinov3)
- [DINOv3 Paper](https://arxiv.org/abs/2508.10104)
- [ArcFace Paper](https://arxiv.org/abs/1801.07698)
- [Knowledge Distillation Survey](https://arxiv.org/abs/2006.05525)
- [ConvNeXt Paper](https://arxiv.org/abs/2201.03545)
