# Phase 1: Detection Model
## PRD_02_DETECTION.md

**Parent Document:** PRD_01_OVERVIEW.md
**Phase:** 1 of 5
**Hardware Target:** Sony IMX500 AI Camera (on-sensor processing)

---

## Objective

Build and deploy a lightweight detection model that:
1. Detects presence/absence of Pokemon cards in frame
2. Outputs bounding box coordinates to crop the card region
3. Runs entirely on IMX500 sensor (24/7 low-power operation)
4. Hands off cropped card image to Hailo-8L for embedding inference

## Two-Stage Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: IMX500 Camera (Always Running, Low Power)        │
│  ┌────────────┐                                             │
│  │  Camera    │ ──► Lightweight YOLO ──► Crop Card         │
│  │  Sensor    │     (on-sensor)          Region            │
│  └────────────┘                               │             │
└────────────────────────────────────────────────┼─────────────┘
                                                 ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: Hailo-8L (Only When Card Detected)               │
│  ┌────────────┐                                             │
│  │  Cropped   │ ──► ConvNeXt-Tiny ──► 768-dim ──► Match    │
│  │  Card      │     Embedding         Embedding  Database  │
│  └────────────┘                                             │
└─────────────────────────────────────────────────────────────┘
```

**Why Two Stages?**
- **Power Efficiency**: IMX500 runs at <1W, Hailo-8L sleeps until card detected
- **Real-time Detection**: On-sensor processing has zero latency to CPU
- **Better Recognition**: Hailo-8L focuses all compute on embedding task
- **Modularity**: Can swap detection/embedding models independently

---

## Detection Approach: Standard Bounding Box

For Stage 1 (IMX500), we use a **standard axis-aligned bounding box**. This is sufficient because:

1. **Embedding model is rotation-invariant**: DINOv3-based models trained with rotation augmentation
2. **IMX500 compatibility**: Standard YOLO detection well-supported in Sony's model zoo
3. **Simplicity**: Easier deployment, lower compute requirements
4. **Crop-and-feed**: We just need a rough card region to pass to Hailo

**Detection Output:**
```
┌──────────────────┐
│   ╱╲             │  ← Axis-aligned bbox
│  ╱  ╲            │     includes some background
│ ╱Card╲           │     but that's OK!
│╱      ╲          │
└──────────────────┘
(x, y, w, h)
```

**Why background noise is acceptable:**
- The embedding model sees the full cropped region
- DINOv3 is trained to focus on salient objects (the card)
- Background is typically neutral (table, hands)
- Training with augmented backgrounds makes model robust

**Rotation handling:**
- Embedding model trained with 0-360° rotation augmentation
- Reference database embeddings include rotated samples
- Cosine similarity works regardless of card orientation

---

## Model Specification

### Architecture
- **Base:** YOLOv8n (nano variant, standard detection)
- **Backbone:** CSPDarknet with C2f modules
- **Parameters:** ~3.2M
- **Quantization:** INT8 for IMX500 deployment
- **IMX500 Support:** Official support via Sony's model conversion tools

**Why YOLOv8n instead of YOLO11n?**
- Proven IMX500 compatibility (Sony's model zoo supports YOLOv8)
- Stable toolchain for INT8 quantization
- Extensive deployment examples available
- YOLO11 support on IMX500 not yet verified

### Detection Output

```python
{
    "class": "pokemon_card",      # Single class: "card"
    "confidence": 0.95,           # Detection confidence
    "bbox": {
        "x1": 120,                # Top-left x
        "y1": 80,                 # Top-left y
        "x2": 520,                # Bottom-right x
        "y2": 400                 # Bottom-right y
    },
    "crop_ready": True            # Ready to pass to Hailo
}
```

**Handoff to Stage 2:**
```python
# IMX500 outputs bbox → Pi crops image → Hailo processes
cropped_card = frame[y1:y2, x1:x2]
embedding = hailo_inference(cropped_card)  # Stage 2
```

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Stage 1 (IMX500 Detection)** | | |
| Inference time | <15ms | On-sensor processing |
| Detection mAP@0.5 | ≥0.90 | Standard COCO metric |
| False positive rate | <2% | Non-cards detected as cards |
| False negative rate | <3% | Cards missed |
| Power consumption | <1W | 24/7 operation |
| **Stage 2 (Hailo-8L Embedding)** | | |
| Inference time | <25ms | ConvNeXt-Tiny on Hailo |
| Embedding accuracy | ≥95% top-1 | On validation set |
| Power consumption | ~2-4W | Only when card detected |
| **End-to-End Latency** | | |
| Detection → Crop → Embed | <50ms | Total pipeline latency |

---

## Training Strategy: Knowledge Distillation

Instead of training from scratch, we'll use **knowledge distillation** from a pre-trained Pokemon TCG card detector.

### Pre-trained Teacher Model

**Source:** [Pokemon-TCGP-Card-Scanner project](https://github.com/1vcian/Pokemon-TCGP-Card-Scanner)
- **Model:** YOLO11n-OBB trained on 10k synthetic Pokemon card images
- **Download:** https://hub.ultralytics.com/models/dQfecRsRsXbAKXOXHLHJ
- **Dataset:** https://hub.ultralytics.com/datasets/8awcqoIQP0jIXIMDOCsC

**Why use a pre-trained model?**
- Already excellent at detecting Pokemon TCG cards
- Trained on 10k diverse synthetic images (rotation, scaling, occlusion, etc.)
- Saves weeks of data collection and annotation
- Proven performance in production use

### Distillation Approach

```
┌────────────────────────────────────────────────────┐
│  Teacher: YOLO11n-OBB (Pre-trained)                │
│  Input: 640×640 image                              │
│  Output: OBB detections (x, y, w, h, θ, conf)      │
└────────────┬───────────────────────────────────────┘
             │ Extract bbox labels (ignore angle)
             │ Generate soft targets (class probs)
             ▼
┌────────────────────────────────────────────────────┐
│  Student: YOLOv8n (Standard Detection)             │
│  Input: 640×640 image                              │
│  Output: Bbox detections (x, y, w, h, conf)        │
│  Loss: α * Hard Loss + (1-α) * Soft Loss           │
└────────────────────────────────────────────────────┘
```

**Distillation Benefits:**
1. Student learns from teacher's knowledge (10k images worth)
2. Handles architecture conversion (YOLO11-OBB → YOLOv8-bbox)
3. Maintains high detection accuracy
4. IMX500-compatible output format
5. Faster than training from scratch

### Training Dataset

We'll reuse the **10k synthetic dataset** from the reference project:
- **Download:** https://hub.ultralytics.com/datasets/8awcqoIQP0jIXIMDOCsC
- **Format:** YOLO OBB format (we'll convert to standard bbox)
- **Augmentations:** Already includes rotation, scaling, perspective, occlusion
- **Diversity:** Multiple cards per image, various backgrounds

**Dataset Statistics:**
| Category | Count | Description |
|----------|-------|-------------|
| Total images | 10,000 | Synthetic generated |
| Cards per image | 1-50 | Variable density |
| Backgrounds | ~1,000 | From Sample Images Repository |
| Augmentations | Full | Rotation, scale, perspective, color, noise |

---

## Training Configuration

### Distillation Training Script

```python
# train_detection_distillation.py

import argparse
from ultralytics import YOLO
import torch
import torch.nn as nn
from pathlib import Path

def convert_obb_to_bbox(obb_labels):
    """Convert OBB labels (x, y, w, h, angle) to standard bbox (x, y, w, h)"""
    # Take axis-aligned bounding box that encompasses the OBB
    # This is a simplified conversion - actual implementation needs rotation matrix
    return obb_labels[..., :4]  # Just take x, y, w, h

def distillation_train(args):
    # Load teacher model (pre-trained YOLO11n-OBB)
    teacher = YOLO('yolo11n-obb.pt')  # Downloaded from reference project
    teacher.model.eval()

    # Load student model (YOLOv8n standard detection)
    student = YOLO('yolov8n.pt')

    # Distillation training configuration
    results = student.train(
        data=args.data_yaml,
        epochs=50,  # Fewer epochs needed with distillation
        imgsz=640,
        batch=16,
        device=0,

        # Distillation parameters
        patience=15,
        save_period=5,

        # Optimizer
        optimizer='AdamW',
        lr0=0.001,  # Lower LR for distillation

        # Augmentation (dataset already has augmentation)
        hsv_h=0.0,  # Disable extra augmentation
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,

        # Output
        project=args.output_dir,
        name='yolov8n_distilled',
        exist_ok=True,
    )

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-yaml', type=str,
                       default='./detection_dataset/data.yaml')
    parser.add_argument('--output-dir', type=str, default='./outputs')
    args = parser.parse_args()

    distillation_train(args)
```

### Dataset Preparation

```python
# prepare_detection_dataset.py

import yaml
from pathlib import Path
import shutil

def convert_obb_dataset_to_bbox(obb_dataset_path, output_path):
    """
    Convert YOLO OBB format to standard bbox format.

    OBB format: class x_center y_center width height angle
    Bbox format: class x_center y_center width height
    """
    obb_path = Path(obb_dataset_path)
    out_path = Path(output_path)
    out_path.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'val']:
        (out_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (out_path / split / 'labels').mkdir(parents=True, exist_ok=True)

        # Copy images
        for img in (obb_path / split / 'images').glob('*'):
            shutil.copy(img, out_path / split / 'images' / img.name)

        # Convert labels (remove angle from OBB format)
        for label_file in (obb_path / split / 'labels').glob('*.txt'):
            with open(label_file) as f:
                lines = f.readlines()

            converted_lines = []
            for line in lines:
                parts = line.strip().split()
                # Keep only: class_id, x_center, y_center, width, height
                # Remove angle (last value in OBB format)
                bbox_line = ' '.join(parts[:5]) + '\n'
                converted_lines.append(bbox_line)

            with open(out_path / split / 'labels' / label_file.name, 'w') as f:
                f.writelines(converted_lines)

    # Create data.yaml
    data_yaml = {
        'path': str(out_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'names': {0: 'card'},
        'nc': 1,
    }

    with open(out_path / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)

    print(f"Dataset converted: {out_path}")
    print(f"  Train images: {len(list((out_path / 'train' / 'images').glob('*')))}")
    print(f"  Val images: {len(list((out_path / 'val' / 'images').glob('*')))}")

if __name__ == '__main__':
    # Download dataset from: https://hub.ultralytics.com/datasets/8awcqoIQP0jIXIMDOCsC
    convert_obb_dataset_to_bbox(
        obb_dataset_path='./pokemon_cards_obb',
        output_path='./detection_dataset'
    )
```

### SageMaker Setup (Optional)

```python
# sagemaker_detection_training.py

from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role

role = get_execution_role()

estimator = PyTorch(
    entry_point='train_detection_distillation.py',
    source_dir='./src/training',
    role=role,
    instance_count=1,
    instance_type='ml.g4dn.xlarge',  # T4 GPU
    framework_version='2.0',
    py_version='py310',
    hyperparameters={
        'data-yaml': '/opt/ml/input/data/dataset/data.yaml',
        'output-dir': '/opt/ml/model',
    },
    output_path='s3://pokemon-card-recognition/models/detection/',
    use_spot_instances=True,
    max_wait=7200,  # 2 hours max
    max_run=3600,   # 1 hour training
)

estimator.fit({
    'dataset': 's3://pokemon-card-recognition/datasets/detection/',
})

from ultralytics import YOLO
import argparse
import os

def train(args):
    # Load pretrained pose model
    model = YOLO('yolov8n-pose.pt')
    
    # Train on card detection
    results = model.train(
        data=os.path.join(args.data_dir, 'dataset.yaml'),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        patience=args.patience,
        device=0,
        
        # Keypoint configuration
        kpt_shape=[4, 2],  # 4 keypoints, 2 coords (x, y)
        
        # Augmentation for robustness
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        translate=0.1,
        scale=0.5,
        perspective=0.001,
        flipud=0.0,  # Don't flip cards vertically
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        
        # Output
        project=args.output_dir,
        name='card_detector',
    )
    
    # Export for IMX500
    model.export(format='onnx', imgsz=640, simplify=True)
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--data_dir', type=str, default='/opt/ml/input/data/train')
    parser.add_argument('--output_dir', type=str, default='/opt/ml/model')
    args = parser.parse_args()
    
    train(args)
```

### Dataset YAML

```yaml
# dataset.yaml

path: /opt/ml/input/data
train: train/images
val: val/images

# Single class
names:
  0: pokemon_card

# 4 keypoints for corners
kpt_shape: [4, 2]

# Flip index mapping (for horizontal flip augmentation)
# TL↔TR, BL↔BR when flipped
flip_idx: [1, 0, 3, 2]
```

---

## Model Export for IMX500

### Step 1: ONNX Export (from training)
```python
from ultralytics import YOLO

model = YOLO('best.pt')
model.export(format='onnx', imgsz=640, simplify=True)
```

### Step 2: IMX500 Conversion
```bash
# Install Sony's IMX500 tools
pip install imx500-converter

# Convert ONNX to IMX500 format
imx500-converter \
    --model card_detector.onnx \
    --output card_detector.rpk \
    --input-size 640 640 \
    --quantize int8
```

### Step 3: Deploy to Camera
```python
# deploy_to_camera.py
from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500

# Initialize camera with model
imx500 = IMX500("card_detector.rpk")
camera = Picamera2(imx500.camera_num)

config = camera.create_preview_configuration(
    controls={"FrameRate": 30},
    buffer_count=4
)
camera.configure(config)
camera.start()

# Detection runs automatically on IMX500
while True:
    metadata = camera.capture_metadata()
    detections = imx500.get_outputs(metadata)
    
    for det in detections:
        if det['confidence'] > 0.7:
            print(f"Card detected: {det['keypoints']}")
```

---

## Inference Pipeline Integration

### Detection Output Processing

```python
# detection_processor.py

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class CardDetection:
    """Processed detection result"""
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    corners: np.ndarray  # 4x2 array of corner points
    keypoint_confidences: np.ndarray  # 4 confidence values
    
    @property
    def is_valid(self) -> bool:
        """Check if detection is usable"""
        return (
            self.confidence > 0.7 and
            np.all(self.keypoint_confidences > 0.5) and
            self._corners_form_quadrilateral()
        )
    
    def _corners_form_quadrilateral(self) -> bool:
        """Validate corners form a reasonable card shape"""
        # Check area is reasonable
        area = self._shoelace_area(self.corners)
        bbox_area = (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
        
        # Card should fill 50-100% of bbox
        return 0.5 < (area / bbox_area) < 1.1
    
    @staticmethod
    def _shoelace_area(corners: np.ndarray) -> float:
        """Calculate area using shoelace formula"""
        n = len(corners)
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += corners[i, 0] * corners[j, 1]
            area -= corners[j, 0] * corners[i, 1]
        return abs(area) / 2


class DetectionProcessor:
    """Process raw IMX500 outputs into CardDetection objects"""
    
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        keypoint_threshold: float = 0.5
    ):
        self.confidence_threshold = confidence_threshold
        self.keypoint_threshold = keypoint_threshold
    
    def process(self, raw_outputs: List[dict]) -> List[CardDetection]:
        """Convert raw IMX500 outputs to CardDetection objects"""
        detections = []
        
        for output in raw_outputs:
            if output.get('confidence', 0) < self.confidence_threshold:
                continue
            
            # Extract keypoints
            kps = np.array(output['keypoints']).reshape(4, 3)  # 4 points, (x, y, conf)
            corners = kps[:, :2]
            kp_confs = kps[:, 2]
            
            # Create detection object
            detection = CardDetection(
                confidence=output['confidence'],
                bbox=tuple(output['bbox']),
                corners=corners,
                keypoint_confidences=kp_confs
            )
            
            if detection.is_valid:
                detections.append(detection)
        
        # Sort by confidence, return best
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


def order_corners(corners: np.ndarray) -> np.ndarray:
    """
    Order corners consistently: TL, TR, BR, BL
    Required for perspective transform
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Top-left has smallest sum (x+y)
    # Bottom-right has largest sum
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    
    # Top-right has smallest difference (y-x)
    # Bottom-left has largest difference
    diff = np.diff(corners, axis=1).flatten()
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]
    
    return rect
```

---

## Acceptance Criteria

### AC-1: Detection Accuracy
```gherkin
GIVEN a frame containing a Pokemon card
WHEN the detection model processes the frame
THEN the model MUST detect the card with confidence ≥0.7
AND the bounding box IoU with ground truth MUST be ≥0.85
```

### AC-2: Keypoint Accuracy
```gherkin
GIVEN a detected Pokemon card
WHEN extracting corner keypoints
THEN each keypoint MUST be within 5 pixels of actual corner
AND keypoint confidence MUST be ≥0.5 for all 4 corners
```

### AC-3: False Positive Rejection
```gherkin
GIVEN a frame with NO Pokemon card (table, hand, other objects)
WHEN the detection model processes the frame
THEN the model MUST NOT output any detection with confidence ≥0.5
```

### AC-4: Rotation Handling
```gherkin
GIVEN a Pokemon card rotated up to 45 degrees
WHEN the detection model processes the frame
THEN detection confidence MUST remain ≥0.7
AND keypoints MUST correctly identify the 4 corners
```

### AC-5: Partial Visibility
```gherkin
GIVEN a Pokemon card with up to 20% outside the frame
WHEN the detection model processes the frame
THEN the model SHOULD detect with confidence ≥0.6
AND visible corners MUST have keypoint confidence ≥0.5
```

### AC-6: Inference Speed
```gherkin
GIVEN the model deployed on IMX500
WHEN processing continuous video frames
THEN inference time MUST be <10ms per frame
AND the camera MUST maintain ≥30 FPS
```

### AC-7: Edge Cases
```gherkin
GIVEN challenging conditions:
  - Multiple cards in frame (detect at least one)
  - Fingers partially covering card (still detect)
  - Low light conditions (detect with reduced confidence)
WHEN the detection model processes the frame
THEN the model SHOULD handle gracefully without crashing
AND SHOULD provide valid keypoints for visible portions
```

---

## Testing Plan

### Unit Tests
```python
# test_detection.py

def test_corner_ordering():
    """Test corners are ordered TL, TR, BR, BL"""
    # Random quadrilateral
    corners = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
    shuffled = corners[np.random.permutation(4)]
    ordered = order_corners(shuffled)
    assert np.allclose(ordered, corners)

def test_detection_validation():
    """Test detection validity checks"""
    valid_detection = CardDetection(
        confidence=0.9,
        bbox=(100, 100, 300, 400),
        corners=np.array([[100, 100], [300, 100], [300, 400], [100, 400]]),
        keypoint_confidences=np.array([0.9, 0.9, 0.9, 0.9])
    )
    assert valid_detection.is_valid
    
    low_conf_detection = CardDetection(
        confidence=0.5,
        bbox=(100, 100, 300, 400),
        corners=np.array([[100, 100], [300, 100], [300, 400], [100, 400]]),
        keypoint_confidences=np.array([0.9, 0.9, 0.9, 0.9])
    )
    assert not low_conf_detection.is_valid

def test_false_positive_rejection():
    """Test that non-cards are rejected"""
    processor = DetectionProcessor(confidence_threshold=0.7)
    
    # Simulate low-confidence outputs for non-card
    raw_outputs = [{'confidence': 0.3, 'bbox': [0,0,100,100], 'keypoints': [...]}]
    detections = processor.process(raw_outputs)
    
    assert len(detections) == 0
```

### Integration Tests
```python
def test_imx500_deployment():
    """Test model runs on actual hardware"""
    from picamera2 import Picamera2
    from picamera2.devices.imx500 import IMX500
    
    imx500 = IMX500("card_detector.rpk")
    camera = Picamera2(imx500.camera_num)
    camera.start()
    
    # Capture and process
    metadata = camera.capture_metadata()
    outputs = imx500.get_outputs(metadata)
    
    assert outputs is not None
    camera.stop()

def test_inference_latency():
    """Test inference meets speed requirements"""
    import time
    
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        metadata = camera.capture_metadata()
        outputs = imx500.get_outputs(metadata)
        latencies.append(time.perf_counter() - start)
    
    avg_latency = np.mean(latencies)
    assert avg_latency < 0.010  # 10ms
```

---

## Deliverables

| Deliverable | Format | Location |
|-------------|--------|----------|
| Trained model | `.pt` | `s3://pokemon-card-recognition/models/detection/` |
| ONNX export | `.onnx` | `s3://pokemon-card-recognition/models/detection/` |
| IMX500 model | `.rpk` | Raspberry Pi |
| Training script | `.py` | Git repo |
| Dataset | Images + YAML | `s3://pokemon-card-recognition/datasets/detection/` |
| Detection processor | `.py` | Git repo |
| Unit tests | `.py` | Git repo |

---

## Dependencies

### Training
- ultralytics >= 8.0
- torch >= 2.0
- sagemaker SDK

### Deployment
- picamera2
- imx500-converter
- numpy

---

## Next Phase

Upon completion of Phase 1, proceed to **PRD_03_EMBEDDING.md** for the embedding model.
