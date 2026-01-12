# Phase 4: Matching Pipeline
## PRD_05_PIPELINE.md

**Parent Document:** PRD_01_OVERVIEW.md  
**Phase:** 4 of 5  
**Duration:** 1 week  
**Target:** End-to-end inference on Raspberry Pi 5

---

## Objective

Build the complete inference pipeline that:
1. Captures frames from AI Camera
2. Detects cards and extracts corners (IMX500)
3. Crops and preprocesses cards (CPU)
4. Computes embeddings (Hailo 8)
5. Searches reference database (CPU)
6. Applies temporal smoothing and rejection logic
7. Outputs final card identification or rejection reason

---

## Pipeline Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         INFERENCE PIPELINE                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │  Camera  │──▶│ Detection│──▶│  Crop &  │──▶│ Embedding│──▶│  Search  │ │
│  │  Capture │   │  (IMX500)│   │  Preproc │   │ (Hailo 8)│   │    &     │ │
│  │          │   │          │   │  (CPU)   │   │          │   │  Smooth  │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│       │              │              │              │              │        │
│       ▼              ▼              ▼              ▼              ▼        │
│   Raw Frame     Detection      Cropped Card   768-dim Vector   CardMatch  │
│   2028x1520     + 4 Corners    224x224        (normalized)     or None    │
│                                                                            │
│  ════════════════════════════════════════════════════════════════════════ │
│                           DECISION GATES                                   │
│  ════════════════════════════════════════════════════════════════════════ │
│                                                                            │
│  Gate 1: Detection               Gate 2: Distance                          │
│  ┌─────────────────────┐        ┌─────────────────────┐                   │
│  │ No card detected?   │        │ Distance > 0.4?     │                   │
│  │ Confidence < 0.7?   │        │                     │                   │
│  │                     │        │ → "Not a known card"│                   │
│  │ → "No card in frame"│        │                     │                   │
│  └─────────────────────┘        └─────────────────────┘                   │
│                                                                            │
│  Gate 3: Temporal Stability                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ Same card predicted for 3+ frames with distance < 0.3?              │  │
│  │                                                                      │  │
│  │ → "Confirmed: [Card Name]"                                          │  │
│  │                                                                      │  │
│  │ If unstable (different predictions): "Stabilizing..."               │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Implementation

### Main Pipeline Class

```python
# card_recognition_pipeline.py

import numpy as np
import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum
from collections import deque

from detection_processor import DetectionProcessor, CardDetection, order_corners
from preprocessing import CardPreprocessor
from hailo_embedding import HailoEmbedder
from reference_database import ReferenceDatabase, CardMatch

logger = logging.getLogger(__name__)


class RecognitionStatus(Enum):
    """Pipeline output states"""
    NO_CARD = "no_card"              # No card detected in frame
    DETECTING = "detecting"           # Card detected, processing
    STABILIZING = "stabilizing"       # Predictions not yet stable
    LOW_CONFIDENCE = "low_confidence" # Match found but uncertain
    UNKNOWN_CARD = "unknown_card"     # Card detected but not in database
    CONFIRMED = "confirmed"           # High-confidence match


@dataclass
class PipelineResult:
    """Complete result from pipeline"""
    status: RecognitionStatus
    card_match: Optional[CardMatch]
    top_matches: List[CardMatch]
    detection_confidence: float
    embedding_distance: float
    temporal_stability: float  # 0-1, how stable the prediction is
    processing_time_ms: float
    frame_count: int
    
    def to_dict(self) -> dict:
        return {
            'status': self.status.value,
            'card': self.card_match.to_dict() if self.card_match else None,
            'alternatives': [m.to_dict() for m in self.top_matches[1:4]],
            'detection_confidence': self.detection_confidence,
            'embedding_distance': self.embedding_distance,
            'temporal_stability': self.temporal_stability,
            'processing_time_ms': self.processing_time_ms,
            'frame_count': self.frame_count,
        }


class TemporalSmoother:
    """
    Smooth predictions over time for stable output.
    
    Prevents flickering between similar cards and ensures
    confident predictions before reporting.
    """
    
    def __init__(
        self,
        stability_threshold: int = 3,      # Frames needed for confirmation
        memory_size: int = 10,             # Frames to remember
        distance_ema_alpha: float = 0.3,   # Smoothing factor
    ):
        self.stability_threshold = stability_threshold
        self.memory_size = memory_size
        self.alpha = distance_ema_alpha
        
        self.prediction_history = deque(maxlen=memory_size)
        self.distance_ema = None
        self.stable_card_id = None
        self.stable_count = 0
        
    def update(self, match: Optional[CardMatch]) -> Tuple[float, bool]:
        """
        Update with new prediction.
        
        Returns:
            Tuple of (stability_score, is_confirmed)
        """
        if match is None:
            self.reset()
            return 0.0, False
        
        card_id = match.card_id
        distance = match.distance
        
        # Update distance EMA
        if self.distance_ema is None:
            self.distance_ema = distance
        else:
            self.distance_ema = self.alpha * distance + (1 - self.alpha) * self.distance_ema
        
        # Track prediction consistency
        self.prediction_history.append(card_id)
        
        if card_id == self.stable_card_id:
            self.stable_count += 1
        else:
            self.stable_card_id = card_id
            self.stable_count = 1
        
        # Calculate stability score
        if len(self.prediction_history) > 0:
            most_common = max(set(self.prediction_history), 
                            key=list(self.prediction_history).count)
            stability = list(self.prediction_history).count(most_common) / len(self.prediction_history)
        else:
            stability = 0.0
        
        is_confirmed = self.stable_count >= self.stability_threshold
        
        return stability, is_confirmed
    
    def reset(self):
        """Reset state (when card leaves frame)"""
        self.prediction_history.clear()
        self.distance_ema = None
        self.stable_card_id = None
        self.stable_count = 0


class CardRecognitionPipeline:
    """
    Complete card recognition pipeline.
    
    Integrates:
    - Detection (IMX500)
    - Preprocessing (CPU)
    - Embedding (Hailo 8)
    - Database search (CPU)
    - Temporal smoothing (CPU)
    """
    
    def __init__(
        self,
        camera,                    # Picamera2 instance with IMX500
        imx500,                    # IMX500 model handler
        embedder: HailoEmbedder,   # Hailo embedding model
        database: ReferenceDatabase,
        
        # Thresholds
        detection_threshold: float = 0.7,
        distance_threshold: float = 0.4,      # Max distance to accept
        high_confidence_threshold: float = 0.25,  # Distance for high confidence
        
        # Temporal
        stability_frames: int = 3,
    ):
        self.camera = camera
        self.imx500 = imx500
        self.embedder = embedder
        self.database = database
        
        self.detection_threshold = detection_threshold
        self.distance_threshold = distance_threshold
        self.high_confidence_threshold = high_confidence_threshold
        
        # Components
        self.detection_processor = DetectionProcessor(
            confidence_threshold=detection_threshold
        )
        self.preprocessor = CardPreprocessor(
            output_size=(224, 224),
            padding_percent=0.03,  # Tight crop!
        )
        self.smoother = TemporalSmoother(
            stability_threshold=stability_frames
        )
        
        self.frame_count = 0
        self._last_detection = None
    
    def process_frame(self) -> PipelineResult:
        """
        Process single frame through entire pipeline.
        
        Returns:
            PipelineResult with status and match information
        """
        start_time = time.perf_counter()
        self.frame_count += 1
        
        # ═══════════════════════════════════════════════════════════
        # STAGE 1: Capture + Detection (IMX500)
        # ═══════════════════════════════════════════════════════════
        
        frame = self.camera.capture_array()
        metadata = self.camera.capture_metadata()
        raw_detections = self.imx500.get_outputs(metadata)
        
        detections = self.detection_processor.process(raw_detections)
        
        # Gate 1: No card detected
        if len(detections) == 0:
            self.smoother.reset()
            self._last_detection = None
            
            return PipelineResult(
                status=RecognitionStatus.NO_CARD,
                card_match=None,
                top_matches=[],
                detection_confidence=0.0,
                embedding_distance=1.0,
                temporal_stability=0.0,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                frame_count=self.frame_count,
            )
        
        # Take best detection
        detection = detections[0]
        self._last_detection = detection
        
        # ═══════════════════════════════════════════════════════════
        # STAGE 2: Preprocessing (CPU)
        # ═══════════════════════════════════════════════════════════
        
        # Order corners and crop
        ordered_corners = order_corners(detection.corners)
        
        tensor, glare_info = self.preprocessor.process(frame, ordered_corners)
        
        # ═══════════════════════════════════════════════════════════
        # STAGE 3: Embedding (Hailo 8)
        # ═══════════════════════════════════════════════════════════
        
        embedding = self.embedder.embed(tensor)
        
        # ═══════════════════════════════════════════════════════════
        # STAGE 4: Database Search (CPU)
        # ═══════════════════════════════════════════════════════════
        
        matches = self.database.search(
            embedding,
            top_k=5,
            distance_threshold=self.distance_threshold
        )
        
        # Gate 2: No match in database (unknown card or not a card)
        if len(matches) == 0:
            self.smoother.reset()
            
            return PipelineResult(
                status=RecognitionStatus.UNKNOWN_CARD,
                card_match=None,
                top_matches=[],
                detection_confidence=detection.confidence,
                embedding_distance=1.0,  # No match found
                temporal_stability=0.0,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                frame_count=self.frame_count,
            )
        
        best_match = matches[0]
        
        # ═══════════════════════════════════════════════════════════
        # STAGE 5: Temporal Smoothing (CPU)
        # ═══════════════════════════════════════════════════════════
        
        stability, is_confirmed = self.smoother.update(best_match)
        
        # Determine final status
        if is_confirmed and best_match.distance < self.high_confidence_threshold:
            status = RecognitionStatus.CONFIRMED
        elif is_confirmed:
            status = RecognitionStatus.LOW_CONFIDENCE
        else:
            status = RecognitionStatus.STABILIZING
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return PipelineResult(
            status=status,
            card_match=best_match if status != RecognitionStatus.STABILIZING else None,
            top_matches=matches,
            detection_confidence=detection.confidence,
            embedding_distance=best_match.distance,
            temporal_stability=stability,
            processing_time_ms=processing_time,
            frame_count=self.frame_count,
        )
    
    def get_debug_info(self) -> dict:
        """Get debugging information about current state"""
        return {
            'frame_count': self.frame_count,
            'last_detection': self._last_detection,
            'smoother_state': {
                'history_length': len(self.smoother.prediction_history),
                'stable_card': self.smoother.stable_card_id,
                'stable_count': self.smoother.stable_count,
                'distance_ema': self.smoother.distance_ema,
            }
        }
    
    def reset(self):
        """Reset pipeline state"""
        self.smoother.reset()
        self.frame_count = 0
        self._last_detection = None
```

---

## Preprocessing Module

```python
# preprocessing.py

import cv2
import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class GlareInfo:
    """Information about detected glare"""
    has_glare: bool
    glare_ratio: float
    severity: str  # 'none', 'mild', 'severe'


class CardPreprocessor:
    """
    Preprocess detected cards for embedding.
    
    Steps:
    1. Perspective warp (from corner keypoints)
    2. Tight crop with minimal padding
    3. Glare detection and mitigation
    4. Normalization for ViT
    """
    
    def __init__(
        self,
        output_size: Tuple[int, int] = (224, 224),
        padding_percent: float = 0.03,
        glare_threshold: int = 245,
    ):
        self.output_size = output_size
        self.padding_percent = padding_percent
        self.glare_threshold = glare_threshold
        
        # Glare handling buffer
        self.frame_buffer = []
        self.buffer_size = 5
    
    def process(
        self,
        frame: np.ndarray,
        corners: np.ndarray
    ) -> Tuple[np.ndarray, GlareInfo]:
        """
        Process frame to extract and prepare card image.
        
        Args:
            frame: Full camera frame (BGR)
            corners: Ordered corner points (4, 2)
            
        Returns:
            Tuple of (preprocessed tensor, glare info)
        """
        # Step 1: Perspective warp
        cropped = self._perspective_crop(frame, corners)
        
        # Step 2: Glare detection
        glare_info = self._detect_glare(cropped)
        
        # Step 3: Glare mitigation (if needed)
        if glare_info.severity == 'severe':
            cropped = self._mitigate_glare(cropped, glare_info)
        
        # Step 4: Normalize for ViT
        tensor = self._normalize(cropped)
        
        return tensor, glare_info
    
    def _perspective_crop(
        self,
        frame: np.ndarray,
        corners: np.ndarray
    ) -> np.ndarray:
        """Crop and dewarp card using corner keypoints"""
        corners = corners.astype(np.float32)
        
        # Calculate output dimensions
        width_top = np.linalg.norm(corners[1] - corners[0])
        width_bottom = np.linalg.norm(corners[2] - corners[3])
        width = int(max(width_top, width_bottom))
        
        height_left = np.linalg.norm(corners[3] - corners[0])
        height_right = np.linalg.norm(corners[2] - corners[1])
        height = int(max(height_left, height_right))
        
        # Enforce card aspect ratio (2.5 x 3.5)
        target_ratio = 2.5 / 3.5
        if width / height > target_ratio:
            width = int(height * target_ratio)
        else:
            height = int(width / target_ratio)
        
        # Minimal padding (3%)
        pad_w = int(width * self.padding_percent)
        pad_h = int(height * self.padding_percent)
        
        # Destination points
        dst = np.array([
            [pad_w, pad_h],
            [width + pad_w, pad_h],
            [width + pad_w, height + pad_h],
            [pad_w, height + pad_h]
        ], dtype=np.float32)
        
        # Compute transform
        M = cv2.getPerspectiveTransform(corners, dst)
        
        # Apply warp
        out_w = width + 2 * pad_w
        out_h = height + 2 * pad_h
        warped = cv2.warpPerspective(
            frame, M, (out_w, out_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # Resize to output size
        resized = cv2.resize(warped, self.output_size, interpolation=cv2.INTER_LANCZOS4)
        
        return resized
    
    def _detect_glare(self, image: np.ndarray) -> GlareInfo:
        """Detect specular highlights"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bright_pixels = np.sum(gray > self.glare_threshold)
        total_pixels = gray.size
        glare_ratio = bright_pixels / total_pixels
        
        if glare_ratio < 0.01:
            severity = 'none'
        elif glare_ratio < 0.05:
            severity = 'mild'
        else:
            severity = 'severe'
        
        return GlareInfo(
            has_glare=(severity != 'none'),
            glare_ratio=glare_ratio,
            severity=severity
        )
    
    def _mitigate_glare(self, image: np.ndarray, glare_info: GlareInfo) -> np.ndarray:
        """Attempt to reduce glare impact"""
        # Simple inpainting for severe glare
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = (gray > self.glare_threshold).astype(np.uint8) * 255
        
        # Dilate mask slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel)
        
        # Inpaint
        result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        return result
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize for ViT input"""
        # BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        # HWC to CHW
        tensor = normalized.transpose(2, 0, 1)
        
        # Add batch dimension
        tensor = np.expand_dims(tensor, axis=0)
        
        return tensor.astype(np.float32)
```

---

## Main Application

```python
# main.py

import argparse
import logging
import json
from pathlib import Path

from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500

from card_recognition_pipeline import CardRecognitionPipeline, RecognitionStatus
from hailo_embedding import HailoEmbedder
from reference_database import ReferenceDatabase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_camera(detection_model_path: str):
    """Initialize camera with detection model"""
    logger.info("Initializing camera and IMX500...")
    
    imx500 = IMX500(detection_model_path)
    camera = Picamera2(imx500.camera_num)
    
    config = camera.create_preview_configuration(
        main={"size": (2028, 1520), "format": "RGB888"},
        controls={"FrameRate": 30},
        buffer_count=4
    )
    camera.configure(config)
    
    return camera, imx500


def main(args):
    # Setup components
    camera, imx500 = setup_camera(args.detection_model)
    
    logger.info("Loading embedding model...")
    embedder = HailoEmbedder(args.embedding_model)
    
    logger.info("Loading reference database...")
    database = ReferenceDatabase(args.database_dir)
    logger.info(f"Database loaded: {database.num_cards} cards")
    
    # Create pipeline
    pipeline = CardRecognitionPipeline(
        camera=camera,
        imx500=imx500,
        embedder=embedder,
        database=database,
        detection_threshold=args.detection_threshold,
        distance_threshold=args.distance_threshold,
        stability_frames=args.stability_frames,
    )
    
    # Start camera
    camera.start()
    logger.info("Camera started. Processing frames...")
    
    last_status = None
    
    try:
        while True:
            # Process frame
            result = pipeline.process_frame()
            
            # Log status changes
            if result.status != last_status:
                logger.info(f"Status: {result.status.value}")
                last_status = result.status
            
            # Handle different states
            if result.status == RecognitionStatus.NO_CARD:
                print("\r[No card]                                        ", end='')
                
            elif result.status == RecognitionStatus.STABILIZING:
                if result.top_matches:
                    print(f"\r[Stabilizing] {result.top_matches[0].name} "
                          f"(d={result.embedding_distance:.3f})        ", end='')
                          
            elif result.status == RecognitionStatus.UNKNOWN_CARD:
                print("\r[Unknown card - not in database]                 ", end='')
                
            elif result.status == RecognitionStatus.LOW_CONFIDENCE:
                match = result.card_match
                print(f"\r[Low confidence] {match.name} ({match.set_name} #{match.number}) "
                      f"conf={match.confidence:.1%}          ")
                      
            elif result.status == RecognitionStatus.CONFIRMED:
                match = result.card_match
                print(f"\n✓ CONFIRMED: {match.name}")
                print(f"  Set: {match.set_name} #{match.number}")
                print(f"  Confidence: {match.confidence:.1%}")
                print(f"  Processing time: {result.processing_time_ms:.1f}ms")
                print()
                
                # Optional: Save result
                if args.output_file:
                    with open(args.output_file, 'a') as f:
                        f.write(json.dumps(result.to_dict()) + '\n')
                
                # Reset for next card
                pipeline.reset()
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        camera.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pokemon Card Recognition')
    
    parser.add_argument('--detection-model', type=str, required=True,
                       help='Path to IMX500 detection model (.rpk)')
    parser.add_argument('--embedding-model', type=str, required=True,
                       help='Path to Hailo embedding model (.hef)')
    parser.add_argument('--database-dir', type=str, required=True,
                       help='Path to reference database directory')
    
    parser.add_argument('--detection-threshold', type=float, default=0.7,
                       help='Detection confidence threshold')
    parser.add_argument('--distance-threshold', type=float, default=0.4,
                       help='Maximum embedding distance for match')
    parser.add_argument('--stability-frames', type=int, default=3,
                       help='Frames needed for confirmation')
    
    parser.add_argument('--output-file', type=str, default=None,
                       help='File to append results to')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    main(args)
```

---

## Output States

### State Machine

```
                    ┌─────────────────┐
        ┌──────────▶│    NO_CARD      │◀─────────────┐
        │           └────────┬────────┘              │
        │                    │                       │
        │           Card detected                    │ Card leaves frame
        │                    │                       │
        │           ┌────────▼────────┐              │
        │           │   DETECTING     │              │
        │           └────────┬────────┘              │
        │                    │                       │
        │      ┌─────────────┼─────────────┐         │
        │      │             │             │         │
        │      ▼             ▼             ▼         │
┌───────┴────────┐  ┌────────────────┐  ┌──────────┴──────┐
│ UNKNOWN_CARD   │  │  STABILIZING   │  │  LOW_CONFIDENCE │
│ (dist > 0.4)   │  │ (counting...)  │  │  (dist > 0.25)  │
└────────────────┘  └───────┬────────┘  └─────────────────┘
                            │
                   3+ stable frames
                   dist < 0.25
                            │
                    ┌───────▼────────┐
                    │   CONFIRMED    │
                    │  (final state) │
                    └────────────────┘
```

### Output JSON Format

```json
{
  "status": "confirmed",
  "card": {
    "card_id": "sv10-025",
    "name": "Pikachu ex",
    "set_name": "Destined Rivals",
    "number": "025",
    "distance": 0.12,
    "confidence": 0.94
  },
  "alternatives": [
    {
      "card_id": "sv9-025",
      "name": "Pikachu",
      "set_name": "Journey Together",
      "number": "025",
      "distance": 0.31,
      "confidence": 0.61
    }
  ],
  "detection_confidence": 0.95,
  "embedding_distance": 0.12,
  "temporal_stability": 1.0,
  "processing_time_ms": 45.2,
  "frame_count": 127
}
```

---

## Acceptance Criteria

### AC-1: No Card Rejection
```gherkin
GIVEN no Pokemon card in the camera frame
WHEN processing the frame
THEN status MUST be "no_card"
AND no card_match should be returned
AND system MUST NOT predict any card
```

### AC-2: Unknown Card Detection
```gherkin
GIVEN a card that is NOT in the reference database
WHEN processing the frame
THEN status MUST be "unknown_card" after stabilization
AND embedding_distance MUST be > 0.4
```

### AC-3: Confirmed Recognition
```gherkin
GIVEN a card that IS in the reference database
AND the card is clearly visible without major occlusion
WHEN processing 3+ consecutive frames
THEN status MUST transition to "confirmed"
AND card_match.name MUST match the actual card
AND embedding_distance MUST be < 0.25
```

### AC-4: Temporal Stability
```gherkin
GIVEN a card being presented to the camera
WHEN predictions are unstable (different cards each frame)
THEN status MUST remain "stabilizing"
AND NO confirmed result should be emitted
```

### AC-5: Processing Speed
```gherkin
GIVEN the complete pipeline running
WHEN processing a frame
THEN total processing time MUST be < 100ms
AND frame rate MUST be ≥ 10 FPS
```

### AC-6: Glare Handling
```gherkin
GIVEN a holo or gold card with visible glare
WHEN processing frames
THEN system SHOULD still reach "confirmed" status
AND correct card SHOULD be in top_matches
```

### AC-7: Context Contamination
```gherkin
GIVEN a card held by fingers (up to 20% occluded)
WHEN processing frames
THEN system SHOULD still identify the card
AND accuracy SHOULD be ≥ 85%
```

---

## Testing Plan

### Integration Tests

```python
def test_no_card_rejection():
    """Empty frame should return NO_CARD"""
    pipeline = create_test_pipeline()
    
    # Simulate empty frame
    result = pipeline.process_test_frame('empty_table.jpg')
    
    assert result.status == RecognitionStatus.NO_CARD
    assert result.card_match is None

def test_known_card_recognition():
    """Known card should be recognized"""
    pipeline = create_test_pipeline()
    
    # Simulate frames of known card
    for _ in range(5):
        result = pipeline.process_test_frame('pikachu_ex.jpg')
    
    assert result.status == RecognitionStatus.CONFIRMED
    assert 'pikachu' in result.card_match.name.lower()

def test_unknown_card_rejection():
    """Card not in database should be flagged"""
    pipeline = create_test_pipeline()
    
    # Use card from set not in database
    for _ in range(5):
        result = pipeline.process_test_frame('unreleased_card.jpg')
    
    assert result.status == RecognitionStatus.UNKNOWN_CARD

def test_processing_speed():
    """Pipeline should be fast enough"""
    pipeline = create_test_pipeline()
    
    times = []
    for _ in range(100):
        result = pipeline.process_frame()
        times.append(result.processing_time_ms)
    
    avg_time = sum(times) / len(times)
    assert avg_time < 100  # <100ms average
```

---

## Deliverables

| Deliverable | Format | Location |
|-------------|--------|----------|
| CardRecognitionPipeline | `.py` | Git repo |
| CardPreprocessor | `.py` | Git repo |
| TemporalSmoother | `.py` | Git repo |
| Main application | `.py` | Git repo |
| Integration tests | `.py` | Git repo |
| Config example | `.json` | Git repo |

---

## Next Phase

Upon completion of Phase 4, proceed to **PRD_06_TRAINING.md** for SageMaker training guide.
