"""IMX500 AI Camera Card Detection

Uses Sony IMX500's on-chip NPU to detect objects in frame.
The NanoDet model runs directly on the camera's AI accelerator.

Output from NanoDet with postprocessing:
- boxes: (N, 4) bounding box coordinates [x1, y1, x2, y2] normalized 0-1
- scores: (N,) confidence scores 0-1
- classes: (N,) COCO class IDs (0-79)
- count: (1,) number of valid detections
"""

import numpy as np
from typing import Optional, List, Tuple

try:
    from picamera2 import Picamera2
    from picamera2.devices import IMX500
    from picamera2.devices.imx500 import NetworkIntrinsics
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False


class IMX500Detector:
    """Card detector using IMX500 AI camera

    Uses NanoDet object detection running on IMX500's NPU.
    Detects rectangular objects that could be cards.
    """

    # COCO classes that might be card-like (rectangular, flat objects)
    CARD_LIKE_CLASSES = {
        73,  # book
        84,  # vase (sometimes cards)
        67,  # cell phone (rectangular)
        63,  # laptop
        62,  # tv
        66,  # keyboard
    }

    def __init__(
        self,
        model_path: str = "/usr/share/imx500-models/imx500_network_nanodet_plus_416x416_pp.rpk",
        confidence_threshold: float = 0.3,
        input_size: Tuple[int, int] = (416, 416),
    ):
        """Initialize IMX500 detector

        Args:
            model_path: Path to NanoDet RPK model
            confidence_threshold: Minimum confidence to accept detection
            input_size: Model input dimensions (width, height)
        """
        if not PICAMERA2_AVAILABLE:
            raise RuntimeError("picamera2 not available")

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size

        self._imx500 = None
        self._camera = None
        self._is_initialized = False

    def initialize(self, camera: 'Picamera2'):
        """Initialize detector with camera instance

        Args:
            camera: Picamera2 instance to attach IMX500 to
        """
        self._camera = camera

        # Create IMX500 device
        self._imx500 = IMX500(self.model_path)

        # Get network intrinsics (input size, labels, etc.)
        intrinsics = self._imx500.network_intrinsics
        if intrinsics:
            print(f"IMX500 Model: {intrinsics.network_name}")
            if hasattr(intrinsics, 'labels'):
                print(f"  Labels: {len(intrinsics.labels)} classes")

        self._is_initialized = True
        print(f"IMX500 detector initialized")
        print(f"  Model: {self.model_path}")
        print(f"  Confidence threshold: {self.confidence_threshold}")

    def detect(self, metadata: dict) -> List[dict]:
        """Extract detections from camera metadata

        The IMX500 runs inference automatically and stores results
        in frame metadata. This method parses those results.

        Args:
            metadata: Frame metadata from picamera2

        Returns:
            List of detections, each with:
                - bbox: (x1, y1, x2, y2) normalized coordinates
                - score: confidence score
                - class_id: COCO class ID
                - class_name: class name string
        """
        if not self._is_initialized:
            return []

        # Get outputs from metadata
        outputs = self._imx500.get_outputs(metadata)
        if outputs is None:
            return []

        # Parse NanoDet outputs
        # outputs[0]: boxes (300, 4) - [x1, y1, x2, y2] normalized
        # outputs[1]: scores (300,)
        # outputs[2]: classes (300,)
        # outputs[3]: count (1,) - number of valid detections

        if len(outputs) < 4:
            return []

        boxes = outputs[0]
        scores = outputs[1]
        classes = outputs[2]
        count = int(outputs[3][0]) if outputs[3].size > 0 else 0

        detections = []

        for i in range(min(count, len(scores))):
            score = float(scores[i])

            if score < self.confidence_threshold:
                continue

            class_id = int(classes[i])
            box = boxes[i]  # [x1, y1, x2, y2] or [cx, cy, w, h]

            # Get class name from COCO labels
            class_name = self._get_class_name(class_id)

            detections.append({
                'bbox': (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                'score': score,
                'class_id': class_id,
                'class_name': class_name,
            })

        return detections

    def detect_cards(self, metadata: dict, frame_size: Tuple[int, int]) -> List[dict]:
        """Detect card-like objects and return pixel coordinates

        Args:
            metadata: Frame metadata from picamera2
            frame_size: (width, height) of the frame

        Returns:
            List of card detections with pixel bboxes
        """
        detections = self.detect(metadata)

        if not detections:
            return []

        w, h = frame_size
        cards = []

        for det in detections:
            # Check if it's a card-like class OR has high confidence
            # (cards might be detected as various objects)
            is_card_like = det['class_id'] in self.CARD_LIKE_CLASSES
            is_confident = det['score'] >= 0.5

            # Accept if card-like OR very confident
            if is_card_like or is_confident:
                # Convert normalized coords to pixels
                x1, y1, x2, y2 = det['bbox']

                # Handle both normalized (0-1) and pixel coords
                if x2 <= 1.0 and y2 <= 1.0:
                    # Normalized - convert to pixels
                    x1_px = int(x1 * w)
                    y1_px = int(y1 * h)
                    x2_px = int(x2 * w)
                    y2_px = int(y2 * h)
                else:
                    # Already pixels
                    x1_px, y1_px = int(x1), int(y1)
                    x2_px, y2_px = int(x2), int(y2)

                # Check aspect ratio - cards are roughly 2.5:3.5 (0.71)
                box_w = x2_px - x1_px
                box_h = y2_px - y1_px

                if box_h > 0:
                    aspect = box_w / box_h
                    # Accept cards in portrait (0.5-0.9) or landscape (1.1-2.0)
                    is_card_aspect = (0.5 <= aspect <= 0.9) or (1.1 <= aspect <= 2.0)
                else:
                    is_card_aspect = False

                cards.append({
                    'bbox': (x1_px, y1_px, x2_px, y2_px),
                    'score': det['score'],
                    'class_id': det['class_id'],
                    'class_name': det['class_name'],
                    'is_card_aspect': is_card_aspect,
                })

        # Sort by score descending
        cards.sort(key=lambda x: x['score'], reverse=True)

        return cards

    def get_best_card_region(
        self,
        metadata: dict,
        frame_size: Tuple[int, int]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Get the best card region from detections

        Args:
            metadata: Frame metadata from picamera2
            frame_size: (width, height) of the frame

        Returns:
            (x1, y1, x2, y2) pixel coordinates or None if no card found
        """
        cards = self.detect_cards(metadata, frame_size)

        if not cards:
            return None

        # Return best detection (highest score)
        return cards[0]['bbox']

    def _get_class_name(self, class_id: int) -> str:
        """Get COCO class name from ID"""
        COCO_CLASSES = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        if 0 <= class_id < len(COCO_CLASSES):
            return COCO_CLASSES[class_id]
        return f"class_{class_id}"


# Quick test
if __name__ == "__main__":
    print("Testing IMX500Detector...")
    print(f"picamera2 available: {PICAMERA2_AVAILABLE}")

    if PICAMERA2_AVAILABLE:
        detector = IMX500Detector()
        print("Detector created successfully")
