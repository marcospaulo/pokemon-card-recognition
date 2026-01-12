"""IMX500 camera capture for Raspberry Pi

Provides hardware interface to Sony IMX500 AI camera for real-time frame capture.
Target: 30 FPS at 1520x1520 resolution.

Includes:
- PiCamera: Basic camera capture
- IMX500AICamera: Camera with AI object detection using on-chip NPU
"""

import numpy as np
import time
from typing import Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.pipeline.interfaces import CameraInterface

try:
    from picamera2 import Picamera2
    from libcamera import controls
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("Warning: picamera2 not available. PiCamera will not work.")

try:
    from picamera2.devices import IMX500
    from picamera2.devices.imx500 import NetworkIntrinsics
    IMX500_AVAILABLE = True
except ImportError:
    IMX500_AVAILABLE = False


class PiCamera(CameraInterface):
    """IMX500 camera interface for Raspberry Pi

    Captures frames from Sony IMX500 AI camera at 1520x1520 resolution.
    Uses Picamera2 API for hardware acceleration.

    Performance target: 30 FPS, ~5ms capture latency
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (1520, 1520),
        fps: int = 30,
        auto_exposure: bool = True,
        auto_white_balance: bool = True,
    ):
        """Initialize Pi camera

        Args:
            resolution: Frame dimensions (width, height)
            fps: Target frame rate
            auto_exposure: Enable auto exposure control
            auto_white_balance: Enable auto white balance
        """
        if not PICAMERA2_AVAILABLE:
            raise RuntimeError(
                "picamera2 not available. Install with: "
                "pip install picamera2"
            )

        self.resolution = resolution
        self.fps = fps
        self.auto_exposure = auto_exposure
        self.auto_white_balance = auto_white_balance

        self._camera = None
        self._is_open = False
        self._frame_count = 0
        self._last_capture_time = 0.0

    def start(self):
        """Initialize and start camera"""
        if self._is_open:
            print("Warning: Camera already started")
            return

        print(f"Initializing IMX500 camera at {self.resolution[0]}x{self.resolution[1]} @ {self.fps} FPS...")

        try:
            # Create Picamera2 instance
            self._camera = Picamera2()

            # Configure camera - use preview config for video/real-time use
            config = self._camera.create_preview_configuration(
                main={"size": self.resolution, "format": "RGB888"},
                controls={
                    "FrameRate": self.fps,
                }
            )

            if self.auto_exposure:
                config["controls"]["AeEnable"] = True
            else:
                config["controls"]["AeEnable"] = False

            if self.auto_white_balance:
                config["controls"]["AwbEnable"] = True
            else:
                config["controls"]["AwbEnable"] = False

            self._camera.configure(config)
            self._camera.start()

            # Wait for camera to warm up
            time.sleep(0.5)

            self._is_open = True
            self._frame_count = 0

            print(f"✓ IMX500 camera ready")

        except Exception as e:
            self._is_open = False
            raise RuntimeError(f"Failed to start camera: {e}")

    def stop(self):
        """Stop camera and release resources"""
        if not self._is_open:
            return

        try:
            if self._camera:
                self._camera.stop()
                self._camera.close()
                self._camera = None

            self._is_open = False
            print(f"Camera stopped. Captured {self._frame_count} frames.")

        except Exception as e:
            print(f"Error stopping camera: {e}")

    def capture_frame(self) -> np.ndarray:
        """Capture single frame from camera

        Returns:
            Frame in BGR format (H, W, 3) for OpenCV compatibility

        Raises:
            RuntimeError: If camera is not started

        Performance: ~3-5ms on Raspberry Pi 5
        """
        if not self._is_open:
            raise RuntimeError("Camera not started. Call start() first.")

        try:
            start_time = time.time()

            # Capture frame from picamera2 - use directly without color conversion
            # picamera2 handles color format internally
            frame = self._camera.capture_array()

            self._frame_count += 1
            self._last_capture_time = (time.time() - start_time) * 1000

            return frame

        except Exception as e:
            raise RuntimeError(f"Frame capture failed: {e}")

    @property
    def is_open(self) -> bool:
        """Check if camera is ready"""
        return self._is_open

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Get camera frame dimensions"""
        return self.resolution

    @property
    def last_capture_time_ms(self) -> float:
        """Get last frame capture latency in milliseconds"""
        return self._last_capture_time

    def set_exposure(self, exposure_us: int):
        """Set manual exposure time

        Args:
            exposure_us: Exposure time in microseconds
        """
        if not self._is_open:
            raise RuntimeError("Camera not started")

        self._camera.set_controls({
            "AeEnable": False,
            "ExposureTime": exposure_us,
        })

    def set_gain(self, gain: float):
        """Set analog gain

        Args:
            gain: Analog gain value (1.0 = no gain)
        """
        if not self._is_open:
            raise RuntimeError("Camera not started")

        self._camera.set_controls({"AnalogueGain": gain})

    def enable_auto_exposure(self, enable: bool = True):
        """Enable/disable auto exposure

        Args:
            enable: True to enable auto exposure
        """
        if not self._is_open:
            raise RuntimeError("Camera not started")

        self._camera.set_controls({"AeEnable": enable})
        self.auto_exposure = enable

    def get_camera_info(self) -> dict:
        """Get camera properties and status

        Returns:
            Dictionary with camera information
        """
        if not self._camera:
            return {"status": "not_initialized"}

        return {
            "status": "open" if self._is_open else "closed",
            "resolution": self.resolution,
            "fps": self.fps,
            "frames_captured": self._frame_count,
            "last_capture_ms": self._last_capture_time,
            "auto_exposure": self.auto_exposure,
            "auto_white_balance": self.auto_white_balance,
        }


class IMX500AICamera(CameraInterface):
    """IMX500 AI camera with on-chip object detection

    Uses Sony IMX500's NPU to run NanoDet object detection directly
    on the camera chip. Returns both frames and AI inference results.

    Performance: ~30 FPS with simultaneous capture and inference
    """

    # COCO classes that might be card-like
    CARD_LIKE_CLASSES = {73, 84, 67, 63, 62, 66}  # book, vase, phone, laptop, tv, keyboard

    def __init__(
        self,
        resolution: Tuple[int, int] = (1080, 1440),
        fps: int = 30,
        model_path: str = "/usr/share/imx500-models/imx500_network_nanodet_plus_416x416_pp.rpk",
        detection_threshold: float = 0.3,
    ):
        """Initialize IMX500 AI camera

        Args:
            resolution: Frame dimensions (width, height)
            fps: Target frame rate
            model_path: Path to NanoDet RPK model for IMX500
            detection_threshold: Minimum confidence for detections
        """
        if not PICAMERA2_AVAILABLE:
            raise RuntimeError("picamera2 not available")
        if not IMX500_AVAILABLE:
            raise RuntimeError("IMX500 support not available")

        self.resolution = resolution
        self.fps = fps
        self.model_path = model_path
        self.detection_threshold = detection_threshold

        self._camera = None
        self._imx500 = None
        self._is_open = False
        self._frame_count = 0
        self._last_capture_time = 0.0
        self._last_metadata = None

    def start(self):
        """Initialize camera with AI model"""
        if self._is_open:
            print("Warning: Camera already started")
            return

        print(f"Initializing IMX500 AI camera...")
        print(f"  Resolution: {self.resolution[0]}x{self.resolution[1]} @ {self.fps} FPS")
        print(f"  Model: {self.model_path}")

        try:
            # Create IMX500 device with model
            self._imx500 = IMX500(self.model_path)

            # Create Picamera2 instance
            self._camera = Picamera2(self._imx500.camera_num)

            # Configure camera
            config = self._camera.create_preview_configuration(
                main={"size": self.resolution, "format": "RGB888"},
                controls={"FrameRate": self.fps},
            )
            self._camera.configure(config)

            # Start camera
            self._camera.start()

            # Wait for warmup
            time.sleep(0.5)

            self._is_open = True
            self._frame_count = 0

            print(f"✓ IMX500 AI camera ready")
            print(f"  Detection threshold: {self.detection_threshold}")

        except Exception as e:
            self._is_open = False
            raise RuntimeError(f"Failed to start IMX500 AI camera: {e}")

    def stop(self):
        """Stop camera and release resources"""
        if not self._is_open:
            return

        try:
            if self._camera:
                self._camera.stop()
                self._camera.close()
                self._camera = None
            self._imx500 = None
            self._is_open = False
            print(f"IMX500 AI camera stopped. Captured {self._frame_count} frames.")

        except Exception as e:
            print(f"Error stopping camera: {e}")

    def capture_frame(self) -> np.ndarray:
        """Capture frame (for compatibility with PiCamera interface)

        Returns:
            RGB frame as numpy array
        """
        frame, _ = self.capture_frame_with_detections()
        return frame

    def capture_frame_with_detections(self) -> Tuple[np.ndarray, list]:
        """Capture frame with AI object detections

        Returns:
            Tuple of (frame, detections) where detections is a list of dicts:
                - bbox: (x1, y1, x2, y2) pixel coordinates
                - score: confidence score
                - class_id: COCO class ID
                - class_name: class name string
        """
        if not self._is_open:
            raise RuntimeError("Camera not started. Call start() first.")

        try:
            start_time = time.time()

            # Capture frame and metadata together for efficiency
            # capture_array with wait=False uses the latest available frame
            (frame, ), metadata = self._camera.capture_arrays(["main"])
            self._last_metadata = metadata

            # Parse detections from metadata
            detections = self._parse_detections(metadata)

            self._frame_count += 1
            self._last_capture_time = (time.time() - start_time) * 1000

            return frame, detections

        except Exception as e:
            raise RuntimeError(f"Frame capture failed: {e}")

    def _parse_detections(self, metadata: dict) -> list:
        """Parse AI inference results from metadata

        Args:
            metadata: Frame metadata containing inference results

        Returns:
            List of detection dicts with bbox, score, class_id, class_name
        """
        if self._imx500 is None:
            return []

        # Get outputs from IMX500
        outputs = self._imx500.get_outputs(metadata)
        if outputs is None or len(outputs) < 4:
            return []

        # NanoDet outputs:
        # [0]: boxes (300, 4) - bounding boxes
        # [1]: scores (300,) - confidence scores
        # [2]: classes (300,) - class IDs
        # [3]: count (1,) - number of valid detections
        boxes = outputs[0]
        scores = outputs[1]
        classes = outputs[2]
        count = int(outputs[3][0]) if outputs[3].size > 0 else 0

        w, h = self.resolution
        detections = []

        for i in range(min(count, len(scores))):
            score = float(scores[i])
            if score < self.detection_threshold:
                continue

            class_id = int(classes[i])
            box = boxes[i]

            # Convert box coordinates to pixels
            # Check if normalized (0-1) or already pixel coords
            if box[2] <= 1.0 and box[3] <= 1.0:
                x1 = int(box[0] * w)
                y1 = int(box[1] * h)
                x2 = int(box[2] * w)
                y2 = int(box[3] * h)
            else:
                x1, y1 = int(box[0]), int(box[1])
                x2, y2 = int(box[2]), int(box[3])

            detections.append({
                'bbox': (x1, y1, x2, y2),
                'score': score,
                'class_id': class_id,
                'class_name': self._get_class_name(class_id),
            })

        return detections

    def get_card_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Get bounding box of detected card-like object

        Uses most recent frame's detections to find card-like objects.

        Returns:
            (x1, y1, x2, y2) pixel coordinates or None if no card found
        """
        if self._last_metadata is None:
            return None

        detections = self._parse_detections(self._last_metadata)

        # Find best card-like detection
        best_card = None
        best_score = 0

        for det in detections:
            # Accept card-like classes OR high-confidence detections
            is_card_like = det['class_id'] in self.CARD_LIKE_CLASSES
            is_confident = det['score'] >= 0.5

            if (is_card_like or is_confident) and det['score'] > best_score:
                # Check aspect ratio (cards are roughly 0.71 portrait)
                x1, y1, x2, y2 = det['bbox']
                box_w = x2 - x1
                box_h = y2 - y1

                if box_h > 0:
                    aspect = box_w / box_h
                    # Accept portrait (0.5-0.9) or landscape (1.1-2.0)
                    if (0.5 <= aspect <= 0.9) or (1.1 <= aspect <= 2.0):
                        best_card = det['bbox']
                        best_score = det['score']

        return best_card

    def _get_class_name(self, class_id: int) -> str:
        """Get COCO class name"""
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

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def frame_size(self) -> Tuple[int, int]:
        return self.resolution

    @property
    def last_capture_time_ms(self) -> float:
        return self._last_capture_time
