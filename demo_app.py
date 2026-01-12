#!/usr/bin/env python3
"""Pokemon Card Recognition Demo Application

Real-time card recognition using:
- IMX500 camera with YOLO11n for card detection
- ONNX EfficientNet-Lite0 for embedding extraction
- uSearch for similarity search across 17,592 cards

Usage:
    python demo_app.py                    # Full pipeline with IMX500 + ONNX
    python demo_app.py --image card.jpg   # Single image mode
    python demo_app.py --no-camera        # Test without camera (ONNX + uSearch only)
"""

import argparse
import time
import json
import sys
import os
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import uSearch
try:
    from usearch.index import Index
    USEARCH_AVAILABLE = True
except ImportError:
    USEARCH_AVAILABLE = False
    print("Warning: usearch not available")

# Import Hailo
try:
    from hailo_platform import HEF, VDevice, FormatType
    from hailo_platform import ConfigureParams, InputVStreamParams, OutputVStreamParams, InferVStreams
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    print("Warning: hailo_platform not available")

# Import ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not available")

# Import camera and IMX500
try:
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("Warning: picamera2 not available")

try:
    from picamera2.devices import IMX500
    from picamera2.devices.imx500 import NetworkIntrinsics
    IMX500_AVAILABLE = True
except ImportError:
    IMX500_AVAILABLE = False
    print("Warning: IMX500 not available")


class ONNXEmbeddingEngine:
    """ONNX-based embedding extraction using EfficientNet-Lite0

    Uses the ONNX model which matches the reference database embeddings.
    """

    def __init__(self, onnx_path: str):
        self.onnx_path = Path(onnx_path)
        self.session = None
        self.input_name = None
        self._inference_count = 0
        self._total_time = 0.0

    def warmup(self):
        """Initialize ONNX Runtime session"""
        if not ONNX_AVAILABLE:
            print("   ONNX Runtime not available - running in simulation mode")
            return

        print(f"   Loading ONNX: {self.onnx_path.name}")

        # Create session with CPU provider
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name

        # Warmup inference
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        self.extract(dummy)

        print(f"   ONNX ready")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ONNX model (ImageNet normalization, NCHW)"""
        # Resize to 224x224
        img = cv2.resize(image, (224, 224))

        # Convert BGR to RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        # Transpose to NCHW and add batch dimension
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        return img

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract 768-dim embedding from image"""
        if not ONNX_AVAILABLE or self.session is None:
            # Simulation mode
            return np.random.randn(768).astype(np.float32)

        start = time.time()

        # Preprocess
        processed = self.preprocess(image)

        # Run inference
        outputs = self.session.run(None, {self.input_name: processed})
        embedding = outputs[0][0]

        # L2 normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        self._inference_count += 1
        self._total_time += time.time() - start

        return embedding

    @property
    def avg_inference_ms(self) -> float:
        if self._inference_count == 0:
            return 0.0
        return (self._total_time / self._inference_count) * 1000


class HailoEmbeddingEngine:
    """Hailo-based embedding extraction using EfficientNet-Lite0

    Note: Requires reference database to be regenerated with Hailo embeddings
    for correct matching. Currently the reference DB uses ONNX embeddings.
    """

    def __init__(self, hef_path: str):
        self.hef_path = Path(hef_path)
        self.device = None
        self.hef = None
        self.network_group = None
        self.input_vstream_params = None
        self.output_vstream_params = None
        self._inference_count = 0
        self._total_time = 0.0

    def warmup(self):
        """Initialize Hailo device and load model"""
        if not HAILO_AVAILABLE:
            print("   Hailo not available - running in simulation mode")
            return

        print(f"   Loading HEF: {self.hef_path.name}")

        # Create virtual device
        self.device = VDevice()

        # Load HEF
        self.hef = HEF(str(self.hef_path))

        # Configure network
        self.network_group = self.device.configure(self.hef)[0]

        # Setup input/output streams
        self.input_vstream_params = InputVStreamParams.make_from_network_group(
            self.network_group, quantized=True, format_type=FormatType.UINT8
        )
        self.output_vstream_params = OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32
        )

        # Create network group params for activation
        self.ng_params = self.network_group.create_params()

        # Warmup inference
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        self.extract(dummy)

        print(f"   Hailo ready")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for Hailo HEF model

        Uses uint8 [0, 255] input to match how reference embeddings were generated.
        The HEF model has quantization built in and expects raw uint8 RGB input.
        """
        # Resize to 224x224
        img = cv2.resize(image, (224, 224))

        # Convert BGR to RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Return as uint8 - matches reference embedding generation
        return img.astype(np.uint8)

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract 768-dim embedding from image"""
        if not HAILO_AVAILABLE or self.network_group is None:
            # Simulation mode
            return np.random.randn(768).astype(np.float32)

        start = time.time()

        # Preprocess
        processed = self.preprocess(image)

        # Run inference with network group activation
        with self.network_group.activate(self.ng_params):
            with InferVStreams(self.network_group, self.input_vstream_params, self.output_vstream_params) as pipeline:
                # input_vstream_params is a dict: {name: VStreamParams}
                input_name = list(self.input_vstream_params.keys())[0]
                input_data = {input_name: processed[np.newaxis, ...]}
                output = pipeline.infer(input_data)
                embedding = list(output.values())[0]  # Shape: (1, 1, 1, 768)

        # Flatten and L2 normalize
        embedding = embedding.flatten()  # Shape: (768,)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        self._inference_count += 1
        self._total_time += time.time() - start

        return embedding

    @property
    def avg_inference_ms(self) -> float:
        if self._inference_count == 0:
            return 0.0
        return (self._total_time / self._inference_count) * 1000


# Alias for backwards compatibility
EmbeddingEngine = ONNXEmbeddingEngine


class YOLOCardDetector:
    """YOLO-based card detection using Ultralytics

    Uses a YOLO model trained on Pokemon cards for detection.
    Supports both regular and OBB (Oriented Bounding Box) models.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.5, imgsz: int = 256):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz  # Smaller = faster (256 gives faster inference, 320 for accuracy)
        self.model = None
        self._detection_count = 0
        self._total_time = 0.0

    def setup(self):
        """Load YOLO model"""
        try:
            from ultralytics import YOLO
            print(f"   Loading YOLO model: {self.model_path.name}")
            self.model = YOLO(str(self.model_path))
            print(f"   YOLO card detector ready")
            return True
        except Exception as e:
            print(f"   Error loading YOLO model: {e}")
            return False

    def detect(self, frame: np.ndarray) -> list:
        """Detect cards in frame

        Args:
            frame: BGR image

        Returns:
            List of detections: [{'box': (x1,y1,x2,y2), 'confidence': float}, ...]
        """
        if self.model is None:
            return []

        start = time.time()

        # Run inference with optimized input size
        results = self.model(frame, imgsz=self.imgsz, verbose=False, conf=self.conf_threshold)

        detections = []
        for result in results:
            # Handle OBB (oriented bounding box) results
            if hasattr(result, 'obb') and result.obb is not None:
                for i in range(len(result.obb)):
                    # Get the axis-aligned bounding box from OBB
                    obb = result.obb[i]
                    # xyxyxyxy gives 4 corners, we take the bounding rect
                    if hasattr(obb, 'xyxyxyxy'):
                        corners = obb.xyxyxyxy.cpu().numpy().reshape(-1, 2)
                        x1, y1 = corners.min(axis=0)
                        x2, y2 = corners.max(axis=0)
                    elif hasattr(obb, 'xyxy'):
                        box = obb.xyxy.cpu().numpy()[0]
                        x1, y1, x2, y2 = box
                    else:
                        continue

                    conf = float(obb.conf.cpu().numpy()[0])
                    detections.append({
                        'box': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': conf,
                        'class_id': 0
                    })
            # Handle regular detection results
            elif hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    xyxy = box.xyxy.cpu().numpy()[0]
                    conf = float(box.conf.cpu().numpy()[0])
                    detections.append({
                        'box': (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                        'confidence': conf,
                        'class_id': int(box.cls.cpu().numpy()[0]) if hasattr(box, 'cls') else 0
                    })

        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        self._detection_count += 1
        self._total_time += time.time() - start

        return detections[:3]  # Return top 3 detections

    @property
    def avg_detection_ms(self) -> float:
        if self._detection_count == 0:
            return 0.0
        return (self._total_time / self._detection_count) * 1000


class IMX500CardDetector:
    """IMX500-based card detection using YOLO11n

    Detects Pokemon cards in the frame and returns bounding boxes.
    Runs on the IMX500 AI camera's built-in NPU.

    NOTE: Requires a YOLO model trained specifically on Pokemon cards.
    The default YOLO11n is trained on COCO and won't detect cards.
    """

    def __init__(self, rpk_path: str, conf_threshold: float = 0.5):
        self.rpk_path = Path(rpk_path)
        self.conf_threshold = conf_threshold
        self.imx500 = None
        self.intrinsics = None
        self._detection_count = 0
        self._total_time = 0.0

    def setup(self, picam2: Picamera2):
        """Initialize IMX500 with YOLO model"""
        if not IMX500_AVAILABLE:
            print("   IMX500 not available - detection disabled")
            return False

        print(f"   Loading YOLO model: {self.rpk_path.name}")

        self.imx500 = IMX500(str(self.rpk_path))

        # Get network intrinsics for coordinate conversion
        self.intrinsics = self.imx500.network_intrinsics
        if self.intrinsics is None:
            self.intrinsics = NetworkIntrinsics()
            self.intrinsics.task = "object detection"

        print(f"   IMX500 ready")
        return True

    def parse_detections(self, outputs: dict, frame_size: tuple) -> list:
        """Parse YOLO outputs into detection list

        Args:
            outputs: Raw outputs from IMX500 (dict with tensor arrays)
            frame_size: (width, height) of the frame

        Returns:
            List of detections: [{'box': (x1,y1,x2,y2), 'confidence': float, 'class_id': int}, ...]
        """
        if outputs is None:
            return []

        # Get the output tensors
        # YOLO format: boxes (N,4), scores (N,), classes (N,), count (1,)
        output_tensors = list(outputs.values()) if isinstance(outputs, dict) else outputs

        if len(output_tensors) < 3:
            return []

        boxes = output_tensors[0]      # Shape: (300, 4) - x1, y1, x2, y2 normalized
        scores = output_tensors[1]     # Shape: (300,) - confidence scores
        class_ids = output_tensors[2]  # Shape: (300,) - class IDs

        detections = []
        frame_w, frame_h = frame_size

        # YOLO input size (from model export)
        input_size = 416

        for i in range(len(scores)):
            conf = float(scores[i])
            if conf < self.conf_threshold:
                continue

            # Get box coordinates (normalized 0-1 from YOLO)
            box = boxes[i]
            x1 = float(box[0]) * frame_w / input_size
            y1 = float(box[1]) * frame_h / input_size
            x2 = float(box[2]) * frame_w / input_size
            y2 = float(box[3]) * frame_h / input_size

            # Clamp to frame bounds
            x1 = max(0, min(frame_w - 1, x1))
            y1 = max(0, min(frame_h - 1, y1))
            x2 = max(0, min(frame_w - 1, x2))
            y2 = max(0, min(frame_h - 1, y2))

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append({
                'box': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': conf,
                'class_id': int(class_ids[i])
            })

        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        return detections

    def get_detections(self, metadata: dict, frame_size: tuple) -> list:
        """Get detections from IMX500 metadata

        Args:
            metadata: Frame metadata from picamera2
            frame_size: (width, height) of the frame

        Returns:
            List of detections
        """
        if not IMX500_AVAILABLE or self.imx500 is None:
            return []

        start = time.time()

        # Get IMX500 outputs from metadata
        outputs = self.imx500.get_outputs(metadata)

        # Debug: print output info occasionally
        if self._detection_count % 100 == 0:
            if outputs is None:
                print(f"[DEBUG] IMX500 outputs: None")
            elif isinstance(outputs, dict):
                print(f"[DEBUG] IMX500 outputs: dict with {len(outputs)} keys")
                for k, v in outputs.items():
                    if hasattr(v, 'shape'):
                        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            elif isinstance(outputs, (list, tuple)):
                print(f"[DEBUG] IMX500 outputs: {type(outputs).__name__} with {len(outputs)} items")
                for i, v in enumerate(outputs):
                    if hasattr(v, 'shape'):
                        print(f"  [{i}]: shape={v.shape}, dtype={v.dtype}")

        detections = self.parse_detections(outputs, frame_size)

        self._detection_count += 1
        self._total_time += time.time() - start

        return detections

    @property
    def avg_detection_ms(self) -> float:
        if self._detection_count == 0:
            return 0.0
        return (self._total_time / self._detection_count) * 1000


class HailoYOLODetector:
    """Hailo-based card detection using YOLOv8s

    Runs YOLOv8s on Hailo NPU for fast object detection (~15ms).
    Filters detections by aspect ratio to find card-like objects.
    """

    def __init__(self, hef_path: str = "/usr/share/hailo-models/yolov8s_h8.hef",
                 conf_threshold: float = 0.3):
        self.hef_path = Path(hef_path)
        self.conf_threshold = conf_threshold
        self._vdevice = None
        self._hef = None
        self._network_group = None
        self._input_vstream_info = None
        self._output_vstream_info = None
        self._detection_count = 0
        self._total_time = 0.0

    def warmup(self):
        """Initialize Hailo device and load model"""
        if not HAILO_AVAILABLE:
            print("   Hailo not available - detection disabled")
            return False

        from hailo_platform import HEF, VDevice, ConfigureParams

        print(f"   Loading Hailo YOLO: {self.hef_path.name}")

        self._hef = HEF(str(self.hef_path))
        self._vdevice = VDevice()

        # Configure network
        configure_params = ConfigureParams.create_from_hef(
            self._hef, interface=HailoStreamInterface.PCIe
        )
        self._network_group = self._vdevice.configure(self._hef, configure_params)[0]

        # Get stream info
        self._input_vstream_info = self._hef.get_input_vstream_infos()[0]
        self._output_vstream_info = self._hef.get_output_vstream_infos()[0]

        print(f"   Hailo YOLO ready (input: {self._input_vstream_info.shape})")
        return True

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for Hailo YOLOv8s (640x640, uint8)"""
        import cv2
        # Resize to 640x640
        resized = cv2.resize(frame, (640, 640))
        return resized.astype(np.uint8)

    def detect(self, frame: np.ndarray) -> list:
        """Detect objects in frame using Hailo YOLOv8s

        Returns:
            List of card-like detections (filtered by aspect ratio)
        """
        if self._network_group is None:
            return []

        start = time.time()
        h, w = frame.shape[:2]

        from hailo_platform import InputVStreamParams, OutputVStreamParams, InferVStreams

        # Preprocess
        input_data = self.preprocess(frame)
        input_data = np.expand_dims(input_data, axis=0)  # Add batch dim

        # Run inference
        input_params = InputVStreamParams.make(self._network_group)
        output_params = OutputVStreamParams.make(self._network_group)

        with InferVStreams(self._network_group, input_params, output_params) as pipeline:
            input_dict = {self._input_vstream_info.name: input_data}
            output = pipeline.infer(input_dict)

        # Parse NMS output: (80 classes, 5 values [x,y,w,h,conf], 100 detections)
        output_data = list(output.values())[0][0]  # Shape: (80, 5, 100)

        detections = []
        for class_id in range(output_data.shape[0]):
            for det_idx in range(output_data.shape[2]):
                conf = output_data[class_id, 4, det_idx]
                if conf < self.conf_threshold:
                    continue

                # Get box coordinates (normalized)
                cx = output_data[class_id, 0, det_idx]
                cy = output_data[class_id, 1, det_idx]
                bw = output_data[class_id, 2, det_idx]
                bh = output_data[class_id, 3, det_idx]

                # Convert to pixel coords
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)

                # Filter by aspect ratio (cards are roughly 2.5:3.5)
                box_w = x2 - x1
                box_h = y2 - y1
                if box_w > 0 and box_h > 0:
                    aspect = box_w / box_h
                    # Card aspect ratio is ~0.71 (2.5/3.5) - allow some tolerance
                    if 0.5 < aspect < 1.0:  # Portrait card
                        detections.append({
                            'box': (x1, y1, x2, y2),
                            'confidence': float(conf),
                            'class_id': class_id
                        })

        # Sort by confidence and take top detection
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        self._detection_count += 1
        self._total_time += time.time() - start

        return detections[:5]  # Return top 5 card-like detections

    @property
    def avg_detection_ms(self) -> float:
        if self._detection_count == 0:
            return 0.0
        return (self._total_time / self._detection_count) * 1000


class CardMatcher:
    """uSearch-based card matching"""

    def __init__(self, reference_path: str):
        self.reference_path = Path(reference_path)
        self.index = None
        self.row_to_card = None
        self.metadata = None
        self.embeddings = None

    def load(self):
        """Load reference database"""
        print(f"   Loading embeddings...")
        self.embeddings = np.load(self.reference_path / "embeddings.npy")
        print(f"   Loaded {self.embeddings.shape[0]} embeddings ({self.embeddings.shape[1]}D)")

        if USEARCH_AVAILABLE:
            print(f"   Loading uSearch index...")
            self.index = Index.restore(str(self.reference_path / "usearch.index"))
        else:
            print(f"   uSearch not available - using brute force search")
            self.index = None

        print(f"   Loading metadata...")
        with open(self.reference_path / "index.json") as f:
            self.row_to_card = json.load(f)
        with open(self.reference_path / "metadata.json") as f:
            self.metadata = json.load(f)

        print(f"   Reference database ready ({len(self.metadata)} cards)")

    def search(self, embedding: np.ndarray, k: int = 5) -> list:
        """Search for similar cards"""
        if self.index is not None:
            # Use uSearch index
            matches = self.index.search(embedding, k)
            row_indices = matches.keys
            distances = matches.distances
        else:
            # Brute force search using L2 distance
            diffs = self.embeddings - embedding
            distances = np.linalg.norm(diffs, axis=1)
            row_indices = np.argsort(distances)[:k]
            distances = distances[row_indices]

        results = []
        for i, (row_idx, distance) in enumerate(zip(row_indices, distances)):
            card_id = self.row_to_card.get(str(int(row_idx)), f"unknown_{row_idx}")
            card_data = self.metadata.get(card_id, {})

            # Convert L2 distance to similarity
            # For normalized embeddings, L2 distance ranges from 0 (identical) to 2 (opposite)
            # Use a more discriminative formula: exponential decay
            similarity = np.exp(-float(distance) * 2.0)  # distance of 0.5 -> ~37%, 0.2 -> ~67%

            results.append({
                'rank': i + 1,
                'card_id': card_id,
                'name': card_data.get('name', 'Unknown'),
                'set': card_data.get('set', ''),
                'similarity': float(similarity),
                'distance': float(distance)
            })

        return results


class DemoApp:
    """Pokemon Card Recognition Demo Application"""

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        card_matcher: CardMatcher,
        card_detector: IMX500CardDetector = None,
        use_camera: bool = True,
        display: bool = True,
        fullscreen: bool = True
    ):
        self.embedding_engine = embedding_engine
        self.card_matcher = card_matcher
        self.card_detector = card_detector
        self.use_camera = use_camera and CAMERA_AVAILABLE
        self.display = display
        self.fullscreen = fullscreen

        self.camera = None
        self.imx500 = None
        self.window_name = "Pokemon Card Recognition"
        self.running = False

        # Stats
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()

    def start_camera(self):
        """Initialize camera"""
        if not self.use_camera:
            return

        print("   Starting camera...")
        self.camera = Picamera2()

        # Configure camera - use RGB888 and convert to BGR for OpenCV
        # Using 1280x720 for good balance of quality and performance
        config = self.camera.create_preview_configuration(
            main={"size": (1280, 720), "format": "RGB888"}
        )
        self.camera.configure(config)
        self.camera.start()
        time.sleep(0.5)
        print("   Camera ready")

    def stop_camera(self):
        """Stop camera"""
        if self.camera:
            self.camera.stop()
            self.camera.close()
            self.camera = None

    def recognize_frame(self, frame: np.ndarray, metadata: dict = None) -> dict:
        """Recognize card(s) in frame

        Args:
            frame: BGR image
            metadata: Frame metadata from picamera2 (unused, kept for compatibility)

        Returns:
            Recognition results with detections and matches
        """
        start = time.time()
        h, w = frame.shape[:2]

        # Get detections from YOLO card detector
        detections = []
        detection_time = 0.0
        if self.card_detector:
            det_start = time.time()
            detections = self.card_detector.detect(frame)
            detection_time = (time.time() - det_start) * 1000

        # If no card detections, use center crop of frame (card-sized region)
        if not detections:
            # Use center region with card aspect ratio (2.5:3.5)
            card_h = int(h * 0.7)  # 70% of frame height
            card_w = int(card_h * 0.714)  # Card aspect ratio
            x1 = (w - card_w) // 2
            y1 = (h - card_h) // 2
            detections = [{'box': (x1, y1, x1 + card_w, y1 + card_h), 'confidence': 0.0, 'class_id': 0}]

        # Process each detection
        card_results = []
        embed_time = 0.0
        search_time = 0.0

        for det in detections[:3]:  # Limit to 3 detections
            x1, y1, x2, y2 = det['box']

            # Clip coordinates to frame bounds (important: negative indices cause bugs!)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            # Crop card region
            card_crop = frame[y1:y2, x1:x2]
            if card_crop.size == 0:
                continue

            # Extract embedding
            embed_start = time.time()
            embedding = self.embedding_engine.extract(card_crop)
            embed_time += (time.time() - embed_start) * 1000

            # Search database
            search_start = time.time()
            matches = self.card_matcher.search(embedding, k=5)
            search_time += (time.time() - search_start) * 1000

            card_results.append({
                'detection': det,
                'matches': matches,
                'top_match': matches[0] if matches else None,
                'crop': card_crop  # For preview display
            })

        total_time = (time.time() - start) * 1000

        # For backwards compatibility, get top result
        top_match = None
        all_matches = []
        card_crop = None
        if card_results:
            top_match = card_results[0].get('top_match')
            all_matches = card_results[0].get('matches', [])
            card_crop = card_results[0].get('crop')

        return {
            'cards': card_results,
            'num_detections': len(detections),
            'results': all_matches,  # Backwards compatibility
            'card_crop': card_crop,  # For preview display
            'timing': {
                'detection_ms': detection_time,
                'embedding_ms': embed_time,
                'search_ms': search_time,
                'total_ms': total_time
            },
            'top_match': top_match
        }

    def draw_results(self, frame: np.ndarray, recognition: dict) -> np.ndarray:
        """Draw recognition results on frame with card preview"""
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Draw detection boxes for each detected card
        cards = recognition.get('cards', [])
        for i, card in enumerate(cards):
            det = card.get('detection', {})
            box = det.get('box')
            top_match = card.get('top_match')

            if box:
                x1, y1, x2, y2 = box
                conf = det.get('confidence', 0)

                # Box color based on detection confidence
                box_color = (0, 255, 0) if conf > 0.7 else (0, 255, 255) if conf > 0.5 else (0, 165, 255)

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3)

                # Draw label above box
                if top_match:
                    label = f"{top_match['name'][:25]} ({top_match['similarity']:.0%})"
                    label_size, _ = cv2.getTextSize(label, font, 0.6, 2)
                    cv2.rectangle(annotated, (x1, y1 - 25), (x1 + label_size[0] + 10, y1), box_color, -1)
                    cv2.putText(annotated, label, (x1 + 5, y1 - 7), font, 0.6, (0, 0, 0), 2)

        # Draw card preview on the right side (proportional to screen size)
        # Pokemon cards have ~2.5:3.5 aspect ratio
        preview_h = int(h * 0.5)  # 50% of frame height
        preview_w = int(preview_h * 0.714)  # Card aspect ratio (2.5/3.5)
        preview_x = w - preview_w - int(w * 0.02)  # 2% margin from right
        preview_y = int(h * 0.03)  # 3% margin from top

        # Draw preview background
        overlay = annotated.copy()
        cv2.rectangle(overlay, (preview_x - 10, preview_y - 10),
                     (preview_x + preview_w + 10, preview_y + preview_h + 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
        cv2.rectangle(annotated, (preview_x - 10, preview_y - 10),
                     (preview_x + preview_w + 10, preview_y + preview_h + 40), (255, 255, 255), 2)

        # Draw "Input to Model" label
        cv2.putText(annotated, "Input to Model", (preview_x, preview_y - 15 + 5),
                   font, 0.5, (200, 200, 200), 1)

        # Draw the cropped card preview
        card_crop = recognition.get('card_crop')
        if card_crop is not None and card_crop.size > 0:
            # Resize crop to fit preview area while maintaining aspect ratio
            crop_h, crop_w = card_crop.shape[:2]
            scale = min(preview_w / crop_w, preview_h / crop_h)
            new_w = int(crop_w * scale)
            new_h = int(crop_h * scale)
            resized = cv2.resize(card_crop, (new_w, new_h))

            # Center in preview area
            offset_x = (preview_w - new_w) // 2
            offset_y = (preview_h - new_h) // 2
            annotated[preview_y + offset_y:preview_y + offset_y + new_h,
                     preview_x + offset_x:preview_x + offset_x + new_w] = resized

            # Draw crop dimensions
            cv2.putText(annotated, f"{crop_w}x{crop_h}",
                       (preview_x + 5, preview_y + preview_h + 25),
                       font, 0.4, (150, 150, 150), 1)
        else:
            # No card detected - show placeholder
            cv2.putText(annotated, "No card", (preview_x + 80, preview_y + preview_h // 2),
                       font, 0.7, (100, 100, 100), 2)

        # Draw info panel for top card (on the left)
        top = recognition.get('top_match')
        if top:
            panel_w = 400
            panel_h = 180
            panel_x = 10
            panel_y = 10

            # Semi-transparent background
            overlay2 = annotated.copy()
            cv2.rectangle(overlay2, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay2, 0.7, annotated, 0.3, 0, annotated)
            cv2.rectangle(annotated, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (255, 255, 255), 2)

            # Draw text
            y_offset = panel_y + 30
            line_height = 28

            # Card name
            cv2.putText(annotated, top['name'][:30], (panel_x + 10, y_offset), font, 0.7, (0, 255, 255), 2)
            y_offset += line_height

            # Set
            cv2.putText(annotated, f"Set: {top['set']}", (panel_x + 10, y_offset), font, 0.5, (255, 255, 255), 1)
            y_offset += line_height

            # Confidence
            conf_color = (0, 255, 0) if top['similarity'] > 0.85 else (0, 255, 255) if top['similarity'] > 0.70 else (0, 0, 255)
            cv2.putText(annotated, f"Confidence: {top['similarity']:.1%}", (panel_x + 10, y_offset), font, 0.6, conf_color, 2)
            y_offset += line_height

            # Timing
            timing = recognition['timing']
            det_ms = timing.get('detection_ms', 0)
            cv2.putText(annotated, f"Det: {det_ms:.0f}ms | Embed: {timing['embedding_ms']:.0f}ms | Search: {timing['search_ms']:.0f}ms",
                        (panel_x + 10, y_offset), font, 0.45, (180, 180, 180), 1)
            y_offset += line_height
            cv2.putText(annotated, f"Total: {timing['total_ms']:.0f}ms | Cards: {recognition.get('num_detections', 0)}",
                        (panel_x + 10, y_offset), font, 0.45, (180, 180, 180), 1)

        # FPS in bottom left
        cv2.putText(annotated, f"FPS: {self.fps:.1f}", (20, h - 20), font, 0.7, (0, 255, 0), 2)

        return annotated

    def run_camera(self):
        """Run live camera recognition"""
        print("\nStarting live recognition...")
        print("Controls: q=quit, s=screenshot, f=fullscreen, a=toggle auto-screenshot")

        if self.display:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            if self.fullscreen:
                # Create fullscreen window
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                # Smaller window that fits most screens
                cv2.resizeWindow(self.window_name, 960, 540)

        self.running = True
        self.last_fps_time = time.time()
        is_fullscreen = self.fullscreen
        last_screenshot_time = 0
        screenshot_interval = 5.0  # Auto screenshot every 5 seconds for debugging

        try:
            while self.running:
                # Capture frame with metadata (for IMX500 outputs)
                request = self.camera.capture_request()
                frame = request.make_array("main")
                metadata = request.get_metadata()
                request.release()

                # Note: Picamera2's "RGB888" format actually outputs BGR byte order
                # (confusing naming from libcamera) - perfect for OpenCV, no conversion needed
                frame_bgr = frame

                # Recognize with metadata
                recognition = self.recognize_frame(frame_bgr, metadata)

                # Draw results
                annotated = self.draw_results(frame_bgr, recognition)

                # Update FPS
                self.frame_count += 1
                now = time.time()
                if now - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count / (now - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = now

                # Display
                if self.display:
                    cv2.imshow(self.window_name, annotated)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # q or ESC
                        break
                    elif key == ord('s'):
                        filename = f"screenshot_{int(time.time())}.jpg"
                        cv2.imwrite(filename, annotated)
                        print(f"Saved: {filename}")
                    elif key == ord('a'):
                        # Toggle auto-screenshot
                        if screenshot_interval > 0:
                            screenshot_interval = 0
                            print("Auto-screenshot disabled")
                        else:
                            screenshot_interval = 5.0
                            print("Auto-screenshot enabled (every 5s)")
                    elif key == ord('f'):
                        # Toggle fullscreen
                        is_fullscreen = not is_fullscreen
                        if is_fullscreen:
                            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        else:
                            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                            cv2.resizeWindow(self.window_name, 1280, 720)

                # Auto-screenshot for debugging
                if screenshot_interval > 0:
                    now_auto = time.time()
                    if now_auto - last_screenshot_time >= screenshot_interval:
                        filename = f"auto_screenshot_{int(now_auto)}.jpg"
                        cv2.imwrite(filename, annotated)
                        print(f"Auto-saved: {filename}")
                        last_screenshot_time = now_auto

        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            if self.display:
                cv2.destroyAllWindows()

    def run_image(self, image_path: str):
        """Recognize single image"""
        print(f"\nProcessing: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image")
            return

        recognition = self.recognize_frame(image)

        print("\n" + "=" * 50)
        print("Recognition Results")
        print("=" * 50)

        for r in recognition['results']:
            print(f"\n#{r['rank']} - {r['name']} ({r['set']})")
            print(f"   Similarity: {r['similarity']:.1%}")

        print("\n" + "-" * 50)
        print("Timing:")
        print(f"  Embedding: {recognition['timing']['embedding_ms']:.1f} ms")
        print(f"  Search:    {recognition['timing']['search_ms']:.1f} ms")
        print(f"  Total:     {recognition['timing']['total_ms']:.1f} ms")

        if self.display:
            annotated = self.draw_results(image, recognition)
            cv2.imshow(self.window_name, annotated)
            print("\nPress any key to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Pokemon Card Recognition Demo')
    parser.add_argument('--image', type=str, help='Path to card image (single image mode)')
    parser.add_argument('--model', type=str,
                       default='models/onnx/pokemon_student_stage2_final.onnx',
                       help='Path to ONNX embedding model')
    parser.add_argument('--detector', type=str,
                       default='models/detection/card_detector_obb.pt',
                       help='Path to YOLO card detector model (.pt file)')
    parser.add_argument('--no-detector', action='store_true',
                       help='Disable IMX500 YOLO detection (use whole frame)')
    parser.add_argument('--hailo', action='store_true',
                       help='Use Hailo HEF model (requires regenerated reference DB)')
    parser.add_argument('--hef-model', type=str,
                       default='models/embedding/pokemon_student_hailo8_with_norm.hef',
                       help='Path to Hailo HEF model (when --hailo is used)')
    parser.add_argument('--reference', type=str,
                       default='data/reference',
                       help='Path to reference database directory')
    parser.add_argument('--no-camera', action='store_true',
                       help='Disable camera (test mode)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display (headless mode)')
    parser.add_argument('--no-fullscreen', action='store_true',
                       help='Disable fullscreen mode')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='Detection confidence threshold (0-1)')

    args = parser.parse_args()

    print("=" * 60)
    print("Pokemon Card Recognition System")
    print("=" * 60)

    # Initialize card detector (YOLO-based)
    card_detector = None
    if not args.no_detector:
        print("\nInitializing card detector...")
        detector_path = Path(args.detector)
        if detector_path.exists():
            card_detector = YOLOCardDetector(
                str(detector_path),
                conf_threshold=args.conf_threshold
            )
            if card_detector.setup():
                print(f"   Card detector ready")
            else:
                card_detector = None
                print("   Running without card detection (using whole frame)")
        else:
            print(f"   Warning: Detector not found: {args.detector}")
            print("   Running without card detection (using whole frame)")

    # Initialize embedding engine
    # Prefer Hailo NPU for faster inference (~15ms vs ~100ms on CPU)
    # Research confirms: Hailo FP32 outputs are compatible with ONNX-generated reference DB
    print("\nInitializing embedding engine...")
    embedding_engine = None

    if args.hailo and HAILO_AVAILABLE:
        print("   Using Hailo NPU (HEF model) - ~15ms inference")
        embedding_engine = HailoEmbeddingEngine(args.hef_model)
    elif not args.hailo and HAILO_AVAILABLE and Path(args.hef_model).exists():
        # Auto-use Hailo if available and model exists
        print("   Auto-detected Hailo NPU - using for faster inference (~15ms)")
        embedding_engine = HailoEmbeddingEngine(args.hef_model)
    else:
        print("   Using ONNX Runtime (CPU) - ~100ms inference")
        embedding_engine = ONNXEmbeddingEngine(args.model)
    embedding_engine.warmup()

    # Initialize card matcher
    print("\nLoading reference database...")
    card_matcher = CardMatcher(args.reference)
    card_matcher.load()

    # Create app
    use_camera = not args.no_camera and not args.image
    app = DemoApp(
        embedding_engine=embedding_engine,
        card_matcher=card_matcher,
        card_detector=card_detector,
        use_camera=use_camera,
        display=not args.no_display,
        fullscreen=not args.no_fullscreen
    )

    if args.image:
        # Single image mode
        app.run_image(args.image)
    elif args.no_camera:
        # Test mode without camera
        print("\nTest mode (no camera)")
        print("Creating dummy image...")
        dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        recognition = app.recognize_frame(dummy)
        print(f"Top match: {recognition['top_match']['name']} ({recognition['top_match']['similarity']:.1%})")
        print(f"Total time: {recognition['timing']['total_ms']:.1f} ms")
    else:
        # Live camera mode
        app.start_camera()
        try:
            app.run_camera()
        finally:
            app.stop_camera()

    print("\nDone!")


if __name__ == "__main__":
    main()
