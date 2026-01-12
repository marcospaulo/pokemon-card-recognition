"""Hailo-8L inference engine for YOLO11-OBB detection

Provides hardware-accelerated inference using Hailo-8L AI accelerator.
Target: 50-70ms inference latency for 640x640 input.
"""

import numpy as np
import time
from typing import List, Optional, Tuple
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.pipeline.interfaces import InferenceEngineInterface
from src.utils.data_structures import Detection

try:
    from hailo_platform import (
        HEF,
        VDevice,
        HailoStreamInterface,
        InferVStreams,
        ConfigureParams,
        InputVStreamParams,
        OutputVStreamParams,
        FormatType,
    )
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    print("Warning: hailo_platform not available. HailoEngine will not work.")


class HailoEngine(InferenceEngineInterface):
    """Hailo-8L inference engine for YOLO11-OBB

    Runs quantized INT8 YOLO11-OBB model on Hailo-8L accelerator.
    Handles preprocessing, inference, and postprocessing of oriented bounding boxes.

    Performance target: 50-70ms per frame (640x640 input)
    """

    def __init__(
        self,
        hef_path: str,
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.45,
        max_detections: int = 20,
    ):
        """Initialize Hailo inference engine

        Args:
            hef_path: Path to compiled HEF model file
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
            nms_iou_threshold: IoU threshold for non-maximum suppression
            max_detections: Maximum number of detections to return
        """
        if not HAILO_AVAILABLE:
            raise RuntimeError(
                "hailo_platform not available. Install HailoRT: "
                "https://github.com/hailo-ai/hailort"
            )

        self.hef_path = Path(hef_path)
        if not self.hef_path.exists():
            raise FileNotFoundError(f"HEF file not found: {hef_path}")

        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = max_detections

        # Hailo components
        self._hef = None
        self._vdevice = None
        self._network_group = None
        self._network_group_params = None
        self._input_vstreams = None
        self._output_vstreams = None

        self._model_loaded = False
        self._input_shape = None
        self._output_shapes = None

        # Performance tracking
        self._inference_count = 0
        self._total_inference_time = 0.0

    def warmup(self):
        """Initialize Hailo device and load model"""
        if self._model_loaded:
            print("Warning: Model already loaded")
            return

        print(f"Loading HEF model: {self.hef_path}")

        try:
            # Load HEF file
            self._hef = HEF(str(self.hef_path))

            # Create virtual device (Hailo-8L)
            self._vdevice = VDevice()

            # Configure network group
            configure_params = ConfigureParams.create_from_hef(
                self._hef, interface=HailoStreamInterface.PCIe
            )
            self._network_group = self._vdevice.configure(self._hef, configure_params)[0]
            self._network_group_params = self._network_group.create_params()

            # Get input/output parameters
            input_vstream_info = self._hef.get_input_vstream_infos()[0]
            output_vstream_infos = self._hef.get_output_vstream_infos()

            # Store input shape
            self._input_shape = input_vstream_info.shape

            # Create input VStream parameters
            self._input_vstreams_params = InputVStreamParams.make_from_network_group(
                self._network_group, quantized=False, format_type=FormatType.UINT8
            )

            # Create output VStream parameters
            self._output_vstreams_params = OutputVStreamParams.make_from_network_group(
                self._network_group, quantized=False, format_type=FormatType.FLOAT32
            )

            self._model_loaded = True

            print(f"✓ Hailo model loaded")
            print(f"  Input shape: {self._input_shape}")
            print(f"  Device: Hailo-8L")

            # Warm up with dummy inference
            print("Running warmup inference...")
            dummy_input = np.zeros(self._input_shape, dtype=np.uint8)
            self.detect(dummy_input)
            print("✓ Warmup complete")

        except Exception as e:
            self._model_loaded = False
            raise RuntimeError(f"Failed to load Hailo model: {e}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLO11-OBB inference on frame

        Args:
            frame: Preprocessed frame (640x640x3, BGR, uint8)

        Returns:
            List of Detection objects with oriented bounding boxes

        Performance: 50-70ms on Hailo-8L
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call warmup() first.")

        start_time = time.time()

        try:
            # Validate input shape
            if frame.shape != tuple(self._input_shape):
                raise ValueError(
                    f"Input shape mismatch. Expected {self._input_shape}, got {frame.shape}"
                )

            # Convert BGR to RGB (YOLO expects RGB)
            frame_rgb = frame[:, :, ::-1].copy()

            # Run inference
            with InferVStreams(
                self._network_group,
                self._input_vstreams_params,
                self._output_vstreams_params,
            ) as infer_pipeline:
                # Send input
                input_dict = {self._input_vstreams_params.keys()[0]: frame_rgb}

                # Get output
                output_dict = infer_pipeline.infer(input_dict)

            # Parse YOLO11-OBB output
            detections = self._parse_yolo_obb_output(output_dict)

            # Apply NMS
            detections = self._apply_nms(detections)

            # Track performance
            inference_time = (time.time() - start_time) * 1000
            self._inference_count += 1
            self._total_inference_time += inference_time

            return detections[:self.max_detections]

        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")

    def _parse_yolo_obb_output(self, output_dict: dict) -> List[Detection]:
        """Parse YOLO11-OBB output tensors to Detection objects

        YOLO11-OBB output format:
        - Shape: (1, num_boxes, 7)
        - Columns: [cx, cy, w, h, angle, confidence, class_id]

        Args:
            output_dict: Raw output from Hailo inference

        Returns:
            List of Detection objects
        """
        detections = []

        # Get output tensor (assumes single output)
        output_key = list(output_dict.keys())[0]
        output = output_dict[output_key]

        # Handle different output formats
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension

        # Parse each detection
        for detection_data in output:
            if len(detection_data) < 7:
                continue

            cx, cy, w, h, angle, confidence, class_id = detection_data[:7]

            # Filter by confidence threshold
            if confidence < self.confidence_threshold:
                continue

            # Create Detection object
            detection = Detection(
                card_name=self._get_class_name(int(class_id)),
                confidence=float(confidence),
                obb=(float(cx), float(cy), float(w), float(h), float(angle)),
            )

            detections.append(detection)

        return detections

    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply non-maximum suppression to remove overlapping boxes

        Args:
            detections: List of detections

        Returns:
            Filtered list after NMS
        """
        if len(detections) <= 1:
            return detections

        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        # NMS logic (simplified - assumes axis-aligned approximation)
        kept = []
        for i, det in enumerate(detections):
            # Check overlap with already kept detections
            overlap = False
            for kept_det in kept:
                if self._compute_iou_obb(det.obb, kept_det.obb) > self.nms_iou_threshold:
                    overlap = True
                    break

            if not overlap:
                kept.append(det)

        return kept

    def _compute_iou_obb(
        self, obb1: Tuple[float, float, float, float, float],
        obb2: Tuple[float, float, float, float, float]
    ) -> float:
        """Compute IoU between two oriented bounding boxes

        Simplified implementation using axis-aligned approximation.
        For production, use rotated IoU calculation.

        Args:
            obb1: (cx, cy, w, h, angle)
            obb2: (cx, cy, w, h, angle)

        Returns:
            IoU score (0.0-1.0)
        """
        cx1, cy1, w1, h1, _ = obb1
        cx2, cy2, w2, h2, _ = obb2

        # Axis-aligned approximation (ignores rotation)
        x1_min, x1_max = cx1 - w1/2, cx1 + w1/2
        y1_min, y1_max = cy1 - h1/2, cy1 + h1/2

        x2_min, x2_max = cx2 - w2/2, cx2 + w2/2
        y2_min, y2_max = cy2 - h2/2, cy2 + h2/2

        # Compute intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Compute union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _get_class_name(self, class_id: int) -> str:
        """Map class ID to Pokemon card name

        TODO: Load from class names file or config

        Args:
            class_id: Class index

        Returns:
            Card name string
        """
        # Placeholder - should load from config
        class_names = {
            0: "Pokemon Card",
        }
        return class_names.get(class_id, f"Unknown_{class_id}")

    @property
    def model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model_loaded

    @property
    def model_input_size(self) -> Tuple[int, int]:
        """Get model input dimensions"""
        if self._input_shape:
            height, width = self._input_shape[:2]
            return (width, height)
        return (640, 640)

    @property
    def avg_inference_time_ms(self) -> float:
        """Get average inference time in milliseconds"""
        if self._inference_count == 0:
            return 0.0
        return self._total_inference_time / self._inference_count

    def get_device_info(self) -> dict:
        """Get Hailo device information

        Returns:
            Dictionary with device stats
        """
        return {
            "model_loaded": self._model_loaded,
            "hef_path": str(self.hef_path),
            "input_shape": self._input_shape,
            "inference_count": self._inference_count,
            "avg_inference_ms": self.avg_inference_time_ms,
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_iou_threshold,
        }

    def __del__(self):
        """Cleanup Hailo resources"""
        if self._vdevice:
            try:
                self._vdevice.release()
            except:
                pass
