"""Abstract base classes for pipeline components

Defines interfaces for:
- Camera capture devices (IMX500, webcam, etc.)
- Inference engines (Hailo, ONNX Runtime, etc.)
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import numpy as np


class CameraInterface(ABC):
    """Abstract interface for camera capture devices

    All camera implementations (PiCamera, IMX500AICamera, USB webcam)
    must implement this interface for consistent frame capture.
    """

    @abstractmethod
    def start(self) -> None:
        """Initialize and start camera capture

        Raises:
            RuntimeError: If camera initialization fails
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop camera and release resources"""
        pass

    @abstractmethod
    def capture_frame(self) -> np.ndarray:
        """Capture single frame from camera

        Returns:
            Frame as numpy array (H, W, 3) in BGR format

        Raises:
            RuntimeError: If capture fails or camera not started
        """
        pass

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """Check if camera is ready for capture"""
        pass

    @property
    @abstractmethod
    def frame_size(self) -> Tuple[int, int]:
        """Get camera frame dimensions (width, height)"""
        pass

    @property
    @abstractmethod
    def last_capture_time_ms(self) -> float:
        """Get last frame capture latency in milliseconds"""
        pass


class InferenceEngineInterface(ABC):
    """Abstract interface for AI inference engines

    All inference implementations (Hailo, ONNX Runtime, TensorRT)
    must implement this interface for consistent model inference.
    """

    @abstractmethod
    def warmup(self) -> None:
        """Initialize device and load model

        Should perform any necessary warmup inference to ensure
        consistent latency for subsequent calls.

        Raises:
            RuntimeError: If initialization fails
        """
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List:
        """Run inference on frame

        Args:
            frame: Preprocessed input frame

        Returns:
            List of detection/embedding results

        Raises:
            RuntimeError: If inference fails
        """
        pass

    @property
    @abstractmethod
    def model_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        pass

    @property
    @abstractmethod
    def model_input_size(self) -> Tuple[int, int]:
        """Get expected model input dimensions (width, height)"""
        pass

    @property
    @abstractmethod
    def avg_inference_time_ms(self) -> float:
        """Get average inference latency in milliseconds"""
        pass

    @abstractmethod
    def get_device_info(self) -> dict:
        """Get device and model information

        Returns:
            Dictionary with device stats, model info, performance metrics
        """
        pass
