#!/usr/bin/env python3
"""Compile Pokemon YOLO-OBB model to Hailo HEF format

This script should be run on an EC2 instance with the Hailo Dataflow Compiler installed.

Prerequisites:
- Hailo Dataflow Compiler (hailo_sdk_client)
- hailo_model_zoo package
- ONNX model: models/detection/card_detector_obb.onnx
- Calibration images in: data/calibration_images/ (100-500 images)

Usage:
  python scripts/compile_yolo_hailo.py --onnx models/detection/card_detector_obb.onnx --output models/detection/card_detector_obb.hef
"""

import argparse
import os
import sys
from pathlib import Path


def compile_with_model_zoo(onnx_path: str, output_path: str, calib_path: str,
                            target: str = "hailo8"):
    """Compile using Hailo Model Zoo (preferred method)"""
    import subprocess

    # Use hailomz compile command
    cmd = [
        "hailomz", "compile",
        "--ckpt", onnx_path,
        "--yaml", "yolov8s",  # Use YOLOv8 config as base
        "--hw-arch", target,
        "--classes", "1",  # Single class: Pokemon card
    ]

    if calib_path and os.path.exists(calib_path):
        cmd.extend(["--calib-path", calib_path])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False

    return True


def compile_with_dfc(onnx_path: str, output_path: str, calib_path: str,
                     target: str = "hailo8"):
    """Compile using Hailo DFC directly"""
    try:
        from hailo_sdk_client import ClientRunner
    except ImportError:
        print("Error: hailo_sdk_client not installed")
        print("Install with: pip install hailo_sdk_client")
        return False

    print(f"Compiling {onnx_path} to {output_path}")
    print(f"Target: {target}")

    # Create runner
    runner = ClientRunner(hw_arch=target)

    # Parse ONNX model
    print("\n[1/4] Parsing ONNX model...")
    hn, npz = runner.translate_onnx_model(
        onnx_path,
        net_name="pokemon_yolo_obb",
        start_node_names=None,
        end_node_names=None,
        net_input_shapes={"images": [1, 3, 640, 640]}
    )

    # Optimize model
    print("\n[2/4] Optimizing model...")
    runner.optimize(hn)

    # Quantize model
    if calib_path and os.path.exists(calib_path):
        print(f"\n[3/4] Quantizing with calibration data from {calib_path}...")
        # Load calibration images
        import numpy as np
        from PIL import Image
        import glob

        calib_images = glob.glob(os.path.join(calib_path, "*.jpg"))[:100]
        calib_data = []
        for img_path in calib_images:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((640, 640))
            arr = np.array(img).astype(np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)  # HWC -> CHW
            calib_data.append(arr)

        calib_dataset = np.stack(calib_data, axis=0)
        runner.quantize(
            hn,
            calib_set={"images": calib_dataset}
        )
    else:
        print("\n[3/4] Quantizing with random calibration (no calib images)...")
        runner.quantize(hn)

    # Compile to HEF
    print("\n[4/4] Compiling to HEF...")
    hef = runner.compile()

    # Save HEF
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(hef)

    print(f"\nSuccess! HEF saved to: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Compile Pokemon YOLO to Hailo HEF")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--output", required=True, help="Output HEF path")
    parser.add_argument("--calib-path", help="Path to calibration images")
    parser.add_argument("--target", default="hailo8", choices=["hailo8", "hailo8l"],
                       help="Target Hailo chip")
    parser.add_argument("--use-model-zoo", action="store_true",
                       help="Use hailomz instead of DFC directly")
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        print(f"Error: ONNX model not found: {args.onnx}")
        sys.exit(1)

    if args.use_model_zoo:
        success = compile_with_model_zoo(args.onnx, args.output, args.calib_path, args.target)
    else:
        success = compile_with_dfc(args.onnx, args.output, args.calib_path, args.target)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
