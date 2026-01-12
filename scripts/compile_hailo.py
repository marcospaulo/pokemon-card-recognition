#!/usr/bin/env python3
"""
Compile ONNX model to Hailo HEF format using Hailo Dataflow Compiler.

This script should run on an EC2 instance with Hailo DFC installed.

Usage:
    python compile_hailo.py --onnx pokemon_student_convnext_tiny.onnx
"""

import argparse
from pathlib import Path


def compile_to_hailo(onnx_path, output_dir, model_name='pokemon_student'):
    """
    Compile ONNX model to Hailo HEF format

    Note: This is a template. Actual Hailo compilation requires:
    1. Hailo Dataflow Compiler installed
    2. Calibration dataset for quantization
    3. Model optimization configuration
    """

    print("="*60)
    print("HAILO MODEL COMPILATION")
    print("="*60)

    try:
        from hailo_sdk_client import ClientRunner
    except ImportError:
        print("❌ ERROR: Hailo SDK not found!")
        print()
        print("Install Hailo Dataflow Compiler:")
        print("  pip install hailo_dataflow_compiler-3.28.0-py3-none-linux_x86_64.whl")
        print()
        print("Get the package from:")
        print("  https://hailo.ai/developer-zone/software-downloads/")
        return False

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input ONNX: {onnx_path}")
    print(f"Output directory: {output_dir}")
    print()

    # Initialize Hailo runner
    print("Initializing Hailo runner...")
    runner = ClientRunner(hw_arch='hailo8l')

    # Load ONNX model
    print("Loading ONNX model...")
    hn, npz = runner.translate_onnx_model(
        onnx_path,
        model_name,
        start_node_names=['input'],
        end_node_names=['embeddings'],
        net_input_shapes={'input': [1, 3, 224, 224]}
    )

    # Optimize model
    print("Optimizing model...")
    runner.optimize(hn)

    # Quantization (requires calibration dataset)
    print()
    print("⚠️  QUANTIZATION STEP")
    print("-" * 60)
    print("For production use, you should provide a calibration dataset")
    print("of representative Pokemon card images.")
    print()
    print("Example calibration dataset structure:")
    print("  calibration_data/")
    print("    ├── image_001.jpg")
    print("    ├── image_002.jpg")
    print("    └── ...")
    print()
    print("Using default quantization (no calibration)...")
    print("-" * 60)
    print()

    # Quantize (simplified - production should use calibration data)
    runner.quantize(
        hn,
        calib_dataset=None,  # TODO: Add calibration dataset
        batch_size=8
    )

    # Compile to HEF
    print("Compiling to HEF...")
    hef_path = output_dir / f'{model_name}.hef'
    runner.compile(
        hn,
        output_file_path=str(hef_path)
    )

    print()
    print("="*60)
    print("✅ COMPILATION SUCCESSFUL")
    print("="*60)
    print(f"HEF file: {hef_path}")
    print(f"Size: {hef_path.stat().st_size / (1024**2):.1f} MB")
    print()
    print("Next steps:")
    print(f"  1. Copy {hef_path.name} to your Raspberry Pi")
    print("  2. Install HailoRT on the Pi")
    print("  3. Run inference with the HEF model")
    print("="*60)

    return True


def main():
    parser = argparse.ArgumentParser(description='Compile ONNX to Hailo HEF')
    parser.add_argument('--onnx', type=str, required=True,
                       help='Path to ONNX model')
    parser.add_argument('--output-dir', type=str, default='./hailo_output',
                       help='Output directory for HEF')
    parser.add_argument('--model-name', type=str, default='pokemon_student',
                       help='Model name')
    parser.add_argument('--calibration-data', type=str, default=None,
                       help='Path to calibration dataset (optional but recommended)')
    args = parser.parse_args()

    if not Path(args.onnx).exists():
        print(f"❌ ERROR: ONNX file not found: {args.onnx}")
        return 1

    success = compile_to_hailo(args.onnx, args.output_dir, args.model_name)
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
