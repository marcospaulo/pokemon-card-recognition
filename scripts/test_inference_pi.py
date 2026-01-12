#!/usr/bin/env python3
"""
Test Pokemon card recognition inference on Raspberry Pi with Hailo-8L

Usage:
    # Test with random image
    python3 test_inference_pi.py --model models/pokemon_student.hef

    # Test with specific image
    python3 test_inference_pi.py --model models/pokemon_student.hef --image card.jpg

    # Benchmark speed
    python3 test_inference_pi.py --model models/pokemon_student.hef --benchmark
"""

import argparse
import time
import numpy as np
from pathlib import Path
from PIL import Image


def preprocess_image(image_path, size=224):
    """Preprocess image for model input"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((size, size), Image.BILINEAR)

    # Convert to numpy array and normalize
    img_array = np.array(img).astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img_array = (img_array - mean) / std

    # CHW format
    img_array = np.transpose(img_array, (2, 0, 1))

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def run_inference_hailo(hef_path, image_array):
    """Run inference using Hailo accelerator"""
    try:
        from hailo_platform import (HEF, ConfigureParams, HailoSchedulingAlgorithm,
                                    HailoStreamInterface, InferVStreams, InputVStreamParams,
                                    OutputVStreamParams, FormatType)

        # Load HEF
        hef = HEF(hef_path)

        # Configure parameters
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)

        # Create input/output stream parameters
        input_vstreams_params = InputVStreamParams.make_from_network_group(hef, quantized=False, format_type=FormatType.FLOAT32)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(hef, quantized=False, format_type=FormatType.FLOAT32)

        # Run inference
        with InferVStreams(hef, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            input_data = {hef.get_input_vstream_names()[0]: image_array}
            output = infer_pipeline.infer(input_data)

            # Get embeddings
            embedding = output[hef.get_output_vstream_names()[0]]
            return embedding

    except ImportError:
        print("❌ ERROR: HailoRT not installed!")
        print()
        print("Install HailoRT:")
        print("  wget https://hailo.ai/developer-zone/software-downloads/hailort/hailort-4.17.0-py3-none-linux_aarch64.whl")
        print("  pip3 install hailort-*.whl")
        return None


def run_inference_cpu(image_array):
    """Fallback: Run inference on CPU using ONNX Runtime"""
    try:
        import onnxruntime as ort

        print("⚠️  Running on CPU (Hailo not available)")

        # Find ONNX model
        onnx_files = list(Path('models').glob('*.onnx'))
        if not onnx_files:
            print("❌ ERROR: No ONNX model found in models/")
            return None

        onnx_path = str(onnx_files[0])
        print(f"Using ONNX model: {onnx_path}")

        session = ort.InferenceSession(onnx_path)
        outputs = session.run(None, {'input': image_array})
        return outputs[0]

    except ImportError:
        print("❌ ERROR: Neither HailoRT nor ONNX Runtime available")
        print("Install ONNX Runtime: pip3 install onnxruntime")
        return None


def benchmark(hef_path, num_runs=100):
    """Benchmark inference speed"""
    print(f"\n{'='*60}")
    print("BENCHMARK MODE")
    print(f"{'='*60}")
    print(f"Runs: {num_runs}")
    print()

    # Create random input
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Warmup
    print("Warming up...")
    for _ in range(10):
        _ = run_inference_hailo(hef_path, dummy_input)

    # Benchmark
    print(f"Running {num_runs} inferences...")
    times = []

    for i in range(num_runs):
        start = time.time()
        _ = run_inference_hailo(hef_path, dummy_input)
        elapsed = time.time() - start
        times.append(elapsed)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{num_runs}")

    # Statistics
    times = np.array(times) * 1000  # Convert to ms
    print()
    print(f"{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Mean:   {times.mean():.2f} ms")
    print(f"Median: {np.median(times):.2f} ms")
    print(f"Min:    {times.min():.2f} ms")
    print(f"Max:    {times.max():.2f} ms")
    print(f"Std:    {times.std():.2f} ms")
    print()
    print(f"Throughput: {1000 / times.mean():.1f} images/second")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Test inference on Raspberry Pi')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to HEF model file')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to test image (optional)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark instead of single inference')
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"❌ ERROR: Model file not found: {args.model}")
        return 1

    print(f"{'='*60}")
    print("POKEMON CARD RECOGNITION - RASPBERRY PI TEST")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print()

    if args.benchmark:
        benchmark(args.model)
        return 0

    # Single inference test
    if args.image:
        if not Path(args.image).exists():
            print(f"❌ ERROR: Image not found: {args.image}")
            return 1

        print(f"Image: {args.image}")
        image_array = preprocess_image(args.image)
    else:
        print("Image: Random test image (no image provided)")
        image_array = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Run inference
    print("\nRunning inference...")
    start = time.time()

    try:
        embedding = run_inference_hailo(args.model, image_array)
    except Exception as e:
        print(f"Hailo inference failed: {e}")
        print("Trying CPU fallback...")
        embedding = run_inference_cpu(image_array)

    elapsed = time.time() - start

    if embedding is not None:
        print(f"\n{'='*60}")
        print("✅ INFERENCE SUCCESSFUL")
        print(f"{'='*60}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Inference time: {elapsed*1000:.2f} ms")
        print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
        print(f"Embedding sample: {embedding[0, :5]}")
        print(f"{'='*60}")
        return 0
    else:
        print("\n❌ INFERENCE FAILED")
        return 1


if __name__ == '__main__':
    exit(main())
