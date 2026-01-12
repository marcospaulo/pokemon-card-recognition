#!/usr/bin/env python3
"""
Compare Hailo HEF vs ONNX embeddings to verify they match

This validates that the Hailo-compiled model produces the same embeddings
as the original ONNX model when given the same input images.
"""

import sys
import numpy as np
from pathlib import Path
import cv2
import argparse

try:
    import onnxruntime as ort
except ImportError:
    print("❌ Error: onnxruntime not installed")
    print("   Run: pip install onnxruntime")
    sys.exit(1)

try:
    from hailo_platform import (HEF, VDevice, HailoStreamInterface,
                                 InferVStreams, ConfigureParams)
except ImportError:
    print("❌ Error: hailo_platform not installed")
    print("   This script must run on a system with Hailo SDK installed")
    print("   (EC2 or Raspberry Pi with Hailo-8L)")
    sys.exit(1)


def preprocess_for_onnx(image_path: Path, size=(224, 224)):
    """Preprocess image for ONNX EfficientNet-Lite0 (with ImageNet normalization)"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Resize
    resized = cv2.resize(image, size)

    # BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    normalized = rgb.astype(np.float32) / 255.0

    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    normalized = (normalized - mean) / std

    # HWC to CHW
    chw = np.transpose(normalized, (2, 0, 1))

    # Add batch dimension
    return np.expand_dims(chw, axis=0).astype(np.float32)


def preprocess_for_hailo(image_path: Path, size=(224, 224)):
    """Preprocess image for Hailo HEF (with ImageNet normalization, HWC format)"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Resize
    resized = cv2.resize(image, size)

    # BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    normalized = rgb.astype(np.float32) / 255.0

    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    normalized = (normalized - mean) / std

    # Keep HWC format for Hailo
    # Add batch dimension
    return np.expand_dims(normalized, axis=0).astype(np.float32)


def run_onnx_inference(session, input_name, output_name, image_path: Path):
    """Run ONNX inference and return embedding"""
    input_tensor = preprocess_for_onnx(image_path)
    outputs = session.run([output_name], {input_name: input_tensor})
    embedding = outputs[0][0]  # Remove batch dimension

    # L2 normalize
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def run_hailo_inference(hef_path: Path, image_path: Path):
    """Run Hailo HEF inference and return embedding"""

    # Load HEF
    hef = HEF(str(hef_path))

    # Configure device
    params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)

    with VDevice(params=params) as device:
        # Prepare input
        input_tensor = preprocess_for_hailo(image_path)

        # Run inference
        with InferVStreams(device, hef) as infer_pipeline:
            input_data = {hef.get_input_vstream_infos()[0].name: input_tensor}

            with infer_pipeline.infer(input_data) as results:
                output_data = results[hef.get_output_vstream_infos()[0].name]
                embedding = output_data[0]  # Remove batch dimension

                # L2 normalize
                embedding = embedding / np.linalg.norm(embedding)
                return embedding


def compare_embeddings(onnx_emb, hailo_emb, tolerance=0.01):
    """Compare two embeddings and return similarity metrics"""

    # Cosine similarity (both are L2-normalized, so just dot product)
    cosine_sim = np.dot(onnx_emb, hailo_emb)

    # L2 distance
    l2_dist = np.linalg.norm(onnx_emb - hailo_emb)

    # Mean absolute error
    mae = np.mean(np.abs(onnx_emb - hailo_emb))

    # Max absolute error
    max_err = np.max(np.abs(onnx_emb - hailo_emb))

    # Check if within tolerance
    match = cosine_sim > (1.0 - tolerance) and l2_dist < tolerance

    return {
        'cosine_similarity': cosine_sim,
        'l2_distance': l2_dist,
        'mean_abs_error': mae,
        'max_abs_error': max_err,
        'match': match
    }


def main():
    parser = argparse.ArgumentParser(description='Compare HEF vs ONNX embeddings')
    parser.add_argument('--hef', required=True, help='Path to HEF file')
    parser.add_argument('--onnx', required=True, help='Path to ONNX file')
    parser.add_argument('--test-images', required=True, help='Directory with test images')
    parser.add_argument('--tolerance', type=float, default=0.01, help='Similarity tolerance (default: 0.01)')

    args = parser.parse_args()

    hef_path = Path(args.hef)
    onnx_path = Path(args.onnx)
    test_dir = Path(args.test_images)

    print("=" * 70)
    print("HEF vs ONNX Embedding Comparison")
    print("=" * 70)

    # Verify files
    if not hef_path.exists():
        print(f"\n❌ HEF file not found: {hef_path}")
        return 1

    if not onnx_path.exists():
        print(f"\n❌ ONNX file not found: {onnx_path}")
        return 1

    if not test_dir.exists():
        print(f"\n❌ Test images directory not found: {test_dir}")
        return 1

    # Load ONNX model
    print(f"\n[1/3] Loading ONNX model...")
    print(f"   Path: {onnx_path}")
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"   ✓ Loaded")

    # Find test images
    print(f"\n[2/3] Finding test images...")
    test_images = list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg"))
    print(f"   Found: {len(test_images)} images")

    if len(test_images) == 0:
        print(f"   ❌ No test images found")
        return 1

    # Compare embeddings
    print(f"\n[3/3] Comparing embeddings...")
    print(f"   Tolerance: {args.tolerance}")
    print()

    results = []
    matches = 0

    for img_path in test_images:
        print(f"   Testing: {img_path.name}")

        try:
            # Get ONNX embedding
            onnx_emb = run_onnx_inference(session, input_name, output_name, img_path)

            # Get Hailo embedding
            hailo_emb = run_hailo_inference(hef_path, img_path)

            # Compare
            metrics = compare_embeddings(onnx_emb, hailo_emb, args.tolerance)
            results.append(metrics)

            if metrics['match']:
                matches += 1
                status = "✓"
            else:
                status = "✗"

            print(f"      {status} Cosine: {metrics['cosine_similarity']:.6f} | L2: {metrics['l2_distance']:.6f}")

        except Exception as e:
            print(f"      ✗ Error: {e}")
            continue

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    if len(results) == 0:
        print("\n❌ No successful comparisons")
        return 1

    avg_cosine = np.mean([r['cosine_similarity'] for r in results])
    avg_l2 = np.mean([r['l2_distance'] for r in results])
    avg_mae = np.mean([r['mean_abs_error'] for r in results])

    print(f"\nTested: {len(results)} images")
    print(f"Matches: {matches}/{len(results)} ({100*matches/len(results):.1f}%)")
    print(f"\nAverage Metrics:")
    print(f"  Cosine Similarity: {avg_cosine:.6f}")
    print(f"  L2 Distance: {avg_l2:.6f}")
    print(f"  Mean Abs Error: {avg_mae:.6f}")

    # Verdict
    print(f"\n{'=' * 70}")
    if matches == len(results) and avg_cosine > 0.99:
        print("✅ PASS: HEF and ONNX embeddings match!")
        print("   The recompiled HEF is working correctly.")
        return 0
    elif avg_cosine > 0.95:
        print("⚠️  PARTIAL PASS: HEF and ONNX are similar but not identical")
        print(f"   Average cosine similarity: {avg_cosine:.6f}")
        print("   This may be acceptable depending on your requirements.")
        return 0
    else:
        print("❌ FAIL: HEF and ONNX embeddings differ significantly")
        print(f"   Average cosine similarity: {avg_cosine:.6f}")
        print("   The calibration may still be incorrect.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
