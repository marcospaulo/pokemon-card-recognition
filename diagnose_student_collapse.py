#!/usr/bin/env python3
"""
Diagnose Student Model Collapse

Checks if the student model is outputting collapsed embeddings
(all inputs produce similar outputs).
"""

import sys
import numpy as np
from pathlib import Path
import cv2
import onnxruntime as ort

def preprocess_image(image_path: Path, input_size=(224, 224)):
    """Preprocess image for model"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    resized = cv2.resize(image, input_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1))
    input_tensor = np.expand_dims(chw, axis=0).astype(np.float32)

    return input_tensor

def main():
    print("=" * 70)
    print("Student Model Collapse Diagnostic")
    print("=" * 70)

    # Load model
    model_path = Path("models/onnx/pokemon_student_stage2_final.onnx")
    print(f"\n[1/3] Loading model...")
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"✅ Model loaded")

    # Get test images
    test_cards_dir = Path("data/test_cards")
    test_images = list(test_cards_dir.glob("*.png"))[:4]
    print(f"\n[2/3] Generating embeddings for {len(test_images)} test cards...")

    embeddings = []
    for img_path in test_images:
        input_tensor = preprocess_image(img_path)
        output = session.run([output_name], {input_name: input_tensor})
        embedding = output[0][0]
        embeddings.append(embedding)
        print(f"   {img_path.name}: shape={embedding.shape}, mean={embedding.mean():.4f}, std={embedding.std():.4f}")

    embeddings = np.array(embeddings)

    # Analyze embeddings
    print(f"\n[3/3] Analyzing embeddings...")
    print(f"\nEmbedding Statistics:")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Mean across all: {embeddings.mean():.6f}")
    print(f"   Std across all:  {embeddings.std():.6f}")

    # Check variance across different cards
    print(f"\nVariance Analysis:")
    per_dimension_variance = embeddings.var(axis=0)  # Variance across cards for each dim
    print(f"   Per-dimension variance (mean): {per_dimension_variance.mean():.8f}")
    print(f"   Per-dimension variance (max):  {per_dimension_variance.max():.8f}")
    print(f"   Per-dimension variance (min):  {per_dimension_variance.min():.8f}")

    # Compute pairwise similarities
    print(f"\nPairwise Cosine Similarities:")
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarity = np.dot(normalized_embeddings[i], normalized_embeddings[j])
            print(f"   Card {i+1} vs Card {j+1}: {similarity:.6f}")

    # Check for collapse
    print(f"\n{'='*70}")
    print("DIAGNOSIS:")
    print(f"{'='*70}")

    avg_variance = per_dimension_variance.mean()
    max_variance = per_dimension_variance.max()

    if avg_variance < 1e-6:
        print(f"❌ SEVERE MODEL COLLAPSE DETECTED")
        print(f"   Per-dimension variance: {avg_variance:.10f} (expected: >0.01)")
        print(f"   The model is outputting nearly identical embeddings for all inputs!")
        print(f"\n   Root causes:")
        print(f"   1. Model weights may have collapsed during distillation")
        print(f"   2. Training may have failed or diverged")
        print(f"   3. Model architecture issue")
        print(f"\n   This explains the ~0.998 similarity scores we saw earlier.")
        status = "COLLAPSED"
    elif avg_variance < 0.001:
        print(f"⚠️  LIKELY MODEL COLLAPSE")
        print(f"   Per-dimension variance: {avg_variance:.10f} (expected: >0.01)")
        print(f"   Embeddings have very low variance - model may be collapsed")
        status = "LIKELY_COLLAPSED"
    elif avg_variance < 0.01:
        print(f"⚠️  LOW EMBEDDING VARIANCE")
        print(f"   Per-dimension variance: {avg_variance:.10f} (expected: >0.1)")
        print(f"   Model may not be distinguishing well between different cards")
        status = "LOW_VARIANCE"
    else:
        print(f"✅ Model appears healthy")
        print(f"   Per-dimension variance: {avg_variance:.6f}")
        status = "HEALTHY"

    # Compare to reference embeddings
    print(f"\nComparing to Reference Database:")
    ref_embeddings = np.load('data/reference/embeddings.npy')
    ref_variance = ref_embeddings.var(axis=0).mean()
    print(f"   Reference embeddings variance: {ref_variance:.6f}")
    print(f"   Student embeddings variance:   {avg_variance:.10f}")
    print(f"   Ratio: {avg_variance / ref_variance:.10f}")

    if avg_variance / ref_variance < 0.001:
        print(f"\n   ⚠️  Student embeddings have {ref_variance / avg_variance:.0f}x LESS variance than reference!")
        print(f"       This confirms the model is collapsed.")

    print(f"\n{'='*70}")
    print(f"FINAL VERDICT: {status}")
    print(f"{'='*70}\n")

    return 0 if status == "HEALTHY" else 1

if __name__ == "__main__":
    sys.exit(main())
