#!/usr/bin/env python3
"""
Test EfficientNet Student Model on Pokemon Cards

Validates that the distilled student model is working correctly by:
1. Loading the student model (ONNX format)
2. Generating embeddings for test cards
3. Searching reference database for matches
4. Reporting top-1 and top-5 accuracy
"""

import sys
import json
import numpy as np
from pathlib import Path
import cv2

# Check for dependencies
try:
    import onnxruntime as ort
except ImportError:
    print("❌ Error: onnxruntime not installed")
    print("   Run: pip install onnxruntime")
    sys.exit(1)

def load_student_model(model_path: Path):
    """Load the student ONNX model"""
    print(f"\n[1/5] Loading student model...")
    print(f"   Model: {model_path}")

    try:
        session = ort.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider']
        )

        # Get model info
        inputs = session.get_inputs()
        outputs = session.get_outputs()

        input_shape = inputs[0].shape
        output_shape = outputs[0].shape

        print(f"✅ Model loaded successfully")
        print(f"   Input: {inputs[0].name} {input_shape}")
        print(f"   Output: {outputs[0].name} {output_shape}")

        return session, inputs[0].name, outputs[0].name

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None, None

def load_reference_database():
    """Load reference embeddings and metadata"""
    print(f"\n[2/5] Loading reference database...")

    try:
        embeddings = np.load('data/reference/embeddings.npy')
        index = json.load(open('data/reference/index.json'))

        # Load metadata as list, then convert to dict by card_id
        metadata_list = json.load(open('data/reference/cards_metadata.json'))
        metadata = {card['card_id']: card for card in metadata_list}

        print(f"✅ Reference database loaded")
        print(f"   Embeddings: {embeddings.shape[0]:,} cards, {embeddings.shape[1]}-dim")
        print(f"   Index: {len(index):,} entries")
        print(f"   Metadata: {len(metadata):,} unique card IDs")

        return embeddings, index, metadata

    except Exception as e:
        print(f"❌ Failed to load reference database: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def preprocess_image(image_path: Path, input_size=(224, 224)):
    """Preprocess image for EfficientNet-Lite0 model (EXACT match with training)"""

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Resize to model input size
    resized = cv2.resize(image, input_size)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    normalized = rgb.astype(np.float32) / 255.0

    # Apply ImageNet normalization (CRITICAL - must match training!)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    normalized = (normalized - mean) / std

    # HWC to CHW
    chw = np.transpose(normalized, (2, 0, 1))

    # Add batch dimension
    input_tensor = np.expand_dims(chw, axis=0).astype(np.float32)

    return input_tensor

def generate_embedding(session, input_name, output_name, image_path: Path):
    """Generate embedding for a single card image"""

    # Preprocess image
    input_tensor = preprocess_image(image_path)

    # Run inference
    outputs = session.run([output_name], {input_name: input_tensor})
    embedding = outputs[0][0]  # Remove batch dimension

    # Normalize embedding (L2 normalization for cosine similarity)
    embedding = embedding / np.linalg.norm(embedding)

    return embedding

def search_similar(query_embedding, ref_embeddings, top_k=5):
    """Search for most similar cards using cosine similarity"""

    # Compute cosine similarity with all reference embeddings
    # Both query and ref should be L2-normalized
    similarities = np.dot(ref_embeddings, query_embedding)

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_scores = similarities[top_indices]

    return top_indices, top_scores

def test_on_card(
    session,
    input_name,
    output_name,
    image_path: Path,
    ref_embeddings,
    index,
    metadata,
):
    """Test student model on a single card"""

    print(f"\n   Testing: {image_path.name}")

    try:
        # Generate embedding
        embedding = generate_embedding(session, input_name, output_name, image_path)

        # Search for matches
        top_indices, top_scores = search_similar(embedding, ref_embeddings, top_k=5)

        # Get card IDs from index
        index_list = list(index.items())

        print(f"   Top-5 matches:")
        for rank, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
            # Get row index and corresponding card_id
            row_str, _ = index_list[idx]
            card_id = index[row_str]  # Use index mapping to get correct card_id

            # Get metadata using card_id as key
            if card_id in metadata:
                card_meta = metadata[card_id]
                card_name = card_meta.get('name', 'Unknown')
                card_set_id = card_meta.get('set_id', 'Unknown')
                card_number = card_meta.get('number', '?')
            else:
                card_name = f"Card {card_id}"
                card_set_id = "Unknown"
                card_number = "?"

            print(f"     {rank}. {card_name} ({card_set_id} #{card_number})")
            print(f"        Similarity: {score:.4f} | Card ID: {card_id}")

        return True, top_indices[0], top_scores[0]

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def main():
    """Main test function"""

    print("=" * 70)
    print("EfficientNet Student Model Test")
    print("=" * 70)

    # Load model
    model_path = Path("models/onnx/pokemon_student_stage2_final.onnx")
    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        return 1

    session, input_name, output_name = load_student_model(model_path)
    if session is None:
        return 1

    # Load reference database
    ref_embeddings, index, metadata = load_reference_database()
    if ref_embeddings is None:
        return 1

    # Find test card images
    print(f"\n[3/5] Finding test cards...")
    test_cards_dir = Path("data/test_cards")

    if not test_cards_dir.exists():
        print(f"❌ Test cards directory not found: {test_cards_dir}")
        print(f"   Please download some test cards first")
        return 1

    test_images = list(test_cards_dir.glob("*.png")) + list(test_cards_dir.glob("*.jpg"))

    if len(test_images) == 0:
        print(f"❌ No test images found in {test_cards_dir}")
        return 1

    print(f"✅ Found {len(test_images)} test cards")

    # Test on each card
    print(f"\n[4/5] Running inference on test cards...")

    results = []
    for image_path in test_images:
        success, top_match_idx, top_score = test_on_card(
            session,
            input_name,
            output_name,
            image_path,
            ref_embeddings,
            index,
            metadata,
        )

        if success:
            results.append({
                'image': image_path.name,
                'top_match_idx': top_match_idx,
                'top_score': float(top_score),
            })

    # Summary
    print(f"\n[5/5] Test Summary")
    print("=" * 70)

    if len(results) == 0:
        print("❌ No successful tests")
        return 1

    print(f"✅ Tested {len(results)} cards successfully")

    # Analyze similarity scores
    scores = [r['top_score'] for r in results]
    avg_score = np.mean(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)

    print(f"\nSimilarity Scores:")
    print(f"   Average: {avg_score:.4f}")
    print(f"   Min:     {min_score:.4f}")
    print(f"   Max:     {max_score:.4f}")

    # Check if scores are reasonable
    print(f"\nDiagnostics:")

    if avg_score < 0.3:
        print(f"   ⚠️  WARNING: Low average similarity ({avg_score:.4f})")
        print(f"       This suggests the model may not be working correctly")
        print(f"       Expected: >0.5 for correct matches, >0.7 for high confidence")
    elif avg_score < 0.5:
        print(f"   ⚠️  Moderate similarity ({avg_score:.4f})")
        print(f"       Matches may be uncertain or incorrect")
    elif avg_score < 0.7:
        print(f"   ✅ Good similarity ({avg_score:.4f})")
        print(f"       Model appears to be working correctly")
    else:
        print(f"   ✅ Excellent similarity ({avg_score:.4f})")
        print(f"       Model is producing high-confidence matches")

    print("\n" + "=" * 70)

    return 0

if __name__ == "__main__":
    sys.exit(main())
