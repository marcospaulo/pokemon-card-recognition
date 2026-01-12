#!/usr/bin/env python3
"""
Pokemon Card Recognition - Real-time Inference on Raspberry Pi 5
Uses IMX500 camera (YOLO detection) + Hailo 8L (embedding) + uSearch (matching)
"""

import argparse
import time
from pathlib import Path
import numpy as np
import json
from PIL import Image
import cv2

try:
    from usearch.index import Index
except ImportError:
    print("⚠️  usearch not installed. Installing...")
    import subprocess
    subprocess.check_call(["pip3", "install", "usearch"])
    from usearch.index import Index

try:
    from hailo_platform import (
        HEF, VDevice, HailoStreamInterface,
        InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
    )
    HAILO_AVAILABLE = True
except ImportError:
    print("⚠️  HailoPlatform not available - will simulate inference")
    HAILO_AVAILABLE = False


class CardRecognizer:
    """Real-time Pokemon card recognition using Hailo NPU"""
    
    def __init__(self, model_path, reference_db_path, confidence_threshold=0.85):
        self.model_path = Path(model_path)
        self.reference_db_path = Path(reference_db_path)
        self.confidence_threshold = confidence_threshold
        
        print("🚀 Initializing Pokemon Card Recognizer...")
        
        # Load reference database
        print("📚 Loading reference database...")
        self._load_reference_database()
        
        # Initialize Hailo
        if HAILO_AVAILABLE:
            print("🧠 Initializing Hailo 8L NPU...")
            self._init_hailo()
        else:
            print("⚠️  Running in simulation mode (Hailo not available)")
        
        print("✅ Recognizer ready!")
    
    def _load_reference_database(self):
        """Load embeddings and uSearch index"""
        # Load embeddings
        embeddings_path = self.reference_db_path / "embeddings.npy"
        self.embeddings = np.load(embeddings_path)
        print(f"   ✅ Loaded {self.embeddings.shape[0]} embeddings ({self.embeddings.shape[1]}D)")
        
        # Load uSearch index
        index_path = self.reference_db_path / "usearch.index"
        self.index = Index.restore(str(index_path))
        print(f"   ✅ Loaded uSearch index")
        
        # Load row -> card_id mapping
        with open(self.reference_db_path / "index.json") as f:
            self.row_to_card = json.load(f)
        
        # Load card metadata
        with open(self.reference_db_path / "metadata.json") as f:
            self.metadata = json.load(f)
        
        print(f"   ✅ Loaded metadata for {len(self.metadata)} cards")
    
    def _init_hailo(self):
        """Initialize Hailo device and load model"""
        try:
            # Get available devices
            devices = VDevice.scan()
            if not devices:
                raise RuntimeError("No Hailo devices found!")
            
            # Use first device
            self.device = VDevice(device_id=None)
            print(f"   ✅ Connected to Hailo device")
            
            # Load HEF
            self.hef = HEF(str(self.model_path))
            print(f"   ✅ Loaded HEF: {self.model_path.name}")
            
            # Configure network group
            self.network_group = self.device.configure(self.hef)[0]
            self.network_group_params = self.network_group.create_params()
            
            # Get input/output info
            self.input_vstreams_params = InputVStreamParams.make_from_network_group(
                self.network_group, quantized=True, format_type=FormatType.UINT8
            )
            self.output_vstreams_params = OutputVStreamParams.make_from_network_group(
                self.network_group, quantized=False, format_type=FormatType.FLOAT32
            )
            
            print(f"   ✅ Network configured")
            
        except Exception as e:
            print(f"   ❌ Hailo initialization failed: {e}")
            raise
    
    def preprocess_image(self, image):
        """Preprocess image for EfficientNet-Lite0 (224x224)"""
        # Resize to 224x224
        img = cv2.resize(image, (224, 224))
        
        # Convert to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # Convert to UINT8 for Hailo (quantized input)
        # Scale to [0, 255] range
        img = ((img + 2.5) / 5.0 * 255).clip(0, 255).astype(np.uint8)
        
        return img
    
    def extract_embedding(self, image):
        """Extract 768-dim embedding using Hailo"""
        if not HAILO_AVAILABLE:
            # Simulation mode - return random embedding
            return np.random.randn(768).astype(np.float32)
        
        # Preprocess
        processed = self.preprocess_image(image)
        
        # Run inference
        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            # Send image
            input_data = {self.input_vstreams_params[0].name: processed[np.newaxis, ...]}
            
            # Get embedding
            output = infer_pipeline.infer(input_data)
            embedding = list(output.values())[0][0]  # Get first output, first batch
            
            # L2 normalize
            embedding = embedding / np.linalg.norm(embedding)
            
        return embedding
    
    def search_card(self, embedding, k=5):
        """Search for nearest neighbors in reference database"""
        # Search using uSearch
        matches = self.index.search(embedding, k)
        
        results = []
        for i, (row_idx, distance) in enumerate(zip(matches.keys, matches.distances)):
            # Get card ID from row index
            card_id = self.row_to_card[str(row_idx)]
            
            # Get metadata
            card_data = self.metadata.get(card_id, {})
            
            # Calculate confidence (cosine similarity)
            # distance is L2, convert to cosine similarity
            confidence = 1.0 - (distance / 2.0)
            
            results.append({
                'rank': i + 1,
                'card_id': card_id,
                'name': card_data.get('name', 'Unknown'),
                'set': card_data.get('set', ''),
                'confidence': float(confidence),
                'distance': float(distance)
            })
        
        return results
    
    def recognize(self, image, return_top_k=5):
        """Full recognition pipeline"""
        start_time = time.time()
        
        # Extract embedding
        embedding_start = time.time()
        embedding = self.extract_embedding(image)
        embedding_time = time.time() - embedding_start
        
        # Search database
        search_start = time.time()
        results = self.search_card(embedding, k=return_top_k)
        search_time = time.time() - search_start
        
        total_time = time.time() - start_time
        
        # Add timing info
        timing = {
            'embedding_ms': embedding_time * 1000,
            'search_ms': search_time * 1000,
            'total_ms': total_time * 1000
        }
        
        return {
            'results': results,
            'timing': timing,
            'top_match': results[0] if results else None
        }


def main():
    parser = argparse.ArgumentParser(description='Pokemon Card Recognition')
    parser.add_argument('--image', type=str, help='Path to card image')
    parser.add_argument('--model', type=str, 
                       default='models/embedding/pokemon_student_efficientnet_lite0_stage2.hef',
                       help='Path to Hailo HEF model')
    parser.add_argument('--reference', type=str, 
                       default='data/reference',
                       help='Path to reference database directory')
    parser.add_argument('--camera', action='store_true', 
                       help='Use camera for real-time recognition')
    parser.add_argument('--confidence', type=float, default=0.85,
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Initialize recognizer
    recognizer = CardRecognizer(
        model_path=args.model,
        reference_db_path=args.reference,
        confidence_threshold=args.confidence
    )
    
    if args.image:
        # Single image mode
        print(f"\n📸 Processing image: {args.image}")
        image = cv2.imread(args.image)
        if image is None:
            print(f"❌ Failed to load image: {args.image}")
            return
        
        result = recognizer.recognize(image)
        
        print("\n" + "="*60)
        print("🎯 Recognition Results")
        print("="*60)
        
        for r in result['results']:
            print(f"\n#{r['rank']} - {r['name']} ({r['set']})")
            print(f"   Confidence: {r['confidence']:.2%}")
            print(f"   Distance: {r['distance']:.4f}")
        
        print("\n" + "="*60)
        print("⏱️  Timing")
        print("="*60)
        print(f"Embedding extraction: {result['timing']['embedding_ms']:.1f} ms")
        print(f"Database search:      {result['timing']['search_ms']:.1f} ms")
        print(f"Total:                {result['timing']['total_ms']:.1f} ms")
        
        if result['top_match']['confidence'] >= args.confidence:
            print(f"\n✅ Match Found: {result['top_match']['name']}")
        else:
            print(f"\n⚠️  Low confidence: {result['top_match']['confidence']:.2%}")
    
    elif args.camera:
        # Real-time camera mode (TODO: integrate with IMX500)
        print("\n📹 Camera mode not yet implemented")
        print("   TODO: Integrate with IMX500 camera + YOLO detection")
    
    else:
        print("❌ Please specify --image or --camera")
        parser.print_help()


if __name__ == "__main__":
    main()
