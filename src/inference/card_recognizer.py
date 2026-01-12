"""
Card Recognition using ONNX embedding model and FAISS similarity search.

This module provides real-time Pokemon card identification by:
1. Computing embeddings using a trained MobileNetV3 model (ONNX)
2. Finding nearest neighbors in a FAISS index of known cards
3. Returning card metadata (name, set, number, etc.)

Usage:
    recognizer = CardRecognizer()
    results = recognizer.recognize(card_image)
    print(results[0]['name'], results[0]['similarity'])
"""

import json
from pathlib import Path
from typing import Optional
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import faiss
except ImportError:
    faiss = None


class CardRecognizer:
    """
    Pokemon card recognition using embedding similarity search.

    Attributes:
        model_path: Path to ONNX embedding model
        index_path: Path to FAISS index
        metadata_path: Path to card metadata JSON
        top_k: Number of similar cards to return
    """

    # Map set codes to full set names
    SET_CODE_TO_NAME = {
        # Base era
        'base1': 'Base Set', 'base2': 'Jungle', 'base3': 'Fossil',
        'base4': 'Team Rocket', 'base5': 'Gym Heroes', 'base6': 'Gym Challenge',
        'basep': 'Base Promo', 'gym1': 'Gym Heroes', 'gym2': 'Gym Challenge',
        'lc': 'Legendary Collection', 'si1': 'Southern Islands',
        # Neo era
        'neo1': 'Neo Genesis', 'neo2': 'Neo Discovery', 'neo3': 'Neo Revelation',
        'neo4': 'Neo Destiny',
        # e-Card era
        'ecard1': 'Expedition', 'ecard2': 'Aquapolis', 'ecard3': 'Skyridge',
        # EX era
        'ex1': 'Ruby & Sapphire', 'ex2': 'Sandstorm', 'ex3': 'Dragon',
        'ex4': 'Team Magma vs Aqua', 'ex5': 'Hidden Legends', 'ex6': 'FireRed LeafGreen',
        'ex7': 'Team Rocket Returns', 'ex8': 'Deoxys', 'ex9': 'Emerald',
        'ex10': 'Unseen Forces', 'ex11': 'Delta Species', 'ex12': 'Legend Maker',
        'ex13': 'Holon Phantoms', 'ex14': 'Crystal Guardians', 'ex15': 'Dragon Frontiers',
        'ex16': 'Power Keepers',
        # Diamond & Pearl era
        'dp1': 'Diamond & Pearl', 'dp2': 'Mysterious Treasures', 'dp3': 'Secret Wonders',
        'dp4': 'Great Encounters', 'dp5': 'Majestic Dawn', 'dp6': 'Legends Awakened',
        'dp7': 'Stormfront',
        # Platinum era
        'pl1': 'Platinum', 'pl2': 'Rising Rivals', 'pl3': 'Supreme Victors',
        'pl4': 'Arceus',
        # HeartGold SoulSilver era
        'hgss1': 'HeartGold SoulSilver', 'hgss2': 'Unleashed', 'hgss3': 'Undaunted',
        'hgss4': 'Triumphant',
        # Black & White era
        'bw1': 'Black & White', 'bw2': 'Emerging Powers', 'bw3': 'Noble Victories',
        'bw4': 'Next Destinies', 'bw5': 'Dark Explorers', 'bw6': 'Dragons Exalted',
        'bw7': 'Boundaries Crossed', 'bw8': 'Plasma Storm', 'bw9': 'Plasma Freeze',
        'bw10': 'Plasma Blast', 'bw11': 'Legendary Treasures',
        # XY era
        'xy1': 'XY', 'xy2': 'Flashfire', 'xy3': 'Furious Fists', 'xy4': 'Phantom Forces',
        'xy5': 'Primal Clash', 'xy6': 'Roaring Skies', 'xy7': 'Ancient Origins',
        'xy8': 'BREAKthrough', 'xy9': 'BREAKpoint', 'xy10': 'Fates Collide',
        'xy11': 'Steam Siege', 'xy12': 'Evolutions',
        # Sun & Moon era
        'sm1': 'Sun & Moon', 'sm2': 'Guardians Rising', 'sm3': 'Burning Shadows',
        'sm4': 'Crimson Invasion', 'sm5': 'Ultra Prism', 'sm6': 'Forbidden Light',
        'sm7': 'Celestial Storm', 'sm8': 'Lost Thunder', 'sm9': 'Team Up',
        'sm10': 'Unbroken Bonds', 'sm11': 'Unified Minds', 'sm12': 'Cosmic Eclipse',
        # Sword & Shield era
        'swsh1': 'Sword & Shield', 'swsh2': 'Rebel Clash', 'swsh3': 'Darkness Ablaze',
        'swsh4': 'Vivid Voltage', 'swsh5': 'Battle Styles', 'swsh6': 'Chilling Reign',
        'swsh7': 'Evolving Skies', 'swsh8': 'Fusion Strike', 'swsh9': 'Brilliant Stars',
        'swsh10': 'Astral Radiance', 'swsh11': 'Lost Origin', 'swsh12': 'Silver Tempest',
        'swsh12pt5': 'Crown Zenith GG', 'swsh13': 'Crown Zenith',
        # SWSH Promos and special sets
        'swshp': 'SWSH Promo', 'cel25': 'Celebrations', 'cel25c': 'Celebrations Classic',
        'pgo': 'Pokemon GO', 'swsh45': 'Shining Fates', 'swsh45sv': 'Shining Fates SV',
        # Scarlet & Violet era
        'sv1': 'Scarlet & Violet', 'sv2': 'Paldea Evolved', 'sv3': 'Obsidian Flames',
        'sv4': '151', 'sv5': 'Temporal Forces', 'sv6': 'Twilight Masquerade',
        'sv7': 'Shrouded Fable', 'sv8': 'Surging Sparks', 'sv9': 'Journey Together',
        'sv10': 'Destined Rivals', 'sv3pt5': 'Paldean Fates', 'sv4pt5': 'Paldean Fates',
        'sv5pt5': 'Prismatic Evolutions', 'sv6pt5': 'Shrouded Fable',
        'sv8pt5': 'Prismatic Evolutions', 'svp': 'SV Promo', 'sve': 'SV Energy',
        # Revisited sets (rsv prefix)
        'rsv10pt5': 'Destined Rivals',
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        top_k: int = 5
    ):
        """
        Initialize card recognizer.
        
        Args:
            model_path: Path to pokemon_embedding.onnx
            index_path: Path to pokemon_faiss.index
            metadata_path: Path to pokemon_index_metadata.json
            top_k: Number of top matches to return
        """
        if ort is None:
            raise ImportError("onnxruntime not installed. Run: pip install onnxruntime")
        if faiss is None:
            raise ImportError("faiss not installed. Run: pip install faiss-cpu")
        
        # Default paths (relative to this file's location)
        base_dir = Path(__file__).parent.parent.parent / "models"
        
        self.model_path = Path(model_path) if model_path else base_dir / "pokemon_embedding.onnx"
        self.index_path = Path(index_path) if index_path else base_dir / "pokemon_faiss.index"
        self.metadata_path = Path(metadata_path) if metadata_path else base_dir / "pokemon_index_metadata.json"
        self.top_k = top_k
        
        # Load model, index, and metadata
        self._load_model()
        self._load_index()
        self._load_metadata()
        
        print(f"CardRecognizer initialized:")
        print(f"  Model: {self.model_path}")
        print(f"  Index: {self.index.ntotal} cards")
        print(f"  Top-K: {self.top_k}")
    
    def _load_model(self):
        """Load ONNX embedding model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Use CPU provider (no GPU on Pi, but could add CoreML for Mac)
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [batch, 3, 384, 384]
    
    def _load_index(self):
        """Load FAISS index."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found: {self.index_path}")
        
        self.index = faiss.read_index(str(self.index_path))
    
    def _load_metadata(self):
        """Load card metadata."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")
        
        with open(self.metadata_path) as f:
            data = json.load(f)
        
        self.class_ids = data['class_ids']  # index position → class_idx
        self.idx_to_class = {int(k): v for k, v in data['idx_to_class'].items()}
        self.card_metadata = data.get('card_metadata', {})
        self.num_classes = data['num_classes']
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.

        Args:
            image: RGB image as numpy array (H, W, 3), uint8 or float

        Returns:
            Preprocessed tensor (1, 3, 384, 384), float32
        """
        import cv2

        # Pad to square first to preserve aspect ratio, then resize to 384x384
        h, w = image.shape[:2]

        if h != w:
            # Determine the larger dimension
            max_dim = max(h, w)

            # Create square canvas with gray padding (neutral color)
            if image.dtype == np.uint8:
                # Use gray (128) for uint8 images
                square = np.full((max_dim, max_dim, 3), 128, dtype=np.uint8)
            else:
                # Use 0.5 for float images
                square = np.full((max_dim, max_dim, 3), 0.5, dtype=image.dtype)

            # Center the image in the square canvas
            y_offset = (max_dim - h) // 2
            x_offset = (max_dim - w) // 2
            square[y_offset:y_offset + h, x_offset:x_offset + w] = image
            image = square

        # Now resize the square image to 384x384
        if image.shape[0] != 384:
            image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_LINEAR)

        # Convert to float and normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        # HWC → CHW and add batch dimension
        image = np.transpose(image, (2, 0, 1))  # (3, 384, 384)
        image = np.expand_dims(image, 0)  # (1, 3, 384, 384)

        return image.astype(np.float32)
    
    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Compute embedding for an image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
        
        Returns:
            L2-normalized embedding (512,)
        """
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Run inference
        embedding = self.session.run(None, {self.input_name: input_tensor})[0]
        
        return embedding[0]  # Remove batch dimension
    
    def recognize(self, image: np.ndarray) -> list[dict]:
        """
        Recognize a Pokemon card from an image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
        
        Returns:
            List of top-k matches, each containing:
                - class_name: Full class identifier
                - card_name: Pokemon/card name
                - set_name: Set identifier
                - card_number: Card number in set
                - similarity: Cosine similarity score (0-1)
        """
        # Get embedding
        embedding = self.get_embedding(image)
        
        # Search FAISS index (inner product = cosine similarity for normalized vectors)
        embedding = embedding.reshape(1, -1).astype(np.float32)
        similarities, indices = self.index.search(embedding, self.top_k)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty results
                continue
            
            class_idx = self.class_ids[idx]
            class_name = self.idx_to_class[class_idx]

            # Parse class name - two formats exist:
            # Format 1: "Abra-en_base_base1_43" -> name="Abra"
            # Format 2: "ex1-100_Magmar_ex" -> name="Magmar ex"
            parts = class_name.split('-')
            first_part = parts[0] if parts else class_name

            # Detect format by checking if first part looks like a set code
            # Set codes start with: ex, dp, bw, xy, sm, swsh, sv, neo, hgss, etc.
            set_prefixes = ('ex', 'dp', 'bw', 'xy', 'sm', 'swsh', 'sv', 'pop', 'np', 'ru', 'neo', 'hgss', 'pl', 'sma', 'cel')
            is_set_first_format = any(first_part.lower().startswith(p) for p in set_prefixes)

            set_name = ''
            card_number = ''

            if is_set_first_format and len(parts) > 1:
                # Format 2: "ex1-100_Magmar_ex"
                set_name = first_part  # e.g., "ex1"
                rest = parts[1]  # e.g., "100_Magmar_ex"
                sub_parts = rest.split('_')
                if len(sub_parts) >= 2:
                    card_number = sub_parts[0]  # e.g., "100"
                    # Name is everything after number, joined with space
                    card_name = ' '.join(sub_parts[1:])  # e.g., "Magmar ex"
                else:
                    card_name = rest
            else:
                # Format 1: "Abra-en_base_base1_43"
                card_name = first_part  # e.g., "Abra"
                if len(parts) > 1:
                    sub_parts = parts[1].split('_')
                    if len(sub_parts) >= 3:
                        set_name = sub_parts[2]  # e.g., "base1"
                    if len(sub_parts) >= 4:
                        card_number = sub_parts[3]  # e.g., "43"

            # Try to get detailed metadata (may not match)
            metadata = self.card_metadata.get(class_name, {})

            # Convert set code to full name
            full_set_name = self.SET_CODE_TO_NAME.get(set_name.lower(), set_name)

            results.append({
                'class_name': class_name,
                'card_name': metadata.get('name', card_name),
                'set_name': metadata.get('set', full_set_name),
                'card_number': metadata.get('number', card_number),
                'similarity': float(sim),
                'metadata': metadata
            })
        
        return results
    
    def recognize_batch(self, images: list[np.ndarray]) -> list[list[dict]]:
        """
        Recognize multiple cards in a batch.
        
        Args:
            images: List of RGB images
        
        Returns:
            List of results for each image
        """
        return [self.recognize(img) for img in images]


# Quick test
if __name__ == "__main__":
    import sys
    
    print("Testing CardRecognizer...")
    
    try:
        recognizer = CardRecognizer()
        
        # Create a dummy test image
        dummy_image = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
        
        print("\nRunning inference on random image...")
        results = recognizer.recognize(dummy_image)
        
        print(f"\nTop {len(results)} matches:")
        for i, r in enumerate(results):
            print(f"  {i+1}. {r['card_name']} (similarity: {r['similarity']:.4f})")
        
        print("\n✓ CardRecognizer working correctly!")
        
    except FileNotFoundError as e:
        print(f"\n⚠ Model files not found: {e}")
        print("  Make sure to copy pokemon_embedding.onnx, pokemon_faiss.index,")
        print("  and pokemon_index_metadata.json to the models/ directory.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
