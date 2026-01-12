# Phase 3: Reference Database
## PRD_04_DATABASE.md

**Parent Document:** PRD_01_OVERVIEW.md
**Phase:** 3 of 5
**Target:** Local JSON → SQLite → (Future: pgvector)

---

## Objective

Build a local reference database that:
1. Stores pre-computed embeddings for all 17,592 cards
2. Supports multiple embedding variants for different deployment targets (Hailo, iOS, Android)
3. Enables fast nearest-neighbor search via uSearch vector index
4. Maps card IDs to metadata (name, set, number, etc.)
5. Works entirely offline

---

## Multi-Student Embedding Architecture

### Knowledge Distillation Overview

This system uses **knowledge distillation** to train lightweight student models that replicate the embedding behavior of a large teacher model. Each student is optimized for a specific deployment target while producing embeddings in the same 768-dimensional space.

| Model Role | Architecture | Embedding Dim | Deployment Target |
|------------|--------------|---------------|-------------------|
| **Teacher** | DINOv3-ViT-L/16 | 768 | Training reference (not deployed) |
| **Hailo Student** | ConvNeXt-Tiny | 768 | Raspberry Pi 5 + Hailo-8L |
| **iOS Student** | ConvNeXt-Base | 768 | iPhone / iPad (CoreML) |
| **Android Student** | ViT-Small | 768 | Android devices (ONNX) |

### Why Multiple Embedding Sets?

Although all models produce 768-dimensional embeddings, **each student model has its own learned embedding space** due to:

1. **Architecture differences** - ConvNeXt-Tiny, ConvNeXt-Base, and ViT-Small have different inductive biases and feature extraction patterns
2. **Quantization effects** - INT8 (Hailo) vs FP16 (iOS/CoreML) vs INT8 (Android/ONNX) introduce different numerical behaviors
3. **Training dynamics** - Each student converges to a slightly different local minimum during distillation
4. **Hardware-specific optimizations** - Model optimizations for each accelerator affect learned representations

**Critical Requirement:** The reference database MUST be generated using the SAME model that will perform inference. Using mismatched embeddings (e.g., teacher embeddings for ConvNeXt-Tiny inference) will degrade matching accuracy significantly.

### Per-Model Similarity Threshold Calibration

Because each student model produces embeddings with different characteristics, **similarity thresholds may need per-model calibration**:

```python
# Example: Calibrated thresholds per model (values determined empirically)
DISTANCE_THRESHOLDS = {
    'teacher': 0.45,    # DINOv3-ViT-L/16 baseline
    'hailo': 0.50,      # ConvNeXt-Tiny may have slightly different distribution
    'ios': 0.48,        # ConvNeXt-Base calibrated threshold
    'android': 0.52,    # ViT-Small calibrated threshold
}
```

During validation, run threshold calibration for each model by:
1. Computing distances between all card embeddings and their augmented versions
2. Computing distances between different cards
3. Finding the optimal threshold that maximizes F1 score for match/no-match classification

```
┌─────────────────────────────────────────────────────────────────┐
│                    EMBEDDING GENERATION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐                                       │
│  │   Teacher Model     │ ─────► teacher_embeddings.npy          │
│  │   DINOv3-ViT-L/16   │        (Validation reference only)     │
│  │   768-dim output    │                                        │
│  └─────────────────────┘                                       │
│           │                                                     │
│           │ Knowledge Distillation                              │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Student Models                         │   │
│  │                                                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │   │
│  │  │   Hailo     │  │    iOS      │  │  Android    │      │   │
│  │  │ ConvNeXt-   │  │ ConvNeXt-   │  │  ViT-Small  │      │   │
│  │  │   Tiny      │  │   Base      │  │             │      │   │
│  │  │  768-dim    │  │  768-dim    │  │  768-dim    │      │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘      │   │
│  │         │                │                │              │   │
│  │         ▼                ▼                ▼              │   │
│  │    hailo_           ios_            android_             │   │
│  │    embeddings       embeddings      embeddings           │   │
│  │    .npy             .npy            .npy                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Deployment Target Matching

| Deployment Target | Student Architecture | Model Format | Embedding File | Use Case |
|-------------------|---------------------|--------------|----------------|----------|
| Raspberry Pi 5 + Hailo-8L | ConvNeXt-Tiny | `.hef` (INT8) | `hailo_embeddings.npy` | Production edge device |
| iPhone / iPad | ConvNeXt-Base | CoreML (FP16) | `ios_embeddings.npy` | iOS mobile app |
| Android devices | ViT-Small | ONNX (INT8) | `android_embeddings.npy` | Android mobile app |
| Validation / Testing | DINOv3-ViT-L/16 | PyTorch (FP32) | `teacher_embeddings.npy` | Accuracy benchmarking |

---

## Architecture

### Local-First Approach

```
┌─────────────────────────────────────────────────────────────────┐
│                    REFERENCE DATABASE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               Embedding Files (per target)               │   │
│  │  ┌────────────────┐ ┌────────────────┐ ┌──────────────┐ │   │
│  │  │ teacher_       │ │ hailo_         │ │ ios_         │ │   │
│  │  │ embeddings.npy │ │ embeddings.npy │ │ embeddings   │ │   │
│  │  │ (17592, 768)   │ │ (17592, 768)   │ │ .npy         │ │   │
│  │  │ ~50MB          │ │ ~50MB          │ │ (17592, 768) │ │   │
│  │  └────────────────┘ └────────────────┘ └──────────────┘ │   │
│  │  ┌──────────────────┐                                    │   │
│  │  │ android_         │                                    │   │
│  │  │ embeddings.npy   │                                    │   │
│  │  │ (17592, 768)     │                                    │   │
│  │  └──────────────────┘                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────┐     ┌─────────────────┐                   │
│  │  usearch.index  │     │  card_metadata  │                   │
│  │                 │     │  .json          │                   │
│  │  Vector search  │     │                 │                   │
│  │  index (HNSW)   │     │  {              │                   │
│  │                 │     │    "card_001":  │                   │
│  │  Per-target:    │     │    {            │                   │
│  │  - hailo.usearch│     │      "name":    │                   │
│  │  - ios.usearch  │     │      "set":     │                   │
│  │  - android...   │     │      "number":  │                   │
│  │                 │     │    },           │                   │
│  └─────────────────┘     └─────────────────┘                   │
│           │                       │                             │
│           └───────────┬───────────┘                             │
│                       ▼                                         │
│           ┌─────────────────────┐                               │
│           │  card_index.json    │                               │
│           │                     │                               │
│           │  Maps row index     │                               │
│           │  in .npy to         │                               │
│           │  card_id in .json   │                               │
│           │                     │                               │
│           │  ["card_001",       │                               │
│           │   "card_002",       │                               │
│           │   ...]              │                               │
│           └─────────────────────┘                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Storage Strategy

**Phase 3a: JSON + NumPy + uSearch (Current)**
- Embeddings: Multiple NumPy `.npy` files (one per deployment target)
- Vector Index: uSearch `.usearch` files (HNSW index for fast ANN search)
- Metadata: JSON file
- Index: JSON array mapping row → card_id
- Search: uSearch approximate nearest neighbor (sub-millisecond)

**Phase 3b: SQLite (Structured, Still Local)**
- Single `.db` file
- Better querying for metadata
- uSearch index for vector similarity

**Future: pgvector (Production Scale)**
- PostgreSQL with vector extension
- HNSW index for fast approximate search
- Cloud-hosted for multi-device access

---

## Data Schema

### Card Metadata (JSON)

```json
{
  "cards": {
    "sv10-001": {
      "name": "Pikachu ex",
      "set_code": "sv10",
      "set_name": "Destined Rivals",
      "number": "001",
      "rarity": "Double Rare",
      "type": "Lightning",
      "hp": "200",
      "artist": "Mitsuhiro Arita",
      "release_date": "2025-01-17",
      "tcgplayer_id": "123456",
      "image_url": "https://...",
      "local_image_path": "images/sv10/001.jpg"
    },
    "sv10-002": {
      "name": "Raichu ex",
      ...
    }
  },
  "metadata": {
    "total_cards": 17592,
    "embedding_dim": 768,
    "teacher_model_version": "dinov3_vit_l16",
    "created_at": "2025-01-15T10:30:00Z",
    "sets": ["sv10", "sv9", "mega1", "mega2", ...],
    "embedding_variants": {
      "teacher": {
        "file": "teacher_embeddings.npy",
        "index_file": "teacher.usearch",
        "architecture": "DINOv3-ViT-L/16",
        "embedding_dim": 768,
        "precision": "fp32",
        "purpose": "validation_reference",
        "distance_threshold": 0.45
      },
      "hailo": {
        "file": "hailo_embeddings.npy",
        "index_file": "hailo.usearch",
        "architecture": "ConvNeXt-Tiny",
        "embedding_dim": 768,
        "precision": "int8",
        "purpose": "hailo8l_deployment",
        "distance_threshold": 0.50
      },
      "ios": {
        "file": "ios_embeddings.npy",
        "index_file": "ios.usearch",
        "architecture": "ConvNeXt-Base",
        "embedding_dim": 768,
        "precision": "fp16",
        "purpose": "ios_deployment",
        "distance_threshold": 0.48
      },
      "android": {
        "file": "android_embeddings.npy",
        "index_file": "android.usearch",
        "architecture": "ViT-Small",
        "embedding_dim": 768,
        "precision": "int8",
        "purpose": "android_deployment",
        "distance_threshold": 0.52
      }
    }
  }
}
```

### Embedding Storage (NumPy)

```python
# Each target has its own embedding file generated by the corresponding model:
#
# | Target   | Architecture     | File                      |
# |----------|------------------|---------------------------|
# | teacher  | DINOv3-ViT-L/16  | teacher_embeddings.npy    |
# | hailo    | ConvNeXt-Tiny    | hailo_embeddings.npy      |
# | ios      | ConvNeXt-Base    | ios_embeddings.npy        |
# | android  | ViT-Small        | android_embeddings.npy    |
#
# All files have the same shape: (17592, 768)
# Dtype: float32
# Size per file: 17592 * 768 * 4 bytes = ~54 MB
#
# IMPORTANT: Each model produces embeddings in its own learned space.
# Always use embeddings that match your inference model!

# Load the appropriate embeddings for your deployment target
target = 'hailo'  # or 'ios', 'android', 'teacher'
embeddings = np.load(f'{target}_embeddings.npy')  # Load into memory
```

### Index Mapping (JSON)

```json
{
  "index_to_card_id": [
    "sv10-001",
    "sv10-002",
    "sv10-003",
    ...
  ],
  "card_id_to_index": {
    "sv10-001": 0,
    "sv10-002": 1,
    "sv10-003": 2,
    ...
  }
}
```

---

## Implementation

### Database Class

```python
# reference_database.py

import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EmbeddingTarget(Enum):
    """Deployment target for embeddings"""
    TEACHER = "teacher"    # Full accuracy reference (validation only)
    HAILO = "hailo"        # Hailo-8L deployment
    IOS = "ios"            # iOS/CoreML deployment
    ANDROID = "android"    # Android/ONNX deployment


@dataclass
class CardMatch:
    """Result of a card search"""
    card_id: str
    name: str
    set_name: str
    number: str
    distance: float
    confidence: float  # 1 - normalized_distance

    def to_dict(self) -> dict:
        return {
            'card_id': self.card_id,
            'name': self.name,
            'set_name': self.set_name,
            'number': self.number,
            'distance': self.distance,
            'confidence': self.confidence,
        }


class ReferenceDatabase:
    """
    Local card reference database for embedding-based matching.

    Supports multiple embedding variants for different deployment targets.
    Each target (Hailo, iOS, Android) has its own embeddings generated by
    the corresponding student model.

    Files required:
    - {target}_embeddings.npy: (N, 768) array of card embeddings
    - {target}.usearch: uSearch vector index for fast ANN search
    - card_metadata.json: Card information
    - card_index.json: Maps embedding row to card_id

    Available targets:
    - teacher: Full DINOv3 model (validation reference only)
    - hailo: Hailo-8L optimized student model
    - ios: CoreML optimized student model
    - android: ONNX optimized student model
    """

    VALID_TARGETS = {'teacher', 'hailo', 'ios', 'android'}

    def __init__(self, data_dir: str, target: str = 'hailo'):
        """
        Initialize the reference database.

        Args:
            data_dir: Path to database directory
            target: Embedding target ('teacher', 'hailo', 'ios', 'android')
                   IMPORTANT: Must match the model used for inference!
        """
        self.data_dir = Path(data_dir)
        self.target = target

        if target not in self.VALID_TARGETS:
            raise ValueError(
                f"Invalid target '{target}'. Must be one of: {self.VALID_TARGETS}"
            )

        # Load target-specific embeddings
        embeddings_path = self.data_dir / f'{target}_embeddings.npy'
        logger.info(f"Loading {target} embeddings from {embeddings_path}")
        self.embeddings = np.load(embeddings_path).astype(np.float32)

        # Normalize embeddings (should already be normalized, but ensure)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / norms

        logger.info(f"Loaded {len(self.embeddings)} embeddings, dim={self.embeddings.shape[1]}")

        # Load uSearch index if available (for fast ANN search)
        usearch_path = self.data_dir / f'{target}.usearch'
        if usearch_path.exists():
            try:
                from usearch.index import Index
                self.usearch_index = Index.restore(str(usearch_path))
                logger.info(f"Loaded uSearch index from {usearch_path}")
            except ImportError:
                logger.warning("uSearch not installed, falling back to brute-force search")
                self.usearch_index = None
        else:
            logger.info(f"No uSearch index found at {usearch_path}, using brute-force")
            self.usearch_index = None

        # Load metadata
        with open(self.data_dir / 'card_metadata.json', 'r') as f:
            self.metadata = json.load(f)

        # Load index
        with open(self.data_dir / 'card_index.json', 'r') as f:
            index_data = json.load(f)
            self.index_to_card_id = index_data['index_to_card_id']
            self.card_id_to_index = index_data['card_id_to_index']
    
    @property
    def num_cards(self) -> int:
        return len(self.embeddings)
    
    @property
    def embedding_dim(self) -> int:
        return self.embeddings.shape[1]
    
    def get_card_info(self, card_id: str) -> Optional[dict]:
        """Get metadata for a card"""
        return self.metadata.get('cards', {}).get(card_id)
    
    def get_embedding(self, card_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific card"""
        idx = self.card_id_to_index.get(card_id)
        if idx is None:
            return None
        return self.embeddings[idx]
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        distance_threshold: float = 0.5
    ) -> List[CardMatch]:
        """
        Find nearest cards to query embedding.

        Uses uSearch index for fast ANN search when available,
        falls back to brute-force cosine similarity otherwise.

        Args:
            query_embedding: 768-dim normalized embedding
            top_k: Number of results to return
            distance_threshold: Maximum distance to consider a match

        Returns:
            List of CardMatch objects, sorted by distance (ascending)
        """
        # Ensure query is normalized
        query_embedding = query_embedding.astype(np.float32)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm

        # Use uSearch index if available (sub-millisecond search)
        if self.usearch_index is not None:
            matches = self.usearch_index.search(query_embedding, top_k)
            top_k_indices = matches.keys
            distances = matches.distances
        else:
            # Fallback: brute-force cosine similarity
            # Compute cosine distances (1 - cosine_similarity)
            # For normalized vectors: distance = 1 - dot_product
            similarities = np.dot(self.embeddings, query_embedding)
            all_distances = 1 - similarities

            # Get top-k indices
            top_k_indices = np.argsort(all_distances)[:top_k]
            distances = all_distances[top_k_indices]

        # Build results
        results = []
        for idx, distance in zip(top_k_indices, distances):
            distance = float(distance)

            if distance > distance_threshold:
                continue

            card_id = self.index_to_card_id[int(idx)]
            card_info = self.get_card_info(card_id) or {}

            # Convert distance to confidence (0-1 scale)
            # distance=0 -> confidence=1, distance=0.5 -> confidence=0
            confidence = max(0, 1 - (distance / distance_threshold))

            results.append(CardMatch(
                card_id=card_id,
                name=card_info.get('name', 'Unknown'),
                set_name=card_info.get('set_name', 'Unknown'),
                number=card_info.get('number', '?'),
                distance=distance,
                confidence=confidence,
            ))

        return results
    
    def search_batch(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 1
    ) -> List[List[CardMatch]]:
        """Search for multiple queries at once (more efficient)"""
        results = []
        
        # Batch similarity computation
        query_embeddings = query_embeddings.astype(np.float32)
        query_embeddings = query_embeddings / np.linalg.norm(
            query_embeddings, axis=1, keepdims=True
        )
        
        all_similarities = np.dot(query_embeddings, self.embeddings.T)
        all_distances = 1 - all_similarities
        
        for i, distances in enumerate(all_distances):
            top_k_indices = np.argsort(distances)[:top_k]
            
            query_results = []
            for idx in top_k_indices:
                card_id = self.index_to_card_id[idx]
                card_info = self.get_card_info(card_id) or {}
                
                query_results.append(CardMatch(
                    card_id=card_id,
                    name=card_info.get('name', 'Unknown'),
                    set_name=card_info.get('set_name', 'Unknown'),
                    number=card_info.get('number', '?'),
                    distance=float(distances[idx]),
                    confidence=max(0, 1 - distances[idx] * 2),
                ))
            
            results.append(query_results)
        
        return results


class DatabaseBuilder:
    """
    Build reference database from card images and embedding model.

    Supports building embeddings for different deployment targets.
    Each target uses its corresponding student model to generate embeddings.
    """

    VALID_TARGETS = {'teacher', 'hailo', 'ios', 'android'}

    def __init__(self, embedder, metadata_source: dict, target: str = 'hailo'):
        """
        Initialize database builder.

        Args:
            embedder: Embedding model instance (must match target!)
            metadata_source: Card metadata dictionary
            target: Deployment target ('teacher', 'hailo', 'ios', 'android')
        """
        if target not in self.VALID_TARGETS:
            raise ValueError(
                f"Invalid target '{target}'. Must be one of: {self.VALID_TARGETS}"
            )

        self.embedder = embedder
        self.metadata_source = metadata_source
        self.target = target

    def build(
        self,
        image_dir: str,
        output_dir: str,
        batch_size: int = 32,
        build_usearch_index: bool = True
    ):
        """
        Build database from card images.

        Args:
            image_dir: Directory containing card images (organized by set)
            output_dir: Where to save database files
            batch_size: Batch size for embedding computation
            build_usearch_index: Whether to build uSearch vector index
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect all images
        image_paths = []
        card_ids = []

        for card_id, info in self.metadata_source['cards'].items():
            image_path = Path(image_dir) / info.get('local_image_path', f'{card_id}.jpg')
            if image_path.exists():
                image_paths.append(str(image_path))
                card_ids.append(card_id)
            else:
                logger.warning(f"Missing image for {card_id}: {image_path}")

        logger.info(f"Processing {len(image_paths)} card images for target: {self.target}")

        # Compute embeddings in batches
        all_embeddings = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = [self.load_and_preprocess(p) for p in batch_paths]
            batch_tensor = np.stack(batch_images)

            # Compute embeddings using target-specific model
            batch_embeddings = self.embedder.embed_batch(batch_tensor)
            all_embeddings.append(batch_embeddings)

            logger.info(f"Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)}")

        # Concatenate all embeddings
        embeddings = np.concatenate(all_embeddings, axis=0)

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        # Save target-specific embeddings
        embeddings_path = output_dir / f'{self.target}_embeddings.npy'
        np.save(embeddings_path, embeddings)
        logger.info(f"Saved {self.target} embeddings: {embeddings.shape}")

        # Build and save uSearch index for fast ANN search
        if build_usearch_index:
            try:
                from usearch.index import Index
                index = Index(ndim=embeddings.shape[1], metric='cos')
                index.add(np.arange(len(embeddings)), embeddings)
                usearch_path = output_dir / f'{self.target}.usearch'
                index.save(str(usearch_path))
                logger.info(f"Saved uSearch index to {usearch_path}")
            except ImportError:
                logger.warning("uSearch not installed, skipping index creation")

        # Save metadata (shared across all targets)
        metadata_path = output_dir / 'card_metadata.json'
        if not metadata_path.exists():
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata_source, f, indent=2)
            logger.info("Saved card metadata")

        # Save index (shared across all targets)
        index_path = output_dir / 'card_index.json'
        if not index_path.exists():
            index_data = {
                'index_to_card_id': card_ids,
                'card_id_to_index': {cid: i for i, cid in enumerate(card_ids)},
            }
            with open(index_path, 'w') as f:
                json.dump(index_data, f, indent=2)
            logger.info("Saved card index")

        logger.info(f"Database built successfully for target '{self.target}' in {output_dir}")
    
    def load_and_preprocess(self, image_path: str) -> np.ndarray:
        """Load and preprocess image for embedding"""
        import cv2
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # CHW format
        image = image.transpose(2, 0, 1)
        
        return image
```

---

## Building the Database

### Step 1: Prepare Metadata

```python
# prepare_metadata.py

import json
from pathlib import Path

def build_metadata_from_your_data(your_card_data: list) -> dict:
    """
    Convert your existing card metadata to database format.

    Assumes you have: card_id, name, set_code, set_name, number, etc.
    """
    metadata = {
        'cards': {},
        'metadata': {
            'total_cards': len(your_card_data),
            'embedding_dim': 768,
            'teacher_model_version': 'dinov3_vit_base',
            'sets': list(set(c['set_code'] for c in your_card_data)),
            'embedding_variants': {
                'teacher': {'file': 'teacher_embeddings.npy', 'model': 'dinov3_vit_base_fp32'},
                'hailo': {'file': 'hailo_embeddings.npy', 'model': 'dinov3_student_hailo_int8'},
                'ios': {'file': 'ios_embeddings.npy', 'model': 'dinov3_student_coreml_fp16'},
                'android': {'file': 'android_embeddings.npy', 'model': 'dinov3_student_onnx_int8'},
            }
        }
    }

    for card in your_card_data:
        card_id = f"{card['set_code']}-{card['number']}"

        metadata['cards'][card_id] = {
            'name': card['name'],
            'set_code': card['set_code'],
            'set_name': card['set_name'],
            'number': card['number'],
            'rarity': card.get('rarity', ''),
            'type': card.get('type', ''),
            'local_image_path': f"images/{card['set_code']}/{card['number']}.jpg",
        }

    return metadata

# Usage
with open('your_existing_card_data.json', 'r') as f:
    your_data = json.load(f)

metadata = build_metadata_from_your_data(your_data)

with open('card_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

### Step 2: Compute Embeddings for Each Target

Each deployment target requires its own embedding file. Use `build_reference_db.py` to generate embeddings for specific targets.

```python
# build_reference_db.py
#
# Usage:
#   python build_reference_db.py --target hailo    # For Hailo-8L deployment
#   python build_reference_db.py --target ios      # For iOS/CoreML deployment
#   python build_reference_db.py --target android  # For Android/ONNX deployment
#   python build_reference_db.py --target teacher  # For validation (full model)

import argparse
from reference_database import DatabaseBuilder
import json

def main():
    parser = argparse.ArgumentParser(description='Build reference database for specific target')
    parser.add_argument('--target', required=True,
                       choices=['teacher', 'hailo', 'ios', 'android'],
                       help='Deployment target for embeddings')
    parser.add_argument('--image-dir', default='./card_images',
                       help='Directory containing card images')
    parser.add_argument('--output-dir', default='./reference_database',
                       help='Output directory for database files')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for embedding computation')
    args = parser.parse_args()

    # Load the appropriate embedder for the target
    if args.target == 'hailo':
        from hailo_embedding import HailoEmbedder
        embedder = HailoEmbedder('models/card_embedding_hailo.hef')
    elif args.target == 'ios':
        from coreml_embedding import CoreMLEmbedder
        embedder = CoreMLEmbedder('models/card_embedding_ios.mlpackage')
    elif args.target == 'android':
        from onnx_embedding import ONNXEmbedder
        embedder = ONNXEmbedder('models/card_embedding_android.onnx')
    elif args.target == 'teacher':
        from teacher_embedding import TeacherEmbedder
        embedder = TeacherEmbedder('models/dinov3_teacher.pth')

    # Load metadata
    with open('card_metadata.json', 'r') as f:
        metadata = json.load(f)

    # Build database for this target
    builder = DatabaseBuilder(embedder, metadata, target=args.target)
    builder.build(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        build_usearch_index=True
    )

    print(f"Database built for target: {args.target}")
    print(f"  Embeddings: {args.output_dir}/{args.target}_embeddings.npy")
    print(f"  uSearch index: {args.output_dir}/{args.target}.usearch")

if __name__ == '__main__':
    main()
```

**Building all targets:**

```bash
# Build embeddings for each deployment target
python build_reference_db.py --target teacher   # Validation reference
python build_reference_db.py --target hailo     # Raspberry Pi deployment
python build_reference_db.py --target ios       # iOS app
python build_reference_db.py --target android   # Android app
```

### Step 3: Validate Database

```python
# validate_database.py

from reference_database import ReferenceDatabase
import numpy as np
import argparse

def validate_database(data_dir: str, target: str):
    """Validate database for a specific target."""

    # Load database with target-specific embeddings
    db = ReferenceDatabase(data_dir, target=target)

    print(f"Target: {target}")
    print(f"Total cards: {db.num_cards}")
    print(f"Embedding dim: {db.embedding_dim}")
    print(f"uSearch index: {'loaded' if db.usearch_index else 'not available'}")

    # Test search with known embedding
    test_embedding = db.get_embedding('sv10-001')  # Pikachu ex
    if test_embedding is not None:
        results = db.search(test_embedding, top_k=5)

        print(f"\nSearch results for Pikachu ex embedding ({target}):")
        for match in results:
            print(f"  {match.name} ({match.set_name} #{match.number})")
            print(f"    Distance: {match.distance:.4f}, Confidence: {match.confidence:.1%}")

        # Test that same card is nearest to itself
        assert results[0].card_id == 'sv10-001', "Self-search failed!"
        assert results[0].distance < 0.01, "Self-distance too high!"

    print(f"\n[OK] Database validation passed for target: {target}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='hailo',
                       choices=['teacher', 'hailo', 'ios', 'android'])
    parser.add_argument('--data-dir', default='./reference_database')
    args = parser.parse_args()

    validate_database(args.data_dir, args.target)
```

**Validate all targets:**

```bash
python validate_database.py --target hailo
python validate_database.py --target ios
python validate_database.py --target android
python validate_database.py --target teacher
```

---

## Performance Optimization

### Memory-Mapped Embeddings (For Large Databases)

```python
class MemoryMappedDatabase(ReferenceDatabase):
    """Use memory mapping for very large databases (100k+ cards)"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
        # Memory-map embeddings (doesn't load into RAM immediately)
        embeddings_path = self.data_dir / 'card_embeddings.npy'
        self.embeddings = np.load(embeddings_path, mmap_mode='r')
        
        # ... rest of initialization
```

### Approximate Nearest Neighbor (Future Optimization)

```python
# For faster search with 100k+ cards, use FAISS or Annoy

import faiss

class FAISSDatabase(ReferenceDatabase):
    """Use FAISS for fast approximate nearest neighbor search"""
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        
        # Build FAISS index
        dim = self.embedding_dim
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine for normalized)
        self.index.add(self.embeddings)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, **kwargs):
        # FAISS search
        query = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query, top_k)
        
        # Convert to CardMatch objects
        # ...
```

---

## File Structure

```
reference_database/
├── teacher_embeddings.npy   # Teacher model embeddings (validation) - 52 MB
├── hailo_embeddings.npy     # Hailo-8L student embeddings - 52 MB
├── ios_embeddings.npy       # iOS/CoreML student embeddings - 52 MB
├── android_embeddings.npy   # Android/ONNX student embeddings - 52 MB
├── teacher.usearch          # uSearch vector index for teacher - ~10 MB
├── hailo.usearch            # uSearch vector index for Hailo - ~10 MB
├── ios.usearch              # uSearch vector index for iOS - ~10 MB
├── android.usearch          # uSearch vector index for Android - ~10 MB
├── card_metadata.json       # Card information (shared) - ~5 MB
├── card_index.json          # Row index <-> card_id mapping (shared) - ~500 KB
└── README.md                # Database documentation

Per-deployment size: ~62 MB (embeddings + index + shared files)
Full database (all targets): ~260 MB
```

**Deployment Notes:**
- Each deployment only needs its target-specific files plus shared metadata
- For Hailo deployment: `hailo_embeddings.npy`, `hailo.usearch`, `card_metadata.json`, `card_index.json`
- For iOS app bundle: `ios_embeddings.npy`, `ios.usearch`, `card_metadata.json`, `card_index.json`
- Teacher embeddings are only needed for validation/benchmarking, not production

---

## Acceptance Criteria

### AC-1: Database Loading
```gherkin
GIVEN a properly formatted reference database
WHEN loading the database for a specific target (hailo, ios, android, teacher)
THEN target-specific embeddings MUST load successfully
AND embedding count MUST match metadata count
AND uSearch index MUST load if available
AND loading time MUST be <5 seconds
```

### AC-2: Search Accuracy
```gherkin
GIVEN a query embedding for a known card
WHEN searching the database with matching target embeddings
THEN the correct card MUST be the top result
AND distance MUST be <0.1 for exact match
```

### AC-3: Search Speed (with uSearch)
```gherkin
GIVEN a query embedding and uSearch index loaded
WHEN searching 17,592 card database
THEN search time MUST be <1ms on Raspberry Pi 5
```

### AC-4: Unknown Rejection
```gherkin
GIVEN a query embedding from non-card image
WHEN searching the database with threshold 0.5
THEN NO results should be returned
OR all results should have distance >0.5
```

### AC-5: Memory Usage
```gherkin
GIVEN a single-target database loaded into memory
THEN total memory usage MUST be <100MB per target
AND the system MUST remain responsive
```

### AC-6: Database Integrity
```gherkin
GIVEN the database files for a target
WHEN validating integrity
THEN embedding count MUST equal index count
AND all card_ids in index MUST exist in metadata
AND no duplicate card_ids MUST exist
AND embedding file name MUST match target pattern
```

### AC-7: Target Consistency
```gherkin
GIVEN embeddings from a student model
WHEN building the reference database
THEN the target parameter MUST match the model type
AND the output file MUST be named {target}_embeddings.npy
AND the uSearch index MUST be named {target}.usearch
```

---

## Testing Plan

### Unit Tests

```python
import pytest
import numpy as np

TARGETS = ['teacher', 'hailo', 'ios', 'android']


@pytest.mark.parametrize('target', TARGETS)
def test_database_loading(target):
    """Test loading database for each target"""
    db = ReferenceDatabase('./test_database', target=target)
    assert db.num_cards == 100  # Test with small set
    assert db.embedding_dim == 768
    assert db.target == target


@pytest.mark.parametrize('target', TARGETS)
def test_search_self(target):
    """Card's own embedding should return itself as top match"""
    db = ReferenceDatabase('./test_database', target=target)

    for card_id in ['sv10-001', 'sv10-002', 'sv10-003']:
        embedding = db.get_embedding(card_id)
        results = db.search(embedding, top_k=1)

        assert len(results) == 1
        assert results[0].card_id == card_id
        assert results[0].distance < 0.01


def test_search_speed_with_usearch():
    """Search with uSearch should be sub-millisecond"""
    import time

    db = ReferenceDatabase('./reference_database', target='hailo')
    assert db.usearch_index is not None, "uSearch index required for speed test"

    query = np.random.randn(768).astype(np.float32)

    start = time.perf_counter()
    for _ in range(1000):
        db.search(query, top_k=5)
    elapsed = (time.perf_counter() - start) / 1000

    assert elapsed < 0.001  # <1ms per search with uSearch


def test_search_speed_brute_force():
    """Search without uSearch should still be reasonable"""
    import time

    db = ReferenceDatabase('./reference_database', target='hailo')
    db.usearch_index = None  # Force brute-force

    query = np.random.randn(768).astype(np.float32)

    start = time.perf_counter()
    for _ in range(100):
        db.search(query, top_k=5)
    elapsed = (time.perf_counter() - start) / 100

    assert elapsed < 0.010  # <10ms per brute-force search


@pytest.mark.parametrize('target', TARGETS)
def test_threshold_filtering(target):
    """High threshold should filter non-matches"""
    db = ReferenceDatabase('./test_database', target=target)

    # Random embedding (not a real card)
    random_query = np.random.randn(768).astype(np.float32)

    results = db.search(random_query, top_k=5, distance_threshold=0.3)

    # Should have few or no results (random vector far from all cards)
    for result in results:
        assert result.distance <= 0.3


def test_invalid_target():
    """Invalid target should raise ValueError"""
    with pytest.raises(ValueError, match="Invalid target"):
        ReferenceDatabase('./test_database', target='invalid')


def test_embedding_file_naming():
    """Verify correct embedding file naming per target"""
    from pathlib import Path

    db_dir = Path('./reference_database')

    for target in TARGETS:
        expected_embedding = db_dir / f'{target}_embeddings.npy'
        expected_usearch = db_dir / f'{target}.usearch'

        # At least one target should exist for testing
        if expected_embedding.exists():
            assert expected_embedding.is_file()
            if expected_usearch.exists():
                assert expected_usearch.is_file()
```

---

## Deliverables

| Deliverable | Format | Location |
|-------------|--------|----------|
| ReferenceDatabase class | `.py` | Git repo |
| DatabaseBuilder class | `.py` | Git repo |
| build_reference_db.py | `.py` | Git repo |
| validate_database.py | `.py` | Git repo |
| Unit tests | `.py` | Git repo |
| teacher_embeddings.npy | `.npy` | reference_database/ |
| hailo_embeddings.npy | `.npy` | reference_database/ |
| ios_embeddings.npy | `.npy` | reference_database/ |
| android_embeddings.npy | `.npy` | reference_database/ |
| {target}.usearch indices | `.usearch` | reference_database/ |
| card_metadata.json | `.json` | reference_database/ |
| card_index.json | `.json` | reference_database/ |

---

## Next Phase

Upon completion of Phase 3, proceed to **PRD_05_PIPELINE.md** for the matching pipeline.
