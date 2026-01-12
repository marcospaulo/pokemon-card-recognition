#!/usr/bin/env python3
"""
Check if reference embeddings are normalized and healthy
"""

import numpy as np
import json

print("=" * 70)
print("Reference Embeddings Analysis")
print("=" * 70)

# Load reference data
embeddings = np.load('data/reference/embeddings.npy')
index = json.load(open('data/reference/index.json'))
metadata = json.load(open('data/reference/cards_metadata.json'))

print(f"\nShape: {embeddings.shape}")
print(f"Dtype: {embeddings.dtype}")

# Check norms
norms = np.linalg.norm(embeddings, axis=1)
print(f"\nNorm Statistics:")
print(f"   Mean: {norms.mean():.6f}")
print(f"   Std:  {norms.std():.6f}")
print(f"   Min:  {norms.min():.6f}")
print(f"   Max:  {norms.max():.6f}")

if norms.std() < 0.01:
    print(f"\n✅ Embeddings are L2-normalized (all norms ≈ 1.0)")
    print(f"   This is EXPECTED and correct for cosine similarity search!")
else:
    print(f"\n⚠️  Embeddings are NOT normalized")

# Check diversity of embeddings
print(f"\nEmbedding Diversity:")
print(f"   Mean: {embeddings.mean():.6f}")
print(f"   Std:  {embeddings.std():.6f}")

# Sample pairwise similarities
print(f"\nRandom Pairwise Similarities (10 samples):")
np.random.seed(42)
for i in range(10):
    idx1, idx2 = np.random.choice(len(embeddings), 2, replace=False)
    sim = np.dot(embeddings[idx1], embeddings[idx2])
    print(f"   Embedding {idx1} vs {idx2}: {sim:.4f}")

# Check if any card IDs have actual names
print(f"\nSample Metadata:")
sample_ids = list(metadata.keys())[:10]
for card_id in sample_ids:
    card = metadata[card_id]
    print(f"   {card_id}: {card}")
    break  # Just show first one to see structure

print("\n" + "=" * 70)
