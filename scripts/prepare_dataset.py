#!/usr/bin/env python3
"""
Create classification dataset from downloaded Pokemon card images.

Takes all card images from PokeTCG_downloader and organizes them into
a classification dataset structure with train/val/test splits.

Structure:
    pokemon_classification_dataset/
    ├── train/
    │   ├── swsh1-1_Celebi_V/
    │   │   └── swsh1-1_Celebi_V.png
    │   └── ...
    ├── val/
    │   └── ...
    ├── test/
    │   └── ...
    ├── class_index.json
    └── card_metadata.json

Usage:
    python create_classification_dataset.py
    python create_classification_dataset.py --dry-run
"""

import os
import json
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Configuration - Updated for new project structure
BASE_DIR = Path(__file__).parent.parent
IMAGES_DIR = BASE_DIR / "data" / "raw" / "card_images"
METADATA_FILE = BASE_DIR / "data" / "reference" / "cards_metadata.json"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "classification"

# Split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.15
TEST_RATIO = 0.05

# Random seed for reproducibility
RANDOM_SEED = 42


def get_all_cards_from_metadata() -> dict:
    """Load all cards from unified cards_metadata.json.

    Uses image_filename as the unique key since card_id has duplicates
    (1,605 cards share card_ids with other cards due to data quality issues).
    """
    all_cards = {}

    with open(METADATA_FILE, 'r') as f:
        cards_list = json.load(f)

    for card in cards_list:
        card_id = card.get("card_id", "")
        image_filename = card.get("image_filename", "")
        if image_filename:
            # Use image_filename (without extension) as unique key
            unique_key = image_filename.replace(".png", "").replace(".jpg", "")
            all_cards[unique_key] = {
                "id": card_id,
                "unique_key": unique_key,
                "name": card.get("name", "Unknown"),
                "supertype": card.get("supertype", ""),
                "subtypes": card.get("subtypes", []),
                "rarity": card.get("rarity", ""),
                "hp": card.get("hp", ""),
                "types": card.get("types", []),
                "image_filename": image_filename,
            }

    return all_cards


def find_image_for_card(card_data: dict) -> Path:
    """Find the image for a card using the image_filename from metadata."""
    image_filename = card_data.get("image_filename", "")
    if image_filename:
        img_path = IMAGES_DIR / image_filename
        if img_path.exists():
            return img_path

    # Fallback: search by card_id
    card_id = card_data.get("id", "")
    for img_file in IMAGES_DIR.glob(f"*{card_id}*"):
        if img_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
            return img_file

    return None


def create_class_name(unique_key: str) -> str:
    """Create a valid class name from unique key (derived from image filename).

    The unique_key is already the image filename without extension,
    which is unique and descriptive (e.g., 'swsh1-1_Celebi_V_high').

    Handles special characters like ! and ? (Unown cards) by replacing
    them with text equivalents to avoid collisions.
    """
    # Replace special characters with text equivalents (for Unown etc.)
    special_chars = {
        "!": "_EXCL",
        "?": "_QUES",
        "♀": "_F",
        "♂": "_M",
        "é": "e",
        "'": "",
    }
    name_clean = unique_key
    for char, replacement in special_chars.items():
        name_clean = name_clean.replace(char, replacement)

    # Clean for use as directory name
    name_clean = name_clean.replace("/", "-").replace(" ", "_")
    name_clean = "".join(c for c in name_clean if c.isalnum() or c in "_-")
    return name_clean


def main():
    parser = argparse.ArgumentParser(description="Create classification dataset from Pokemon card images")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be created")
    parser.add_argument("--force", action="store_true", help="Overwrite existing dataset")
    args = parser.parse_args()

    print("=" * 70)
    print("Pokemon Card Classification Dataset Creator")
    print("=" * 70)

    # Check if output exists
    if OUTPUT_DIR.exists() and not args.dry_run:
        if args.force:
            print(f"\nRemoving existing dataset: {OUTPUT_DIR}")
            shutil.rmtree(OUTPUT_DIR)
        else:
            print(f"\nOutput directory already exists: {OUTPUT_DIR}")
            print("Use --force to overwrite or delete manually.")
            return

    # Load card database
    print("\n1. Loading card database from unified metadata...")
    all_cards = get_all_cards_from_metadata()
    print(f"   Found {len(all_cards)} cards in metadata")

    # Find images for all cards
    print("\n2. Finding card images...")
    cards_with_images = []
    missing_images = []

    for unique_key, card_data in tqdm(all_cards.items(), desc="Scanning"):
        img_path = find_image_for_card(card_data)
        if img_path:
            cards_with_images.append({
                **card_data,
                "image_path": img_path,
                "class_name": create_class_name(unique_key)
            })
        else:
            missing_images.append(unique_key)

    print(f"   Found images for {len(cards_with_images)} cards")
    print(f"   Missing images for {len(missing_images)} cards")

    if missing_images and len(missing_images) <= 20:
        print(f"   Missing: {missing_images}")
    elif missing_images:
        print(f"   First 10 missing: {missing_images[:10]}")

    if len(cards_with_images) < 17500:
        print(f"\n   WARNING: Expected ~17,592 cards but only found {len(cards_with_images)}")
        print(f"   Card images should be at: {IMAGES_DIR}")

    # Split into train/val/test
    print("\n3. Splitting into train/val/test...")
    random.seed(RANDOM_SEED)
    random.shuffle(cards_with_images)

    total = len(cards_with_images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_cards = cards_with_images[:train_end]
    val_cards = cards_with_images[train_end:val_end]
    test_cards = cards_with_images[val_end:]

    print(f"   Train: {len(train_cards)} cards ({100*len(train_cards)/total:.1f}%)")
    print(f"   Val:   {len(val_cards)} cards ({100*len(val_cards)/total:.1f}%)")
    print(f"   Test:  {len(test_cards)} cards ({100*len(test_cards)/total:.1f}%)")

    if args.dry_run:
        print("\n[DRY RUN] Would create dataset with:")
        print(f"   Total classes: {len(cards_with_images)}")
        print(f"   Train classes: {len(train_cards)}")
        print(f"   Val classes:   {len(val_cards)}")
        print(f"   Test classes:  {len(test_cards)}")
        return

    # Create directory structure
    print("\n4. Creating dataset structure...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "train").mkdir(exist_ok=True)
    (OUTPUT_DIR / "val").mkdir(exist_ok=True)
    (OUTPUT_DIR / "test").mkdir(exist_ok=True)

    # Copy images to dataset
    print("\n5. Copying images...")

    def copy_cards(cards, split_name):
        split_dir = OUTPUT_DIR / split_name
        for card in tqdm(cards, desc=f"  {split_name}"):
            class_dir = split_dir / card["class_name"]
            class_dir.mkdir(exist_ok=True)

            # Copy image
            dest_path = class_dir / card["image_path"].name
            shutil.copy2(card["image_path"], dest_path)

    copy_cards(train_cards, "train")
    copy_cards(val_cards, "val")
    copy_cards(test_cards, "test")

    # Create class index
    print("\n6. Creating class_index.json...")
    all_class_names = sorted([c["class_name"] for c in cards_with_images])
    class_to_idx = {name: idx for idx, name in enumerate(all_class_names)}

    class_index = {
        "num_classes": len(all_class_names),
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(idx): name for name, idx in class_to_idx.items()},
        "train_classes": len(train_cards),
        "val_classes": len(val_cards),
        "test_classes": len(test_cards),
    }

    with open(OUTPUT_DIR / "class_index.json", "w") as f:
        json.dump(class_index, f, indent=2)

    # Create card metadata
    print("7. Creating card_metadata.json...")
    card_metadata = {}
    for card in cards_with_images:
        card_metadata[card["class_name"]] = {
            "card_id": card["id"],
            "name": card["name"],
            "supertype": card["supertype"],
            "subtypes": card["subtypes"],
            "rarity": card["rarity"],
            "hp": card.get("hp", ""),
            "types": card.get("types", []),
        }

    with open(OUTPUT_DIR / "card_metadata.json", "w") as f:
        json.dump(card_metadata, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("DATASET CREATION COMPLETE")
    print("=" * 70)
    print(f"\nDataset location: {OUTPUT_DIR}")
    print(f"\nStructure:")
    print(f"  {OUTPUT_DIR}/")
    print(f"  ├── train/         ({len(train_cards)} classes)")
    print(f"  ├── val/           ({len(val_cards)} classes)")
    print(f"  ├── test/          ({len(test_cards)} classes)")
    print(f"  ├── class_index.json")
    print(f"  └── card_metadata.json")
    print(f"\nTotal classes: {len(cards_with_images)}")
    print(f"\nNext steps:")
    print(f"  1. Upload to S3: aws s3 sync {OUTPUT_DIR} s3://your-bucket/classification_dataset/")
    print(f"  2. Launch training on SageMaker")
    print(f"  3. Or run locally: python src/training/sagemaker/sagemaker_train_combined.py")


if __name__ == "__main__":
    main()
