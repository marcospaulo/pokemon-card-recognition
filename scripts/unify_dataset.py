#!/usr/bin/env python3
"""
Unify Pokemon card dataset.

Creates a single unified metadata file mapping all 17,592 card images
to their metadata from the per-set JSON files.

Output:
- data/reference/cards_metadata.json - Complete metadata for all cards
- data/reference/card_index.json - card_id -> image filename mapping
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict

# Paths
BASE_DIR = Path(__file__).parent.parent
CARD_IMAGES_DIR = BASE_DIR / "data" / "raw" / "card_images"
METADATA_BY_SET_DIR = BASE_DIR / "data" / "raw" / "metadata" / "by_set"
OUTPUT_DIR = BASE_DIR / "data" / "reference"


def parse_image_filename(filename: str) -> dict:
    """
    Parse card image filename to extract card info.

    Pattern 1 (old): {Name}-en_{era}_{set}_{number}_high.png
    Example: Alakazam-en_base_base1_1_high.png -> {name: Alakazam, set: base1, number: 1}

    Pattern 2 (new): {set}-{number}_{Name}_high.png or {set}-{number}_{Name}.png
    Example: sv8-87_Dedenne_high.png -> {name: Dedenne, set: sv8, number: 87}
    """
    # Pattern 1: {Name}-en_{era}_{set}_{number}_high.png (older format)
    # Check this FIRST because it's more specific (contains "-en_")
    match = re.match(r"(.+)-en_([^_]+)_([^_]+)_(\d+)_high\.png", filename)
    if match:
        name, era, set_id, number = match.groups()
        return {
            "filename": filename,
            "name": name,
            "set_id": set_id,
            "number": number,
            "card_id": f"{set_id}-{number}"
        }

    # Pattern 2: {set}-{number}_{Name}[_high].png (newer format, most common)
    # Allow special chars in number (e.g., ? and ! for Unown cards)
    match = re.match(r"([a-zA-Z0-9]+)-([a-zA-Z0-9?!]+)_(.+?)(?:_high)?\.png", filename)
    if match:
        set_id, number, name = match.groups()
        # Clean up name (replace underscores with spaces)
        name = name.replace("_", " ").strip()
        return {
            "filename": filename,
            "name": name,
            "set_id": set_id,
            "number": number,
            "card_id": f"{set_id}-{number}"
        }

    return None


def load_all_set_metadata(metadata_dir: Path) -> dict:
    """
    Load all per-set metadata files and index by card_id.

    Handles two formats:
    1. Array of cards: [{...}, {...}]
    2. Object with cards array: {"cardCount": {...}, "cards": [...]}

    When the same card_id appears in multiple files, prefer the entry
    with more metadata (more keys = more complete data).
    """
    metadata_by_id = {}

    for json_file in metadata_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # Handle both formats
                if isinstance(data, list):
                    # Format 1: Array of cards
                    cards = data
                elif isinstance(data, dict) and "cards" in data:
                    # Format 2: Object with cards array
                    cards = data["cards"]
                else:
                    print(f"  Warning: Unknown format in {json_file.name}")
                    continue

                for card in cards:
                    if isinstance(card, dict):
                        card_id = card.get("id", "")
                        if card_id:
                            # Prefer entry with more keys (more complete metadata)
                            existing = metadata_by_id.get(card_id)
                            if existing is None or len(card) > len(existing):
                                metadata_by_id[card_id] = card
        except Exception as e:
            print(f"  Warning: Could not load {json_file.name}: {e}")

    return metadata_by_id


def main():
    print("=" * 60)
    print("UNIFYING POKEMON CARD DATASET")
    print("=" * 60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Parse all image filenames
    print("\n[1/4] Parsing image filenames...")
    image_files = list(CARD_IMAGES_DIR.glob("*.png"))
    print(f"  Found {len(image_files)} image files")

    parsed_images = []
    parse_errors = []
    for img_file in image_files:
        parsed = parse_image_filename(img_file.name)
        if parsed:
            parsed_images.append(parsed)
        else:
            parse_errors.append(img_file.name)

    print(f"  Successfully parsed: {len(parsed_images)}")
    if parse_errors:
        print(f"  Parse errors: {len(parse_errors)}")
        for err in parse_errors[:5]:
            print(f"    - {err}")

    # Step 2: Load all set metadata
    print("\n[2/4] Loading set metadata...")
    set_metadata = load_all_set_metadata(METADATA_BY_SET_DIR)
    print(f"  Loaded metadata for {len(set_metadata)} unique card IDs")

    # Step 3: Match images to metadata
    print("\n[3/4] Matching images to metadata...")
    unified_cards = []
    full_metadata = 0
    partial_metadata = 0
    no_metadata = []

    for img_info in parsed_images:
        card_id = img_info["card_id"]
        metadata = set_metadata.get(card_id, {})

        unified_card = {
            "card_id": card_id,
            "image_filename": img_info["filename"],
            "name": metadata.get("name", img_info["name"]),
            "set_id": img_info["set_id"],
            "number": img_info["number"],
            # Include all available metadata
            "supertype": metadata.get("supertype", ""),
            "subtypes": metadata.get("subtypes", []),
            "hp": metadata.get("hp", ""),
            "types": metadata.get("types", []),
            "rarity": metadata.get("rarity", ""),
            "artist": metadata.get("artist", ""),
            "attacks": metadata.get("attacks", []),
            "weaknesses": metadata.get("weaknesses", []),
            "retreatCost": metadata.get("retreatCost", []),
            "evolvesFrom": metadata.get("evolvesFrom", ""),
            "evolvesTo": metadata.get("evolvesTo", []),
            "abilities": metadata.get("abilities", []),
            "flavorText": metadata.get("flavorText", ""),
            "nationalPokedexNumbers": metadata.get("nationalPokedexNumbers", []),
            "legalities": metadata.get("legalities", {}),
        }

        unified_cards.append(unified_card)

        # Track metadata quality
        if metadata.get("supertype"):
            full_metadata += 1
        elif metadata:
            partial_metadata += 1
        else:
            no_metadata.append(card_id)

    print(f"  Full metadata (with supertype): {full_metadata}")
    print(f"  Partial metadata (name only): {partial_metadata}")
    print(f"  No metadata: {len(no_metadata)}")
    if no_metadata:
        print(f"  Sample missing IDs: {no_metadata[:5]}")

    # Step 4: Save unified data
    print("\n[4/4] Saving unified dataset...")

    # Full metadata
    metadata_path = OUTPUT_DIR / "cards_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(unified_cards, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {metadata_path}")
    print(f"  Total cards: {len(unified_cards)}")

    # Card index (card_id -> image filename)
    card_index = {card["card_id"]: card["image_filename"] for card in unified_cards}
    index_path = OUTPUT_DIR / "card_index.json"
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(card_index, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {index_path}")

    # Summary stats
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total cards: {len(unified_cards)}")
    print(f"  With full metadata: {full_metadata}")
    print(f"  With partial metadata: {partial_metadata}")
    print(f"  Without metadata: {len(no_metadata)}")

    # Count by set
    sets = defaultdict(int)
    for card in unified_cards:
        sets[card["set_id"]] += 1
    print(f"  Total sets: {len(sets)}")
    print(f"  Top 5 sets by card count:")
    for set_id, count in sorted(sets.items(), key=lambda x: -x[1])[:5]:
        print(f"    {set_id}: {count} cards")

    print("\nDone!")


if __name__ == "__main__":
    main()
