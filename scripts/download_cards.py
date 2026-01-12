#!/usr/bin/env python3
"""
Download ALL Pokemon TCG cards from GitHub JSON data
Downloads high-res images and saves metadata
"""

import json
import requests
from pathlib import Path
from tqdm import tqdm
import time

# Directories
REPO_DIR = Path("/Users/marcos/dev/raspberry-pi/pokemon-tcg-data/cards/en")
OUTPUT_DIR = Path("/Users/marcos/dev/raspberry-pi/PokeTCG_downloader/assets")
IMAGES_DIR = OUTPUT_DIR / "card_images"
METADATA_DIR = OUTPUT_DIR / "metadata_dir"
LABELS_DIR = OUTPUT_DIR / "image_labels"

# Create directories
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

def download_card(card_data, set_id):
    """Download single card image and metadata"""
    card_id = card_data.get("id", "unknown")
    card_name = card_data.get("name", "unknown").replace("/", "-").replace(" ", "_")

    # Get high-res image URL
    images = card_data.get("images", {})
    image_url = images.get("large")

    if not image_url:
        return False

    # Download image
    filename = f"{card_id}_{card_name}_high.png"
    filepath = IMAGES_DIR / filename

    if filepath.exists():
        return True  # Already downloaded

    try:
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            filepath.write_bytes(response.content)

            # Save metadata
            metadata_filename = f"labels_{card_id}_{card_name}.json"
            metadata_filepath = LABELS_DIR / metadata_filename
            metadata_filepath.write_text(json.dumps(card_data, indent=2))

            return True
    except Exception as e:
        print(f"  ERROR downloading {card_id}: {e}")

    return False

def main():
    # Get all JSON files
    json_files = sorted(REPO_DIR.glob("*.json"))
    print(f"Found {len(json_files)} set JSON files\n")

    total_cards = 0
    total_downloaded = 0

    for json_file in json_files:
        set_id = json_file.stem
        print(f"{'='*60}")
        print(f"Processing set: {set_id}")
        print(f"{'='*60}")

        # Load cards from JSON
        try:
            with open(json_file) as f:
                cards = json.load(f)
        except Exception as e:
            print(f"ERROR loading {json_file}: {e}")
            continue

        print(f"  Cards in set: {len(cards)}")

        # Download each card
        success_count = 0
        for card in tqdm(cards, desc=f"  Downloading {set_id}", unit="card"):
            if download_card(card, set_id):
                success_count += 1
            time.sleep(0.05)  # Rate limiting

        total_cards += len(cards)
        total_downloaded += success_count

        print(f"  Downloaded: {success_count}/{len(cards)} cards\n")

        # Save set metadata
        metadata_file = METADATA_DIR / f"cards_metadata_{set_id}.json"
        metadata_file.write_text(json.dumps(cards, indent=2))

    print(f"\n{'='*60}")
    print(f"COMPLETE!")
    print(f"{'='*60}")
    print(f"Total cards processed: {total_cards}")
    print(f"Successfully downloaded: {total_downloaded}")
    print(f"\nImages saved to: {IMAGES_DIR}")
    print(f"Metadata saved to: {METADATA_DIR}")
    print(f"Labels saved to: {LABELS_DIR}")

if __name__ == "__main__":
    main()
