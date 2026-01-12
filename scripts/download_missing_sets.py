#!/usr/bin/env python3
"""
Download missing Pokemon TCG sets after Prismatic Evolutions
Uses Pokemon TCG API (pokemontcg.io)
"""

import requests
import json
import os
from pathlib import Path
import time

# API Configuration
API_BASE_URL = "https://api.pokemontcg.io/v2"
API_KEY = "a8503903-6a75-4196-9960-83f9574d6f7b"

# Missing sets to download
MISSING_SETS = [
    "sv09",  # Journey Together (March 2025)
    "sv10",  # Destined Rivals (May 2025)
    # Note: Black Bolt & White Flare and Ascended Heroes may have different set IDs
    # Will attempt common naming patterns
]

# Output directories
BASE_DIR = Path("assets")
IMAGES_DIR = BASE_DIR / "card_images"
METADATA_DIR = BASE_DIR / "metadata_dir"
LABELS_DIR = BASE_DIR / "image_labels"

# Create directories
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

def get_headers():
    """Get API headers with API key if available"""
    headers = {}
    if API_KEY:
        headers["X-Api-Key"] = API_KEY
    return headers

def get_all_cards_in_set(set_code):
    """Download all cards from a specific set"""
    print(f"\n{'='*60}")
    print(f"Downloading set: {set_code}")
    print(f"{'='*60}")

    url = f"{API_BASE_URL}/cards"
    params = {"q": f"set.id:{set_code}", "pageSize": 250}
    headers = get_headers()

    all_cards = []
    page = 1

    while True:
        params["page"] = page
        response = requests.get(url, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()
            cards = data.get("data", [])

            if not cards:
                break

            all_cards.extend(cards)
            print(f"  Downloaded page {page}: {len(cards)} cards")

            # Check if there are more pages
            total_count = data.get("totalCount", 0)
            if len(all_cards) >= total_count:
                break

            page += 1
            time.sleep(0.1)  # Rate limiting
        else:
            print(f"  ERROR: {response.status_code} - {response.text}")
            break

    print(f"  Total cards in {set_code}: {len(all_cards)}")
    return all_cards

def download_card_image(card_data, set_code):
    """Download single card image"""
    card_id = card_data.get("id")
    card_name = card_data.get("name", "unknown").replace("/", "-")

    # Get high-res image URL
    images = card_data.get("images", {})
    image_url = images.get("large") or images.get("small")

    if not image_url:
        print(f"  No image URL for {card_id}")
        return False

    # Download image
    filename = f"{card_id}_{card_name}.png"
    filepath = IMAGES_DIR / filename

    if filepath.exists():
        return True  # Already downloaded

    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            filepath.write_bytes(response.content)
            return True
    except Exception as e:
        print(f"  ERROR downloading {card_id}: {e}")

    return False

def save_card_metadata(card_data, set_code):
    """Save card metadata as JSON"""
    card_id = card_data.get("id")
    card_name = card_data.get("name", "unknown").replace("/", "-")

    filename = f"labels_{card_id}_{card_name}.json"
    filepath = LABELS_DIR / filename

    try:
        filepath.write_text(json.dumps(card_data, indent=2))
        return True
    except Exception as e:
        print(f"  ERROR saving metadata for {card_id}: {e}")
        return False

def download_set(set_code):
    """Download complete set: metadata and images"""
    # Get all cards
    cards = get_all_cards_in_set(set_code)

    if not cards:
        print(f"  No cards found for {set_code}")
        return

    # Save set metadata
    metadata_file = METADATA_DIR / f"cards_metadata_{set_code}.json"
    metadata_file.write_text(json.dumps(cards, indent=2))
    print(f"  Saved metadata: {len(cards)} cards")

    # Download images and labels
    print(f"\n  Downloading images...")
    success_count = 0
    for i, card in enumerate(cards, 1):
        if download_card_image(card, set_code):
            save_card_metadata(card, set_code)
            success_count += 1

        if i % 10 == 0:
            print(f"    Progress: {i}/{len(cards)} ({success_count} successful)")

        time.sleep(0.05)  # Rate limiting

    print(f"  ✓ Downloaded {success_count}/{len(cards)} cards from {set_code}")

def main():
    if not API_KEY:
        print("WARNING: No API key set. Rate limits will be very low.")
        print("Get an API key at: https://pokemontcg.io/")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    print(f"\n{'='*60}")
    print("Pokemon TCG - Missing Sets Downloader")
    print(f"{'='*60}")
    print(f"Downloading {len(MISSING_SETS)} sets after Prismatic Evolutions")

    for set_code in MISSING_SETS:
        download_set(set_code)
        time.sleep(1)  # Pause between sets

    print(f"\n{'='*60}")
    print("✓ All missing sets downloaded!")
    print(f"{'='*60}")
    print(f"\nImages saved to: {IMAGES_DIR}")
    print(f"Metadata saved to: {METADATA_DIR}")
    print(f"Labels saved to: {LABELS_DIR}")

if __name__ == "__main__":
    main()
