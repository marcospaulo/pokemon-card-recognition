#!/usr/bin/env python3
"""
Extract metadata from card images that are missing metadata.

This script processes cards that don't have full metadata and
stores the extracted information to be merged back.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
CARDS_METADATA = BASE_DIR / "data" / "reference" / "cards_metadata.json"
EXTRACTED_METADATA = BASE_DIR / "data" / "reference" / "extracted_metadata.json"


def load_extracted_metadata():
    """Load extracted metadata from JSON file."""
    if EXTRACTED_METADATA.exists():
        with open(EXTRACTED_METADATA, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


# Legacy hardcoded metadata (kept for reference)
EXTRACTED_LEGACY = {
    "swsh35-26": {
        "name": "Machamp",
        "supertype": "Pokémon",
        "subtypes": ["Stage 2"],
        "hp": "170",
        "types": ["Fighting"],
        "evolvesFrom": "Machoke",
        "attacks": [
            {
                "name": "Macho Revenge",
                "cost": ["Fighting", "Colorless"],
                "damage": "20×",
                "text": "This attack does 20 damage for each Fighting Pokémon in your discard pile."
            },
            {
                "name": "Dynamite Punch",
                "cost": ["Fighting", "Fighting", "Colorless"],
                "damage": "200",
                "text": "This Pokémon also does 50 damage to itself."
            }
        ],
        "weaknesses": [{"type": "Psychic", "value": "×2"}],
        "retreatCost": ["Colorless", "Colorless"],
        "artist": "Anesaki Dynamic",
        "rarity": "Rare",
        "nationalPokedexNumbers": [68]
    },
    "swsh3-132": {
        "name": "Copperajah",
        "supertype": "Pokémon",
        "subtypes": ["Stage 1"],
        "hp": "190",
        "types": ["Metal"],
        "evolvesFrom": "Cufant",
        "abilities": [
            {
                "name": "Antibacterial Skin",
                "type": "Ability",
                "text": "This Pokémon can't be affected by any Special Conditions."
            }
        ],
        "attacks": [
            {
                "name": "Vengeful Stomp",
                "cost": ["Metal", "Colorless", "Colorless"],
                "damage": "120+",
                "text": "If your Benched Pokémon have any damage counters on them, this attack does 120 more damage."
            }
        ],
        "weaknesses": [{"type": "Fire", "value": "×2"}],
        "resistances": [{"type": "Grass", "value": "-30"}],
        "retreatCost": ["Colorless", "Colorless", "Colorless", "Colorless"],
        "artist": "Kouki Saitou",
        "rarity": "Rare",
        "nationalPokedexNumbers": [879]
    },
    "lc-75": {
        "name": "Exeggcute",
        "supertype": "Pokémon",
        "subtypes": ["Basic"],
        "hp": "50",
        "types": ["Grass"],
        "attacks": [
            {
                "name": "Hypnosis",
                "cost": ["Psychic"],
                "damage": "",
                "text": "The Defending Pokémon is now Asleep."
            },
            {
                "name": "Leech Seed",
                "cost": ["Grass", "Grass"],
                "damage": "20",
                "text": "Unless all damage from this attack is prevented, you may remove 1 damage counter from Exeggcute."
            }
        ],
        "weaknesses": [{"type": "Fire", "value": "×2"}],
        "retreatCost": ["Colorless"],
        "artist": "Mitsuhiro Arita",
        "rarity": "Common",
        "nationalPokedexNumbers": [102]
    },
    "xyp-XY98": {
        "name": "M Aerodactyl-EX",
        "supertype": "Pokémon",
        "subtypes": ["MEGA", "EX"],
        "hp": "210",
        "types": ["Colorless"],
        "evolvesFrom": "Aerodactyl-EX",
        "attacks": [
            {
                "name": "Rock Drill Dive",
                "cost": ["Colorless", "Colorless", "Colorless"],
                "damage": "110",
                "text": "This attack does 10 damage to each Benched Pokémon (both yours and your opponent's). (Don't apply Weakness and Resistance for Benched Pokémon.)"
            }
        ],
        "weaknesses": [{"type": "Lightning", "value": "×2"}],
        "resistances": [{"type": "Fighting", "value": "-20"}],
        "retreatCost": [],
        "artist": "5ban Graphics",
        "rarity": "Rare Ultra",
        "nationalPokedexNumbers": [142]
    }
}


def merge_metadata():
    """Merge extracted metadata into the main cards_metadata.json file."""
    # Load extracted metadata from JSON file
    extracted_data = load_extracted_metadata()
    print(f"Loaded {len(extracted_data)} cards from extracted_metadata.json")

    with open(CARDS_METADATA, 'r') as f:
        cards = json.load(f)

    updated = 0
    for card in cards:
        card_id = card["card_id"]
        if card_id in extracted_data:
            extracted = extracted_data[card_id]
            # Update fields from extracted data
            for key, value in extracted.items():
                if value:  # Only update if we have a value
                    card[key] = value
            updated += 1

    # Save updated metadata
    with open(CARDS_METADATA, 'w') as f:
        json.dump(cards, f, indent=2, ensure_ascii=False)

    print(f"Updated {updated} cards with extracted metadata")

    # Count remaining cards without full metadata
    remaining = sum(1 for card in cards if not card.get("supertype"))
    print(f"Remaining cards needing metadata: {remaining}")


if __name__ == "__main__":
    merge_metadata()
