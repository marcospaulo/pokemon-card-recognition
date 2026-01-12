#!/usr/bin/env python3
"""
Deduplicate cards_metadata.json to keep only one entry per unique card_id

This fixes the recurring issue where duplicate image downloads (different naming conventions)
create duplicate metadata entries for the same card_id.
"""

import json
from pathlib import Path
from collections import defaultdict
import shutil
from datetime import datetime

def deduplicate_metadata(input_path: Path, output_path: Path = None):
    """
    Remove duplicate card_id entries from metadata, keeping only the first occurrence

    Args:
        input_path: Path to original cards_metadata.json
        output_path: Path to save deduplicated version (default: overwrite original after backup)
    """

    print("=" * 70)
    print("Deduplicating Metadata")
    print("=" * 70)

    # Load metadata
    print(f"\n[1/4] Loading metadata from: {input_path}")
    with open(input_path, 'r') as f:
        metadata_list = json.load(f)

    print(f"   Total entries: {len(metadata_list)}")

    # Group by card_id
    print(f"\n[2/4] Analyzing duplicates...")
    by_card_id = defaultdict(list)
    for entry in metadata_list:
        card_id = entry.get('card_id')
        if card_id:
            by_card_id[card_id].append(entry)

    duplicates = {cid: entries for cid, entries in by_card_id.items() if len(entries) > 1}

    print(f"   Unique card_ids: {len(by_card_id)}")
    print(f"   Duplicate card_ids: {len(duplicates)}")
    print(f"   Total duplicate entries: {sum(len(e) - 1 for e in duplicates.values())}")

    # Create deduplicated list (keep first occurrence of each card_id)
    print(f"\n[3/4] Deduplicating...")
    seen_card_ids = set()
    deduplicated = []
    removed_count = 0

    for entry in metadata_list:
        card_id = entry.get('card_id')
        if card_id not in seen_card_ids:
            seen_card_ids.add(card_id)
            deduplicated.append(entry)
        else:
            removed_count += 1

    print(f"   Kept: {len(deduplicated)} entries")
    print(f"   Removed: {removed_count} duplicate entries")

    # Save deduplicated metadata
    print(f"\n[4/4] Saving deduplicated metadata...")

    if output_path is None:
        # Create backup of original
        backup_path = input_path.with_suffix(f'.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        print(f"   Creating backup: {backup_path}")
        shutil.copy2(input_path, backup_path)
        output_path = input_path

    with open(output_path, 'w') as f:
        json.dump(deduplicated, f, indent=2)

    print(f"   Saved to: {output_path}")

    # Verify
    with open(output_path, 'r') as f:
        verified = json.load(f)

    card_ids_check = [e.get('card_id') for e in verified]
    unique_check = len(set(card_ids_check))

    print(f"\n✅ Verification:")
    print(f"   Entries in file: {len(verified)}")
    print(f"   Unique card_ids: {unique_check}")

    if len(verified) == unique_check:
        print(f"   ✓ No duplicates remaining!")
    else:
        print(f"   ⚠️  Warning: Still have duplicates!")

    print("\n" + "=" * 70)
    print("✅ Deduplication complete!")
    print("=" * 70)

    return True

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Deduplicate cards_metadata.json')
    parser.add_argument(
        '--input',
        type=str,
        default='data/reference/cards_metadata.json',
        help='Path to input metadata file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output file (default: overwrite input after backup)'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        return 1

    success = deduplicate_metadata(input_path, output_path)
    return 0 if success else 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
