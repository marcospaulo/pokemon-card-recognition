#!/usr/bin/env python3
"""
Reorganize dataset to train on ALL 17,592 cards
Combines train/val/test into single train folder
"""

import shutil
from pathlib import Path

data_dir = Path(__file__).parent.parent / 'data' / 'processed' / 'classification'

print("Reorganizing dataset to train on ALL cards...")
print(f"Data directory: {data_dir}")

# Count current structure
train_dir = data_dir / 'train'
val_dir = data_dir / 'val'
test_dir = data_dir / 'test'

train_cards = len(list(train_dir.iterdir())) if train_dir.exists() else 0
val_cards = len(list(val_dir.iterdir())) if val_dir.exists() else 0
test_cards = len(list(test_dir.iterdir())) if test_dir.exists() else 0

print(f"\nCurrent structure:")
print(f"  Train: {train_cards} cards")
print(f"  Val: {val_cards} cards")
print(f"  Test: {test_cards} cards")
print(f"  Total: {train_cards + val_cards + test_cards} cards")

# Create backup
backup_dir = data_dir.parent / 'classification_backup'
if not backup_dir.exists():
    print(f"\nCreating backup at: {backup_dir}")
    shutil.copytree(data_dir, backup_dir)
    print("✓ Backup created")
else:
    print(f"\nBackup already exists at: {backup_dir}")

# Move all cards to train
print("\nMoving all cards to train folder...")
moved = 0

for source_dir in [val_dir, test_dir]:
    if source_dir.exists():
        for card_folder in source_dir.iterdir():
            if card_folder.is_dir():
                dest = train_dir / card_folder.name
                if not dest.exists():
                    shutil.move(str(card_folder), str(dest))
                    moved += 1
                else:
                    print(f"  Warning: {card_folder.name} already exists in train, skipping")

print(f"✓ Moved {moved} card folders to train")

# Count final structure
final_train_cards = len(list(train_dir.iterdir()))
print(f"\nFinal structure:")
print(f"  Train: {final_train_cards} cards")
print(f"  Val: 0 cards (empty)")
print(f"  Test: 0 cards (empty)")

print("\n✅ Dataset reorganized! All cards now in train folder")
print(f"Backup available at: {backup_dir}")
