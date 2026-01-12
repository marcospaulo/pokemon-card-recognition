# Pokemon Card Training Split Strategy

## The Problem with Traditional Splits

**WRONG Approach (typical classification):**
```
17,592 cards → split into:
  - 14,000 cards in train/
  - 1,796 cards in val/
  - 1,796 cards in test/
```

❌ **This doesn't work because:**
- Each image is a SPECIFIC card that must be recognized
- If a card is only in val/test, the model never learns it
- During inference, we need to recognize ALL 17,592 cards
- These aren't interchangeable examples of classes

## The CORRECT Approach

**For Pokemon card recognition:**

```
All 17,592 cards → MUST be in training
  - train/: 17,592 cards (ALL cards - REQUIRED)
  - val/:   17,592 cards (COPY for validation metrics)
  - test/:  17,592 cards (COPY for final evaluation)
```

✅ **Why this works:**
- Model trains on ALL cards (can recognize everything)
- Validation set measures training progress (same data, different metrics)
- Test set provides final performance evaluation
- No card is "held out" from learning

## What Are We Actually Validating?

**Not generalization to unseen cards** (that's impossible - we need to recognize specific cards)

**Instead, we're validating:**
1. **Convergence**: Is the loss decreasing properly?
2. **Overfitting detection**: Is training loss << validation loss?
3. **Embedding quality**: Are similar cards clustered together?
4. **Confidence calibration**: Are high confidence predictions actually correct?

## Implementation Strategy

### Option 1: Symlinks (Recommended - No Space Waste)
```bash
# Train has the actual data (12.5 GB)
data/processed/classification/train/

# Val and test use symlinks (negligible space)
data/processed/classification/val/ → symlinks to train/
data/processed/classification/test/ → symlinks to train/
```

### Option 2: Hardlinks (Alternative)
```bash
# All three share the same underlying files
# No extra space used, same file on disk
ln train/card_id/image.png val/card_id/image.png
```

### Option 3: Actual Copies (Only if filesystem doesn't support links)
```bash
# Duplicates data (37.5 GB total instead of 12.5 GB)
cp -r train/ val/
cp -r train/ test/
```

## Current State

```
data/processed/classification/
├── train/     # 17,592 cards ✅
├── val/       # 0 cards ❌
└── test/      # 0 cards ❌
```

**Status**: Only train/ exists with all data

## Recommended Action

**Create symlinks for val and test:**

```bash
cd data/processed/classification/

# Create val directory structure
mkdir -p val
cd train
for card_dir in */; do
    card_id="${card_dir%/}"
    mkdir -p "../val/$card_id"

    # Symlink all images
    for img in "$card_dir"*.{png,jpg,jpeg} 2>/dev/null; do
        [ -f "$img" ] && ln -s "../../train/$img" "../val/$card_id/"
    done
done

# Repeat for test/
# (same process)
```

**Result:**
- train/: 17,592 cards (12.5 GB)
- val/: 17,592 cards (symlinks, ~0 GB)
- test/: 17,592 cards (symlinks, ~0 GB)
- Total space: 12.5 GB (no duplication)

## Training Workflow

```python
# During training
for epoch in range(num_epochs):
    # Train on all cards
    train_loss = train_epoch(train_loader)

    # Validate on same cards (different metrics)
    val_loss = validate_epoch(val_loader)
    val_accuracy = compute_accuracy(val_loader)
    val_embedding_quality = compute_embedding_quality(val_loader)

    # Check for overfitting
    if val_loss > train_loss * 1.5:
        print("Warning: Possible overfitting detected")

    # Save checkpoint if val metrics improve
    if val_accuracy > best_accuracy:
        save_checkpoint(model)
```

## Why This Makes Sense

**Think of it like learning faces:**
- You need to learn ALL faces you'll encounter
- Can't hold out some faces for "validation"
- But you still want to track if you're memorizing vs learning features
- Validation set helps monitor this on the same data

**For Pokemon cards:**
- Need to learn ALL 17,592 specific cards
- Can't hold out cards from training
- But want to track embedding quality, confidence calibration
- Validation/test sets enable this monitoring

## Metrics to Track

**Training Metrics:**
- Cross-entropy loss
- Triplet loss
- Learning rate

**Validation Metrics:**
- Top-1 accuracy (exact match)
- Top-5 accuracy (in top 5 matches)
- Mean reciprocal rank (MRR)
- Embedding cluster quality
- Intra-class distance (for duplicate variants)
- Inter-class distance (between different cards)

**Test Metrics (final evaluation):**
- Same as validation, but run once at the end
- Used for final model selection
- Reported in papers/documentation

## Summary

✅ **Correct approach:**
- All 17,592 cards in train/ (required for recognition)
- Copy/symlink to val/ and test/ for metrics
- No cards held out from learning

❌ **Wrong approach:**
- Split cards into train/val/test
- Model can't recognize cards it never saw
- Defeats the purpose of the system

**The goal is to recognize specific cards, not generalize to unseen cards.**
