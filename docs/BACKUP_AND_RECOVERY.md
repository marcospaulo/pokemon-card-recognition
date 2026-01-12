# Pokemon Card Classification - Backup & Recovery Guide

**Last Updated:** 2026-01-07

## Critical Data Locations

### 1. Source of Truth: Original Card Images
**Location:** `/Users/marcos/dev/raspberry-pi/PokeTCG_downloader/assets/card_images/`
- **Total Files:** 17,455 card images (*_high.png)
- **Purpose:** Master source for all training data
- **Recovery:** Can regenerate entire dataset from these images
- **DO NOT DELETE** this directory without multiple verified backups

### 2. Classification Dataset (Training-Ready Structure)
**Primary Location:** `/Users/marcos/dev/raspberry-pi/pokemon_classification_dataset/`
- **Total Classes:** 16,469 unique Pokemon cards
- **Structure:**
  - `train/` - 13,175 classes (13,934 images)
  - `val/` - 2,470 classes (2,649 images)
  - `test/` - 824 classes (872 images)
  - `card_metadata.json` - Card information
  - `class_index.json` - Class-to-index mapping

**How to Regenerate:**
```bash
cd /Users/marcos/dev/raspberry-pi/training_prep
python3 prepare_classification_dataset.py
```

### 3. AWS S3 Bucket (Training Input)
**Bucket:** `pokemon-card-training-us-east-2`
**Region:** `us-east-2` (Ohio)
**Path:** `s3://pokemon-card-training-us-east-2/classification_dataset/`
- **Purpose:** Training data for SageMaker jobs
- **Important:** ALL training must use us-east-2 region

**How to Upload:**
```bash
aws s3 sync /Users/marcos/dev/raspberry-pi/pokemon_classification_dataset/ \
  s3://pokemon-card-training-us-east-2/classification_dataset/ \
  --region us-east-2 \
  --delete
```

**Verification:**
```bash
# Should return 17592
aws s3 ls s3://pokemon-card-training-us-east-2/classification_dataset/train/ --region us-east-2 | wc -l
```

### 4. Cloudflare R2 Backup (Long-term Storage)
**Bucket:** `pokemon-cards`
**Endpoint:** `https://5e76b23340eb966f6fdfa8d687df8f59.r2.cloudflarestorage.com`
**Access Key ID:** `b05c3c9c1d6e049d27cbe4edc0f4e6b6`
**Status:** Currently has SSL issues, needs fixing

### 5. PostgreSQL Database (Card Metadata)
**Database:** seek-apps PostgreSQL
**Table:** `pokemon_cards`
**Records:** 19,783 unique cards
- Contains full metadata but NOT images
- Can help verify card counts and metadata

## Recovery Procedures

### Scenario 1: Lost S3 Bucket
**Steps:**
1. Verify local dataset exists: `ls /Users/marcos/dev/raspberry-pi/pokemon_classification_dataset/train/ | wc -l` (should be 17592)
2. If local dataset missing, regenerate from source images:
   ```bash
   cd /Users/marcos/dev/raspberry-pi/training_prep
   python3 prepare_classification_dataset.py
   ```
3. Create new S3 bucket in us-east-2:
   ```bash
   aws s3 mb s3://pokemon-card-training-us-east-2 --region us-east-2
   ```
4. Upload complete dataset:
   ```bash
   aws s3 sync /Users/marcos/dev/raspberry-pi/pokemon_classification_dataset/ \
     s3://pokemon-card-training-us-east-2/classification_dataset/ \
     --region us-east-2
   ```
5. Verify: `aws s3 ls s3://pokemon-card-training-us-east-2/classification_dataset/train/ --region us-east-2 | wc -l`

### Scenario 2: Lost Local Dataset
**Steps:**
1. Verify source images exist: `find /Users/marcos/dev/raspberry-pi/PokeTCG_downloader/assets/card_images -name "*.png" | wc -l` (should be 17592)
2. Regenerate dataset:
   ```bash
   cd /Users/marcos/dev/raspberry-pi/training_prep
   python3 prepare_classification_dataset.py
   ```
3. Verify: `ls /Users/marcos/dev/raspberry-pi/pokemon_classification_dataset/train/ | wc -l` (should be 17592)

### Scenario 3: Lost Source Images
**Critical:** This is catastrophic - source images cannot be regenerated
**Recovery:**
1. Check Cloudflare R2 backup (if SSL fixed)
2. Download from AWS S3 us-east-2 (if still exists)
3. Last resort: Re-download all Pokemon cards from Pokemon TCG API

## Verification Checklist

Before any destructive operations (deletion, bucket migration, etc.):

- [ ] Verify local source images: 17,592 files
- [ ] Verify local classification dataset: 17,592 classes
- [ ] Verify S3 bucket has complete data: 17,592 train classes
- [ ] Create temporary backup if needed
- [ ] Test recovery procedure on sample data first

## Critical Rules

1. **NEVER** delete source data before verifying destination has everything
2. **ALWAYS** verify file counts match expected values
3. **ALWAYS** use us-east-2 region for AWS resources
4. **NEVER** delete `/Users/marcos/dev/raspberry-pi/PokeTCG_downloader/assets/card_images/`
5. **ALWAYS** keep local classification dataset as backup

## Current Status (2026-01-07)

- ✅ Source images: 17,455 files in `/Users/marcos/dev/raspberry-pi/PokeTCG_downloader/assets/card_images/`
- ✅ Local classification dataset: 16,469 classes in `/Users/marcos/dev/raspberry-pi/pokemon_classification_dataset/`
- 🔄 S3 us-east-2: Currently uploading (in progress)
- ❌ Cloudflare R2: Has SSL issues, needs fixing
- ✅ PostgreSQL: 19,783 cards metadata uploaded

## Mistakes to Avoid

1. ❌ **DON'T**: Delete us-west-2 bucket before confirming us-east-2 has everything
2. ❌ **DON'T**: Assume `aws s3 sync` completed successfully without verification
3. ❌ **DON'T**: Delete local backups until cloud backups are verified
4. ✅ **DO**: Always verify destination before deleting source
5. ✅ **DO**: Keep multiple backups (local + S3 + R2)
