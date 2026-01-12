# AWS Organization Plan - Pokemon Card Recognition

## Current State (2026-01-11 - UPDATED 8:15 PM)

### S3 Buckets (Both in us-east-2 ✓)
```
pokemon-card-training-us-east-2/
├── models/
│   ├── embedding/teacher/    # DINOv3 (5.7 GB)
│   └── embedding/student/    # EfficientNet (412 MB)
└── data/                     # ✅ BACKUP COMPLETE (25.9 GiB, 52,956 files)
    ├── raw/                  # 17,572 card images (13 GB)
    ├── processed/            # 17,576 classification images (13 GB)
    ├── calibration/          # 1,024 Hailo calibration images (734 MB)
    └── reference/            # Reference database (106 MB)
        ├── embeddings.npy    # 51.5 MB - 17,592 x 768 embeddings
        ├── usearch.index     # 54.0 MB - Vector search index
        ├── index.json        # 652 KB - Row → card_id mapping
        └── metadata.json     # 543 KB - Card metadata

pokemon-card-training-east/
└── classification_dataset/   # 17,592 card images (old MobileNetV3 runs)
```

### SageMaker Resources
- ❌ No SageMaker Project
- ✅ Model Registry: "pokemon-card-recognition-models" (3 versions)
- ✅ Training Jobs: Multiple completed jobs
- ❌ No Data Pipeline configured

### Local Data (39 GB)
```
data/
├── calibration/      734 MB  # Hailo compilation calibration images (1,024 cards)
├── processed/         25 GB  # train/val/test splits (17,592 cards x 3 splits)
├── raw/               13 GB  # Original card images (17,592 cards)
└── reference/         21 MB  # Metadata JSON only
```

**Problem:** Only `pokemon-card-training-east` has the classification dataset backed up. The main bucket has NO data!

---

## Proposed Organization

### 1. Consolidate S3 Buckets (Single Source of Truth)

**Keep:** `s3://pokemon-card-training-us-east-2/` (primary bucket, us-east-2)
**Archive:** `pokemon-card-training-east` (redundant, can delete after migration)

**New Structure:**
```
s3://pokemon-card-training-us-east-2/
├── data/
│   ├── raw/                           # 13 GB - Original 17,592 card images
│   │   ├── card_images/              # Single source of truth
│   │   └── metadata/                 # Card metadata JSON
│   ├── processed/
│   │   ├── classification/           # 25 GB - train/val/test splits
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── detection/                # YOLO training data
│   ├── calibration/                  # 734 MB - Hailo calibration images
│   └── reference/                    # Embeddings database (TO BE CREATED)
│       ├── embeddings.npy            # 17,592 x 768 embeddings
│       ├── usearch.index             # Vector search index
│       ├── index.json                # Row -> card_id mapping
│       └── metadata.json             # Card details
│
├── models/
│   ├── detection/
│   │   └── yolo11n-obb/
│   │       ├── yolo11n-obb.pt        # PyTorch weights
│   │       └── yolo11n-obb.onnx      # ONNX export
│   ├── embedding/
│   │   ├── teacher/                  # DINOv3 (5.7 GB)
│   │   │   └── pokemon-card-dinov3-teacher-*/
│   │   │       └── output/model.tar.gz
│   │   └── student/                  # EfficientNet (412 MB + 14 MB HEF)
│   │       ├── pytorch-training-*/
│   │       │   └── output/model.tar.gz
│   │       ├── student_stage2_final.onnx
│   │       └── pokemon_student_efficientnet_lite0_stage2.hef
│   └── registry/                     # Organized by version
│       ├── v1-dinov3-teacher/
│       ├── v2-efficientnet-stage1/
│       └── v3-efficientnet-stage2-production/
│
└── pipelines/                        # SageMaker Pipeline definitions
    ├── training/
    │   ├── teacher_training.py
    │   └── student_training.py
    └── inference/
        └── batch_inference.py
```

### 2. Create SageMaker Project

**Project Name:** `pokemon-card-recognition`
**Purpose:** Centralize all ML workflows
**Components:**
- Data pipeline (import from S3)
- Training pipelines (teacher + student distillation)
- Model registry integration
- Batch inference for embeddings generation

**IAM Permissions Needed:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::pokemon-card-training-us-east-2",
        "arn:aws:s3:::pokemon-card-training-us-east-2/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:CreateProcessingJob",
        "sagemaker:CreateModel",
        "sagemaker:CreateModelPackage",
        "sagemaker:DescribeModelPackage"
      ],
      "Resource": "*"
    }
  ]
}
```

### 3. Set Up Data Pipeline

**SageMaker Data Wrangler Flow:**
```
Source: s3://pokemon-card-training-us-east-2/data/raw/card_images/
    ↓
Transform: Resize to 224x224, normalize, augment
    ↓
Output: s3://pokemon-card-training-us-east-2/data/processed/classification/
```

**Processing Job:**
```python
# scripts/prepare_sagemaker_dataset.py
from sagemaker.processing import ScriptProcessor

processor = ScriptProcessor(
    role=sagemaker_role,
    instance_type='ml.m5.xlarge',
    instance_count=1,
    base_job_name='pokemon-data-prep'
)

processor.run(
    code='scripts/prepare_classification_dataset.py',
    inputs=[
        ProcessingInput(
            source='s3://pokemon-card-training-us-east-2/data/raw/card_images/',
            destination='/opt/ml/processing/input'
        )
    ],
    outputs=[
        ProcessingOutput(
            source='/opt/ml/processing/output/train',
            destination='s3://pokemon-card-training-us-east-2/data/processed/classification/train/'
        ),
        ProcessingOutput(
            source='/opt/ml/processing/output/val',
            destination='s3://pokemon-card-training-us-east-2/data/processed/classification/val/'
        ),
        ProcessingOutput(
            source='/opt/ml/processing/output/test',
            destination='s3://pokemon-card-training-us-east-2/data/processed/classification/test/'
        )
    ]
)
```

---

## Local Data Cleanup Plan

### What to Keep Locally (14 GB)
```
✅ data/raw/card_images/              # 13 GB - Source of truth
✅ data/reference/                    # 21 MB - Metadata
✅ models/                            # 1.1 GB - Compiled models
```

### What to Delete (25 GB savings)
```
❌ data/processed/classification/     # 25 GB - Can regenerate from raw + backed up on S3
❌ data/calibration/                  # 734 MB - Already used, backed up on EC2
```

### Backup Before Deleting
```bash
# Backup processed data to S3
aws s3 sync data/processed/ s3://pokemon-card-training-us-east-2/data/processed/ \
  --exclude "*.DS_Store"

# Backup calibration data to S3
aws s3 sync data/calibration/ s3://pokemon-card-training-us-east-2/data/calibration/ \
  --exclude "*.DS_Store"

# Verify backup
aws s3 ls s3://pokemon-card-training-us-east-2/data/processed/ --recursive | wc -l
aws s3 ls s3://pokemon-card-training-us-east-2/data/calibration/ --recursive | wc -l

# Delete local copies
rm -rf data/processed/
rm -rf data/calibration/
```

**Result:** 39 GB → 14 GB (64% reduction)

---

## Implementation Steps

### Phase 1: S3 Organization ✅ COMPLETE
1. ✅ Verify both buckets in us-east-2
2. ✅ Upload raw data to S3 (13 GB, 17,572 files)
3. ✅ Upload processed data to S3 (13 GB, 17,576 files)
4. ✅ Upload calibration data to S3 (734 MB, 1,024 files)
5. ✅ Upload reference database to S3 (106 MB, 15 files)
6. ✅ Verify all data backed up (25.9 GiB total)
7. ⏳ Archive/delete `pokemon-card-training-east` bucket (pending)

### Phase 2: SageMaker Project Setup (1-2 hours)
1. Create SageMaker Project: `pokemon-card-recognition`
2. Grant IAM permissions for S3 access
3. Set up Data Wrangler flow
4. Create data processing pipeline
5. Link Model Registry to project

### Phase 3: Local Cleanup (15 min)
1. Verify S3 backups complete
2. Delete `data/processed/` (25 GB)
3. Delete `data/calibration/` (734 MB)
4. Keep `data/raw/` as local source of truth

### Phase 4: Generate Reference Embeddings (2-3 hours)
1. Extract PyTorch weights from model.tar.gz
2. Run `build_usearch_index.py` on all 17,592 cards
3. Upload embeddings to S3
4. Deploy to Raspberry Pi

---

## Migration Commands

```bash
# 1. Backup everything to S3
aws s3 sync data/raw/ s3://pokemon-card-training-us-east-2/data/raw/ --exclude "*.DS_Store"
aws s3 sync data/processed/ s3://pokemon-card-training-us-east-2/data/processed/ --exclude "*.DS_Store"
aws s3 sync data/calibration/ s3://pokemon-card-training-us-east-2/data/calibration/ --exclude "*.DS_Store"

# 2. Verify backup sizes match
du -sh data/raw/ && aws s3 ls s3://pokemon-card-training-us-east-2/data/raw/ --recursive --human-readable --summarize
du -sh data/processed/ && aws s3 ls s3://pokemon-card-training-us-east-2/data/processed/ --recursive --human-readable --summarize

# 3. Clean up local storage
rm -rf data/processed/
rm -rf data/calibration/

# 4. Create SageMaker Project
aws sagemaker create-project \
  --project-name pokemon-card-recognition \
  --project-description "Pokemon card detection and recognition ML pipeline" \
  --service-catalog-provisioning-details '{"ProductId":"prod-xxxxxxxxx"}'
```

---

## Final State

### S3 (Single Bucket)
- `pokemon-card-training-us-east-2`: 40+ GB organized data

### SageMaker
- Project: `pokemon-card-recognition`
- Model Registry: Linked to project
- Pipelines: Training + inference

### Local (Minimal)
- 14 GB essential files only
- Can regenerate everything from S3

---

## Benefits

1. **Single Source of Truth:** All data in one S3 bucket
2. **Organized:** Clear structure for data/models/pipelines
3. **Efficient:** 64% local storage reduction
4. **Scalable:** SageMaker pipelines for automation
5. **Recoverable:** Everything backed up on S3
6. **Cost-Effective:** Reduced local storage costs
