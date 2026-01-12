# Data Integration Plan - Consolidate Everything Under Project Structure

## Current State Analysis

### What the Other Agent Created ✅
```
s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/
├── models/
│   ├── dinov3-teacher/v1.0/         # 5.6 GB - Teacher model
│   ├── efficientnet-student/
│   │   └── stage2/v2.0/              # 97.5 MB - PyTorch + ONNX
│   └── efficientnet-hailo/
│       ├── v2.1/                     # 13.8 MB - HEF file
│       └── calibration/              # EMPTY (placeholder)
├── experiments/mlflow/               # Experiment tracking
├── analytics/                        # Dashboards & metrics
├── pipelines/                        # Training/inference pipelines
├── profiling/                        # Model profiling data
├── metadata/                         # Project manifest
└── README.md                         # Documentation
```

### What I Uploaded (Outside Project) ⚠️
```
s3://pokemon-card-training-us-east-2/data/
├── raw/                              # 13 GB - 17,592 card images
├── processed/classification/         # 13 GB - train/val/test splits
├── calibration/                      # 734 MB - 1,024 Hailo calibration images
└── reference/                        # 106 MB - Embeddings + uSearch index
    ├── embeddings.npy                # 51.5 MB - 17,592 x 768 embeddings
    ├── usearch.index                 # 54.0 MB - Vector search index
    ├── index.json                    # 652 KB - Row → card_id mapping
    └── metadata.json                 # 543 KB - Card metadata
```

**Total**: 26.8 GB (52,994 files) currently scattered outside the project structure

---

## Problem

The data I uploaded is at the **root level** (`s3://.../data/`) instead of being organized under the **project structure** (`s3://.../project/pokemon-card-recognition/`).

This creates:
1. **Disorganization**: Data is in two places
2. **No clear ownership**: Hard to tell what belongs to which project
3. **Difficult cleanup**: Can't delete the project cleanly
4. **Missing integrations**: Calibration data not linked to Hailo model

---

## Proposed Integration

Move everything under the unified project structure:

### Target Structure
```
s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/
├── models/
│   ├── dinov3-teacher/v1.0/
│   │   └── model.tar.gz              # ✅ Already here (5.6 GB)
│   ├── efficientnet-student/stage2/v2.0/
│   │   ├── student_stage2_final.pt   # ✅ Already here (74.7 MB)
│   │   └── student_stage2_final.onnx # ✅ Already here (22.8 MB)
│   └── efficientnet-hailo/
│       ├── v2.1/
│       │   └── pokemon_student_efficientnet_lite0_stage2.hef  # ✅ Already here (13.8 MB)
│       └── calibration/              # ⬅️ MOVE: 734 MB (1,024 images)
│           └── [1,024 calibration images]
│
├── data/                             # ⬅️ CREATE NEW
│   ├── raw/                          # ⬅️ MOVE: 13 GB (17,592 images)
│   │   └── card_images/
│   │       └── [17,592 PNG files]
│   ├── processed/                    # ⬅️ MOVE: 13 GB (17,592 images)
│   │   └── classification/
│   │       ├── train/                # 17,592 card directories
│   │       ├── val/                  # (empty - can be generated)
│   │       └── test/                 # (empty - can be generated)
│   └── reference/                    # ⬅️ MOVE: 106 MB (embeddings + index)
│       ├── embeddings.npy            # 51.5 MB - For inference
│       ├── usearch.index             # 54.0 MB - Fast vector search
│       ├── index.json                # Card ID mapping
│       └── metadata.json             # Card metadata
│
├── experiments/mlflow/               # ✅ Already organized
├── analytics/                        # ✅ Already organized
├── pipelines/                        # ✅ Already organized
├── profiling/                        # ✅ Already organized
├── metadata/                         # ✅ Already organized
└── README.md                         # ✅ Already exists
```

---

## Migration Commands

### Option 1: Copy (Safe - Keep Originals)
```bash
# 1. Move calibration data to Hailo model directory
aws s3 sync s3://pokemon-card-training-us-east-2/data/calibration/ \
  s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-hailo/calibration/

# 2. Move raw data
aws s3 sync s3://pokemon-card-training-us-east-2/data/raw/ \
  s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/raw/

# 3. Move processed data
aws s3 sync s3://pokemon-card-training-us-east-2/data/processed/ \
  s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/processed/

# 4. Move reference database
aws s3 sync s3://pokemon-card-training-us-east-2/data/reference/ \
  s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/

# 5. Verify all files copied correctly
aws s3 ls s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/ --recursive --summarize

# 6. After verification, delete old data directory
aws s3 rm s3://pokemon-card-training-us-east-2/data/ --recursive
```

### Option 2: Move (Faster - No Duplication)
```bash
# Use AWS CLI with --no-copy-source-if-match to move efficiently
# This requires AWS CLI v2 with server-side move support
```

---

## Benefits of Integration

### 1. **Single Source of Truth**
All project assets in one place:
- Models: `project/pokemon-card-recognition/models/`
- Data: `project/pokemon-card-recognition/data/`
- Experiments: `project/pokemon-card-recognition/experiments/`
- Analytics: `project/pokemon-card-recognition/analytics/`

### 2. **Clear Model-Data Relationships**
- Calibration images linked to Hailo model
- Reference database linked to inference pipeline
- Training data linked to experiments

### 3. **Easy Project Management**
```bash
# Clone entire project
aws s3 sync s3://.../project/pokemon-card-recognition/ ./local-project/

# Delete entire project cleanly
aws s3 rm s3://.../project/pokemon-card-recognition/ --recursive

# Share project with team
# Just share one S3 path
```

### 4. **SageMaker Integration**
- Data Wrangler can reference `project/.../data/`
- Pipelines can reference `project/.../models/`
- MLFlow experiments stay organized

### 5. **Cost Optimization**
- Easier to apply lifecycle policies to the entire project
- Transition old data to Glacier as a unit
- Track costs per project (not scattered)

---

## Updated Project Manifest

After migration, update `metadata/project_manifest.json`:

```json
{
  "project": {
    "name": "pokemon-card-recognition",
    "version": "2.0.0",
    "data": {
      "raw": {
        "path": "data/raw/card_images/",
        "size_gb": 13.0,
        "num_files": 17592,
        "description": "Original high-resolution Pokemon card images"
      },
      "processed": {
        "path": "data/processed/classification/",
        "size_gb": 13.0,
        "num_files": 17592,
        "description": "Preprocessed training/validation/test splits"
      },
      "reference": {
        "path": "data/reference/",
        "size_mb": 106,
        "description": "Production inference database (embeddings + uSearch index)",
        "files": {
          "embeddings": "embeddings.npy",
          "index": "usearch.index",
          "mapping": "index.json",
          "metadata": "metadata.json"
        }
      },
      "calibration": {
        "path": "models/efficientnet-hailo/calibration/",
        "size_mb": 734,
        "num_files": 1024,
        "description": "Hailo compilation calibration dataset"
      }
    }
  }
}
```

---

## Estimated Migration Time

Based on S3 sync performance:
- **Calibration** (734 MB): ~30 seconds
- **Raw data** (13 GB): ~2-3 minutes
- **Processed data** (13 GB): ~2-3 minutes
- **Reference** (106 MB): ~10 seconds

**Total**: ~5-7 minutes for full migration

---

## Verification Checklist

After migration:
- [ ] Verify file counts match (52,994 files)
- [ ] Verify total size matches (26.8 GB)
- [ ] Test reference database access from new location
- [ ] Update any scripts/notebooks pointing to old paths
- [ ] Delete old `data/` directory after 24h verification period
- [ ] Update project manifest
- [ ] Update README.md with new data paths

---

## Final Project Size

After integration:
```
s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/
├── models/         6.5 GB  (teacher + student + hailo + calibration)
├── data/          26.1 GB  (raw + processed + reference)
├── experiments/      1 MB  (MLFlow metadata)
├── analytics/        1 MB  (dashboards)
├── metadata/        10 KB  (manifests)
└── README.md         3 KB

Total: ~32.6 GB (Everything in one place)
```

---

## Recommendation

**Execute Option 1 (Copy then Delete)** for safety:
1. Copy all data to project structure
2. Verify file counts and sizes match
3. Test reference database from new location
4. Wait 24 hours
5. Delete old `data/` directory

This ensures no data loss during migration.
