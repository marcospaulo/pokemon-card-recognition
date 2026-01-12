# AWS Organization

Complete AWS infrastructure setup including S3, SageMaker, and IAM configuration.

---

## Overview

All project resources are organized in AWS **us-east-2** region under the `pokemon-card-training-us-east-2` S3 bucket with a unified project structure.

**AWS Account:** marcospaulo (943271038849)
**Primary Bucket:** `s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/`
**Total Size:** 31.7 GB (53,068 objects)

---

## S3 Bucket Structure

```
s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/
│
├── metadata/
│   └── project_manifest.json          # Project metadata and versioning
│
├── models/                             # 5.7 GB total
│   ├── dinov3-teacher/v1.0/
│   │   └── model.tar.gz               # 5.6 GB - Teacher model (86M params)
│   │
│   ├── efficientnet-student/
│   │   └── stage2/v2.0/
│   │       ├── student_stage2_final.pt      # 75 MB - PyTorch weights
│   │       └── student_stage2_final.onnx    # 23 MB - ONNX export
│   │
│   └── efficientnet-hailo/
│       ├── v2.1/
│       │   └── pokemon_student_efficientnet_lite0_stage2.hef  # 13.8 MB
│       └── calibration/                # 734 MB - 1,024 calibration images
│
├── data/                               # 25.2 GB total (51,970 files)
│   ├── raw/                            # 13 GB
│   │   └── card_images/                # 17,592 original PNG files
│   │
│   ├── processed/                      # 13 GB
│   │   └── classification/             # 17,592 processed training images
│   │
│   ├── calibration/                    # See models/efficientnet-hailo/calibration/
│   │
│   └── reference/                      # 106 MB - Production inference database
│       ├── embeddings.npy              # 51.5 MB - 17,592 x 768 embeddings
│       ├── usearch.index               # 54.0 MB - ARM-optimized vector search
│       ├── index.json                  # 652 KB - Row → card_id mapping
│       └── metadata.json               # 543 KB - Card metadata
│
├── experiments/mlflow/                 # MLFlow experiment tracking
│   └── experiments_index.json
│
├── profiling/                          # 117 MB - SageMaker Profiler outputs
│   ├── teacher/2026-01-10/            # 44.3 MB
│   └── student_stage2/2026-01-11/     # 72.8 MB
│
├── analytics/                          # ~2 MB
│   ├── dashboards/
│   │   └── dashboard_config.json
│   └── metrics/
│       ├── model_performance.csv
│       ├── compression_metrics.csv
│       ├── cost_breakdown.csv
│       ├── model_lineage.json
│       ├── storage_metrics.csv
│       └── summary.json
│
└── pipelines/                          # Future: CI/CD pipelines
    ├── training/
    ├── inference/
    └── deployment/
```

---

## Data Migration History

### Initial State (2026-01-11 Morning)
Data was scattered across multiple locations:
- Root-level `data/` directory
- Multiple duplicates
- Unorganized structure

### Migration Process
**Goal:** Consolidate everything under unified project structure

**Actions Taken:**
1. Created organized project structure at `project/pokemon-card-recognition/`
2. Server-side S3 sync (no local bandwidth usage)
3. Migrated 25.2 GiB (51,970 files) in 4 parallel operations:
   - Raw card images: 13 GB (17,592 files)
   - Processed data: 13 GB (17,592 files)
   - Calibration data: 734 MB (1,024 files)
   - Reference database: 106 MB (15 files)
4. Verified migration success
5. Cleaned up old root-level data

**Result:** ✅ Clean, organized, single source of truth

See **[Data Integration](../Project-History/Data-Integration.md)** for detailed migration story.

---

## S3 Access Patterns

### Quick Access Links

**AWS Console:**
- [S3 Project Root](https://s3.console.aws.amazon.com/s3/buckets/pokemon-card-training-us-east-2?region=us-east-2&prefix=project/pokemon-card-recognition/)
- [Models Directory](https://s3.console.aws.amazon.com/s3/buckets/pokemon-card-training-us-east-2?prefix=project/pokemon-card-recognition/models/)
- [Data Directory](https://s3.console.aws.amazon.com/s3/buckets/pokemon-card-training-us-east-2?prefix=project/pokemon-card-recognition/data/)

### AWS CLI Commands

```bash
# List project structure
aws s3 ls s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/ --recursive --human-readable

# Download reference database (required for inference)
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/ ./data/reference/

# Download Hailo model for Raspberry Pi
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-hailo/v2.1/pokemon_student_efficientnet_lite0_stage2.hef ./

# Download all models
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/ ./models/

# Download raw training data (warning: 13 GB)
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/raw/ ./data/raw/
```

---

## SageMaker Integration

### Model Registry

**Model Package Group:** `pokemon-card-recognition-models`

**Registered Models:**
1. **Teacher Model (v4)** - DINOv3 ViT-B/14
   - ARN: `arn:aws:sagemaker:us-east-2:943271038849:model-package/pokemon-card-recognition-models/4`
   - Size: 5.6 GB
   - Parameters: 86M

2. **Student Model (v5)** - EfficientNet-Lite0 Stage 2
   - ARN: `arn:aws:sagemaker:us-east-2:943271038849:model-package/pokemon-card-recognition-models/5`
   - Size: 75 MB
   - Parameters: 4.7M

**Console Access:**
- [Model Package Group](https://console.aws.amazon.com/sagemaker/home?region=us-east-2#/model-package-groups/pokemon-card-recognition-models)
- [Training Jobs](https://console.aws.amazon.com/sagemaker/home?region=us-east-2#/jobs)

### Training Jobs History

| Job Name | Date | Instance | Duration | Cost |
|----------|------|----------|----------|------|
| dinov3-teacher-2026-01-10 | 2026-01-10 | ml.g5.2xlarge | 3.5h | $5.32 |
| efficientnet-stage1-2026-01-10 | 2026-01-10 | ml.g5.2xlarge | 2h | $3.04 |
| efficientnet-stage2-2026-01-11 | 2026-01-11 | ml.g5.2xlarge | 2h | $3.04 |

**Total Training Cost:** $11.40

---

## IAM Configuration

### Service Role

**Role Name:** `SageMaker-MarcosAdmin-ExecutionRole`
**ARN:** `arn:aws:iam::943271038849:role/SageMaker-MarcosAdmin-ExecutionRole`

**Attached Policies:**
- AmazonSageMakerFullAccess
- AmazonS3FullAccess
- IAMFullAccess
- CloudWatchFullAccess
- AWSCloudFormationFullAccess

**Capabilities:**
- ✅ Create/manage SageMaker resources
- ✅ Full S3 bucket access
- ✅ CloudWatch logging and monitoring
- ✅ Model Registry management
- ✅ Training job execution

**Trust Policy:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

See **[Access Control](Access-Control.md)** for detailed IAM documentation.

---

## Storage Costs

### Current Usage (2026-01-11)

| Component | Size | Monthly Cost | Annual Cost |
|-----------|------|--------------|-------------|
| **Models** | 5.7 GB | $0.13 | $1.56 |
| **Data** | 25.2 GB | $0.58 | $6.96 |
| **Profiling** | 117 MB | $0.003 | $0.04 |
| **Analytics** | 2 MB | $0.0001 | $0.001 |
| **Total** | **31.7 GB** | **$0.73** | **$8.76** |

**Pricing:** $0.023/GB/month (S3 Standard in us-east-2)

### Cost Optimization

**Lifecycle Policies (Planned):**
- Move profiling data to Glacier after 90 days
- Archive old training checkpoints
- **Potential Savings:** ~40% (~$3.50/year)

See **[Cost Analysis](Cost-Analysis.md)** for detailed breakdown.

---

## Data Integrity

### Verification Checklist

```bash
# Verify raw data count
aws s3 ls s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/raw/card_images/ --recursive | wc -l
# Expected: 17,592

# Verify processed data
aws s3 ls s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/processed/classification/ --recursive | grep -c "/$"
# Expected: 17,592

# Verify reference database files
aws s3 ls s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/
# Expected: embeddings.npy, usearch.index, index.json, metadata.json

# Check model files
aws s3 ls s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/ --recursive --human-readable
# Expected: All model versions present
```

---

## Backup and Recovery

### Current State
- ✅ All data on S3 (99.999999999% durability)
- ✅ Versioning enabled on critical files
- ✅ Cross-region replication: NOT enabled (single region sufficient)

### Recovery Procedures

**If local data is lost:**
```bash
# Restore complete project
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/ ./pokemon-card-recognition/
```

**If specific component is lost:**
```bash
# Restore models only
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/ ./models/

# Restore reference database only
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/ ./data/reference/
```

**If S3 bucket is accidentally deleted:**
- Contact AWS Support immediately (within 30 days for recovery)
- Recent backups may be available in AWS backups
- Local copies on development machine can be re-uploaded

---

## Monitoring

### CloudWatch Integration

**Log Groups:**
- `/aws/sagemaker/TrainingJobs` - Training job logs
- `/aws/sagemaker/Endpoints` - Inference endpoint logs (if deployed)

**Metrics:**
- Training job CPU/GPU utilization
- S3 bucket size and object count
- Data transfer metrics

**Access:**
- [CloudWatch Console](https://console.aws.amazon.com/cloudwatch/home?region=us-east-2)

---

## Best Practices

### When Adding New Data
```bash
# Always upload to organized structure
aws s3 cp new_model.hef s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/new-model/v1.0/

# Use sync for directories
aws s3 sync ./new_data/ s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/new_data/
```

### When Downloading Data
```bash
# Use sync instead of cp for directories (incremental)
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/ ./data/reference/

# Use --dryrun to preview changes
aws s3 sync s3://bucket/path/ ./local/ --dryrun
```

### When Cleaning Up
```bash
# NEVER delete from S3 without local backup
# First download, then verify, then delete

# Wrong:
aws s3 rm s3://bucket/data/ --recursive  # ❌ Risky!

# Right:
aws s3 sync s3://bucket/data/ ./backup/   # ✅ Backup first
# Verify backup is complete
aws s3 rm s3://bucket/data/ --recursive   # ✅ Then delete
```

---

## Related Documentation

- **[Data Management](S3-Data-Management.md)** - Detailed S3 structure
- **[Access Control](Access-Control.md)** - IAM roles and permissions
- **[Cost Analysis](Cost-Analysis.md)** - Pricing breakdown
- **[Organization Journey](../Project-History/Organization-Journey.md)** - How we organized everything

---

**Last Updated:** 2026-01-11
