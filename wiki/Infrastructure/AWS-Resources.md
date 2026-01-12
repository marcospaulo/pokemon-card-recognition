# AWS Resources

> **Account**: 943271038849
> **Region**: us-east-2 (Ohio)
> **Total Storage**: 31.7 GB on S3
> **Training Cost**: $2.80 (one-time)

[← Back to Wiki Home](../Home.md)

---

## Overview

All training data, models, and reference databases are stored on AWS S3. Training was performed on AWS SageMaker. This page documents the **actual AWS resources** currently deployed.

---

## S3 Storage

### Main Bucket

**Bucket**: `pokemon-card-training-us-east-2`
**Region**: us-east-2
**Total Size**: 31.7 GB
**Versioning**: Disabled
**Encryption**: AES-256 (server-side)

### Directory Structure

```
s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/

├── data/                                    # 25.2 GB
│   ├── raw/
│   │   └── card_images/                     # 12.6 GB (17,592 PNG files)
│   │
│   ├── processed/
│   │   ├── classification/                  # 12.5 GB
│   │   │   ├── train/                       # 17,592 card directories
│   │   │   ├── val/                         # Empty (not needed)
│   │   │   └── test/                        # Empty (not needed)
│   │   │
│   │   └── detection/                       # (future YOLO dataset)
│   │
│   └── reference/                           # 128 MB
│       ├── embeddings.npy                   # 52 MB - [17592, 768] embeddings
│       ├── usearch.index                    # 55 MB - HNSW vector index
│       ├── index.json                       # 374 KB - row → card_id mapping
│       └── metadata.json                    # 4.8 MB - 15,987 card details
│
├── models/                                  # 6.5 GB
│   ├── dinov3-teacher/v1.0/
│   │   └── dinov3_vit_base_patch14.pth      # 5.6 GB (teacher model)
│   │
│   ├── efficientnet-student/stage2/v2.0/
│   │   ├── efficientnet_lite0_student.pt    # 75 MB (PyTorch checkpoint)
│   │   └── efficientnet_lite0_student.onnx  # 23 MB (ONNX export)
│   │
│   └── efficientnet-hailo/
│       ├── pokemon_student_efficientnet_lite0_stage2.hef  # 14 MB (Hailo)
│       └── calibration/                     # 734 MB (1,024 cal images)
│
├── experiments/                             # Training artifacts
├── pipelines/                               # SageMaker pipelines
└── README.md                                # Project documentation
```

---

## Storage Breakdown

| Category | Size | Files | Purpose |
|----------|------|-------|---------|
| **Raw Images** | 12.6 GB | 17,592 | Original card scans |
| **Processed Data** | 12.5 GB | ~17,592 | Training-ready datasets |
| **Teacher Model** | 5.6 GB | 1 | DINOv2 for distillation |
| **Student Models** | 112 MB | 3 | PyTorch, ONNX, HEF |
| **Reference DB** | 128 MB | 4 | Production embeddings |
| **Calibration** | 734 MB | 1,024 | Hailo quantization |
| **Misc** | ~20 MB | - | Configs, logs |
| **TOTAL** | **31.7 GB** | **~36,000** | - |

---

## Download Commands

### Download Reference Database (Required for Inference)

```bash
# Reference database - 128 MB (REQUIRED)
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/ \
  ./data/reference/
```

**Contents**:
- `embeddings.npy` - 17,592 card embeddings
- `usearch.index` - Fast vector search index
- `index.json` - Card ID mapping
- `metadata.json` - Card names, sets, rarity

### Download Models

```bash
# Hailo HEF model - 14 MB (for Raspberry Pi)
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-hailo/pokemon_student_efficientnet_lite0_stage2.hef \
  ./models/embedding/

# ONNX model - 23 MB (for testing)
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-student/stage2/v2.0/efficientnet_lite0_student.onnx \
  ./models/

# PyTorch checkpoint - 75 MB (for retraining)
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-student/stage2/v2.0/efficientnet_lite0_student.pt \
  ./models/
```

### Download Raw Data (Optional - for Development)

```bash
# Raw card images - 12.6 GB
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/raw/ \
  ./data/raw/

# Processed training data - 12.5 GB
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/processed/ \
  ./data/processed/
```

---

## IAM Configuration

### Users

| User | Created | Purpose | Access Level |
|------|---------|---------|--------------|
| **marcospaulo** | - | Main account | Admin (root-level) |
| **raspberry-pi-user** | Jan 11, 2026 | Raspberry Pi AWS access | Admin (programmatic) |

### Raspberry Pi User Credentials

**User**: `raspberry-pi-user`
**ARN**: `arn:aws:iam::943271038849:user/raspberry-pi-user`
**Access Key ID**: `[Stored in ~/.aws/credentials on Pi]`
**Permissions**: AmazonS3ReadOnlyAccess policy

**Purpose**: Allows Raspberry Pi to download data from S3 without browser login

**Configured on Pi**:
```bash
# Credentials stored at
~/.aws/credentials
~/.aws/config

# Verify
aws sts get-caller-identity
```

---

## SageMaker

### Training Jobs

**Most Recent Job**:
- **Name**: `pokemon-efficientnet-distillation-2026-01-10`
- **Instance**: ml.g4dn.xlarge (NVIDIA T4, 16GB VRAM)
- **Duration**: 3.8 hours
- **Cost**: $2.80
- **Status**: ✅ Completed
- **Output**: EfficientNet-Lite0 v2.0 (96.8% accuracy)

### Model Registry

**Package Group**: `pokemon-card-embedding`

| Version | Date | Accuracy | Status |
|---------|------|----------|--------|
| v1.0 | Dec 2025 | 89.3% | Archived |
| v2.0 | Jan 2026 | 96.8% | ✅ Approved (deployed) |

**Current Deployed**: v2.0

---

## AWS CLI Setup

### Installation

```bash
# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

### Configuration

```bash
# Configure with raspberry-pi-user credentials
aws configure

# Enter the credentials you received from AWS admin:
AWS Access Key ID: [provided by AWS admin]
AWS Secret Access Key: [provided by AWS admin]
Default region: us-east-2
Default output format: json
```

### Verification

```bash
# Test access
aws sts get-caller-identity

# Expected output:
# {
#   "UserId": "AIDA5XH2VZ6AR3ZLAQYZF",
#   "Account": "943271038849",
#   "Arn": "arn:aws:iam::943271038849:user/raspberry-pi-user"
# }

# Test S3 access
aws s3 ls s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/
```

---

## Cost Analysis

### One-Time Costs

| Service | Usage | Cost |
|---------|-------|------|
| **SageMaker Training** | 3.8 hours on ml.g4dn.xlarge | $2.80 |
| **S3 Storage (setup)** | 31.7 GB upload | ~$0.10 |
| **Data Transfer** | Minimal (within region) | ~$0.05 |
| **TOTAL** | - | **$2.95** |

### Monthly Costs

| Service | Usage | Cost/Month |
|---------|-------|------------|
| **S3 Storage** | 31.7 GB | **$0.73** |
| **S3 Requests** | Minimal (downloads) | <$0.01 |
| **Data Transfer OUT** | ~1 GB (to Pi) | $0.09 |
| **TOTAL** | - | **$0.82/month** |

**Annual**: ~$9.84/year for storage

### Cost Optimization

**Current status**: ✅ Already optimized
- Training done once ($2.80), no recurring cost
- S3 storage minimal (~$0.73/month)
- No EC2 instances (runs on Raspberry Pi)
- No RDS (local SQLite/JSON)

**If storage costs become an issue**:
- Move to S3 Glacier: $0.73 → $0.13/month (82% savings)
- Delete intermediate artifacts: 31.7 GB → ~15 GB
- Archive old training checkpoints

---

## Access URLs

### S3 Console

**Bucket**: https://s3.console.aws.amazon.com/s3/buckets/pokemon-card-training-us-east-2?region=us-east-2

**Reference DB**: https://s3.console.aws.amazon.com/s3/buckets/pokemon-card-training-us-east-2?prefix=project/pokemon-card-recognition/data/reference/

**Models**: https://s3.console.aws.amazon.com/s3/buckets/pokemon-card-training-us-east-2?prefix=project/pokemon-card-recognition/models/

### SageMaker Console

**Training Jobs**: https://console.aws.amazon.com/sagemaker/home?region=us-east-2#/jobs

**Model Registry**: https://console.aws.amazon.com/sagemaker/home?region=us-east-2#/model-packages

### IAM Console

**Users**: https://console.aws.amazon.com/iam/home#/users

**raspberry-pi-user**: https://console.aws.amazon.com/iam/home#/users/details/raspberry-pi-user

---

## Data Versioning

### Current Versions

| Dataset/Model | Version | Date | Status |
|---------------|---------|------|--------|
| Raw images | v1.0 | Dec 2025 | Stable |
| Reference DB | v2.0 | Jan 11, 2026 | ✅ Current |
| EfficientNet student | v2.0 | Jan 10, 2026 | ✅ Deployed |
| Metadata | v2.1 | Jan 11, 2026 | ✅ Fixed (15,987 cards) |

### Change Log

**v2.1 (Jan 11, 2026)** - Metadata fix
- Fixed metadata.json (was 7 cards, now 15,987)
- Fixed index.json (now maps to card_ids correctly)
- All 17,592 embeddings have valid metadata

**v2.0 (Jan 10, 2026)** - Model upgrade
- EfficientNet-Lite0 with DINOv2 distillation
- 96.8% accuracy (up from 89.3%)
- Deployed to Raspberry Pi

---

## Backup Strategy

### What's Backed Up

✅ **On S3** (primary storage):
- Raw card images (12.6 GB)
- Processed datasets (12.5 GB)
- All model checkpoints (6.5 GB)
- Reference database (128 MB)

✅ **On GitHub** (code only):
- Source code
- Configuration files
- Documentation (wiki)

✅ **On Raspberry Pi** (deployment):
- HEF model (14 MB)
- Reference database (111 MB)
- Latest code (git)

### Recovery Procedures

**Scenario 1: Raspberry Pi failure**
```bash
# Flash new SD card, install OS
# Clone repo
git clone git@github.com:marcospaulo/pokemon-card-recognition.git
# Download data
aws s3 sync s3://.../data/reference/ ./data/reference/
aws s3 cp s3://.../models/.../model.hef ./models/embedding/
```

**Scenario 2: S3 data loss** (unlikely)
```bash
# Re-upload from local backup
aws s3 sync ./data/ s3://.../data/
```

---

## Security

### S3 Bucket Policy

- **Public access**: Blocked
- **Encryption**: AES-256 (at rest)
- **Versioning**: Disabled (to save costs)
- **Access**: IAM users only

### IAM Best Practices

- ✅ Separate user for Raspberry Pi (not root account)
- ✅ Access keys rotated if compromised
- ✅ No hardcoded credentials in code
- ✅ Credentials stored in `~/.aws/` (not in repo)

---

## Related Documentation

- **[Raspberry Pi Setup](../Deployment/Raspberry-Pi-Setup.md)** - How to download data
- **[Dataset Structure](../Reference/Dataset.md)** - Data organization
- **[Training Guide](../Development/Training.md)** - SageMaker training

---

*Last updated: January 11, 2026*
*Account: 943271038849*
*Region: us-east-2*
