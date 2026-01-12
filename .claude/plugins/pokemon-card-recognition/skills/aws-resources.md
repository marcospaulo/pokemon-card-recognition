---
skill_name: pokemon-card-recognition:aws-resources
description: Detailed AWS resources, S3 structure, IAM permissions, and access configuration
tags: [aws, s3, sagemaker, iam, access]
---

# AWS Resources & Access Configuration

## Account Details

- **Account ID:** 943271038849
- **Region:** us-east-2 (Ohio)
- **Owner:** Marcos Paulo (marcospaulo)
- **Service Account:** `SageMaker-MarcosAdmin-ExecutionRole`
- **Service Account ARN:** `arn:aws:iam::943271038849:role/SageMaker-MarcosAdmin-ExecutionRole`

---

## S3 Bucket Structure

### Main Bucket
```
s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/
```

### Complete Directory Layout

```
project/pokemon-card-recognition/
│
├── README.md                          # Project overview
│
├── metadata/
│   └── project_manifest.json          # v1.1.0 - Complete project metadata
│
├── models/                            # 5.7 GB total
│   ├── dinov3-teacher/v1.0/
│   │   └── model.tar.gz              # 5.6 GB - SageMaker model package
│   │
│   ├── efficientnet-student/
│   │   ├── stage1/v1.0/              # (transitional - not preserved)
│   │   └── stage2/v2.0/
│   │       ├── student_stage2_final.pt      # 74.7 MB - PyTorch checkpoint
│   │       └── student_stage2_final.onnx    # 22.8 MB - ONNX export
│   │
│   └── efficientnet-hailo/v2.1/
│       └── pokemon_student_efficientnet_lite0_stage2.hef  # 13.8 MB - Hailo NPU
│
├── data/                             # 26.7 GB total
│   ├── raw/                          # 13 GB
│   │   └── card_images/              # 17,592 original card images
│   │
│   ├── processed/                    # 13 GB
│   │   └── classification/           # 17,592 processed for training
│   │
│   ├── calibration/                  # 734 MB
│   │   └── [1,024 images]            # For Hailo INT8 calibration
│   │
│   └── reference/                    # 106 MB - Production inference database
│       ├── embeddings.npy            # 51.5 MB - 17,592 x 768 embeddings
│       ├── usearch.index             # 54.0 MB - ARM-optimized vector index
│       ├── index.json                # 652 KB - Row → card_id mapping
│       └── metadata.json             # 543 KB - Card details (name, set, etc.)
│
├── experiments/mlflow/               # MLFlow tracking
│   └── experiments_index.json
│
├── profiling/                        # 117 MB - SageMaker Profiler outputs
│   ├── teacher/2026-01-10/           # 44.3 MB - Teacher training metrics
│   └── student_stage2/2026-01-11/    # 72.8 MB - Student training metrics
│
├── analytics/                        # ~2 MB
│   ├── dashboards/
│   │   └── dashboard_config.json
│   └── metrics/
│       ├── model_performance.csv     # Model comparison metrics
│       ├── compression_metrics.csv   # Compression ratios
│       ├── cost_breakdown.csv        # Detailed costs
│       ├── model_lineage.json        # Parent-child relationships
│       ├── storage_metrics.csv       # Storage by component
│       └── summary.json              # Overall summary
│
└── pipelines/                        # Future automation
    ├── training/
    ├── inference/
    └── deployment/
```

### Storage Breakdown
- **Total:** ~31.7 GB (53,068 objects)
- **Models:** 5.7 GB
- **Data:** 26.7 GB (raw + processed + calibration + reference)
- **Profiling:** 117 MB
- **Analytics:** ~2 MB
- **Metadata:** ~10 MB

---

## SageMaker Resources

### Model Registry

**Model Package Group:**
```
Name: pokemon-card-recognition-models
ARN: arn:aws:sagemaker:us-east-2:943271038849:model-package-group/pokemon-card-recognition-models
Status: Active
```

**Registered Models:**
```
1. Model #4: DINOv3 Teacher
   ARN: arn:aws:sagemaker:us-east-2:943271038849:model-package/pokemon-card-recognition-models/4
   Status: Approved
   Type: Teacher (embedding generation)

2. Model #5: EfficientNet Student Stage 2
   ARN: arn:aws:sagemaker:us-east-2:943271038849:model-package/pokemon-card-recognition-models/5
   Status: Approved
   Type: Student (compressed)
```

### SageMaker User Profile

```yaml
Profile Name: marcospaulo
Domain ID: d-slzqikvnlai2
Domain ARN: arn:aws:sagemaker:us-east-2:943271038849:domain/d-slzqikvnlai2
Status: InService
Execution Role: SageMaker-MarcosAdmin-ExecutionRole
```

**Studio URL:**
```
https://d-slzqikvnlai2.studio.us-east-2.sagemaker.aws
```

---

## IAM Service Account Permissions

### SageMaker-MarcosAdmin-ExecutionRole

**Role ARN:**
```
arn:aws:iam::943271038849:role/SageMaker-MarcosAdmin-ExecutionRole
```

**Description:**
"Full admin execution role for marcospaulo SageMaker Studio profile"

### Attached AWS Managed Policies (10)

1. **AmazonSageMakerFullAccess**
   - All SageMaker operations
   - Training jobs, endpoints, pipelines, Model Registry

2. **AmazonS3FullAccess**
   - All S3 operations
   - Read, write, delete, bucket management

3. **IAMFullAccess**
   - Create/modify/delete IAM roles
   - Attach/detach policies
   - Pass roles to services

4. **CloudWatchFullAccess**
   - View logs and metrics
   - Create dashboards and alarms
   - Full monitoring access

5. **AWSCloudFormationFullAccess**
   - Infrastructure as Code
   - Stack creation and management

6. **AWSLambda_FullAccess**
   - Create and manage Lambda functions
   - For automation and pipelines

7. **AWSCodePipeline_FullAccess**
   - CI/CD pipeline management
   - Deployment automation

8. **AWSCodeBuildAdminAccess**
   - Build automation
   - Container builds

9. **AWSServiceCatalogAdminFullAccess**
   - Service catalog management
   - Product provisioning

10. **IAMReadOnlyAccess**
    - View IAM resources
    - Audit permissions

### Custom Inline Policies (2)

**1. SageMakerCompleteAdminAccess**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "sagemaker:*",
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    }
  ]
}
```

**2. AdditionalAdminPermissions**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "iam:PassRole",
        "iam:CreateRole",
        "iam:AttachRolePolicy",
        "logs:*",
        "events:*",
        "sns:*",
        "kms:*"
      ],
      "Resource": "*"
    }
  ]
}
```

### Key Permissions Summary

```json
{
  "sagemaker:*",      // Everything in SageMaker
  "s3:*",             // Everything in S3
  "iam:*",            // Everything in IAM
  "cloudformation:*", // Infrastructure management
  "cloudwatch:*",     // Monitoring and logs
  "lambda:*",         // Serverless functions
  "codepipeline:*",   // CI/CD pipelines
  "codebuild:*",      // Build automation
  "ecr:*",            // Container registries
  "logs:*",           // CloudWatch logs
  "events:*",         // EventBridge
  "sns:*",            // Notifications
  "kms:*"             // Encryption keys
}
```

**Access Level:** FULL ADMIN - Can create, modify, delete, and deploy everything

---

## S3 Bucket Policies

### Current Bucket Policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "MarcosAdminFullAccess",
      "Effect": "Allow",
      "Principal": {
        "AWS": [
          "arn:aws:iam::943271038849:role/SageMaker-MarcosAdmin-ExecutionRole",
          "arn:aws:iam::943271038849:root"
        ]
      },
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::pokemon-card-training-us-east-2",
        "arn:aws:s3:::pokemon-card-training-us-east-2/*"
      ]
    },
    {
      "Sid": "ProjectSpecificAccess",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::943271038849:role/SageMaker-MarcosAdmin-ExecutionRole"
      },
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket",
        "s3:GetBucketLocation",
        "s3:GetObjectVersion",
        "s3:PutObjectAcl"
      ],
      "Resource": [
        "arn:aws:s3:::pokemon-card-training-us-east-2/project/pokemon-card-recognition/*",
        "arn:aws:s3:::pokemon-card-training-us-east-2"
      ]
    }
  ]
}
```

### Bucket Tags

```yaml
Project: pokemon-card-recognition
Owner: marcospaulo
ServiceAccount: SageMaker-MarcosAdmin-ExecutionRole
Environment: production
ManagedBy: claude-code
```

---

## S3 Lifecycle Policies

### Configured Rules

**1. ArchiveOldTrainingOutputs**
```yaml
Status: Enabled
Prefix: models/embedding/
Transition: → Glacier after 90 days
Noncurrent Versions: → Glacier after 30 days
Purpose: Archive old training job outputs
Savings: ~$1.90/year
```

**2. ArchiveOldProfilingData**
```yaml
Status: Enabled
Prefix: project/pokemon-card-recognition/profiling/
Transition: → Glacier after 180 days
Purpose: Archive profiling metrics
Savings: ~$0.20/year
```

**3. DeleteIncompleteMultipartUploads**
```yaml
Status: Enabled
Prefix: (all)
Action: Delete incomplete uploads after 7 days
Purpose: Prevent abandoned upload costs
Savings: ~$0.18/year
```

**Total Savings:** $2.28/year from lifecycle policies

---

## Console Access Links

### SageMaker

**Studio:**
```
https://d-slzqikvnlai2.studio.us-east-2.sagemaker.aws
```

**Model Registry:**
```
https://console.aws.amazon.com/sagemaker/home?region=us-east-2#/model-package-groups/pokemon-card-recognition-models
```

**Training Jobs:**
```
https://console.aws.amazon.com/sagemaker/home?region=us-east-2#/jobs
```

**Endpoints:**
```
https://console.aws.amazon.com/sagemaker/home?region=us-east-2#/endpoints
```

### S3

**Project Root:**
```
https://s3.console.aws.amazon.com/s3/buckets/pokemon-card-training-us-east-2?prefix=project/pokemon-card-recognition/
```

**Models:**
```
https://s3.console.aws.amazon.com/s3/buckets/pokemon-card-training-us-east-2?prefix=project/pokemon-card-recognition/models/
```

**Data:**
```
https://s3.console.aws.amazon.com/s3/buckets/pokemon-card-training-us-east-2?prefix=project/pokemon-card-recognition/data/
```

**Analytics:**
```
https://s3.console.aws.amazon.com/s3/buckets/pokemon-card-training-us-east-2?prefix=project/pokemon-card-recognition/analytics/
```

### CloudWatch

**Dashboards:**
```
https://console.aws.amazon.com/cloudwatch/home?region=us-east-2#dashboards:
```

**Log Groups:**
```
https://console.aws.amazon.com/cloudwatch/home?region=us-east-2#logsV2:log-groups
```

**Metrics:**
```
https://console.aws.amazon.com/cloudwatch/home?region=us-east-2#metricsV2:
```

### IAM

**Execution Role:**
```
https://console.aws.amazon.com/iam/home?region=us-east-2#/roles/SageMaker-MarcosAdmin-ExecutionRole
```

---

## Access Verification

### Run Verification Script

```bash
bash scripts/verify_project_access.sh
```

**Tests (10 total):**
1. ✓ S3 Bucket Read Access
2. ✓ S3 Write Access
3. ✓ SageMaker Model Registry Access
4. ✓ IAM Role Access
5. ✓ CloudWatch Logs Access
6. ✓ SageMaker Training Jobs Access
7. ✓ S3 Lifecycle Policy Access
8. ✓ S3 Bucket Tagging Access
9. ✓ Download Project Manifest
10. ✓ Model File Access

### Manual Verification Commands

**Check SageMaker Profile:**
```bash
aws sagemaker describe-user-profile \
  --domain-id d-slzqikvnlai2 \
  --user-profile-name marcospaulo \
  --region us-east-2
```

**Check IAM Role Policies:**
```bash
aws iam list-attached-role-policies \
  --role-name SageMaker-MarcosAdmin-ExecutionRole
```

**Check S3 Bucket Policy:**
```bash
aws s3api get-bucket-policy \
  --bucket pokemon-card-training-us-east-2 \
  | jq -r '.Policy' | jq .
```

**Check Lifecycle Policies:**
```bash
aws s3api get-bucket-lifecycle-configuration \
  --bucket pokemon-card-training-us-east-2
```

---

## Important Notes

### SageMaker Profile Selection

When accessing SageMaker Studio, you'll be prompted to select the **"marcospaulo"** user profile. This is CORRECT and INTENDED:

1. **AWS SSO login:** You log in as `marcospaulo` with AdministratorAccess
2. **SageMaker User Profile:** Inside SageMaker, you select the `marcospaulo` profile
3. **Execution Role:** This profile uses `SageMaker-MarcosAdmin-ExecutionRole` for all operations

**Why the prompt exists:**
- One SageMaker domain can have multiple user profiles
- Each profile can have different execution roles and permissions
- Your profile (`marcospaulo`) is configured with FULL ADMIN ACCESS

Selecting "marcospaulo" when prompted gives you complete control over all project resources.

---

## Grant Additional Access Script

If you need to add explicit permissions or tag resources:

```bash
python3 scripts/grant_full_project_access.py
```

**This script:**
1. Adds explicit S3 bucket policies
2. Tags Model Registry resources with ownership
3. Verifies all access permissions

**Note:** Requires authenticated AWS CLI session (run `aws sso login` first if expired)

---

**Last Updated:** 2026-01-12
**Access Level:** ✅ FULL ADMIN
**Verification:** ✅ All 10 tests passing
