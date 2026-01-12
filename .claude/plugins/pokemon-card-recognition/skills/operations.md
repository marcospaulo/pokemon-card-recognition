---
skill_name: pokemon-card-recognition:operations
description: Common operations, commands, and workflows for downloading models, deploying endpoints, and managing data
tags: [operations, commands, deployment, workflows]
---

# Operations & Workflows

## Quick Operations

### Download Models

**Teacher Model (SageMaker package):**
```bash
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/dinov3-teacher/v1.0/model.tar.gz ./
tar -xzf model.tar.gz
```

**Student PyTorch Checkpoint:**
```bash
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-student/stage2/v2.0/student_stage2_final.pt ./
```

**Student ONNX Model:**
```bash
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-student/stage2/v2.0/student_stage2_final.onnx ./
```

**Hailo HEF (for Raspberry Pi):**
```bash
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-hailo/v2.1/pokemon_student_efficientnet_lite0_stage2.hef ./
```

### Download Reference Database

**Complete database (106 MB):**
```bash
mkdir -p data/reference
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/ ./data/reference/
```

**Individual files:**
```bash
# Embeddings (51.5 MB)
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/embeddings.npy ./

# uSearch index (54.0 MB)
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/usearch.index ./

# Index mapping (652 KB)
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/index.json ./

# Card metadata (543 KB)
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/metadata.json ./
```

### View Project Metadata

**Project Manifest:**
```bash
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/metadata/project_manifest.json - | jq .
```

**Analytics Summary:**
```bash
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/analytics/metrics/summary.json - | jq .
```

**Model Performance:**
```bash
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/analytics/metrics/model_performance.csv - | column -t -s,
```

**Cost Breakdown:**
```bash
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/analytics/metrics/cost_breakdown.csv - | column -t -s,
```

---

## SageMaker Operations

### Model Registry

**List all models:**
```bash
aws sagemaker list-model-packages \
  --model-package-group-name pokemon-card-recognition-models \
  --region us-east-2
```

**Describe specific model:**
```bash
# Teacher (Model #4)
aws sagemaker describe-model-package \
  --model-package-name arn:aws:sagemaker:us-east-2:943271038849:model-package/pokemon-card-recognition-models/4 \
  --region us-east-2

# Student Stage 2 (Model #5)
aws sagemaker describe-model-package \
  --model-package-name arn:aws:sagemaker:us-east-2:943271038849:model-package/pokemon-card-recognition-models/5 \
  --region us-east-2
```

**Update approval status:**
```bash
aws sagemaker update-model-package \
  --model-package-arn arn:aws:sagemaker:us-east-2:943271038849:model-package/pokemon-card-recognition-models/5 \
  --approval-status Approved \
  --region us-east-2
```

### Training Jobs

**List training jobs:**
```bash
aws sagemaker list-training-jobs \
  --region us-east-2 \
  --max-results 10 \
  --sort-by CreationTime \
  --sort-order Descending
```

**Describe training job:**
```bash
# Teacher training
aws sagemaker describe-training-job \
  --training-job-name pokemon-card-dinov3-teacher-2026-01-10-13-31-34-937 \
  --region us-east-2

# Student Stage 2 training
aws sagemaker describe-training-job \
  --training-job-name pytorch-training-2026-01-11-23-31-10-757 \
  --region us-east-2
```

### Endpoints

**List endpoints:**
```bash
aws sagemaker list-endpoints \
  --region us-east-2
```

**Deploy model to endpoint:**
```python
import boto3
from sagemaker import ModelPackage

role = 'arn:aws:iam::943271038849:role/SageMaker-MarcosAdmin-ExecutionRole'

# Load model from registry
model = ModelPackage(
    role=role,
    model_package_arn='arn:aws:sagemaker:us-east-2:943271038849:model-package/pokemon-card-recognition-models/5'
)

# Deploy to endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.c5.xlarge',
    endpoint_name='pokemon-student-stage2',
    wait=True
)

# Test inference
import numpy as np
test_embedding = predictor.predict(test_image)
print(f"Embedding shape: {test_embedding.shape}")  # (768,)
```

**Delete endpoint:**
```bash
aws sagemaker delete-endpoint \
  --endpoint-name pokemon-student-stage2 \
  --region us-east-2

aws sagemaker delete-endpoint-config \
  --endpoint-config-name pokemon-student-stage2 \
  --region us-east-2
```

---

## Data Operations

### Download Training Data

**Raw card images (13 GB):**
```bash
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/raw/card_images/ ./data/raw/
```

**Processed classification dataset (13 GB):**
```bash
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/processed/classification/ ./data/processed/
```

**Hailo calibration data (734 MB):**
```bash
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/calibration/ ./data/calibration/
```

### List Data Files

**Count images by directory:**
```bash
# Raw images
aws s3 ls s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/raw/card_images/ --recursive | wc -l
# Expected: 17,592

# Processed images
aws s3 ls s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/processed/classification/ --recursive | wc -l
# Expected: 17,592

# Calibration images
aws s3 ls s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/calibration/ --recursive | wc -l
# Expected: 1,024
```

**List all models:**
```bash
aws s3 ls s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/ --recursive --human-readable
```

---

## Profiling Data

### Download Profiling Outputs

**Teacher profiling (44.3 MB):**
```bash
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/profiling/teacher/2026-01-10/ ./profiling/teacher/
```

**Student Stage 2 profiling (72.8 MB):**
```bash
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/profiling/student_stage2/2026-01-11/ ./profiling/student/
```

### Analyze Profiling Data

**View system metrics:**
```bash
# CPU utilization
grep "CPUUtilization" profiling/teacher/*.json | jq .

# GPU memory
grep "GPUMemoryUtilization" profiling/teacher/*.json | jq .

# IO wait
grep "DiskUtilization" profiling/teacher/*.json | jq .
```

---

## CloudWatch Logs

### View Training Logs

**Teacher training logs:**
```bash
aws logs tail /aws/sagemaker/TrainingJobs \
  --log-stream-name-prefix pokemon-card-dinov3-teacher-2026-01-10-13-31-34-937 \
  --region us-east-2 \
  --follow
```

**Student training logs:**
```bash
aws logs tail /aws/sagemaker/TrainingJobs \
  --log-stream-name-prefix pytorch-training-2026-01-11-23-31-10-757 \
  --region us-east-2 \
  --follow
```

### View Endpoint Logs

**Inference logs:**
```bash
aws logs tail /aws/sagemaker/Endpoints/pokemon-student-stage2 \
  --region us-east-2 \
  --follow
```

---

## Raspberry Pi Deployment

### Transfer Files to Pi

**Transfer Hailo model:**
```bash
# Download from S3
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-hailo/v2.1/pokemon_student_efficientnet_lite0_stage2.hef ./

# Transfer to Raspberry Pi
scp pokemon_student_efficientnet_lite0_stage2.hef pi@raspberrypi:/home/pi/models/
```

**Transfer reference database:**
```bash
# Download from S3
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/ ./data/reference/

# Transfer to Raspberry Pi
scp -r data/reference/ pi@raspberrypi:/home/pi/inference/reference/
```

### Setup Hailo Runtime on Pi

**Install Hailo SDK:**
```bash
# On Raspberry Pi
wget https://hailo.ai/downloads/hailo-sdk-v3.27.0-rpi.tar.gz
tar -xzf hailo-sdk-v3.27.0-rpi.tar.gz
cd hailo-sdk
sudo ./install.sh

# Verify installation
hailortcli scan
# Expected: Hailo-8L detected
```

**Install Python dependencies:**
```bash
pip install hailo-platform numpy opencv-python usearch
```

### Run Inference on Pi

**Python inference script:**
```python
#!/usr/bin/env python3
from hailo_platform import HailoDevice, InferenceContext
import numpy as np
import cv2
from usearch.index import Index
import json

# Load Hailo model
device = HailoDevice()
model = device.create_infer_model('/home/pi/models/pokemon_student_efficientnet_lite0_stage2.hef')

# Load reference database
index = Index.restore('/home/pi/inference/reference/usearch.index')
with open('/home/pi/inference/reference/index.json') as f:
    index_mapping = json.load(f)
with open('/home/pi/inference/reference/metadata.json') as f:
    metadata = json.load(f)

def recognize_card(image_path):
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC → CHW
    image = np.expand_dims(image, axis=0)   # Add batch dimension

    # Run inference on Hailo NPU
    with InferenceContext(model) as ctx:
        embedding = ctx.run(image)  # Shape: (1, 768)

    # Search for similar cards
    matches = index.search(embedding[0], k=5)

    # Print top 5 matches
    print("\nTop 5 matches:")
    for i, (row_id, distance) in enumerate(zip(matches.keys, matches.distances)):
        card_id = index_mapping[str(row_id)]
        card_info = metadata[card_id]
        similarity = 1 - distance
        print(f"{i+1}. {card_info['name']} (Set: {card_info['set']}) - Similarity: {similarity:.3f}")

    return matches

# Test with a card image
recognize_card('/home/pi/test_card.jpg')
```

**Run:**
```bash
python3 recognize_card.py
```

---

## Training New Models

### Launch Teacher Training Job

```python
import sagemaker
from sagemaker.estimator import Estimator

role = 'arn:aws:iam::943271038849:role/SageMaker-MarcosAdmin-ExecutionRole'

# Define estimator
estimator = Estimator(
    image_uri='763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:2.0.1-gpu-py310',
    role=role,
    instance_count=1,
    instance_type='ml.p4d.24xlarge',
    volume_size=100,
    max_run=3600,
    output_path='s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/dinov3-teacher/',
    hyperparameters={
        'model': 'facebook/dinov3-base',
        'batch-size': 128,
        'learning-rate': 5e-5,
        'epochs': 10,
        'image-size': 518
    }
)

# Start training
estimator.fit({
    'training': 's3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/processed/classification/'
})
```

### Launch Student Training Job

```python
# Define student estimator
student_estimator = Estimator(
    image_uri='763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:2.0.1-gpu-py310',
    role=role,
    instance_count=1,
    instance_type='ml.p4d.24xlarge',
    volume_size=100,
    max_run=3600,
    output_path='s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-student/',
    hyperparameters={
        'model': 'efficientnet-lite0',
        'teacher-model': 's3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/dinov3-teacher/v1.0/model.tar.gz',
        'batch-size': 256,
        'learning-rate': 1e-3,
        'epochs': 20,
        'image-size': 224,
        'distillation': True,
        'temperature': 0.5
    }
)

# Start training
student_estimator.fit({
    'training': 's3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/processed/classification/'
})
```

### Register Model to Registry

```python
# Register student model
model_package = student_estimator.register(
    model_package_group_name='pokemon-card-recognition-models',
    inference_instances=['ml.c5.xlarge', 'ml.c5.2xlarge'],
    transform_instances=['ml.m5.xlarge'],
    content_types=['application/x-image'],
    response_types=['application/json'],
    approval_status='PendingManualApproval',
    description='EfficientNet-Lite0 student model (Stage 2) with hard negative mining'
)

print(f"Model registered: {model_package.model_package_arn}")
```

---

## Verification & Testing

### Run Full Access Verification

```bash
bash scripts/verify_project_access.sh
```

**Expected output:**
```
Testing: S3 Bucket Read Access... ✓ PASS
Testing: S3 Write Access... ✓ PASS
Testing: SageMaker Model Registry Access... ✓ PASS
Testing: IAM Role Access... ✓ PASS
Testing: CloudWatch Logs Access... ✓ PASS
Testing: SageMaker Training Jobs Access... ✓ PASS
Testing: S3 Lifecycle Policy Access... ✓ PASS
Testing: S3 Bucket Tagging Access... ✓ PASS
Testing: Download Project Manifest... ✓ PASS
Testing: Model File Access... ✓ PASS

Passed: 10
Failed: 0
✓ All tests passed! Full admin access confirmed.
```

### Verify SageMaker Profile

```bash
aws sagemaker describe-user-profile \
  --domain-id d-slzqikvnlai2 \
  --user-profile-name marcospaulo \
  --region us-east-2 \
  | jq .
```

### Verify IAM Permissions

```bash
aws iam list-attached-role-policies \
  --role-name SageMaker-MarcosAdmin-ExecutionRole \
  | jq .
```

---

## Lifecycle Management

### View Lifecycle Policies

```bash
aws s3api get-bucket-lifecycle-configuration \
  --bucket pokemon-card-training-us-east-2 \
  | jq .
```

### Restore from Glacier

**If files have been archived to Glacier:**
```bash
# Initiate restore (takes 3-5 hours)
aws s3api restore-object \
  --bucket pokemon-card-training-us-east-2 \
  --key project/pokemon-card-recognition/profiling/teacher/2026-01-10/file.json \
  --restore-request Days=7

# Check restore status
aws s3api head-object \
  --bucket pokemon-card-training-us-east-2 \
  --key project/pokemon-card-recognition/profiling/teacher/2026-01-10/file.json \
  | jq '.Restore'

# After restore completes, download
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/profiling/teacher/2026-01-10/file.json ./
```

---

## Batch Operations

### Sync All Models Locally

```bash
mkdir -p models
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/ ./models/ \
  --exclude "*" \
  --include "dinov3-teacher/v1.0/*" \
  --include "efficientnet-student/stage2/v2.0/*" \
  --include "efficientnet-hailo/v2.1/*"
```

### Sync All Analytics

```bash
mkdir -p analytics
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/analytics/ ./analytics/
```

### Sync Profiling Data

```bash
mkdir -p profiling
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/profiling/ ./profiling/
```

---

## Cleanup Operations

### Delete Endpoint (to save costs)

```bash
# Delete endpoint
aws sagemaker delete-endpoint --endpoint-name pokemon-student-stage2 --region us-east-2

# Delete endpoint config
aws sagemaker delete-endpoint-config --endpoint-config-name pokemon-student-stage2 --region us-east-2

# Note: This does NOT delete the model from Model Registry
```

### Delete Old Training Job Outputs

**Careful:** This deletes original training outputs. Models are preserved in organized structure.

```bash
# List old training outputs
aws s3 ls s3://pokemon-card-training-us-east-2/models/embedding/ --recursive

# Delete (if you're sure)
aws s3 rm s3://pokemon-card-training-us-east-2/models/embedding/ --recursive
```

---

## Cost Monitoring

### View S3 Storage Costs

```bash
# Get bucket size
aws s3 ls s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/ --recursive --summarize

# Estimate monthly cost (S3 Standard: $0.023/GB)
# Current: 31.7 GB × $0.023 = $0.729/month
# After lifecycle (60% to Glacier): ~$0.30/month
```

### View Training Job Costs

```bash
# List all training jobs with instance types
aws sagemaker list-training-jobs --region us-east-2 | jq '.TrainingJobSummaries[] | {JobName: .TrainingJobName, Instance: .ResourceConfig.InstanceType, Duration: .TrainingTimeInSeconds, Status: .TrainingJobStatus}'
```

---

**Last Updated:** 2026-01-12
**Common Operations:** Download models, deploy endpoints, train new models, access data
**Verification:** 10 automated tests in `scripts/verify_project_access.sh`
