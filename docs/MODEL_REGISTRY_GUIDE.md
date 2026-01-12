# SageMaker Model Registry Guide

Complete guide for tracking and organizing all Pokemon Card Recognition models.

## Overview

All models (teacher + student) are tracked in a unified **Model Package Group** with:
- ✅ Full training metadata
- ✅ Model lineage (teacher → student)
- ✅ Performance metrics
- ✅ Architecture information
- ✅ Approval workflow
- ✅ Version control

## Quick Start

### Step 1: Setup Project Infrastructure

Create the Model Package Group and MLflow experiment:

```bash
python scripts/setup_sagemaker_project.py
```

This creates:
- **Model Package Group**: `pokemon-card-recognition-models`
- **MLflow Experiment**: `pokemon-card-recognition-training`
- **Metadata Template**: `config/model_metadata_template.json`

### Step 2: Register Teacher Model (Retroactive)

Register your existing DINOv3 teacher model:

```bash
python scripts/register_teacher_model.py
```

This extracts:
- Model architecture (DINOv3-ViT-L/16, 304M params)
- Training metrics (top1_acc, top5_acc, etc.)
- Training job information
- Hyperparameters used

### Step 3: Train Student (Auto-Registration)

Student training automatically registers models:

```bash
python scripts/launch_student_distillation_8xA100.py
```

Each stage registers a new model version:
- **Stage 1**: `student-efficientnet-lite0-stage1-v1`
- **Stage 2**: `student-efficientnet-lite0-stage2-v1`

## Model Naming Convention

All models follow this pattern:

```
{model_type}-{architecture}-{stage}-{version}
```

### Examples:

**Teacher Model:**
```
teacher-dinov3-vitl16-phase2-v1
```

**Student Models:**
```
student-efficientnet-lite0-stage1-v1
student-efficientnet-lite0-stage2-v1
```

**Exported Models:**
```
student-efficientnet-lite0-onnx-v1      # ONNX export
student-efficientnet-lite0-hailo8l-v1   # Hailo HEF
```

## Viewing Models

### CLI

List all model versions:

```bash
aws sagemaker list-model-packages \
  --model-package-group-name pokemon-card-recognition-models \
  --sort-by CreationTime \
  --sort-order Descending
```

Get specific model details:

```bash
aws sagemaker describe-model-package \
  --model-package-name <model-package-arn>
```

### Console

View in SageMaker Console:

```
https://console.aws.amazon.com/sagemaker/home?region=us-east-2#/model-packages/pokemon-card-recognition-models
```

## Model Metadata

Each registered model includes:

### Teacher Model

```json
{
  "Architecture": "DINOv3-ViT-L/16",
  "Parameters": "304000000",
  "EmbeddingDim": "768",
  "NumClasses": "17592",
  "ModelType": "teacher",
  "Purpose": "knowledge_distillation_source",
  "InferenceTarget": "cloud-gpu",
  "TrainingApproach": "Self-supervised + ArcFace fine-tuning"
}
```

### Student Model

```json
{
  "Architecture": "efficientnet_lite0",
  "Parameters": "4700000",
  "EmbeddingDim": "768",
  "NumClasses": "17592",
  "ModelType": "student",
  "Stage": "stage1",
  "TeacherModel": "dinov3_vitl16",
  "BestSimilarity": "0.9234",
  "HailoCompatible": "true",
  "Normalization": "BatchNorm",
  "Purpose": "edge_deployment_hailo8l",
  "InferenceTarget": "raspberry-pi-hailo8l"
}
```

## Model Approval Workflow

### Teacher Models

- **Auto-approved**: Teacher models are automatically approved since they're used for distillation only

### Student Models

- **Stage 1**: `PendingManualApproval` (intermediate checkpoint)
- **Stage 2**: `PendingManualApproval` → Manual approval required before production deployment

### Approve a Model

```bash
aws sagemaker update-model-package \
  --model-package-arn <model-package-arn> \
  --model-approval-status Approved
```

## Model Lineage

Track the full distillation pipeline:

```
DINOv3-ViT-L/16 (304M params)
    ↓ [Knowledge Distillation]
EfficientNet-Lite0 Stage 1 (4.7M params)
    ↓ [Task-Specific Fine-Tuning]
EfficientNet-Lite0 Stage 2 (4.7M params)
    ↓ [ONNX Export]
ONNX Model (Hailo-compatible)
    ↓ [Hailo Compilation]
HEF Model (Hailo-8L bytecode)
```

## Querying Models

### Get Latest Approved Student Model

```python
import boto3

sm_client = boto3.client('sagemaker')

response = sm_client.list-model-packages(
    ModelPackageGroupName='pokemon-card-recognition-models',
    ModelApprovalStatus='Approved',
    SortBy='CreationTime',
    SortOrder='Descending',
    MaxResults=10
)

# Filter for student models
student_models = [
    pkg for pkg in response['ModelPackageSummaryList']
    if 'student' in pkg.get('ModelPackageDescription', '').lower()
]

latest_student = student_models[0]
print(f"Latest approved student: {latest_student['ModelPackageArn']}")
```

### Get All Model Versions

```python
import boto3

sm_client = boto3.client('sagemaker')

# Paginate through all versions
paginator = sm_client.get_paginator('list_model_packages')
page_iterator = paginator.paginate(
    ModelPackageGroupName='pokemon-card-recognition-models'
)

all_models = []
for page in page_iterator:
    all_models.extend(page['ModelPackageSummaryList'])

print(f"Total model versions: {len(all_models)}")
for model in all_models:
    print(f"  - {model['ModelPackageArn']}")
```

## Integration with Training

### Automatic Registration

Student training automatically registers models at the end of each stage:

```python
# In train_student_distillation.py (end of training)

register_student_model(
    model_path=final_path,
    model_metadata={
        'architecture': 'efficientnet_lite0',
        'stage': 'stage1',
        'best_similarity': 0.9234,
        'teacher_model': 'dinov3_vitl16',
        # ... full metadata
    }
)
```

### Manual Registration

If automatic registration fails, register manually:

```python
import boto3

sm_client = boto3.client('sagemaker')

response = sm_client.create_model_package(
    ModelPackageGroupName='pokemon-card-recognition-models',
    ModelPackageDescription='...',
    InferenceSpecification={...},
    ModelApprovalStatus='PendingManualApproval',
    CustomerMetadataProperties={...},
    Tags=[...]
)
```

## MLflow Integration

All training runs log to MLflow:

### View Runs

```python
import mlflow

mlflow.set_tracking_uri('s3://pokemon-card-training-us-east-2/mlflow')
mlflow.set_experiment('pokemon-card-recognition-training')

# Get all runs
runs = mlflow.search_runs(experiment_names=['pokemon-card-recognition-training'])

print(f"Total runs: {len(runs)}")
print(runs[['run_id', 'metrics.cosine_similarity', 'tags.stage']])
```

### Compare Models

```python
import mlflow

# Load specific runs
teacher_run = mlflow.get_run('teacher-run-id')
student_stage1_run = mlflow.get_run('student-stage1-run-id')
student_stage2_run = mlflow.get_run('student-stage2-run-id')

# Compare metrics
print("Teacher Top1:", teacher_run.data.metrics['top1_acc'])
print("Student Stage 1 Similarity:", student_stage1_run.data.metrics['cosine_similarity'])
print("Student Stage 2 Similarity:", student_stage2_run.data.metrics['cosine_similarity'])
```

## Troubleshooting

### Model Registration Fails

If automatic registration fails during training:

1. **Check permissions**: Ensure SageMaker execution role has `sagemaker:CreateModelPackage` permission
2. **Check Model Package Group exists**: Run `setup_sagemaker_project.py` first
3. **Manual registration**: Use `register_teacher_model.py` or `register_student_model.py` manually

### Can't Find Model

If a model isn't showing up:

```bash
# List all model packages (including unapproved)
aws sagemaker list-model-packages \
  --model-package-group-name pokemon-card-recognition-models \
  --max-results 50
```

### Model Approval Issues

If you can't approve a model:

1. Check IAM permissions: `sagemaker:UpdateModelPackage`
2. Verify model exists: `describe-model-package`
3. Check current status: Model must be `PendingManualApproval`

## Best Practices

### 1. Always Register Models

- ✅ Register all training runs (even failed ones for debugging)
- ✅ Include full metadata (architecture, hyperparameters, metrics)
- ✅ Tag models appropriately (stage, purpose, target hardware)

### 2. Use Approval Workflow

- ✅ Only approve models that pass validation
- ✅ Stage 2 student models require explicit approval before deployment
- ✅ Document approval decision in model description

### 3. Track Lineage

- ✅ Link student models to their teacher
- ✅ Include distillation configuration in metadata
- ✅ Track ONNX → HEF compilation lineage

### 4. Clean Naming

- ✅ Follow naming convention: `{type}-{arch}-{stage}-{version}`
- ✅ Use descriptive model descriptions
- ✅ Include performance metrics in description

### 5. Monitor Metrics

- ✅ Track cosine similarity (student vs teacher)
- ✅ Monitor retrieval accuracy
- ✅ Compare across versions

## Summary

With Model Registry, you now have:

- ✅ **Single source of truth** for all models
- ✅ **Complete lineage** tracking (teacher → student)
- ✅ **Version control** with approval workflow
- ✅ **Full metadata** (architecture, metrics, hyperparameters)
- ✅ **Organized structure** with consistent naming
- ✅ **Easy queries** via CLI, Python, or Console

All models are properly tracked and organized for production deployment!
