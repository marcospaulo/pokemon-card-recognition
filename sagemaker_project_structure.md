# Pokemon Card Recognition - Unified SageMaker Project Structure

## Project Overview

**Project Name:** `pokemon-card-recognition`
**Project ID:** TBD (created by SageMaker)
**Description:** End-to-end Pokemon card recognition system with teacher-student distillation and edge deployment

## Project Components

### 1. Model Registry Organization

**Model Package Group:** `pokemon-card-recognition-models`

#### Model Variants

| Model Name | Version | Type | Size | Purpose | Status |
|------------|---------|------|------|---------|--------|
| `dinov3-teacher` | v1.0 | Teacher | 1.1 GB ONNX | Cloud inference, teacher for distillation | ✅ Trained |
| `efficientnet-student-stage1` | v1.0 | Student | 200 MB | General feature distillation | ✅ Trained |
| `efficientnet-student-stage2` | v2.0 | Student | 200 MB | Task-specific fine-tuned | ✅ Trained |
| `efficientnet-hailo-optimized` | v2.1 | Optimized | 14.5 MB HEF | Edge deployment (Raspberry Pi + Hailo-8L) | ✅ Compiled |

### 2. S3 Organization Structure

```
s3://pokemon-card-training-us-east-2/
├── project/
│   └── pokemon-card-recognition/
│       ├── metadata/
│       │   ├── project_manifest.json           # Master project metadata
│       │   ├── model_lineage.json              # Model relationships (teacher→student)
│       │   ├── training_history.json           # All training runs
│       │   └── performance_benchmarks.json     # Performance metrics
│       │
│       ├── models/
│       │   ├── dinov3-teacher/
│       │   │   ├── v1.0/
│       │   │   │   ├── model.tar.gz            # SageMaker deployment artifact
│       │   │   │   ├── pytorch/phase2_checkpoint.pt
│       │   │   │   ├── onnx/dinov3_teacher.onnx
│       │   │   │   ├── metrics.json
│       │   │   │   └── model_card.md
│       │   │   └── training_config.json
│       │   │
│       │   ├── efficientnet-student/
│       │   │   ├── stage1/
│       │   │   │   ├── v1.0/
│       │   │   │   │   ├── model.tar.gz
│       │   │   │   │   ├── student_stage1.pt
│       │   │   │   │   ├── student_stage1.onnx
│       │   │   │   │   ├── metrics.json
│       │   │   │   │   └── distillation_report.json
│       │   │   │   └── training_config.json
│       │   │   │
│       │   │   └── stage2/
│       │   │       ├── v2.0/
│       │   │       │   ├── model.tar.gz
│       │   │       │   ├── student_stage2_final.pt
│       │   │       │   ├── student_stage2_final.onnx
│       │   │       │   ├── metrics.json
│       │   │       │   └── distillation_report.json
│       │   │       └── training_config.json
│       │   │
│       │   └── efficientnet-hailo/
│       │       ├── v2.1/
│       │       │   ├── pokemon_student_efficientnet_lite0_stage2.hef
│       │       │   ├── compilation_report.json
│       │       │   ├── profiling_results.json
│       │       │   └── hailo_config.yaml
│       │       └── calibration/
│       │           ├── calibration_dataset.npy
│       │           └── calibration_config.json
│       │
│       ├── experiments/
│       │   └── mlflow/
│       │       ├── pokemon-card-dinov3-teacher/
│       │       │   ├── run_<id>/
│       │       │   │   ├── metrics/
│       │       │   │   ├── params/
│       │       │   │   ├── artifacts/
│       │       │   │   └── meta.yaml
│       │       │   └── meta.yaml
│       │       │
│       │       └── pokemon-card-student-distillation/
│       │           ├── stage1/
│       │           │   └── run_<id>/
│       │           └── stage2/
│       │               └── run_<id>/
│       │
│       ├── profiling/
│       │   ├── teacher/
│       │   │   ├── sagemaker_debugger/
│       │   │   │   ├── rules/
│       │   │   │   └── tensors/
│       │   │   └── sagemaker_profiler/
│       │   │       ├── system/
│       │   │       ├── framework/
│       │   │       └── reports/profiler-report.html
│       │   │
│       │   ├── student_stage1/
│       │   └── student_stage2/
│       │
│       ├── analytics/
│       │   ├── dashboards/
│       │   │   ├── training_overview.json          # QuickSight/custom dashboard
│       │   │   ├── model_performance.json
│       │   │   └── cost_tracking.json
│       │   │
│       │   ├── reports/
│       │   │   ├── model_comparison_report.pdf     # Teacher vs Student
│       │   │   ├── distillation_analysis.pdf       # Stage 1 vs Stage 2
│       │   │   ├── edge_deployment_benchmarks.pdf  # Hailo performance
│       │   │   └── training_summary.pdf
│       │   │
│       │   └── metrics/
│       │       ├── accuracy_over_time.csv
│       │       ├── loss_convergence.csv
│       │       ├── inference_latency.csv
│       │       └── model_sizes.csv
│       │
│       ├── data/
│       │   ├── raw/                    # Soft link to main data location
│       │   ├── processed/              # Soft link to main data location
│       │   ├── calibration/            # Soft link to main data location
│       │   └── reference/              # Soft link to main data location
│       │
│       ├── pipelines/
│       │   ├── training/
│       │   │   ├── teacher_pipeline.py
│       │   │   └── student_pipeline.py
│       │   │
│       │   ├── inference/
│       │   │   ├── batch_inference_pipeline.py
│       │   │   └── realtime_endpoint_pipeline.py
│       │   │
│       │   └── deployment/
│       │       ├── model_packaging_pipeline.py
│       │       └── edge_deployment_pipeline.py
│       │
│       └── checkpoints/
│           ├── teacher/
│           ├── student_stage1/
│           └── student_stage2/
```

### 3. MLFlow Organization

**Tracking URI:** `s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/experiments/mlflow`

#### Experiments

1. **pokemon-card-dinov3-teacher**
   - Run ID: `<generated_by_mlflow>`
   - Metrics: top1_accuracy, top5_accuracy, loss
   - Artifacts: model checkpoints, embeddings, tensorboard logs

2. **pokemon-card-student-distillation-stage1**
   - Run ID: `<generated_by_mlflow>`
   - Metrics: feature_loss, kl_loss, attention_loss, highfreq_loss
   - Artifacts: stage1 checkpoints, distillation analysis

3. **pokemon-card-student-distillation-stage2**
   - Run ID: `<generated_by_mlflow>`
   - Metrics: task_accuracy, distillation_gap, cosine_similarity
   - Artifacts: stage2 checkpoints, final model

### 4. Model Lineage & Relationships

```
DINOv3 Teacher (v1.0)
    ↓ distillation (Stage 1)
EfficientNet Student Stage 1 (v1.0)
    ↓ fine-tuning (Stage 2)
EfficientNet Student Stage 2 (v2.0)
    ↓ compilation + quantization
EfficientNet Hailo Optimized (v2.1)
```

### 5. Project Metadata Schema

**File:** `project_manifest.json`

```json
{
  "project_name": "pokemon-card-recognition",
  "project_id": "<sagemaker_project_id>",
  "created_at": "2026-01-10T13:31:34Z",
  "last_updated": "2026-01-11T20:00:00Z",
  "version": "1.0.0",
  "description": "Pokemon card recognition using DINOv3 teacher-student distillation",

  "models": {
    "dinov3-teacher": {
      "version": "v1.0",
      "type": "teacher",
      "architecture": "DINOv3-ViT-L/16",
      "parameters": "304M",
      "embedding_dim": 768,
      "input_size": "224x224",
      "framework": "PyTorch 2.8.0",
      "training_job": "<job_name>",
      "training_date": "2026-01-10",
      "s3_path": "s3://.../models/dinov3-teacher/v1.0/",
      "model_registry_arn": "<arn>",
      "metrics": {
        "top1_accuracy": 0.0,
        "top5_accuracy": 0.0,
        "embedding_quality": "high"
      }
    },
    "efficientnet-student-stage1": {
      "version": "v1.0",
      "type": "student",
      "parent_model": "dinov3-teacher:v1.0",
      "architecture": "EfficientNet-Lite0",
      "parameters": "4.7M",
      "compression_ratio": "64.7x",
      "training_approach": "knowledge_distillation",
      "training_date": "2026-01-11",
      "metrics": {
        "accuracy_gap": "<percentage>",
        "cosine_similarity_to_teacher": "<score>"
      }
    },
    "efficientnet-student-stage2": {
      "version": "v2.0",
      "type": "student",
      "parent_model": "efficientnet-student-stage1:v1.0",
      "architecture": "EfficientNet-Lite0",
      "parameters": "4.7M",
      "training_approach": "task_specific_finetuning",
      "training_date": "2026-01-11",
      "metrics": {
        "top1_accuracy": "<score>",
        "top5_accuracy": "<score>"
      }
    },
    "efficientnet-hailo-optimized": {
      "version": "v2.1",
      "type": "optimized",
      "parent_model": "efficientnet-student-stage2:v2.0",
      "architecture": "EfficientNet-Lite0",
      "format": "Hailo HEF",
      "quantization": "INT8",
      "size": "14.5MB",
      "target_hardware": "Hailo-8L",
      "compilation_date": "2026-01-11",
      "performance": {
        "inference_latency_ms": 10,
        "throughput_fps": 100,
        "power_consumption_w": 2.5
      }
    }
  },

  "experiments": {
    "mlflow_tracking_uri": "s3://.../experiments/mlflow",
    "experiments": [
      {
        "name": "pokemon-card-dinov3-teacher",
        "run_count": 1,
        "best_run_id": "<id>"
      },
      {
        "name": "pokemon-card-student-distillation",
        "run_count": 2,
        "stages": ["stage1", "stage2"]
      }
    ]
  },

  "datasets": {
    "training": {
      "s3_path": "s3://.../data/processed/classification/",
      "num_classes": 17592,
      "train_samples": 14073,
      "val_samples": 2638,
      "test_samples": 881,
      "size_gb": 25
    },
    "calibration": {
      "s3_path": "s3://.../data/calibration/",
      "num_samples": 1024,
      "size_mb": 734
    }
  },

  "infrastructure": {
    "s3_bucket": "pokemon-card-training-us-east-2",
    "region": "us-east-2",
    "training_instance": "ml.p4d.24xlarge",
    "compilation_instance": "m5.2xlarge (i-0fb03883eddb631d8)"
  },

  "cost_tracking": {
    "total_training_cost_usd": 11.0,
    "teacher_training_cost": 4.0,
    "student_stage1_cost": 4.0,
    "student_stage2_cost": 3.0,
    "compilation_cost": 0.5
  }
}
```

## Implementation Steps

### Phase 1: Project Creation
1. Create SageMaker Project
2. Set up S3 structure
3. Configure IAM roles and permissions

### Phase 2: Model Registration
1. Register DINOv3 Teacher to Model Registry
2. Register EfficientNet Student Stage 1
3. Register EfficientNet Student Stage 2
4. Register Hailo Optimized model
5. Link models with lineage relationships

### Phase 3: Data Organization
1. Migrate MLFlow experiments to unified location
2. Organize profiling data
3. Consolidate analytics and reports
4. Update data paths in metadata

### Phase 4: Documentation & Automation
1. Create project README
2. Generate model cards for each variant
3. Set up automated reporting
4. Create dashboard for monitoring

## Access & Usage

### Viewing Project
```python
import sagemaker
sm_client = sagemaker.Session().sagemaker_client
project = sm_client.describe_project(ProjectName='pokemon-card-recognition')
```

### Accessing Models
```python
from sagemaker import ModelPackage
model = ModelPackage(
    role=role,
    model_package_arn='arn:aws:sagemaker:...:model-package/dinov3-teacher/1'
)
```

### Viewing Experiments
```python
import mlflow
mlflow.set_tracking_uri('s3://.../experiments/mlflow')
experiment = mlflow.get_experiment_by_name('pokemon-card-dinov3-teacher')
```

## Benefits of This Organization

1. **Single Source of Truth**: All models, data, and experiments in one project
2. **Model Lineage**: Clear parent-child relationships between models
3. **Version Control**: Track all model versions and experiments
4. **Cost Tracking**: Unified cost attribution per model
5. **Easy Discovery**: Find any asset quickly through project structure
6. **Reproducibility**: Complete training configurations and artifacts
7. **Compliance**: Model cards and documentation for each model
8. **Collaboration**: Team members can easily navigate the project
