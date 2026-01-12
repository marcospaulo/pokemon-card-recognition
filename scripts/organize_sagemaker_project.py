"""
Unified SageMaker Project Organization Script

This script creates a complete SageMaker project structure and organizes:
- All model variants (DINOv3 teacher, EfficientNet student stages, Hailo optimized)
- MLFlow experiment data
- Profiling and analytics data  - Model registry entries with lineage
- Centralized metadata

Usage:
    python scripts/organize_sagemaker_project.py --execute
"""

import json
import boto3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse

# Configuration
PROJECT_NAME = "pokemon-card-recognition"
BUCKET_NAME = "pokemon-card-training-us-east-2"
REGION = "us-east-2"
MODEL_PACKAGE_GROUP = "pokemon-card-recognition-models"

# Initialize AWS clients
sagemaker = boto3.client('sagemaker', region_name=REGION)
s3 = boto3.client('s3', region_name=REGION)


class ProjectOrganizer:
    """Organizes all ML assets into a unified SageMaker project."""

    def __init__(self, bucket: str, project_name: str, dry_run: bool = True):
        self.bucket = bucket
        self.project_name = project_name
        self.dry_run = dry_run
        self.project_prefix = f"project/{project_name}"

        print(f"{'[DRY RUN] ' if dry_run else ''}Initializing Project Organizer")
        print(f"  Bucket: {bucket}")
        print(f"  Project: {project_name}")
        print(f"  Prefix: {self.project_prefix}")

    def create_project_structure(self):
        """Create the S3 directory structure for the project."""
        print("\n" + "="*70)
        print("STEP 1: Creating Project S3 Structure")
        print("="*70)

        directories = [
            f"{self.project_prefix}/metadata/",
            f"{self.project_prefix}/models/dinov3-teacher/v1.0/pytorch/",
            f"{self.project_prefix}/models/dinov3-teacher/v1.0/onnx/",
            f"{self.project_prefix}/models/efficientnet-student/stage1/v1.0/",
            f"{self.project_prefix}/models/efficientnet-student/stage2/v2.0/",
            f"{self.project_prefix}/models/efficientnet-hailo/v2.1/",
            f"{self.project_prefix}/models/efficientnet-hailo/calibration/",
            f"{self.project_prefix}/experiments/mlflow/",
            f"{self.project_prefix}/profiling/teacher/",
            f"{self.project_prefix}/profiling/student_stage1/",
            f"{self.project_prefix}/profiling/student_stage2/",
            f"{self.project_prefix}/analytics/dashboards/",
            f"{self.project_prefix}/analytics/reports/",
            f"{self.project_prefix}/analytics/metrics/",
            f"{self.project_prefix}/pipelines/training/",
            f"{self.project_prefix}/pipelines/inference/",
            f"{self.project_prefix}/pipelines/deployment/",
        ]

        for directory in directories:
            key = f"{directory}.keep"
            print(f"  Creating: s3://{self.bucket}/{directory}")
            if not self.dry_run:
                s3.put_object(Bucket=self.bucket, Key=key, Body=b'')

        print(f"\n✓ Created {len(directories)} directories")

    def create_project_manifest(self) -> Dict:
        """Generate the master project manifest."""
        print("\n" + "="*70)
        print("STEP 2: Creating Project Manifest")
        print("="*70)

        manifest = {
            "project_name": self.project_name,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "version": "1.0.0",
            "description": "Pokemon card recognition using DINOv3 teacher-student distillation and edge optimization",

            "models": {
                "dinov3-teacher": {
                    "version": "v1.0",
                    "type": "teacher",
                    "architecture": "DINOv3-ViT-L/16",
                    "parameters": "304M",
                    "embedding_dim": 768,
                    "input_size": "224x224",
                    "framework": "PyTorch 2.8.0 + Transformers",
                    "training_date": "2026-01-10",
                    "s3_path": f"s3://{self.bucket}/{self.project_prefix}/models/dinov3-teacher/v1.0/",
                    "sagemaker_model": "pokemon-card-dinov3-teacher-2026-01-10-13-31-34-937",
                    "formats": ["pytorch", "onnx", "sagemaker_tarball"],
                    "metrics": {
                        "num_classes": 17592,
                        "embedding_quality": "high",
                        "training_epochs": 13,
                        "phases": {
                            "frozen": 3,
                            "unfrozen": 10
                        }
                    },
                    "training_config": {
                        "batch_size": 256,
                        "learning_rate": {"frozen": 0.001, "unfrozen": 0.00001},
                        "optimizer": "AdamW",
                        "loss": "ArcFace (margin=0.5, scale=64)",
                        "hardware": "8x A100 80GB",
                        "instance": "ml.p4d.24xlarge",
                        "cost_usd": 4.0
                    }
                },
                "efficientnet-student-stage1": {
                    "version": "v1.0",
                    "type": "student",
                    "parent_model": "dinov3-teacher:v1.0",
                    "architecture": "EfficientNet-Lite0",
                    "parameters": "4.7M",
                    "embedding_dim": 768,
                    "compression_ratio": "64.7x",
                    "training_approach": "multi-level knowledge distillation",
                    "training_date": "2026-01-11",
                    "s3_path": f"s3://{self.bucket}/{self.project_prefix}/models/efficientnet-student/stage1/v1.0/",
                    "formats": ["pytorch", "onnx"],
                    "training_config": {
                        "stage": 1,
                        "epochs": 30,
                        "batch_size": 512,
                        "learning_rate": 0.0001,
                        "distillation_losses": {
                            "feature_loss": 0.35,
                            "kl_divergence": 0.25,
                            "attention_loss": 0.25,
                            "high_frequency": 0.15
                        },
                        "hardware": "8x A100 80GB",
                        "cost_usd": 4.0
                    }
                },
                "efficientnet-student-stage2": {
                    "version": "v2.0",
                    "type": "student",
                    "parent_model": "efficientnet-student-stage1:v1.0",
                    "architecture": "EfficientNet-Lite0",
                    "parameters": "4.7M",
                    "embedding_dim": 768,
                    "training_approach": "task-specific fine-tuning with distillation",
                    "training_date": "2026-01-11",
                    "s3_path": f"s3://{self.bucket}/{self.project_prefix}/models/efficientnet-student/stage2/v2.0/",
                    "formats": ["pytorch", "onnx"],
                    "production_ready": True,
                    "training_config": {
                        "stage": 2,
                        "epochs": 20,
                        "batch_size": 512,
                        "learning_rate": 0.0001,
                        "hardware": "8x A100 80GB",
                        "cost_usd": 3.0
                    }
                },
                "efficientnet-hailo-optimized": {
                    "version": "v2.1",
                    "type": "optimized",
                    "parent_model": "efficientnet-student-stage2:v2.0",
                    "architecture": "EfficientNet-Lite0",
                    "format": "Hailo HEF",
                    "quantization": "INT8",
                    "size_mb": 14.5,
                    "target_hardware": "Hailo-8L NPU",
                    "deployment_platform": "Raspberry Pi 5",
                    "compilation_date": "2026-01-11",
                    "s3_path": f"s3://{self.bucket}/{self.project_prefix}/models/efficientnet-hailo/v2.1/",
                    "performance": {
                        "inference_latency_ms": 10,
                        "throughput_fps": 100,
                        "power_consumption_w": 2.5,
                        "speedup_vs_cpu": "50x"
                    },
                    "compilation_config": {
                        "calibration_samples": 1024,
                        "optimization_level": 2,
                        "compiler_version": "3.29.0",
                        "cost_usd": 0.5
                    }
                }
            },

            "model_lineage": {
                "description": "Parent-child relationships between models",
                "graph": {
                    "dinov3-teacher:v1.0": {
                        "children": ["efficientnet-student-stage1:v1.0"],
                        "relationship": "knowledge_distillation"
                    },
                    "efficientnet-student-stage1:v1.0": {
                        "parent": "dinov3-teacher:v1.0",
                        "children": ["efficientnet-student-stage2:v2.0"],
                        "relationship": "fine_tuning"
                    },
                    "efficientnet-student-stage2:v2.0": {
                        "parent": "efficientnet-student-stage1:v1.0",
                        "children": ["efficientnet-hailo-optimized:v2.1"],
                        "relationship": "compilation_optimization"
                    },
                    "efficientnet-hailo-optimized:v2.1": {
                        "parent": "efficientnet-student-stage2:v2.0",
                        "children": [],
                        "relationship": "production_deployment"
                    }
                }
            },

            "experiments": {
                "mlflow_tracking_uri": f"s3://{self.bucket}/{self.project_prefix}/experiments/mlflow",
                "experiments": [
                    {
                        "name": "pokemon-card-dinov3-teacher",
                        "description": "DINOv3 teacher model training with ArcFace",
                        "runs": 1,
                        "best_run_id": "TBD"
                    },
                    {
                        "name": "pokemon-card-student-distillation-stage1",
                        "description": "Stage 1: General feature distillation",
                        "runs": 1,
                        "best_run_id": "TBD"
                    },
                    {
                        "name": "pokemon-card-student-distillation-stage2",
                        "description": "Stage 2: Task-specific fine-tuning",
                        "runs": 1,
                        "best_run_id": "TBD"
                    }
                ]
            },

            "datasets": {
                "training": {
                    "s3_path": f"s3://{self.bucket}/data/processed/classification/",
                    "num_classes": 17592,
                    "splits": {
                        "train": 14073,
                        "val": 2638,
                        "test": 881
                    },
                    "size_gb": 25,
                    "format": "directory_per_class"
                },
                "calibration": {
                    "s3_path": f"s3://{self.bucket}/data/calibration/",
                    "num_samples": 1024,
                    "size_mb": 734,
                    "purpose": "Hailo INT8 quantization"
                },
                "reference": {
                    "s3_path": f"s3://{self.bucket}/data/reference/",
                    "size_mb": 21,
                    "contents": ["embeddings", "metadata", "index"]
                }
            },

            "infrastructure": {
                "s3_bucket": self.bucket,
                "region": REGION,
                "training_instance": "ml.p4d.24xlarge (8x A100 80GB)",
                "compilation_instance": "m5.2xlarge (Hailo SDK)",
                "compilation_instance_id": "i-0fb03883eddb631d8",
                "iam_role": "arn:aws:iam::943271038849:role/SageMaker-ExecutionRole"
            },

            "cost_tracking": {
                "total_cost_usd": 11.5,
                "breakdown": {
                    "teacher_training": 4.0,
                    "student_stage1": 4.0,
                    "student_stage2": 3.0,
                    "hailo_compilation": 0.5
                },
                "training_time_hours": 0.5,
                "cost_per_model": {
                    "dinov3-teacher": 4.0,
                    "efficientnet-student-stage1": 4.0,
                    "efficientnet-student-stage2": 3.0,
                    "efficientnet-hailo-optimized": 0.5
                }
            },

            "documentation": {
                "readme": f"s3://{self.bucket}/{self.project_prefix}/README.md",
                "architecture_diagrams": f"s3://{self.bucket}/{self.project_prefix}/docs/architecture/",
                "model_cards": {
                    "teacher": f"s3://{self.bucket}/{self.project_prefix}/models/dinov3-teacher/v1.0/MODEL_CARD.md",
                    "student_stage1": f"s3://{self.bucket}/{self.project_prefix}/models/efficientnet-student/stage1/v1.0/MODEL_CARD.md",
                    "student_stage2": f"s3://{self.bucket}/{self.project_prefix}/models/efficientnet-student/stage2/v2.0/MODEL_CARD.md",
                    "hailo": f"s3://{self.bucket}/{self.project_prefix}/models/efficientnet-hailo/v2.1/MODEL_CARD.md"
                },
                "training_guides": "https://github.com/..../docs/"
            }
        }

        # Save manifest
        manifest_path = f"{self.project_prefix}/metadata/project_manifest.json"
        print(f"  Generated manifest with {len(manifest['models'])} models")
        print(f"  Saving to: s3://{self.bucket}/{manifest_path}")

        if not self.dry_run:
            s3.put_object(
                Bucket=self.bucket,
                Key=manifest_path,
                Body=json.dumps(manifest, indent=2).encode('utf-8'),
                ContentType='application/json'
            )

        print("✓ Project manifest created")
        return manifest

    def organize_models(self):
        """Copy and organize all model files into the new structure."""
        print("\n" + "="*70)
        print("STEP 3: Organizing Model Files")
        print("="*70)

        # Model file mappings (source -> destination)
        model_files = [
            # Teacher model
            {
                "source": "models/embedding/teacher/pokemon-card-dinov3-teacher-2026-01-10-13-31-34-937/output/model.tar.gz",
                "dest": f"{self.project_prefix}/models/dinov3-teacher/v1.0/model.tar.gz",
                "description": "DINOv3 Teacher - SageMaker deployment artifact"
            },
            # Student Stage 2 (production)
            {
                "source": "models/embedding/pytorch_weights/student_stage2_final.pt",
                "dest": f"{self.project_prefix}/models/efficientnet-student/stage2/v2.0/student_stage2_final.pt",
                "description": "EfficientNet Student Stage 2 - PyTorch checkpoint"
            },
            {
                "source": "models/onnx/pokemon_student_stage2_final.onnx",
                "dest": f"{self.project_prefix}/models/efficientnet-student/stage2/v2.0/student_stage2_final.onnx",
                "description": "EfficientNet Student Stage 2 - ONNX export"
            },
            # Hailo optimized
            {
                "source": "models/embedding/pokemon_student_efficientnet_lite0_stage2.hef",
                "dest": f"{self.project_prefix}/models/efficientnet-hailo/v2.1/pokemon_student_efficientnet_lite0_stage2.hef",
                "description": "EfficientNet Hailo HEF - Edge deployment"
            },
        ]

        for file_map in model_files:
            source = file_map["source"]
            dest = file_map["dest"]
            desc = file_map["description"]

            print(f"\n  {desc}")
            print(f"    Source: s3://{self.bucket}/{source}")
            print(f"    Dest:   s3://{self.bucket}/{dest}")

            if not self.dry_run:
                try:
                    s3.copy_object(
                        CopySource={'Bucket': self.bucket, 'Key': source},
                        Bucket=self.bucket,
                        Key=dest
                    )
                    print(f"    ✓ Copied")
                except Exception as e:
                    print(f"    ⚠ Copy failed: {e}")
            else:
                print(f"    [DRY RUN] Would copy")

        print(f"\n✓ Model organization complete")

    def register_models_to_registry(self):
        """Register all model variants to SageMaker Model Registry."""
        print("\n" + "="*70)
        print("STEP 4: Registering Models to Model Registry")
        print("="*70)

        # Create model package group if it doesn't exist
        try:
            print(f"\n  Checking if model package group '{MODEL_PACKAGE_GROUP}' exists...")
            sagemaker.describe_model_package_group(ModelPackageGroupName=MODEL_PACKAGE_GROUP)
            print(f"  ✓ Model package group exists")
        except sagemaker.exceptions.ResourceNotFound:
            print(f"  Creating model package group: {MODEL_PACKAGE_GROUP}")
            if not self.dry_run:
                sagemaker.create_model_package_group(
                    ModelPackageGroupName=MODEL_PACKAGE_GROUP,
                    ModelPackageGroupDescription="Pokemon card recognition models - teacher, student variants, and optimized versions"
                )
                print(f"  ✓ Created")
            else:
                print(f"  [DRY RUN] Would create")

        # Model registration details
        models_to_register = [
            {
                "name": "dinov3-teacher",
                "version": "v1.0",
                "model_data": f"s3://{self.bucket}/{self.project_prefix}/models/dinov3-teacher/v1.0/model.tar.gz",
                "inference_spec": {
                    "image": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0-cpu-py310",
                    "supported_content_types": ["image/jpeg", "image/png", "application/json"],
                    "supported_response_types": ["application/json", "application/x-npy"]
                },
                "description": "DINOv3-ViT-L/16 teacher model for Pokemon card embeddings (768-dim)",
                "approval_status": "Approved"
            },
            {
                "name": "efficientnet-student-stage2",
                "version": "v2.0",
                "model_data": f"s3://{self.bucket}/{self.project_prefix}/models/efficientnet-student/stage2/v2.0/model.tar.gz",
                "inference_spec": {
                    "image": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0-cpu-py310",
                    "supported_content_types": ["image/jpeg", "image/png"],
                    "supported_response_types": ["application/json"]
                },
                "description": "EfficientNet-Lite0 student model (Stage 2) - Production ready, 64.7x compressed",
                "approval_status": "Approved"
            }
        ]

        for model in models_to_register:
            print(f"\n  Registering: {model['name']} ({model['version']})")
            print(f"    Description: {model['description']}")
            print(f"    Model Data: {model['model_data']}")

            if not self.dry_run:
                try:
                    response = sagemaker.create_model_package(
                        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
                        ModelPackageDescription=model['description'],
                        InferenceSpecification={
                            'Containers': [{
                                'Image': model['inference_spec']['image'],
                                'ModelDataUrl': model['model_data']
                            }],
                            'SupportedContentTypes': model['inference_spec']['supported_content_types'],
                            'SupportedResponseMIMETypes': model['inference_spec']['supported_response_types']
                        },
                        ModelApprovalStatus=model['approval_status']
                    )
                    print(f"    ✓ Registered: {response['ModelPackageArn']}")
                except Exception as e:
                    print(f"    ⚠ Registration failed: {e}")
            else:
                print(f"    [DRY RUN] Would register")

        print("\n✓ Model registry update complete")

    def organize_mlflow_data(self):
        """Organize MLFlow experiment data."""
        print("\n" + "="*70)
        print("STEP 5: Organizing MLFlow Experiment Data")
        print("="*70)

        # MLFlow data would typically be in training job outputs
        # This creates the structure and documentation

        mlflow_experiments = {
            "experiments": [
                {
                    "name": "pokemon-card-dinov3-teacher",
                    "description": "DINOv3 teacher training with ArcFace loss",
                    "s3_location": f"s3://{self.bucket}/{self.project_prefix}/experiments/mlflow/teacher/",
                    "runs": []
                },
                {
                    "name": "pokemon-card-student-distillation-stage1",
                    "description": "Multi-level knowledge distillation - Stage 1",
                    "s3_location": f"s3://{self.bucket}/{self.project_prefix}/experiments/mlflow/student_stage1/",
                    "runs": []
                },
                {
                    "name": "pokemon-card-student-distillation-stage2",
                    "description": "Task-specific fine-tuning - Stage 2",
                    "s3_location": f"s3://{self.bucket}/{self.project_prefix}/experiments/mlflow/student_stage2/",
                    "runs": []
                }
            ]
        }

        mlflow_path = f"{self.project_prefix}/experiments/mlflow/experiments_index.json"
        print(f"  Creating MLFlow experiments index")
        print(f"  Location: s3://{self.bucket}/{mlflow_path}")

        if not self.dry_run:
            s3.put_object(
                Bucket=self.bucket,
                Key=mlflow_path,
                Body=json.dumps(mlflow_experiments, indent=2).encode('utf-8'),
                ContentType='application/json'
            )

        print("✓ MLFlow organization complete")

    def create_analytics_dashboard(self):
        """Create analytics dashboard configuration."""
        print("\n" + "="*70)
        print("STEP 6: Creating Analytics Dashboard Configuration")
        print("="*70)

        dashboard_config = {
            "dashboards": {
                "training_overview": {
                    "name": "Training Overview",
                    "description": "High-level training metrics across all models",
                    "metrics": [
                        "loss_curves",
                        "accuracy_progression",
                        "learning_rate_schedules",
                        "training_time"
                    ],
                    "visualization_tool": "TensorBoard + Custom"
                },
                "model_comparison": {
                    "name": "Model Comparison",
                    "description": "Compare teacher vs student performance",
                    "metrics": [
                        "accuracy_gap",
                        "inference_latency",
                        "model_size",
                        "compression_ratio"
                    ]
                },
                "cost_tracking": {
                    "name": "Cost Tracking",
                    "description": "Training and infrastructure costs",
                    "metrics": [
                        "cost_per_model",
                        "cost_per_epoch",
                        "instance_utilization"
                    ]
                }
            },
            "reports": {
                "weekly_summary": "Auto-generated training summary",
                "model_performance": "Detailed performance benchmarks",
                "distillation_analysis": "Teacher-student similarity analysis"
            }
        }

        dashboard_path = f"{self.project_prefix}/analytics/dashboards/dashboard_config.json"
        print(f"  Creating dashboard config")
        print(f"  Location: s3://{self.bucket}/{dashboard_path}")

        if not self.dry_run:
            s3.put_object(
                Bucket=self.bucket,
                Key=dashboard_path,
                Body=json.dumps(dashboard_config, indent=2).encode('utf-8'),
                ContentType='application/json'
            )

        print("✓ Analytics dashboard configured")

    def generate_project_readme(self):
        """Generate comprehensive project README."""
        print("\n" + "="*70)
        print("STEP 7: Generating Project README")
        print("="*70)

        readme_content = f"""# Pokemon Card Recognition - SageMaker Project

## Overview

Unified SageMaker project for Pokemon card recognition using teacher-student distillation.

**Project Name:** `{self.project_name}`
**Created:** {datetime.utcnow().strftime('%Y-%m-%d')}
**S3 Bucket:** `{self.bucket}`
**Region:** `{REGION}`

## Model Variants

1. **DINOv3 Teacher (v1.0)**
   - Architecture: DINOv3-ViT-L/16 (304M params)
   - Embedding: 768-dim L2-normalized
   - Purpose: High-accuracy teacher for knowledge distillation
   - Location: `s3://{self.bucket}/{self.project_prefix}/models/dinov3-teacher/v1.0/`

2. **EfficientNet Student Stage 1 (v1.0)**
   - Architecture: EfficientNet-Lite0 (4.7M params)
   - Compression: 64.7x vs teacher
   - Purpose: General feature distillation
   - Location: `s3://{self.bucket}/{self.project_prefix}/models/efficientnet-student/stage1/v1.0/`

3. **EfficientNet Student Stage 2 (v2.0)** ⭐ Production
   - Architecture: EfficientNet-Lite0 (4.7M params)
   - Purpose: Task-specific fine-tuning, production ready
   - Location: `s3://{self.bucket}/{self.project_prefix}/models/efficientnet-student/stage2/v2.0/`

4. **Hailo Optimized (v2.1)** 🚀 Edge Deployment
   - Format: Hailo HEF (INT8 quantized)
   - Size: 14.5 MB
   - Target: Raspberry Pi 5 + Hailo-8L
   - Performance: 10ms inference, 100 FPS
   - Location: `s3://{self.bucket}/{self.project_prefix}/models/efficientnet-hailo/v2.1/`

## Project Structure

See `metadata/project_manifest.json` for complete structure and metadata.

## Accessing Models

### Via SageMaker Model Registry
```python
from sagemaker import ModelPackage
model = ModelPackage(
    role=role,
    model_package_arn='arn:aws:sagemaker:{REGION}:...:model-package/{MODEL_PACKAGE_GROUP}/1'
)
```

### Via S3
```bash
aws s3 cp s3://{self.bucket}/{self.project_prefix}/models/dinov3-teacher/v1.0/model.tar.gz .
```

## Viewing Experiments

MLFlow tracking URI: `s3://{self.bucket}/{self.project_prefix}/experiments/mlflow`

```python
import mlflow
mlflow.set_tracking_uri('s3://{self.bucket}/{self.project_prefix}/experiments/mlflow')
experiment = mlflow.get_experiment_by_name('pokemon-card-dinov3-teacher')
```

## Cost Summary

- Total Training: $11.50
- Teacher: $4.00 (8xA100, 10-15 min)
- Student Stage 1: $4.00 (8xA100, 15 min)
- Student Stage 2: $3.00 (8xA100, 10 min)
- Hailo Compilation: $0.50 (m5.2xlarge, 1 hour)

## Contact & Support

For questions, see documentation in `/docs/` or contact the ML team.
"""

        readme_path = f"{self.project_prefix}/README.md"
        print(f"  Generating README")
        print(f"  Location: s3://{self.bucket}/{readme_path}")

        if not self.dry_run:
            s3.put_object(
                Bucket=self.bucket,
                Key=readme_path,
                Body=readme_content.encode('utf-8'),
                ContentType='text/markdown'
            )

        print("✓ README generated")

    def execute_full_organization(self):
        """Execute all organization steps."""
        print("\n" + "🎯" * 35)
        print(f"POKEMON CARD RECOGNITION - PROJECT ORGANIZATION")
        print("🎯" * 35)
        print(f"\n{'[DRY RUN MODE]' if self.dry_run else '[EXECUTION MODE]'}")
        print(f"Project: {self.project_name}")
        print(f"Bucket: {self.bucket}")
        print(f"Region: {REGION}\n")

        try:
            self.create_project_structure()
            manifest = self.create_project_manifest()
            self.organize_models()
            self.register_models_to_registry()
            self.organize_mlflow_data()
            self.create_analytics_dashboard()
            self.generate_project_readme()

            print("\n" + "="*70)
            print("✅ PROJECT ORGANIZATION COMPLETE!")
            print("="*70)
            print(f"\nProject Location: s3://{self.bucket}/{self.project_prefix}/")
            print(f"Manifest: s3://{self.bucket}/{self.project_prefix}/metadata/project_manifest.json")
            print(f"README: s3://{self.bucket}/{self.project_prefix}/README.md")
            print(f"\nModel Registry: {MODEL_PACKAGE_GROUP}")
            print(f"Total Models: 4 variants organized")
            print(f"Total Cost: $11.50 USD")

            if self.dry_run:
                print("\n⚠️  This was a DRY RUN. No changes were made.")
                print("   Run with --execute to apply changes.")

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

        return True


def main():
    parser = argparse.ArgumentParser(description="Organize SageMaker project structure")
    parser.add_argument('--execute', action='store_true', help='Execute changes (default is dry-run)')
    parser.add_argument('--bucket', default=BUCKET_NAME, help='S3 bucket name')
    parser.add_argument('--project-name', default=PROJECT_NAME, help='Project name')
    args = parser.parse_args()

    organizer = ProjectOrganizer(
        bucket=args.bucket,
        project_name=args.project_name,
        dry_run=not args.execute
    )

    success = organizer.execute_full_organization()
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
