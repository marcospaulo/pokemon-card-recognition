#!/usr/bin/env python3
"""
Setup SageMaker Project for Pokemon Card Recognition

Creates a unified MLOps project to track:
1. Teacher model (DINOv3-ViT-L/16) - retroactively registered
2. Student models (EfficientNet-Lite0) - Stage 1 & Stage 2
3. Model lineage (teacher → student distillation)
4. All training metrics, hyperparameters, and artifacts

This provides a single source of truth for all model versions.
"""

import boto3
import sagemaker
from sagemaker.model import ModelPackage
from datetime import datetime
import json

# Initialize clients
session = sagemaker.Session()
sm_client = boto3.client('sagemaker')
sts = boto3.client('sts')

# Get account details
account_id = sts.get_caller_identity()['Account']
region = session.boto_region_name

# Project configuration
PROJECT_NAME = "pokemon-card-recognition"
MODEL_PACKAGE_GROUP_NAME = f"{PROJECT_NAME}-models"

print("="*70)
print("SageMaker Project Setup: Pokemon Card Recognition")
print("="*70)
print(f"Account: {account_id}")
print(f"Region: {region}")
print(f"Project: {PROJECT_NAME}")
print(f"Model Package Group: {MODEL_PACKAGE_GROUP_NAME}")
print()


def create_model_package_group():
    """
    Create Model Package Group for version control.

    This is like a "repository" for model versions - all teacher and student
    models will be registered here with full lineage tracking.
    """
    print("[1/3] Creating Model Package Group...")

    try:
        response = sm_client.create_model_package_group(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
            ModelPackageGroupDescription=(
                "Pokemon Card Recognition Models: DINOv3 teacher and "
                "EfficientNet-Lite0 student models with full distillation lineage"
            ),
            Tags=[
                {'Key': 'Project', 'Value': PROJECT_NAME},
                {'Key': 'Team', 'Value': 'ML'},
                {'Key': 'Purpose', 'Value': 'Pokemon card embedding and recognition'},
                {'Key': 'Architecture', 'Value': 'Knowledge Distillation'},
            ]
        )
        print(f"  ✅ Created Model Package Group: {MODEL_PACKAGE_GROUP_NAME}")
        print(f"  ARN: {response['ModelPackageGroupArn']}")
        return response['ModelPackageGroupArn']

    except sm_client.exceptions.ResourceInUse:
        print(f"  ℹ️  Model Package Group already exists: {MODEL_PACKAGE_GROUP_NAME}")
        # Get existing group
        response = sm_client.describe_model_package_group(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME
        )
        print(f"  ARN: {response['ModelPackageGroupArn']}")
        return response['ModelPackageGroupArn']
    except Exception as e:
        print(f"  ❌ Error: {e}")
        raise


def create_mlflow_experiment():
    """
    Create MLflow experiment for unified tracking.

    All training runs (teacher + student) will log to this experiment.
    """
    print("\n[2/3] Setting up MLflow experiment...")

    import mlflow

    # Set tracking URI (SageMaker-managed MLflow)
    tracking_uri = f"arn:aws:sagemaker:{region}:{account_id}:mlflow-tracking-server/pokemon-card-tracking"

    try:
        # Try to use managed MLflow if available
        mlflow.set_tracking_uri(tracking_uri)
        print(f"  ✅ Using SageMaker-managed MLflow")
        print(f"  Tracking URI: {tracking_uri}")
    except Exception:
        # Fall back to file-based tracking
        tracking_uri = f"s3://pokemon-card-training-{region}/mlflow"
        mlflow.set_tracking_uri(tracking_uri)
        print(f"  ℹ️  Using S3-backed MLflow")
        print(f"  Tracking URI: {tracking_uri}")

    # Create experiment
    experiment_name = f"{PROJECT_NAME}-training"
    try:
        mlflow.create_experiment(
            name=experiment_name,
            tags={
                'project': PROJECT_NAME,
                'purpose': 'Track teacher and student model training',
            }
        )
        print(f"  ✅ Created experiment: {experiment_name}")
    except Exception:
        print(f"  ℹ️  Experiment already exists: {experiment_name}")

    return experiment_name, tracking_uri


def create_model_metadata_template():
    """
    Create standardized metadata template for model registration.

    This ensures all models are registered with complete information.
    """
    print("\n[3/3] Creating model metadata template...")

    metadata_template = {
        "teacher_model": {
            "model_name": "dinov3-vitl16-teacher",
            "architecture": "DINOv3-ViT-L/16",
            "parameters": "304M",
            "embedding_dim": 768,
            "training_approach": "Self-supervised + ArcFace fine-tuning",
            "training_stages": ["phase1_general", "phase2_pokemon_cards"],
            "input_size": "224x224",
            "framework": "PyTorch 2.8.0 + Transformers",
            "purpose": "Teacher model for knowledge distillation",
            "inference_target": "Cloud/GPU",
            "metrics_tracked": [
                "top1_accuracy",
                "top5_accuracy",
                "embedding_similarity",
                "retrieval_accuracy"
            ]
        },
        "student_model": {
            "model_name": "efficientnet-lite0-student",
            "architecture": "EfficientNet-Lite0",
            "parameters": "4.7M",
            "embedding_dim": 768,
            "training_approach": "Knowledge distillation from DINOv3",
            "training_stages": ["stage1_general", "stage2_task_specific"],
            "teacher_model": "dinov3-vitl16-teacher",
            "distillation_losses": {
                "feature": 0.35,
                "kl_divergence": 0.25,
                "attention": 0.25,
                "high_frequency": 0.15
            },
            "input_size": "224x224",
            "framework": "PyTorch 2.8.0 + timm",
            "purpose": "Edge deployment on Hailo-8L",
            "inference_target": "Raspberry Pi + Hailo-8L",
            "hailo_compatible": True,
            "normalization": "BatchNorm",
            "metrics_tracked": [
                "cosine_similarity_to_teacher",
                "student_top1",
                "teacher_top1",
                "gap",
                "feature_loss",
                "kl_loss",
                "attention_loss",
                "highfreq_loss"
            ]
        },
        "naming_convention": {
            "pattern": "{model_type}-{architecture}-{stage}-{version}",
            "examples": {
                "teacher": "teacher-dinov3-vitl16-phase2-v1",
                "student_stage1": "student-efficientnet-lite0-stage1-v1",
                "student_stage2": "student-efficientnet-lite0-stage2-v1",
                "onnx_export": "student-efficientnet-lite0-onnx-v1",
                "hailo_hef": "student-efficientnet-lite0-hailo8l-v1"
            }
        }
    }

    # Save template
    template_path = "config/model_metadata_template.json"
    import os
    os.makedirs(os.path.dirname(template_path), exist_ok=True)

    with open(template_path, 'w') as f:
        json.dump(metadata_template, f, indent=2)

    print(f"  ✅ Created metadata template: {template_path}")
    print(f"  Teacher: {metadata_template['teacher_model']['model_name']}")
    print(f"  Student: {metadata_template['student_model']['model_name']}")
    print(f"  Naming convention defined ✓")

    return metadata_template


def print_next_steps(experiment_name, tracking_uri):
    """Print next steps for user."""
    print("\n" + "="*70)
    print("✅ SETUP COMPLETE")
    print("="*70)
    print("\nModel Package Group created:")
    print(f"  • Name: {MODEL_PACKAGE_GROUP_NAME}")
    print(f"  • View in console: https://console.aws.amazon.com/sagemaker/home?region={region}#/model-packages/{MODEL_PACKAGE_GROUP_NAME}")

    print("\nMLflow Experiment created:")
    print(f"  • Name: {experiment_name}")
    print(f"  • URI: {tracking_uri}")

    print("\nNext steps:")
    print("  1. Register teacher model:")
    print("     python scripts/register_teacher_model.py")
    print()
    print("  2. Student training will auto-register models:")
    print("     python scripts/launch_student_distillation_8xA100.py")
    print()
    print("  3. View all models:")
    print(f"     aws sagemaker list-model-packages \\")
    print(f"       --model-package-group-name {MODEL_PACKAGE_GROUP_NAME}")
    print()


def main():
    """Setup SageMaker project infrastructure."""
    try:
        # Create model package group (version control)
        mpg_arn = create_model_package_group()

        # Setup MLflow experiment
        experiment_name, tracking_uri = create_mlflow_experiment()

        # Create metadata template
        metadata = create_model_metadata_template()

        # Print next steps
        print_next_steps(experiment_name, tracking_uri)

        return {
            'model_package_group': MODEL_PACKAGE_GROUP_NAME,
            'model_package_group_arn': mpg_arn,
            'mlflow_experiment': experiment_name,
            'mlflow_tracking_uri': tracking_uri,
            'metadata_template': metadata
        }

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        raise


if __name__ == '__main__':
    result = main()
