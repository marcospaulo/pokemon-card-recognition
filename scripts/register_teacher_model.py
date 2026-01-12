#!/usr/bin/env python3
"""
Register Teacher Model to SageMaker Model Registry

Retroactively registers the trained DINOv3-ViT-L/16 teacher model with:
- Full training metadata (hyperparameters, metrics)
- Model lineage (training job, datasets used)
- Architecture information
- Approval workflow for production deployment

This creates a versioned artifact in the Model Registry for tracking.
"""

import boto3
import sagemaker
from sagemaker.model import ModelPackage
import json
import tarfile
import tempfile
from pathlib import Path
import torch

# Initialize
session = sagemaker.Session()
sm_client = boto3.client('sagemaker')
s3_client = boto3.client('s3')
sts = boto3.client('sts')

account_id = sts.get_caller_identity()['Account']
region = session.boto_region_name

# Configuration
MODEL_PACKAGE_GROUP_NAME = "pokemon-card-recognition-models"
TEACHER_S3_PATH = "s3://pokemon-card-training-us-east-2/models/embedding/teacher/pokemon-card-dinov3-teacher-2026-01-10-13-31-34-937/output/model.tar.gz"


def extract_model_metadata(model_tar_path):
    """
    Extract metadata from teacher model archive.

    Downloads and inspects the model checkpoint to extract:
    - Model architecture details
    - Training configuration
    - Performance metrics
    - Number of parameters
    """
    print("\n[1/4] Extracting model metadata...")

    # Parse S3 path
    bucket = model_tar_path.split('/')[2]
    key = '/'.join(model_tar_path.split('/')[3:])

    # Download model.tar.gz
    with tempfile.TemporaryDirectory() as tmpdir:
        local_tar = Path(tmpdir) / 'model.tar.gz'
        print(f"  Downloading: {model_tar_path}")
        s3_client.download_file(bucket, key, str(local_tar))

        # Extract
        extract_dir = Path(tmpdir) / 'model'
        extract_dir.mkdir()
        with tarfile.open(local_tar, 'r:gz') as tar:
            tar.extractall(extract_dir)

        print(f"  Extracted {len(list(extract_dir.iterdir()))} files")

        # Find checkpoint file
        checkpoint_files = [
            'phase2_checkpoint.pt',
            'checkpoint.pt',
            'model.pt',
            'pytorch_model.bin'
        ]

        checkpoint_path = None
        for fname in checkpoint_files:
            candidate = extract_dir / fname
            if candidate.exists():
                checkpoint_path = candidate
                break

        if not checkpoint_path:
            raise FileNotFoundError(f"No checkpoint found in model archive. Files: {list(extract_dir.iterdir())}")

        print(f"  Loading checkpoint: {checkpoint_path.name}")

        # Try loading with weights_only=True first (safer, avoids custom class issues)
        checkpoint = None
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            print(f"  ✅ Loaded checkpoint with weights_only=True")
        except Exception as e:
            print(f"  ⚠️  weights_only=True failed: {type(e).__name__}")
            try:
                # Try loading just the keys to count parameters
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                print(f"  ✅ Loaded checkpoint with weights_only=False")
            except AttributeError as attr_err:
                print(f"  ⚠️  Full checkpoint loading failed (custom classes): {attr_err}")
                print(f"  ℹ️  Using known metadata for DINOv3-ViT-L/16")
                checkpoint = None  # Use default metadata

        # Extract metadata (use defaults if checkpoint is None or doesn't have keys)
        metadata = {
            'architecture': 'DINOv3-ViT-L/16',
            'model_name': 'dinov3_vitl16',
            'embedding_dim': checkpoint.get('embedding_dim', 768) if checkpoint and isinstance(checkpoint, dict) else 768,
            'num_classes': checkpoint.get('num_classes', 17592) if checkpoint and isinstance(checkpoint, dict) else 17592,
            'parameters': 304000000,  # Known: DINOv3-ViT-L/16 has 304M parameters
            'training_epochs': checkpoint.get('epoch', 'Unknown') if checkpoint and isinstance(checkpoint, dict) else 'Unknown',
            'framework': 'PyTorch 2.8.0',
            'training_approach': 'Self-supervised DINOv3 + ArcFace fine-tuning',
        }

        # Extract metrics if available
        metrics = {}
        if checkpoint and isinstance(checkpoint, dict):
            metric_keys = ['top1_acc', 'top5_acc', 'val_loss', 'train_loss']
            for key in metric_keys:
                if key in checkpoint:
                    metrics[key] = float(checkpoint[key])

        metadata['final_metrics'] = metrics

        print(f"  ✅ Architecture: {metadata['architecture']}")
        print(f"  ✅ Parameters: {metadata['parameters']:,}")
        print(f"  ✅ Embedding dim: {metadata['embedding_dim']}")
        if metrics:
            print(f"  ✅ Metrics: {metrics}")

        return metadata


def get_training_job_metadata(s3_path):
    """
    Extract training job metadata from S3 path.

    SageMaker training jobs follow pattern:
    s3://bucket/pytorch-training-YYYY-MM-DD-HH-MM-SS-mmm/output/model.tar.gz
    """
    print("\n[2/4] Extracting training job info...")

    # Parse training job name from S3 path
    parts = s3_path.split('/')
    training_job_name = None

    # Try standard SageMaker format: s3://bucket/pytorch-training-YYYY-MM-DD-HH-MM-SS-mmm/output/model.tar.gz
    for part in parts:
        if part.startswith('pytorch-training-'):
            training_job_name = part
            break

    # Try custom format: s3://bucket/models/embedding/teacher/pokemon-card-dinov3-teacher-YYYY-MM-DD-HH-MM-SS-mmm/output/model.tar.gz
    if not training_job_name:
        for part in parts:
            if 'teacher' in part.lower() and any(c.isdigit() for c in part):
                training_job_name = part
                break

    if not training_job_name:
        print("  ⚠️  Could not parse training job name from S3 path")
        return {}

    try:
        # Get training job details
        response = sm_client.describe_training_job(TrainingJobName=training_job_name)

        metadata = {
            'training_job_name': training_job_name,
            'training_job_arn': response['TrainingJobArn'],
            'instance_type': response['ResourceConfig']['InstanceType'],
            'instance_count': response['ResourceConfig']['InstanceCount'],
            'training_time_seconds': response['TrainingTimeInSeconds'],
            'billable_time_seconds': response['BillableTimeInSeconds'],
            'creation_time': response['CreationTime'].isoformat(),
            'training_end_time': response['TrainingEndTime'].isoformat() if 'TrainingEndTime' in response else None,
        }

        # Extract hyperparameters
        if 'HyperParameters' in response:
            metadata['hyperparameters'] = response['HyperParameters']

        # Extract input data config
        if 'InputDataConfig' in response:
            metadata['input_data'] = [
                {
                    'channel': channel['ChannelName'],
                    's3_uri': channel['DataSource']['S3DataSource']['S3Uri']
                }
                for channel in response['InputDataConfig']
            ]

        print(f"  ✅ Training job: {training_job_name}")
        print(f"  ✅ Instance: {metadata['instance_count']}x {metadata['instance_type']}")
        print(f"  ✅ Training time: {metadata['training_time_seconds']}s")

        return metadata

    except Exception as e:
        print(f"  ⚠️  Could not fetch training job details: {e}")
        return {'training_job_name': training_job_name}


def register_teacher_model(model_metadata, training_metadata):
    """
    Register teacher model to SageMaker Model Registry.

    Creates a model package with full metadata and lineage tracking.
    """
    print("\n[3/4] Registering teacher model to Model Registry...")

    # Combine metadata
    full_metadata = {
        **model_metadata,
        **training_metadata,
        'model_type': 'teacher',
        'purpose': 'Teacher model for knowledge distillation to edge-deployable student',
        'inference_target': 'Cloud/GPU',
        'registration_date': Path(__file__).stat().st_mtime,
    }

    # Create model package
    try:
        response = sm_client.create_model_package(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
            ModelPackageDescription=(
                "DINOv3-ViT-L/16 teacher model trained on 17,592 Pokemon cards. "
                "Self-supervised learning + ArcFace fine-tuning. "
                "Used for knowledge distillation to EfficientNet-Lite0 student."
            ),
            InferenceSpecification={
                'Containers': [
                    {
                        'Image': f'763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference:2.0.1-gpu-py310',
                        'ModelDataUrl': TEACHER_S3_PATH,
                        'Framework': 'PYTORCH',
                        'FrameworkVersion': '2.0.1',
                    }
                ],
                'SupportedContentTypes': ['application/x-image'],
                'SupportedResponseMIMETypes': ['application/json'],
            },
            ModelApprovalStatus='Approved',  # Auto-approve teacher model
            MetadataProperties={
                'CommitId': 'N/A',  # Could link to git commit
                'Repository': 'pokemon-card-recognition',
                'GeneratedBy': training_metadata.get('training_job_name', 'manual'),
            },
            CustomerMetadataProperties={
                'Architecture': str(full_metadata.get('architecture', 'Unknown')),
                'ModelName': str(full_metadata.get('model_name', 'Unknown')),
                'Parameters': str(full_metadata.get('parameters', 'Unknown')),
                'EmbeddingDim': str(full_metadata.get('embedding_dim', '768')),
                'NumClasses': str(full_metadata.get('num_classes', '17592')),
                'ModelType': 'teacher',
                'Purpose': 'knowledge_distillation_source',
                'Framework': 'PyTorch-2.8.0',
                'TrainingJob': str(training_metadata.get('training_job_name', 'manual-registration')),
            },
            # Note: Tags are not supported on model package versions, only on the Model Package Group
        )

        model_package_arn = response['ModelPackageArn']
        print(f"  ✅ Registered teacher model")
        print(f"  ARN: {model_package_arn}")

        return model_package_arn

    except Exception as e:
        print(f"  ❌ Registration failed: {e}")
        raise


def print_summary(model_package_arn, model_metadata):
    """Print registration summary."""
    print("\n[4/4] Registration complete!")
    print("="*70)
    print("✅ TEACHER MODEL REGISTERED")
    print("="*70)
    print(f"\nModel Package ARN:")
    print(f"  {model_package_arn}")
    print(f"\nArchitecture:")
    print(f"  {model_metadata['architecture']} ({model_metadata['parameters']:,} parameters)")
    print(f"\nView in SageMaker Console:")
    print(f"  https://console.aws.amazon.com/sagemaker/home?region={region}#/model-packages/{MODEL_PACKAGE_GROUP_NAME}")
    print(f"\nNext steps:")
    print(f"  1. Student training will automatically link to this teacher model")
    print(f"  2. View model lineage: teacher → student distillation")
    print(f"  3. Track all model versions in one place")
    print()


def main():
    """Register teacher model retroactively."""
    print("="*70)
    print("Register Teacher Model to SageMaker Model Registry")
    print("="*70)
    print(f"Teacher model: {TEACHER_S3_PATH}")
    print(f"Model Package Group: {MODEL_PACKAGE_GROUP_NAME}")
    print()

    try:
        # Extract model metadata
        model_metadata = extract_model_metadata(TEACHER_S3_PATH)

        # Get training job metadata
        training_metadata = get_training_job_metadata(TEACHER_S3_PATH)

        # Register to Model Registry
        model_package_arn = register_teacher_model(model_metadata, training_metadata)

        # Print summary
        print_summary(model_package_arn, model_metadata)

        return model_package_arn

    except Exception as e:
        print(f"\n❌ Registration failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    model_package_arn = main()
    if model_package_arn:
        print(f"✅ Success! Model registered: {model_package_arn}")
    else:
        print("❌ Failed to register model")
        exit(1)
