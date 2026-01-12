#!/usr/bin/env python3
"""
Register Stage 1 Student Model to SageMaker Model Registry

Registers the EfficientNet-Lite0 Stage 1 student model with:
- Distillation metadata from DINOv3 teacher
- Training configuration
- Hailo-8L compatibility information
"""

import boto3
import sagemaker
from pathlib import Path

# Initialize
session = sagemaker.Session()
sm_client = boto3.client('sagemaker')
sts = boto3.client('sts')

account_id = sts.get_caller_identity()['Account']
region = session.boto_region_name

# Configuration
MODEL_PACKAGE_GROUP_NAME = "pokemon-card-recognition-models"
STUDENT_S3_PATH = "s3://pokemon-card-training-us-east-2/models/embedding/student/pytorch-training-2026-01-11-22-26-25-713/output/model.tar.gz"
TRAINING_JOB_NAME = "pytorch-training-2026-01-11-22-26-25-713"


def get_training_metadata():
    """Get training job metadata."""
    print("\n[1/2] Extracting training job info...")

    try:
        response = sm_client.describe_training_job(TrainingJobName=TRAINING_JOB_NAME)

        metadata = {
            'training_job_name': TRAINING_JOB_NAME,
            'training_job_arn': response['TrainingJobArn'],
            'instance_type': response['ResourceConfig']['InstanceType'],
            'instance_count': response['ResourceConfig']['InstanceCount'],
            'training_time_seconds': response['TrainingTimeInSeconds'],
            'hyperparameters': response.get('HyperParameters', {}),
        }

        print(f"  ✅ Training job: {TRAINING_JOB_NAME}")
        print(f"  ✅ Instance: {metadata['instance_count']}x {metadata['instance_type']}")
        print(f"  ✅ Training time: {metadata['training_time_seconds']}s")

        return metadata

    except Exception as e:
        print(f"  ⚠️  Could not fetch training job details: {e}")
        return {'training_job_name': TRAINING_JOB_NAME}


def register_student_model(training_metadata):
    """Register Stage 1 student model to Model Registry."""
    print("\n[2/2] Registering Stage 1 student model to Model Registry...")

    # Get hyperparameters
    hp = training_metadata.get('hyperparameters', {})
    student_model = hp.get('student-model', 'efficientnet_lite0').strip('"')
    teacher_model = hp.get('teacher-model', 'dinov3_vitl16').strip('"')
    epochs = hp.get('epochs-stage1', '30').strip('"')

    # Get teacher model ARN for lineage
    try:
        teacher_packages = sm_client.list_model_packages(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
            MaxResults=5
        )
        teacher_arn = None
        for pkg in teacher_packages.get('ModelPackageSummaryList', []):
            if 'teacher' in pkg.get('ModelPackageDescription', '').lower():
                teacher_arn = pkg['ModelPackageArn']
                break
    except:
        teacher_arn = None

    try:
        response = sm_client.create_model_package(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
            ModelPackageDescription=(
                f"EfficientNet-Lite0 Stage 1 student model distilled from DINOv3-ViT-L/16 teacher. "
                f"Pure knowledge distillation training ({epochs} epochs). "
                f"4.7M parameters optimized for Hailo-8L edge deployment."
            ),
            InferenceSpecification={
                'Containers': [
                    {
                        'Image': f'763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference:2.0.1-gpu-py310',
                        'ModelDataUrl': STUDENT_S3_PATH,
                        'Framework': 'PYTORCH',
                        'FrameworkVersion': '2.0.1',
                    }
                ],
                'SupportedContentTypes': ['application/x-image'],
                'SupportedResponseMIMETypes': ['application/json'],
            },
            ModelApprovalStatus='Approved',
            MetadataProperties={
                'CommitId': 'N/A',
                'Repository': 'pokemon-card-recognition',
                'GeneratedBy': training_metadata.get('training_job_name', 'manual'),
            },
            CustomerMetadataProperties={
                'ModelType': 'student',
                'Stage': 'stage1',
                'Architecture': 'EfficientNet-Lite0',
                'Parameters': '4700000',
                'EmbeddingDim': '768',
                'TeacherModel': teacher_model,
                'TeacherARN': str(teacher_arn) if teacher_arn else 'Unknown',
                'DistillationEpochs': epochs,
                'HailoCompatible': 'true',
                'Normalization': 'BatchNorm',
                'Framework': 'PyTorch-2.8.0',
                'TrainingJob': str(training_metadata.get('training_job_name', 'manual-registration')),
            },
        )

        model_package_arn = response['ModelPackageArn']
        print(f"  ✅ Registered Stage 1 student model")
        print(f"  ARN: {model_package_arn}")

        return model_package_arn

    except Exception as e:
        print(f"  ❌ Registration failed: {e}")
        raise


def print_summary(model_package_arn):
    """Print registration summary."""
    print("\n" + "="*70)
    print("✅ STAGE 1 STUDENT MODEL REGISTERED")
    print("="*70)
    print(f"\nModel Package ARN:")
    print(f"  {model_package_arn}")
    print(f"\nArchitecture:")
    print(f"  EfficientNet-Lite0 (4.7M parameters)")
    print(f"\nView in SageMaker Console:")
    print(f"  https://console.aws.amazon.com/sagemaker/home?region={region}#/model-packages/{MODEL_PACKAGE_GROUP_NAME}")
    print(f"\nNext steps:")
    print(f"  1. Launch Stage 2 fine-tuning with card labels")
    print(f"  2. Register Stage 2 model")
    print(f"  3. Export to ONNX for Hailo compilation")
    print()


def main():
    """Register Stage 1 student model."""
    print("="*70)
    print("Register Stage 1 Student Model to SageMaker Model Registry")
    print("="*70)
    print(f"Student model: {STUDENT_S3_PATH}")
    print(f"Model Package Group: {MODEL_PACKAGE_GROUP_NAME}")
    print()

    try:
        # Get training job metadata
        training_metadata = get_training_metadata()

        # Register to Model Registry
        model_package_arn = register_student_model(training_metadata)

        # Print summary
        print_summary(model_package_arn)

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
