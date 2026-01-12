#!/usr/bin/env python3
"""
Register Stage 2 Student Model to SageMaker Model Registry

Registers the final EfficientNet-Lite0 Stage 2 model with:
- Task-specific fine-tuning metadata (ArcFace on 17,592 classes)
- Training configuration
- Links to Stage 1 and teacher models
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
STUDENT_S3_PATH = "s3://pokemon-card-training-us-east-2/models/embedding/student/pytorch-training-2026-01-11-23-31-10-757/output/model.tar.gz"
TRAINING_JOB_NAME = "pytorch-training-2026-01-11-23-31-10-757"


def get_training_metadata():
    """Get training job metadata."""
    print("\n[1/3] Extracting training job info...")

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
        print(f"  ✅ Training time: {metadata['training_time_seconds']}s (~{metadata['training_time_seconds']//60} min)")

        return metadata

    except Exception as e:
        print(f"  ⚠️  Could not fetch training job details: {e}")
        return {'training_job_name': TRAINING_JOB_NAME}


def get_stage1_model_arn():
    """Get Stage 1 model ARN for lineage."""
    print("\n[2/3] Getting Stage 1 model ARN for lineage...")
    try:
        packages = sm_client.list_model_packages(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
            MaxResults=10
        )
        for pkg in packages.get('ModelPackageSummaryList', []):
            desc = pkg.get('ModelPackageDescription', '')
            if 'Stage 1' in desc or 'stage1' in desc.lower():
                print(f"  ✅ Found Stage 1: {pkg['ModelPackageArn']}")
                return pkg['ModelPackageArn']
    except:
        pass

    print(f"  ⚠️  Could not find Stage 1 model")
    return None


def register_student_model(training_metadata, stage1_arn):
    """Register Stage 2 student model to Model Registry."""
    print("\n[3/3] Registering Stage 2 student model to Model Registry...")

    # Get hyperparameters
    hp = training_metadata.get('hyperparameters', {})
    student_model = hp.get('student-model', 'efficientnet_lite0').strip('"')
    teacher_model = hp.get('teacher-model', 'dinov3_vitl16').strip('"')
    epochs = hp.get('epochs-stage2', '20').strip('"')

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
                f"EfficientNet-Lite0 Stage 2 FINAL student model with task-specific fine-tuning. "
                f"Distilled from DINOv3-ViT-L/16 teacher + ArcFace fine-tuning on 17,592 Pokemon cards ({epochs} epochs). "
                f"4.7M parameters optimized for Raspberry Pi + Hailo-8L edge deployment. "
                f"Ready for ONNX export and HEF compilation."
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
                'Stage': 'stage2',
                'Status': 'final',
                'Architecture': 'EfficientNet-Lite0',
                'Parameters': '4700000',
                'EmbeddingDim': '768',
                'NumClasses': '17592',
                'TeacherModel': teacher_model,
                'TeacherARN': str(teacher_arn) if teacher_arn else 'Unknown',
                'Stage1ARN': str(stage1_arn) if stage1_arn else 'Unknown',
                'FineTuningEpochs': epochs,
                'LossFunction': 'ArcFace',
                'HailoCompatible': 'true',
                'Normalization': 'BatchNorm',
                'Framework': 'PyTorch-2.8.0',
                'TrainingJob': str(training_metadata.get('training_job_name', 'manual-registration')),
                'DeploymentTarget': 'RaspberryPi-Hailo8L',
                'NextStep': 'ONNX-Export',
            },
        )

        model_package_arn = response['ModelPackageArn']
        print(f"  ✅ Registered Stage 2 final student model")
        print(f"  ARN: {model_package_arn}")

        return model_package_arn

    except Exception as e:
        print(f"  ❌ Registration failed: {e}")
        raise


def print_summary(model_package_arn):
    """Print registration summary."""
    print("\n" + "="*70)
    print("✅ STAGE 2 FINAL STUDENT MODEL REGISTERED")
    print("="*70)
    print(f"\nModel Package ARN:")
    print(f"  {model_package_arn}")
    print(f"\nArchitecture:")
    print(f"  EfficientNet-Lite0 (4.7M parameters)")
    print(f"\nTraining:")
    print(f"  Stage 1: 30 epochs (knowledge distillation)")
    print(f"  Stage 2: 20 epochs (ArcFace fine-tuning on 17,592 classes)")
    print(f"\nView in SageMaker Console:")
    print(f"  https://console.aws.amazon.com/sagemaker/home?region={region}#/model-packages/{MODEL_PACKAGE_GROUP_NAME}")
    print(f"\nModel Registry Status:")
    print(f"  Version 1: DINOv3-ViT-L/16 Teacher (304M params)")
    print(f"  Version 2: EfficientNet-Lite0 Stage 1 (4.7M params)")
    print(f"  Version 3: EfficientNet-Lite0 Stage 2 FINAL (4.7M params) ← THIS ONE")
    print(f"\nNext steps:")
    print(f"  1. Export to ONNX: python scripts/export_student_to_onnx.py")
    print(f"  2. Compile to Hailo HEF")
    print(f"  3. Deploy to Raspberry Pi")
    print()


def main():
    """Register Stage 2 student model."""
    print("="*70)
    print("Register Stage 2 FINAL Student Model to SageMaker Model Registry")
    print("="*70)
    print(f"Student model: {STUDENT_S3_PATH}")
    print(f"Model Package Group: {MODEL_PACKAGE_GROUP_NAME}")
    print()

    try:
        # Get training job metadata
        training_metadata = get_training_metadata()

        # Get Stage 1 model ARN
        stage1_arn = get_stage1_model_arn()

        # Register to Model Registry
        model_package_arn = register_student_model(training_metadata, stage1_arn)

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
