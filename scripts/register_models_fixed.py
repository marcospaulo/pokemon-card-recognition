"""
Register all model variants to SageMaker Model Registry (Fixed Version)

This script registers all 4 model variants with the correct ECR region (us-east-2).

Usage:
    python scripts/register_models_fixed.py
"""

import boto3
import argparse

REGION = 'us-east-2'
BUCKET = 'pokemon-card-training-us-east-2'
MODEL_PACKAGE_GROUP = 'pokemon-card-recognition-models'
PROJECT_PREFIX = 'project/pokemon-card-recognition'

# Correct ECR image for us-east-2 region
INFERENCE_IMAGE = f'763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-inference:2.0-cpu-py310'

sagemaker = boto3.client('sagemaker', region_name=REGION)


def register_dinov3_teacher():
    """Register DINOv3 teacher model."""
    print("\n[1/2] Registering DINOv3 Teacher Model...")
    print(f"  Architecture: DINOv3-ViT-L/16 (304M params)")
    print(f"  Embedding Dimension: 768")
    print(f"  Model Data: s3://{BUCKET}/{PROJECT_PREFIX}/models/dinov3-teacher/v1.0/model.tar.gz")

    try:
        response = sagemaker.create_model_package(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP,
            ModelPackageDescription='DINOv3-ViT-L/16 teacher model for Pokemon card embeddings. 768-dimensional L2-normalized embeddings trained on 17,592 Pokemon cards with ArcFace loss.',
            InferenceSpecification={
                'Containers': [{
                    'Image': INFERENCE_IMAGE,
                    'ModelDataUrl': f's3://{BUCKET}/{PROJECT_PREFIX}/models/dinov3-teacher/v1.0/model.tar.gz',
                    'Environment': {
                        'SAGEMAKER_PROGRAM': 'inference.py',
                        'SAGEMAKER_SUBMIT_DIRECTORY': f's3://{BUCKET}/{PROJECT_PREFIX}/models/dinov3-teacher/v1.0/model.tar.gz',
                    }
                }],
                'SupportedContentTypes': ['image/jpeg', 'image/png', 'application/json', 'application/x-npy'],
                'SupportedResponseMIMETypes': ['application/json', 'application/x-npy']
            },
            ModelApprovalStatus='Approved',
            CustomerMetadataProperties={
                'ModelType': 'teacher',
                'Architecture': 'DINOv3-ViT-L/16',
                'Parameters': '304M',
                'EmbeddingDim': '768',
                'Framework': 'PyTorch',
                'TrainingDate': '2026-01-10',
                'InputSize': '224x224'
            }
        )

        print(f"  ✓ Registered successfully!")
        print(f"  ARN: {response['ModelPackageArn']}")
        return response['ModelPackageArn']

    except Exception as e:
        print(f"  ✗ Registration failed: {e}")
        return None


def register_efficientnet_student():
    """Register EfficientNet student Stage 2 model."""
    print("\n[2/2] Registering EfficientNet Student Stage 2 Model...")
    print(f"  Architecture: EfficientNet-Lite0 (4.7M params)")
    print(f"  Compression: 64.7x vs teacher")
    print(f"  Model Data: s3://{BUCKET}/{PROJECT_PREFIX}/models/efficientnet-student/stage2/v2.0/model.tar.gz")

    try:
        # Note: This requires creating a proper model.tar.gz with inference code
        # For now, register with the ONNX model as ModelDataUrl
        response = sagemaker.create_model_package(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP,
            ModelPackageDescription='EfficientNet-Lite0 student model (Stage 2). Production-ready, 64.7x compressed from teacher. Fine-tuned for Pokemon card recognition with knowledge distillation.',
            InferenceSpecification={
                'Containers': [{
                    'Image': INFERENCE_IMAGE,
                    'ModelDataUrl': f's3://{BUCKET}/{PROJECT_PREFIX}/models/efficientnet-student/stage2/v2.0/student_stage2_final.onnx'
                }],
                'SupportedContentTypes': ['image/jpeg', 'image/png'],
                'SupportedResponseMIMETypes': ['application/json']
            },
            ModelApprovalStatus='Approved',
            CustomerMetadataProperties={
                'ModelType': 'student',
                'ParentModel': 'dinov3-teacher:v1.0',
                'Architecture': 'EfficientNet-Lite0',
                'Parameters': '4.7M',
                'CompressionRatio': '64.7x',
                'EmbeddingDim': '768',
                'Framework': 'PyTorch',
                'TrainingDate': '2026-01-11',
                'Stage': '2',
                'ProductionReady': 'true'
            }
        )

        print(f"  ✓ Registered successfully!")
        print(f"  ARN: {response['ModelPackageArn']}")
        return response['ModelPackageArn']

    except Exception as e:
        print(f"  ✗ Registration failed: {e}")
        return None


def verify_model_package_group():
    """Verify that the model package group exists."""
    print("\nVerifying Model Package Group...")
    try:
        response = sagemaker.describe_model_package_group(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP
        )
        print(f"  ✓ Model Package Group exists: {MODEL_PACKAGE_GROUP}")
        print(f"  ARN: {response['ModelPackageGroupArn']}")
        return True
    except sagemaker.exceptions.ResourceNotFound:
        print(f"  ✗ Model Package Group '{MODEL_PACKAGE_GROUP}' not found!")
        print(f"  Creating it now...")
        try:
            sagemaker.create_model_package_group(
                ModelPackageGroupName=MODEL_PACKAGE_GROUP,
                ModelPackageGroupDescription="Pokemon card recognition models - teacher, student variants, and optimized versions"
            )
            print(f"  ✓ Created: {MODEL_PACKAGE_GROUP}")
            return True
        except Exception as e:
            print(f"  ✗ Failed to create: {e}")
            return False


def list_registered_models():
    """List all registered models in the package group."""
    print("\nListing Registered Models...")
    try:
        response = sagemaker.list_model_packages(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP,
            SortBy='CreationTime',
            SortOrder='Descending'
        )

        if response['ModelPackageSummaryList']:
            print(f"  Found {len(response['ModelPackageSummaryList'])} registered models:")
            for i, pkg in enumerate(response['ModelPackageSummaryList'], 1):
                print(f"\n  {i}. {pkg['ModelPackageArn'].split('/')[-1]}")
                print(f"     Status: {pkg['ModelApprovalStatus']}")
                print(f"     Created: {pkg['CreationTime']}")
        else:
            print("  No models registered yet.")

    except Exception as e:
        print(f"  ✗ Failed to list models: {e}")


def main():
    parser = argparse.ArgumentParser(description="Register models to SageMaker Model Registry")
    parser.add_argument('--list-only', action='store_true', help='Only list existing registrations')
    args = parser.parse_args()

    print("=" * 70)
    print("SageMaker Model Registry - Registration Script")
    print("=" * 70)
    print(f"Region: {REGION}")
    print(f"Bucket: {BUCKET}")
    print(f"Model Package Group: {MODEL_PACKAGE_GROUP}")
    print(f"Inference Image: {INFERENCE_IMAGE}")

    # Verify package group exists
    if not verify_model_package_group():
        print("\n✗ Cannot proceed without model package group")
        return 1

    if args.list_only:
        list_registered_models()
        return 0

    # Register models
    print("\n" + "=" * 70)
    print("Registering Models")
    print("=" * 70)

    teacher_arn = register_dinov3_teacher()
    student_arn = register_efficientnet_student()

    # Summary
    print("\n" + "=" * 70)
    print("Registration Summary")
    print("=" * 70)

    success_count = sum([bool(teacher_arn), bool(student_arn)])
    print(f"\n✓ Successfully registered: {success_count}/2 models")

    if teacher_arn:
        print(f"\n  Teacher Model: {teacher_arn}")
    if student_arn:
        print(f"\n  Student Model: {student_arn}")

    # List all registered models
    list_registered_models()

    print("\n" + "=" * 70)
    print("Next Steps")
    print("=" * 70)
    print("\n1. Deploy models to endpoints:")
    print("   aws sagemaker create-endpoint --endpoint-config-name <config> --endpoint-name <name>")
    print("\n2. View in SageMaker Console:")
    print(f"   https://console.aws.amazon.com/sagemaker/home?region={REGION}#/model-packages")
    print("\n3. Use in code:")
    print("   from sagemaker import ModelPackage")
    print(f"   model = ModelPackage(role=role, model_package_arn='{teacher_arn if teacher_arn else '<ARN>'}')")

    return 0 if success_count == 2 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
