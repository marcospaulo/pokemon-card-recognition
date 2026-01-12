#!/usr/bin/env python3
"""
Create a Simple SageMaker Project

Uses the simplest approach to create a SageMaker Project that aggregates
all Pokemon Card Recognition resources.
"""

import boto3
from botocore.exceptions import ClientError

sm_client = boto3.client('sagemaker', region_name='us-east-2')
sts = boto3.client('sts')

account_id = sts.get_caller_identity()['Account']
region = 'us-east-2'

PROJECT_NAME = "pokemon-card-recognition"
PROJECT_DESCRIPTION = (
    "Pokemon Card Recognition ML Project: Knowledge distillation from "
    "DINOv3-ViT-L/16 teacher (304M) to EfficientNet-Lite0 student (4.7M) "
    "for Raspberry Pi + Hailo-8L edge deployment."
)

print("="*70)
print("Creating SageMaker Project")
print("="*70)
print(f"Project: {PROJECT_NAME}")
print(f"Region: {region}")
print()

try:
    # Try using TemplateProviders (newer approach that doesn't require Service Catalog)
    response = sm_client.create_project(
        ProjectName=PROJECT_NAME,
        ProjectDescription=PROJECT_DESCRIPTION,
        Tags=[
            {'Key': 'Purpose', 'Value': 'KnowledgeDistillation'},
            {'Key': 'Architecture', 'Value': 'DINOv3-EfficientNet'},
            {'Key': 'DeploymentTarget', 'Value': 'RaspberryPi-Hailo8L'},
            {'Key': 'TeacherModel', 'Value': 'DINOv3-ViT-L-16'},
            {'Key': 'StudentModel', 'Value': 'EfficientNet-Lite0'},
            {'Key': 'Classes', 'Value': '17592'},
        ]
    )

    print("✅ Project created successfully!")
    print(f"   ARN: {response['ProjectArn']}")
    print(f"   ID: {response['ProjectId']}")
    print()
    print("View in SageMaker Console:")
    print(f"   https://console.aws.amazon.com/sagemaker/home?region={region}#/projects/{PROJECT_NAME}")
    print()

except ClientError as e:
    error_code = e.response['Error']['Code']
    error_msg = e.response['Error']['Message']

    print(f"❌ Failed to create project")
    print(f"   Error: {error_code}")
    print(f"   Message: {error_msg}")
    print()

    if 'ServiceCatalogProvisioningDetails' in error_msg or 'TemplateProviders' in error_msg:
        print("ℹ️  SageMaker Projects require one of:")
        print("   1. AWS Service Catalog templates (for MLOps CI/CD)")
        print("   2. Template providers (custom project templates)")
        print()
        print("Since we don't need CI/CD pipelines, our Model Package Group")
        print("provides equivalent organization for tracking models, training,")
        print("and metadata.")
        print()
        print("Current setup:")
        print(f"   ✅ Model Package Group: pokemon-card-recognition-models")
        print(f"   ✅ Console: https://console.aws.amazon.com/sagemaker/home?region={region}#/model-packages/pokemon-card-recognition-models")
        print()

    exit(1)
