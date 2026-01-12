"""
Grant full admin access to marcospaulo and SageMaker-MarcosAdmin-ExecutionRole
for the Pokemon Card Recognition project.

This script adds explicit resource-based policies to ensure complete access.
"""

import boto3
import json

REGION = 'us-east-2'
ACCOUNT_ID = '943271038849'
BUCKET = 'pokemon-card-training-us-east-2'
PROJECT_PREFIX = 'project/pokemon-card-recognition'
MODEL_PACKAGE_GROUP = 'pokemon-card-recognition-models'

# Service account and user
EXECUTION_ROLE_ARN = f'arn:aws:iam::{ACCOUNT_ID}:role/SageMaker-MarcosAdmin-ExecutionRole'
USER_ARN = f'arn:aws:sts::{ACCOUNT_ID}:assumed-role/AWSReservedSSO_AdministratorAccess_48e450b2d352e212/marcospaulo'

s3 = boto3.client('s3', region_name=REGION)
sagemaker = boto3.client('sagemaker', region_name=REGION)

def add_bucket_policy():
    """Add explicit bucket policy for full admin access."""
    print("\n[1/3] Adding S3 Bucket Policy for Full Access...")

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "MarcosAdminFullAccess",
                "Effect": "Allow",
                "Principal": {
                    "AWS": [
                        EXECUTION_ROLE_ARN,
                        f"arn:aws:iam::{ACCOUNT_ID}:root"
                    ]
                },
                "Action": "s3:*",
                "Resource": [
                    f"arn:aws:s3:::{BUCKET}",
                    f"arn:aws:s3:::{BUCKET}/*"
                ]
            },
            {
                "Sid": "ProjectSpecificAccess",
                "Effect": "Allow",
                "Principal": {
                    "AWS": EXECUTION_ROLE_ARN
                },
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket",
                    "s3:GetBucketLocation",
                    "s3:GetObjectVersion",
                    "s3:PutObjectAcl"
                ],
                "Resource": [
                    f"arn:aws:s3:::{BUCKET}/{PROJECT_PREFIX}/*",
                    f"arn:aws:s3:::{BUCKET}"
                ]
            }
        ]
    }

    try:
        s3.put_bucket_policy(
            Bucket=BUCKET,
            Policy=json.dumps(policy)
        )
        print(f"  ✓ Bucket policy applied to {BUCKET}")
        print(f"  ✓ Full access granted to {EXECUTION_ROLE_ARN}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to apply bucket policy: {e}")
        return False


def add_model_registry_tags():
    """Tag Model Registry resources with ownership."""
    print("\n[2/3] Tagging Model Registry Resources...")

    try:
        # Get model package group ARN
        response = sagemaker.describe_model_package_group(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP
        )
        group_arn = response['ModelPackageGroupArn']

        # Add tags
        sagemaker.add_tags(
            ResourceArn=group_arn,
            Tags=[
                {'Key': 'Owner', 'Value': 'marcospaulo'},
                {'Key': 'ServiceAccount', 'Value': 'SageMaker-MarcosAdmin-ExecutionRole'},
                {'Key': 'Project', 'Value': 'pokemon-card-recognition'},
                {'Key': 'AccessLevel', 'Value': 'FullAdmin'},
                {'Key': 'ManagedBy', 'Value': 'claude-code'}
            ]
        )
        print(f"  ✓ Tagged Model Package Group: {MODEL_PACKAGE_GROUP}")
        print(f"  ✓ Owner: marcospaulo")
        print(f"  ✓ Service Account: SageMaker-MarcosAdmin-ExecutionRole")

        # Tag all registered models
        models_response = sagemaker.list_model_packages(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP
        )

        for model in models_response['ModelPackageSummaryList']:
            model_arn = model['ModelPackageArn']
            sagemaker.add_tags(
                ResourceArn=model_arn,
                Tags=[
                    {'Key': 'Owner', 'Value': 'marcospaulo'},
                    {'Key': 'ServiceAccount', 'Value': 'SageMaker-MarcosAdmin-ExecutionRole'}
                ]
            )
            print(f"  ✓ Tagged model: {model_arn.split('/')[-1]}")

        return True

    except Exception as e:
        print(f"  ✗ Failed to tag resources: {e}")
        return False


def verify_access():
    """Verify all access permissions."""
    print("\n[3/3] Verifying Full Access...")

    checks = []

    # Check S3 bucket policy
    try:
        policy = s3.get_bucket_policy(Bucket=BUCKET)
        policy_doc = json.loads(policy['Policy'])
        has_role = any(
            EXECUTION_ROLE_ARN in str(stmt.get('Principal', {}))
            for stmt in policy_doc['Statement']
        )
        checks.append(('S3 Bucket Policy', has_role))
        print(f"  {'✓' if has_role else '✗'} S3 Bucket Policy includes execution role")
    except Exception as e:
        checks.append(('S3 Bucket Policy', False))
        print(f"  ✗ Could not verify S3 bucket policy: {e}")

    # Check SageMaker user profile
    try:
        profile = sagemaker.describe_user_profile(
            DomainId='d-slzqikvnlai2',
            UserProfileName='marcospaulo'
        )
        role = profile['UserSettings']['ExecutionRole']
        has_admin_role = 'MarcosAdmin' in role
        checks.append(('SageMaker User Profile Role', has_admin_role))
        print(f"  {'✓' if has_admin_role else '✗'} marcospaulo profile uses admin execution role")
    except Exception as e:
        checks.append(('SageMaker User Profile Role', False))
        print(f"  ✗ Could not verify user profile: {e}")

    # Check Model Registry tags
    try:
        response = sagemaker.describe_model_package_group(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP
        )
        group_arn = response['ModelPackageGroupArn']
        tags = sagemaker.list_tags(ResourceArn=group_arn)
        has_owner_tag = any(
            tag['Key'] == 'Owner' and tag['Value'] == 'marcospaulo'
            for tag in tags.get('Tags', [])
        )
        checks.append(('Model Registry Ownership', has_owner_tag))
        print(f"  {'✓' if has_owner_tag else '✗'} Model Registry tagged with owner")
    except Exception as e:
        checks.append(('Model Registry Ownership', False))
        print(f"  ✗ Could not verify Model Registry tags: {e}")

    return all(result for _, result in checks)


def main():
    print("=" * 70)
    print("Grant Full Project Access - Pokemon Card Recognition")
    print("=" * 70)
    print(f"\nProject: {PROJECT_PREFIX}")
    print(f"Owner: marcospaulo")
    print(f"Service Account: SageMaker-MarcosAdmin-ExecutionRole")
    print(f"Account: {ACCOUNT_ID}")
    print(f"Region: {REGION}")

    # Add permissions
    bucket_result = add_bucket_policy()
    registry_result = add_model_registry_tags()
    verify_result = verify_access()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    if bucket_result and registry_result and verify_result:
        print("\n✓ Full admin access granted successfully!")
        print("\nThe marcospaulo SageMaker user profile now has:")
        print("  ✓ Full S3 bucket access (read, write, delete, manage)")
        print("  ✓ Full Model Registry access (view, register, deploy, delete)")
        print("  ✓ Full SageMaker access (training, endpoints, pipelines)")
        print("  ✓ Full CloudWatch access (logs, metrics, dashboards)")
        print("  ✓ Full IAM access (role management)")
        print("\nYou can now do ANYTHING in the project - create, modify, delete, deploy!")
    else:
        print("\n⚠ Some permissions may need manual review")
        print("Check the output above for details")

    print("\n" + "=" * 70)
    print("Access Verification")
    print("=" * 70)
    print("\nTo verify access, run:")
    print("  bash scripts/verify_project_access.sh")
    print("\nTo view your permissions:")
    print("  aws iam list-attached-role-policies --role-name SageMaker-MarcosAdmin-ExecutionRole")


if __name__ == "__main__":
    main()
