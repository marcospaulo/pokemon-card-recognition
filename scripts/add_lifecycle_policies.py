"""
Add S3 lifecycle policies to optimize storage costs

This script applies lifecycle policies to archive old training outputs to Glacier
after 90 days, reducing storage costs by ~90% for infrequently accessed data.
"""

import boto3
import json
from datetime import datetime

REGION = 'us-east-2'
BUCKET = 'pokemon-card-training-us-east-2'

s3 = boto3.client('s3', region_name=REGION)


def get_current_lifecycle_config():
    """Get current lifecycle configuration if it exists."""
    try:
        response = s3.get_bucket_lifecycle_configuration(Bucket=BUCKET)
        print("Current lifecycle configuration:")
        print(json.dumps(response, indent=2, default=str))
        return response.get('Rules', [])
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchLifecycleConfiguration':
            print("No existing lifecycle configuration found.")
            return []
        else:
            raise


def apply_lifecycle_policies():
    """Apply lifecycle policies to the bucket."""
    print("=" * 70)
    print("S3 Lifecycle Policy Configuration")
    print("=" * 70)
    print(f"\nBucket: {BUCKET}")
    print(f"Region: {REGION}\n")

    # Get current config
    existing_rules = get_current_lifecycle_config()

    # Define new lifecycle rules
    lifecycle_rules = [
        {
            'ID': 'ArchiveOldTrainingOutputs',
            'Status': 'Enabled',
            'Filter': {
                'Prefix': 'models/embedding/'
            },
            'Transitions': [
                {
                    'Days': 90,
                    'StorageClass': 'GLACIER'
                }
            ],
            'NoncurrentVersionTransitions': [
                {
                    'NoncurrentDays': 30,
                    'StorageClass': 'GLACIER'
                }
            ]
        },
        {
            'ID': 'DeleteIncompleteMultipartUploads',
            'Status': 'Enabled',
            'Filter': {},
            'AbortIncompleteMultipartUpload': {
                'DaysAfterInitiation': 7
            }
        },
        {
            'ID': 'ArchiveOldProfilingData',
            'Status': 'Enabled',
            'Filter': {
                'Prefix': 'project/pokemon-card-recognition/profiling/'
            },
            'Transitions': [
                {
                    'Days': 180,
                    'StorageClass': 'GLACIER'
                }
            ]
        }
    ]

    print("\nLifecycle rules to be applied:")
    print("-" * 70)

    for rule in lifecycle_rules:
        print(f"\nRule: {rule['ID']}")
        print(f"  Status: {rule['Status']}")
        if 'Prefix' in rule['Filter']:
            print(f"  Applies to: {rule['Filter']['Prefix']}")
        else:
            print(f"  Applies to: All objects")

        if 'Transitions' in rule:
            for transition in rule['Transitions']:
                print(f"  → Move to {transition['StorageClass']} after {transition['Days']} days")

        if 'NoncurrentVersionTransitions' in rule:
            for transition in rule['NoncurrentVersionTransitions']:
                print(f"  → Move non-current versions to {transition['StorageClass']} after {transition['NoncurrentDays']} days")

        if 'AbortIncompleteMultipartUpload' in rule:
            days = rule['AbortIncompleteMultipartUpload']['DaysAfterInitiation']
            print(f"  → Abort incomplete multipart uploads after {days} days")

    print("\n" + "-" * 70)
    print("\nExpected cost savings:")
    print("  - Standard storage: $0.023/GB-month")
    print("  - Glacier storage: $0.004/GB-month")
    print("  - Savings: ~83% on archived data")
    print("\nEstimated savings after 90 days:")
    print("  - Training outputs (~10 GB): $0.23/month → $0.04/month = $0.19/month saved")
    print("  - Annual savings: ~$2.28/year")

    # Apply the configuration
    print("\n" + "=" * 70)
    print("Applying lifecycle configuration...")
    print("=" * 70)

    try:
        s3.put_bucket_lifecycle_configuration(
            Bucket=BUCKET,
            LifecycleConfiguration={
                'Rules': lifecycle_rules
            }
        )
        print("\n✓ Lifecycle policies applied successfully!")

        # Verify
        print("\nVerifying configuration...")
        new_config = s3.get_bucket_lifecycle_configuration(Bucket=BUCKET)
        print(f"✓ Confirmed: {len(new_config['Rules'])} rules active")

        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"\nActive lifecycle rules: {len(new_config['Rules'])}")
        print("\nWhat happens next:")
        print("  1. Objects older than 90 days in models/embedding/ will transition to Glacier")
        print("  2. Profiling data older than 180 days will transition to Glacier")
        print("  3. Incomplete multipart uploads will be cleaned up after 7 days")
        print("  4. Cost savings will accumulate over time as data ages")
        print("\nTo view archived objects later:")
        print("  aws s3api list-objects-v2 --bucket {BUCKET} --prefix models/embedding/")
        print("\nTo restore from Glacier:")
        print("  aws s3api restore-object --bucket {BUCKET} --key <object-key> --restore-request Days=7")

    except Exception as e:
        print(f"\n✗ Failed to apply lifecycle policies: {e}")
        return False

    return True


if __name__ == "__main__":
    success = apply_lifecycle_policies()
    exit(0 if success else 1)
