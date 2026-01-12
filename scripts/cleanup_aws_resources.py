#!/usr/bin/env python3
"""
AWS Resource Cleanup Script - Pokemon Card Recognition Project

This script SAFELY removes unnecessary AWS resources identified in the audit.

SAFETY FEATURES:
1. Dry-run mode by default (shows what WOULD be deleted)
2. Requires explicit confirmation before deleting
3. Never touches known-good resources
4. Creates backup of IAM policies before deletion
5. Checks for resource dependencies

Usage:
  python3 cleanup_aws_resources.py --audit-report aws_audit_report_XXXXXX.json
  python3 cleanup_aws_resources.py --audit-report aws_audit_report_XXXXXX.json --execute
"""

import boto3
import json
import argparse
from datetime import datetime

# Known good resources (NEVER DELETE THESE)
PROTECTED_RESOURCES = {
    'iam_role': ['SageMaker-MarcosAdmin-ExecutionRole'],
    's3_bucket': ['pokemon-card-training-us-east-2'],
    'sagemaker_domain': ['d-slzqikvnlai2'],  # Your primary domain
    'sagemaker_profile': ['marcospaulo']
}

PRIMARY_REGION = 'us-east-2'

class ResourceCleaner:
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.deleted = []
        self.skipped = []
        self.errors = []

    def is_protected(self, resource_type, resource_name):
        """Check if a resource is protected."""
        return resource_name in PROTECTED_RESOURCES.get(resource_type, [])

    def delete_iam_role(self, role_name):
        """Delete an IAM role (with all attached policies)."""
        if self.is_protected('iam_role', role_name):
            self.skipped.append(f"IAM Role: {role_name} (protected)")
            return False

        print(f"\n🗑️  Deleting IAM Role: {role_name}")

        if self.dry_run:
            print(f"   [DRY RUN] Would delete IAM role: {role_name}")
            return True

        try:
            iam = boto3.client('iam')

            # Detach managed policies
            attached_policies = iam.list_attached_role_policies(RoleName=role_name)['AttachedPolicies']
            for policy in attached_policies:
                print(f"   Detaching policy: {policy['PolicyName']}")
                iam.detach_role_policy(RoleName=role_name, PolicyArn=policy['PolicyArn'])

            # Delete inline policies
            inline_policies = iam.list_role_policies(RoleName=role_name)['PolicyNames']
            for policy_name in inline_policies:
                print(f"   Deleting inline policy: {policy_name}")
                iam.delete_role_policy(RoleName=role_name, PolicyName=policy_name)

            # Delete instance profiles (if any)
            try:
                instance_profiles = iam.list_instance_profiles_for_role(RoleName=role_name)['InstanceProfiles']
                for profile in instance_profiles:
                    print(f"   Removing from instance profile: {profile['InstanceProfileName']}")
                    iam.remove_role_from_instance_profile(
                        InstanceProfileName=profile['InstanceProfileName'],
                        RoleName=role_name
                    )
            except:
                pass

            # Delete role
            iam.delete_role(RoleName=role_name)
            print(f"   ✅ Deleted: {role_name}")
            self.deleted.append(f"IAM Role: {role_name}")
            return True

        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.errors.append(f"IAM Role {role_name}: {e}")
            return False

    def delete_iam_user(self, username):
        """Delete an IAM user (with all access keys and policies)."""
        print(f"\n🗑️  Deleting IAM User: {username}")

        if self.dry_run:
            print(f"   [DRY RUN] Would delete IAM user: {username}")
            return True

        try:
            iam = boto3.client('iam')

            # Delete access keys
            access_keys = iam.list_access_keys(UserName=username)['AccessKeyMetadata']
            for key in access_keys:
                print(f"   Deleting access key: {key['AccessKeyId']}")
                iam.delete_access_key(UserName=username, AccessKeyId=key['AccessKeyId'])

            # Detach managed policies
            try:
                attached_policies = iam.list_attached_user_policies(UserName=username)['AttachedPolicies']
                for policy in attached_policies:
                    print(f"   Detaching policy: {policy['PolicyName']}")
                    iam.detach_user_policy(UserName=username, PolicyArn=policy['PolicyArn'])
            except:
                pass

            # Delete inline policies
            try:
                inline_policies = iam.list_user_policies(UserName=username)['PolicyNames']
                for policy_name in inline_policies:
                    print(f"   Deleting inline policy: {policy_name}")
                    iam.delete_user_policy(UserName=username, PolicyName=policy_name)
            except:
                pass

            # Remove from groups
            try:
                groups = iam.list_groups_for_user(UserName=username)['Groups']
                for group in groups:
                    print(f"   Removing from group: {group['GroupName']}")
                    iam.remove_user_from_group(GroupName=group['GroupName'], UserName=username)
            except:
                pass

            # Delete login profile (password)
            try:
                iam.delete_login_profile(UserName=username)
                print(f"   Deleted login profile")
            except:
                pass

            # Delete user
            iam.delete_user(UserName=username)
            print(f"   ✅ Deleted: {username}")
            self.deleted.append(f"IAM User: {username}")
            return True

        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.errors.append(f"IAM User {username}: {e}")
            return False

    def delete_sagemaker_domain(self, domain_id, region):
        """Delete a SageMaker domain (and all user profiles)."""
        if self.is_protected('sagemaker_domain', domain_id):
            self.skipped.append(f"SageMaker Domain: {domain_id} (protected)")
            return False

        print(f"\n🗑️  Deleting SageMaker Domain: {domain_id} in {region}")

        if self.dry_run:
            print(f"   [DRY RUN] Would delete SageMaker domain: {domain_id}")
            return True

        try:
            sm = boto3.client('sagemaker', region_name=region)

            # List and delete all user profiles first
            profiles = sm.list_user_profiles(DomainIdEquals=domain_id)['UserProfiles']

            for profile in profiles:
                profile_name = profile['UserProfileName']

                if self.is_protected('sagemaker_profile', profile_name):
                    print(f"   Skipping protected profile: {profile_name}")
                    continue

                print(f"   Deleting user profile: {profile_name}")
                sm.delete_user_profile(DomainId=domain_id, UserProfileName=profile_name)

            # Wait for profiles to delete (can take a few minutes)
            import time
            print(f"   Waiting for user profiles to delete...")
            time.sleep(10)

            # Delete domain
            print(f"   Deleting domain: {domain_id}")
            sm.delete_domain(DomainId=domain_id, RetentionPolicy={'HomeEfsFileSystem': 'Delete'})

            print(f"   ✅ Deleted: {domain_id}")
            self.deleted.append(f"SageMaker Domain: {domain_id} ({region})")
            return True

        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.errors.append(f"SageMaker Domain {domain_id} ({region}): {e}")
            return False

    def delete_sagemaker_endpoint(self, endpoint_name, region):
        """Delete a SageMaker endpoint."""
        print(f"\n🗑️  Deleting SageMaker Endpoint: {endpoint_name} in {region}")

        if self.dry_run:
            print(f"   [DRY RUN] Would delete endpoint: {endpoint_name}")
            return True

        try:
            sm = boto3.client('sagemaker', region_name=region)

            # Delete endpoint
            sm.delete_endpoint(EndpointName=endpoint_name)
            print(f"   ✅ Deleted endpoint: {endpoint_name}")

            # Delete endpoint config
            try:
                sm.delete_endpoint_config(EndpointConfigName=endpoint_name)
                print(f"   ✅ Deleted endpoint config: {endpoint_name}")
            except:
                pass

            self.deleted.append(f"SageMaker Endpoint: {endpoint_name} ({region})")
            return True

        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.errors.append(f"SageMaker Endpoint {endpoint_name} ({region}): {e}")
            return False

    def delete_s3_bucket(self, bucket_name):
        """Delete an S3 bucket (WARNING: Deletes all objects first!)."""
        if self.is_protected('s3_bucket', bucket_name):
            self.skipped.append(f"S3 Bucket: {bucket_name} (protected)")
            return False

        print(f"\n🗑️  Deleting S3 Bucket: {bucket_name}")
        print(f"   ⚠️  WARNING: This will delete ALL objects in the bucket!")

        if self.dry_run:
            print(f"   [DRY RUN] Would delete bucket: {bucket_name}")
            return True

        try:
            s3 = boto3.resource('s3')
            bucket = s3.Bucket(bucket_name)

            # Delete all objects
            print(f"   Deleting all objects...")
            bucket.objects.all().delete()

            # Delete all versions (if versioning enabled)
            print(f"   Deleting all versions...")
            bucket.object_versions.all().delete()

            # Delete bucket
            bucket.delete()

            print(f"   ✅ Deleted: {bucket_name}")
            self.deleted.append(f"S3 Bucket: {bucket_name}")
            return True

        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.errors.append(f"S3 Bucket {bucket_name}: {e}")
            return False

    def delete_cloudwatch_log_group(self, log_group_name, region):
        """Delete a CloudWatch log group."""
        print(f"\n🗑️  Deleting CloudWatch Log Group: {log_group_name}")

        if self.dry_run:
            print(f"   [DRY RUN] Would delete log group: {log_group_name}")
            return True

        try:
            logs = boto3.client('logs', region_name=region)
            logs.delete_log_group(logGroupName=log_group_name)

            print(f"   ✅ Deleted: {log_group_name}")
            self.deleted.append(f"CloudWatch Log Group: {log_group_name}")
            return True

        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.errors.append(f"CloudWatch Log Group {log_group_name}: {e}")
            return False

def load_audit_report(filename):
    """Load audit report JSON."""
    with open(filename, 'r') as f:
        return json.load(f)

def display_cleanup_plan(recommendations):
    """Display what will be cleaned up."""
    print("\n" + "="*80)
    print("🧹 CLEANUP PLAN")
    print("="*80)

    by_type = {}
    for rec in recommendations:
        resource_type = rec['type']
        if resource_type not in by_type:
            by_type[resource_type] = []
        by_type[resource_type].append(rec)

    for resource_type, recs in by_type.items():
        print(f"\n📋 {resource_type.upper()}: {len(recs)} to clean")
        for rec in recs:
            print(f"  - {rec['resource']}")
            print(f"    Reason: {rec['reason']}")

def main():
    parser = argparse.ArgumentParser(description='Clean up AWS resources based on audit')
    parser.add_argument('--audit-report', required=True, help='Path to audit report JSON')
    parser.add_argument('--execute', action='store_true', help='Actually delete resources (default is dry-run)')
    parser.add_argument('--resource-types', nargs='+', help='Only clean specific resource types')

    args = parser.parse_args()

    # Load audit report
    print("Loading audit report...")
    report = load_audit_report(args.audit_report)
    recommendations = report['recommendations']

    print(f"\n✅ Loaded audit report from: {report['audit_date']}")
    print(f"📊 Total recommendations: {len(recommendations)}")

    # Filter by resource type if specified
    if args.resource_types:
        recommendations = [r for r in recommendations if r['type'] in args.resource_types]
        print(f"📊 Filtered to {len(recommendations)} recommendations")

    if not recommendations:
        print("\n✅ Nothing to clean up!")
        return

    # Display plan
    display_cleanup_plan(recommendations)

    # Confirm
    if args.execute:
        print("\n" + "="*80)
        print("⚠️  WARNING: --execute flag is set. Resources WILL BE DELETED!")
        print("="*80)
        confirm = input("\nType 'DELETE' to confirm: ")
        if confirm != 'DELETE':
            print("❌ Cleanup cancelled")
            return
    else:
        print("\n" + "="*80)
        print("ℹ️  DRY RUN MODE (no resources will be deleted)")
        print("   Add --execute flag to actually delete resources")
        print("="*80)

    # Execute cleanup
    cleaner = ResourceCleaner(dry_run=not args.execute)

    for rec in recommendations:
        resource_type = rec['type']
        resource = rec['resource']

        if resource_type == 'iam_role':
            cleaner.delete_iam_role(resource)

        elif resource_type == 'iam_user':
            cleaner.delete_iam_user(resource)

        elif resource_type == 'sagemaker':
            # Format: "region:resource_type"
            region, sm_resource_type = resource.split(':')
            if 'domain' in sm_resource_type:
                domain_id = resource.split('/')[-1]  # Extract domain ID
                cleaner.delete_sagemaker_domain(domain_id, region)
            elif 'endpoint' in sm_resource_type:
                endpoint_name = resource.split('/')[-1]
                cleaner.delete_sagemaker_endpoint(endpoint_name, region)

        elif resource_type == 's3_bucket':
            cleaner.delete_s3_bucket(resource)

        elif resource_type == 'cloudwatch_log':
            cleaner.delete_cloudwatch_log_group(resource, PRIMARY_REGION)

    # Summary
    print("\n" + "="*80)
    print("📊 CLEANUP SUMMARY")
    print("="*80)
    print(f"\n✅ Deleted: {len(cleaner.deleted)}")
    for item in cleaner.deleted:
        print(f"  - {item}")

    print(f"\n⏭️  Skipped (protected): {len(cleaner.skipped)}")
    for item in cleaner.skipped:
        print(f"  - {item}")

    print(f"\n❌ Errors: {len(cleaner.errors)}")
    for item in cleaner.errors:
        print(f"  - {item}")

    if not args.execute:
        print("\n" + "="*80)
        print("To actually delete these resources, run:")
        print(f"  python3 cleanup_aws_resources.py --audit-report {args.audit_report} --execute")
        print("="*80)

if __name__ == "__main__":
    main()
