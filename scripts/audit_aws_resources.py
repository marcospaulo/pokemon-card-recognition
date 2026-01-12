#!/usr/bin/env python3
"""
Comprehensive AWS Resource Audit for Pokemon Card Recognition Project

This script inventories all AWS resources that may have been created during
the project across all regions, identifying what's in use vs what can be cleaned up.

Focus areas:
1. IAM Roles (especially SageMaker execution roles)
2. IAM Users
3. SageMaker resources (domains, user profiles, endpoints, training jobs)
4. S3 buckets
5. CloudWatch Log Groups
6. EC2 resources (if any)
"""

import boto3
import json
from datetime import datetime
from collections import defaultdict

# Project-specific identifiers
PROJECT_KEYWORDS = ['pokemon', 'card', 'recognition', 'marcos', 'sagemaker', 'dinov3', 'efficientnet', 'hailo']
PRIMARY_REGION = 'us-east-2'
KNOWN_GOOD_ROLE = 'SageMaker-MarcosAdmin-ExecutionRole'
KNOWN_GOOD_BUCKET = 'pokemon-card-training-us-east-2'

def get_all_regions():
    """Get list of all AWS regions."""
    ec2 = boto3.client('ec2', region_name='us-east-1')
    regions = ec2.describe_regions()['Regions']
    return [r['RegionName'] for r in regions]

def is_project_related(name):
    """Check if a resource name is related to this project."""
    name_lower = name.lower()
    return any(keyword in name_lower for keyword in PROJECT_KEYWORDS)

def audit_iam_roles():
    """Audit all IAM roles."""
    print("\n" + "="*80)
    print("IAM ROLES AUDIT")
    print("="*80)

    iam = boto3.client('iam')

    roles = iam.list_roles()['Roles']

    project_roles = []
    other_roles = []

    for role in roles:
        role_name = role['RoleName']
        created = role['CreateDate']

        # Get attached policies
        attached_policies = iam.list_attached_role_policies(RoleName=role_name)['AttachedPolicies']

        # Get inline policies
        inline_policies = iam.list_role_policies(RoleName=role_name)['PolicyNames']

        role_info = {
            'name': role_name,
            'arn': role['Arn'],
            'created': created.isoformat(),
            'age_days': (datetime.now(created.tzinfo) - created).days,
            'attached_policies': len(attached_policies),
            'inline_policies': len(inline_policies),
            'policies': [p['PolicyName'] for p in attached_policies],
            'path': role['Path']
        }

        if is_project_related(role_name):
            project_roles.append(role_info)
        else:
            other_roles.append(role_info)

    print(f"\n📋 PROJECT-RELATED ROLES: {len(project_roles)}")
    for role in project_roles:
        in_use = "✅ IN USE" if role['name'] == KNOWN_GOOD_ROLE else "❓ CHECK"
        print(f"\n{in_use} - {role['name']}")
        print(f"  Created: {role['created']} ({role['age_days']} days ago)")
        print(f"  Attached Policies: {role['attached_policies']}")
        print(f"  Inline Policies: {role['inline_policies']}")
        if role['policies']:
            print(f"  Policies: {', '.join(role['policies'][:3])}...")

    print(f"\n📋 OTHER AWS SERVICE ROLES: {len(other_roles)}")
    print("(AWS-managed roles like AWSServiceRole* are normal and should NOT be deleted)")

    # Count AWS service roles
    aws_service_roles = [r for r in other_roles if r['name'].startswith('AWSServiceRole')]
    print(f"  AWS Service Roles: {len(aws_service_roles)} (keep these)")

    return {
        'project_roles': project_roles,
        'other_roles': other_roles,
        'aws_service_roles': len(aws_service_roles)
    }

def audit_iam_users():
    """Audit all IAM users."""
    print("\n" + "="*80)
    print("IAM USERS AUDIT")
    print("="*80)

    iam = boto3.client('iam')

    users = iam.list_users()['Users']

    user_details = []

    for user in users:
        username = user['UserName']
        created = user['CreateDate']

        # Get access keys
        try:
            access_keys = iam.list_access_keys(UserName=username)['AccessKeyMetadata']
        except:
            access_keys = []

        # Get attached policies
        try:
            attached_policies = iam.list_attached_user_policies(UserName=username)['AttachedPolicies']
        except:
            attached_policies = []

        # Get groups
        try:
            groups = iam.list_groups_for_user(UserName=username)['Groups']
        except:
            groups = []

        # Get last activity (password last used)
        password_last_used = user.get('PasswordLastUsed')

        user_info = {
            'name': username,
            'arn': user['Arn'],
            'created': created.isoformat(),
            'age_days': (datetime.now(created.tzinfo) - created).days,
            'access_keys': len(access_keys),
            'attached_policies': len(attached_policies),
            'groups': len(groups),
            'password_last_used': password_last_used.isoformat() if password_last_used else 'Never',
            'path': user['Path']
        }

        user_details.append(user_info)

    print(f"\n📋 TOTAL IAM USERS: {len(user_details)}")

    if not user_details:
        print("  No IAM users found (SSO-only setup - GOOD!)")
        return {'users': [], 'sso_only': True}

    for user in user_details:
        print(f"\n❓ {user['name']}")
        print(f"  Created: {user['created']} ({user['age_days']} days ago)")
        print(f"  Access Keys: {user['access_keys']}")
        print(f"  Policies: {user['attached_policies']}")
        print(f"  Groups: {user['groups']}")
        print(f"  Password Last Used: {user['password_last_used']}")

    return {'users': user_details, 'sso_only': False}

def audit_sagemaker_by_region():
    """Audit SageMaker resources across all regions."""
    print("\n" + "="*80)
    print("SAGEMAKER RESOURCES BY REGION")
    print("="*80)

    regions = get_all_regions()
    region_resources = defaultdict(lambda: defaultdict(int))

    for region in regions:
        print(f"\n🌍 Checking {region}...", end=" ")

        try:
            sm = boto3.client('sagemaker', region_name=region)

            # Domains
            domains = sm.list_domains()['Domains']
            region_resources[region]['domains'] = len(domains)

            if domains:
                for domain in domains:
                    print(f"\n  ✅ Domain: {domain['DomainId']} ({domain['DomainName']})")

                    # User profiles
                    try:
                        profiles = sm.list_user_profiles(DomainIdEquals=domain['DomainId'])['UserProfiles']
                        region_resources[region]['user_profiles'] += len(profiles)
                        for profile in profiles:
                            in_use = "✅ IN USE" if region == PRIMARY_REGION and profile['UserProfileName'] == 'marcospaulo' else "❓ CHECK"
                            print(f"    {in_use} Profile: {profile['UserProfileName']}")
                    except Exception as e:
                        print(f"    ⚠ Could not list profiles: {e}")

            # Training jobs (last 100)
            training_jobs = sm.list_training_jobs(MaxResults=100)['TrainingJobSummaries']
            region_resources[region]['training_jobs'] = len(training_jobs)

            # Endpoints
            endpoints = sm.list_endpoints()['Endpoints']
            region_resources[region]['endpoints'] = len(endpoints)

            # Model packages
            try:
                model_packages = sm.list_model_packages(MaxResults=100)['ModelPackageSummaryList']
                region_resources[region]['model_packages'] = len(model_packages)
            except:
                region_resources[region]['model_packages'] = 0

            # Print summary for this region
            total = sum(region_resources[region].values())
            if total > 0:
                print(f"\n  📊 Summary: {total} resources")
                for resource_type, count in region_resources[region].items():
                    if count > 0:
                        print(f"    - {resource_type}: {count}")
            else:
                print("(empty)")

        except Exception as e:
            print(f"⚠ Error: {e}")

    return dict(region_resources)

def audit_s3_buckets():
    """Audit S3 buckets."""
    print("\n" + "="*80)
    print("S3 BUCKETS AUDIT")
    print("="*80)

    s3 = boto3.client('s3')

    buckets = s3.list_buckets()['Buckets']

    bucket_details = []

    for bucket in buckets:
        bucket_name = bucket['Name']
        created = bucket['CreationDate']

        # Get bucket region
        try:
            location = s3.get_bucket_location(Bucket=bucket_name)['LocationConstraint']
            region = location if location else 'us-east-1'
        except:
            region = 'unknown'

        # Get bucket size (approximate)
        try:
            # Note: This is slow for large buckets, we'll skip exact size
            size_estimate = "unknown"
        except:
            size_estimate = "unknown"

        bucket_info = {
            'name': bucket_name,
            'region': region,
            'created': created.isoformat(),
            'age_days': (datetime.now(created.tzinfo) - created).days,
            'project_related': is_project_related(bucket_name)
        }

        bucket_details.append(bucket_info)

    print(f"\n📋 TOTAL S3 BUCKETS: {len(bucket_details)}")

    project_buckets = [b for b in bucket_details if b['project_related']]
    other_buckets = [b for b in bucket_details if not b['project_related']]

    print(f"\n✅ PROJECT-RELATED BUCKETS: {len(project_buckets)}")
    for bucket in project_buckets:
        in_use = "✅ IN USE" if bucket['name'] == KNOWN_GOOD_BUCKET else "❓ CHECK"
        print(f"  {in_use} - {bucket['name']} ({bucket['region']})")
        print(f"    Created: {bucket['created']} ({bucket['age_days']} days ago)")

    print(f"\n📋 OTHER BUCKETS: {len(other_buckets)}")
    for bucket in other_buckets[:5]:  # Show first 5
        print(f"  - {bucket['name']} ({bucket['region']})")
    if len(other_buckets) > 5:
        print(f"  ... and {len(other_buckets) - 5} more")

    return {
        'project_buckets': project_buckets,
        'other_buckets': other_buckets
    }

def audit_cloudwatch_logs():
    """Audit CloudWatch log groups in primary region."""
    print("\n" + "="*80)
    print("CLOUDWATCH LOG GROUPS (us-east-2)")
    print("="*80)

    logs = boto3.client('logs', region_name=PRIMARY_REGION)

    try:
        log_groups = logs.describe_log_groups()['logGroups']

        project_logs = []
        other_logs = []

        for lg in log_groups:
            log_info = {
                'name': lg['logGroupName'],
                'created': lg['creationTime'],
                'size_bytes': lg.get('storedBytes', 0),
                'retention': lg.get('retentionInDays', 'Never expire')
            }

            if is_project_related(lg['logGroupName']) or '/aws/sagemaker/' in lg['logGroupName']:
                project_logs.append(log_info)
            else:
                other_logs.append(log_info)

        print(f"\n📋 PROJECT-RELATED LOG GROUPS: {len(project_logs)}")
        for log in project_logs[:10]:  # Show first 10
            size_mb = log['size_bytes'] / (1024 * 1024)
            print(f"  - {log['name']}")
            print(f"    Size: {size_mb:.2f} MB, Retention: {log['retention']}")

        if len(project_logs) > 10:
            print(f"  ... and {len(project_logs) - 10} more")

        print(f"\n📋 OTHER LOG GROUPS: {len(other_logs)}")

        return {
            'project_logs': project_logs,
            'other_logs': other_logs
        }
    except Exception as e:
        print(f"⚠ Error: {e}")
        return {'project_logs': [], 'other_logs': []}

def generate_cleanup_recommendations(audit_results):
    """Generate cleanup recommendations based on audit."""
    print("\n" + "="*80)
    print("🧹 CLEANUP RECOMMENDATIONS")
    print("="*80)

    recommendations = []

    # IAM Roles
    project_roles = audit_results['iam_roles']['project_roles']
    roles_to_check = [r for r in project_roles if r['name'] != KNOWN_GOOD_ROLE]

    if roles_to_check:
        print(f"\n❓ IAM ROLES TO REVIEW: {len(roles_to_check)}")
        for role in roles_to_check:
            print(f"  - {role['name']} ({role['age_days']} days old)")
            recommendations.append({
                'type': 'iam_role',
                'resource': role['name'],
                'action': 'review_and_delete',
                'reason': f"Project-related role, not the known good role ({KNOWN_GOOD_ROLE})"
            })
    else:
        print(f"\n✅ IAM ROLES: Clean! Only {KNOWN_GOOD_ROLE} found")

    # IAM Users
    if not audit_results['iam_users']['sso_only']:
        print(f"\n❓ IAM USERS: {len(audit_results['iam_users']['users'])} found")
        print("  Consider: You're using SSO (marcospaulo), these IAM users may be unnecessary")
        for user in audit_results['iam_users']['users']:
            print(f"  - {user['name']} (last used: {user['password_last_used']})")
            recommendations.append({
                'type': 'iam_user',
                'resource': user['name'],
                'action': 'review_and_delete',
                'reason': 'IAM user may be unnecessary with SSO setup'
            })
    else:
        print(f"\n✅ IAM USERS: Clean! SSO-only setup (no IAM users)")

    # SageMaker regions
    empty_regions = []
    populated_regions = []

    for region, resources in audit_results['sagemaker_regions'].items():
        total = sum(resources.values())
        if total > 0:
            if region != PRIMARY_REGION:
                empty_regions.append(region)
                for resource_type, count in resources.items():
                    if count > 0:
                        recommendations.append({
                            'type': 'sagemaker',
                            'resource': f"{region}:{resource_type}",
                            'action': 'review_and_delete',
                            'reason': f'SageMaker resources in non-primary region ({PRIMARY_REGION})'
                        })
            populated_regions.append((region, total))

    if empty_regions:
        print(f"\n❓ SAGEMAKER IN WRONG REGIONS: {len(empty_regions)}")
        print(f"  Primary region is {PRIMARY_REGION}, but found resources in:")
        for region in empty_regions:
            print(f"  - {region}")
    else:
        print(f"\n✅ SAGEMAKER REGIONS: Clean! Only in {PRIMARY_REGION}")

    # S3 Buckets
    project_buckets = audit_results['s3_buckets']['project_buckets']
    buckets_to_check = [b for b in project_buckets if b['name'] != KNOWN_GOOD_BUCKET]

    if buckets_to_check:
        print(f"\n❓ S3 BUCKETS TO REVIEW: {len(buckets_to_check)}")
        for bucket in buckets_to_check:
            print(f"  - {bucket['name']} ({bucket['region']}, {bucket['age_days']} days old)")
            recommendations.append({
                'type': 's3_bucket',
                'resource': bucket['name'],
                'action': 'review_and_delete',
                'reason': f"Project-related bucket, not the known good bucket ({KNOWN_GOOD_BUCKET})"
            })
    else:
        print(f"\n✅ S3 BUCKETS: Clean! Only {KNOWN_GOOD_BUCKET} found")

    return recommendations

def save_audit_report(audit_results, recommendations):
    """Save audit report to file."""
    report = {
        'audit_date': datetime.now().isoformat(),
        'primary_region': PRIMARY_REGION,
        'known_good_resources': {
            'role': KNOWN_GOOD_ROLE,
            'bucket': KNOWN_GOOD_BUCKET
        },
        'audit_results': audit_results,
        'recommendations': recommendations
    }

    filename = f"aws_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n💾 Audit report saved to: {filename}")

    return filename

def main():
    print("="*80)
    print("AWS RESOURCE AUDIT - Pokemon Card Recognition Project")
    print("="*80)
    print(f"\nPrimary Region: {PRIMARY_REGION}")
    print(f"Known Good Role: {KNOWN_GOOD_ROLE}")
    print(f"Known Good Bucket: {KNOWN_GOOD_BUCKET}")
    print("\nThis audit will:")
    print("  1. Inventory IAM roles and users")
    print("  2. Check SageMaker resources across ALL regions")
    print("  3. List S3 buckets")
    print("  4. Check CloudWatch log groups")
    print("  5. Generate cleanup recommendations")

    input("\nPress Enter to continue...")

    # Run audits
    audit_results = {}

    audit_results['iam_roles'] = audit_iam_roles()
    audit_results['iam_users'] = audit_iam_users()
    audit_results['sagemaker_regions'] = audit_sagemaker_by_region()
    audit_results['s3_buckets'] = audit_s3_buckets()
    audit_results['cloudwatch_logs'] = audit_cloudwatch_logs()

    # Generate recommendations
    recommendations = generate_cleanup_recommendations(audit_results)

    # Save report
    report_file = save_audit_report(audit_results, recommendations)

    print("\n" + "="*80)
    print("✅ AUDIT COMPLETE")
    print("="*80)
    print(f"\nTotal Recommendations: {len(recommendations)}")
    print(f"\nDetailed report saved to: {report_file}")
    print("\nNext steps:")
    print("  1. Review the recommendations above")
    print("  2. Check the detailed JSON report")
    print("  3. Run cleanup script to remove unnecessary resources")

if __name__ == "__main__":
    main()
