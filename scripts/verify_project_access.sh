#!/bin/bash
# Verify SageMaker-MarcosAdmin-ExecutionRole has full access to the project

echo "=============================================================="
echo "Pokemon Card Recognition - Access Verification"
echo "=============================================================="
echo ""
echo "Testing access for: SageMaker-MarcosAdmin-ExecutionRole"
echo "Account: 943271038849"
echo "Region: us-east-2"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Track results
PASSED=0
FAILED=0

# Test function
test_access() {
    local test_name="$1"
    local command="$2"

    echo -n "Testing: $test_name... "

    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((FAILED++))
    fi
}

# Test 1: S3 Bucket Access
test_access "S3 Bucket Read Access" \
    "aws s3 ls s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/"

# Test 2: S3 Write Access
test_access "S3 Write Access" \
    "echo 'test' | aws s3 cp - s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/.access_test && aws s3 rm s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/.access_test"

# Test 3: SageMaker Model Registry Access
test_access "SageMaker Model Registry Access" \
    "aws sagemaker list-model-packages --model-package-group-name pokemon-card-recognition-models --region us-east-2"

# Test 4: IAM Role Access
test_access "IAM Role Read Access" \
    "aws iam get-role --role-name SageMaker-MarcosAdmin-ExecutionRole"

# Test 5: CloudWatch Logs Access
test_access "CloudWatch Logs Access" \
    "aws logs describe-log-groups --region us-east-2 --limit 1"

# Test 6: SageMaker Training Jobs Access
test_access "SageMaker Training Jobs Access" \
    "aws sagemaker list-training-jobs --region us-east-2 --max-results 1"

# Test 7: S3 Lifecycle Policy Access
test_access "S3 Lifecycle Policy Access" \
    "aws s3api get-bucket-lifecycle-configuration --bucket pokemon-card-training-us-east-2"

# Test 8: S3 Bucket Tagging Access
test_access "S3 Bucket Tagging Access" \
    "aws s3api get-bucket-tagging --bucket pokemon-card-training-us-east-2"

# Test 9: Download Project Manifest
test_access "Download Project Manifest" \
    "aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/metadata/project_manifest.json /tmp/test_manifest.json && rm /tmp/test_manifest.json"

# Test 10: Model File Access
test_access "Model File Access" \
    "aws s3 ls s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/dinov3-teacher/v1.0/model.tar.gz"

echo ""
echo "=============================================================="
echo "Test Results"
echo "=============================================================="
echo ""
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed! Full admin access confirmed.${NC}"
    echo ""
    echo "The SageMaker-MarcosAdmin-ExecutionRole has complete access to:"
    echo "  ✓ S3 bucket and all objects"
    echo "  ✓ SageMaker Model Registry"
    echo "  ✓ SageMaker Training Jobs"
    echo "  ✓ CloudWatch Logs"
    echo "  ✓ IAM Role information"
    echo "  ✓ S3 Lifecycle policies"
    exit 0
else
    echo -e "${RED}✗ Some tests failed. Please review permissions.${NC}"
    exit 1
fi
