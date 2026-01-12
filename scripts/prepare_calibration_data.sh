#!/bin/bash
# Download Pokemon card images from S3 for calibration

set -e

CALIB_DIR="${HOME_DIR:-/home/ubuntu}/pokemon_calibration"
mkdir -p "$CALIB_DIR"

echo "Downloading Pokemon card images from S3..."

# Download 128 random Pokemon card images
aws s3 ls s3://pokemon-card-images/ --no-sign-request 2>/dev/null | \
    grep -E '\.(jpg|jpeg|png)$' | \
    awk '{print $4}' | \
    shuf | \
    head -128 | \
    while read img; do
        aws s3 cp "s3://pokemon-card-images/$img" "$CALIB_DIR/" --no-sign-request 2>/dev/null || true
    done

# Count downloaded images
COUNT=$(ls -1 "$CALIB_DIR" | wc -l)
echo "Downloaded $COUNT Pokemon card images to $CALIB_DIR"

if [ "$COUNT" -lt 32 ]; then
    echo "ERROR: Only downloaded $COUNT images, need at least 32"
    exit 1
fi

echo "✓ Calibration data ready"
