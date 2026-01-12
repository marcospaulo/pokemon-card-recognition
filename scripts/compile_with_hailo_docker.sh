#!/bin/bash
# Compile ONNX model using Hailo Docker container on EC2
set -e

# Use absolute paths
HOME_DIR="/home/ubuntu"

echo "======================================"
echo "Hailo Model Compilation (Docker)"
echo "======================================"

# Step 1: Download Pokemon card images for calibration
echo ""
echo "[1/5] Preparing calibration data..."
CALIB_DIR="${HOME_DIR}/pokemon_calibration"
mkdir -p "$CALIB_DIR"

echo "      Downloading Pokemon card images from S3..."
aws s3 ls s3://pokemon-card-images/ --no-sign-request 2>/dev/null | \
    grep -E '\.(jpg|jpeg|png)$' | \
    awk '{print $4}' | \
    shuf | \
    head -128 | \
    while read img; do
        aws s3 cp "s3://pokemon-card-images/$img" "$CALIB_DIR/" --no-sign-request 2>/dev/null || true
    done

# Count downloaded images
COUNT=$(ls -1 "$CALIB_DIR"/*.jpg "$CALIB_DIR"/*.jpeg "$CALIB_DIR"/*.png 2>/dev/null | wc -l)
echo "      Downloaded $COUNT Pokemon card images"

if [ "$COUNT" -lt 32 ]; then
    echo "      WARNING: Only downloaded $COUNT images, using SDK defaults as fallback"
    CALIB_DIR=""
else
    echo "      ✓ Using $COUNT Pokemon card images for calibration"
fi

# Step 2: Check if Docker image is loaded
echo ""
echo "[2/5] Loading Docker image..."
IMAGE_NAME=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep hailo | head -1)

if [ -z "$IMAGE_NAME" ]; then
    echo "      Loading Hailo Docker image (first time)..."
    docker load -i ${HOME_DIR}/hailo_docker.tar.gz
    IMAGE_NAME=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep hailo | head -1)
else
    echo "      ✓ Using cached Docker image: $IMAGE_NAME"
fi

# Step 3: Prepare workspace with proper permissions
echo ""
echo "[3/5] Preparing workspace..."
WORKSPACE="${HOME_DIR}/hailo-workspace"
mkdir -p "$WORKSPACE"
chmod 777 "$WORKSPACE"  # Ensure Docker can write

# Copy files to workspace
cp ${HOME_DIR}/pokemon_student_efficientnet_lite0.onnx "$WORKSPACE/"
cp ${HOME_DIR}/hailo_compile.py "$WORKSPACE/"

echo "      ✓ Workspace ready"

# Step 4: Run compilation
echo ""
echo "[4/5] Starting Hailo compilation..."
echo "      This will take 20-30 minutes..."
echo ""

# Build Docker run command with calibration data mounted if available
DOCKER_CMD="docker run --rm \
    -v $WORKSPACE:/workspace \
    -w /workspace"

if [ -n "$CALIB_DIR" ]; then
    DOCKER_CMD="$DOCKER_CMD -v $CALIB_DIR:/calibration_data:ro"
    ENV_CALIB="-e CALIB_DATA_PATH=/calibration_data"
else
    ENV_CALIB=""
fi

# Run compilation
$DOCKER_CMD $ENV_CALIB $IMAGE_NAME \
    /bin/bash -c "python3 /workspace/hailo_compile.py"

# Step 5: Verify output
echo ""
echo "[5/5] Verifying output..."
if [ -f "$WORKSPACE/pokemon_student.hef" ]; then
    cp "$WORKSPACE/pokemon_student.hef" "${HOME_DIR}/"
    echo "✓ HEF file ready: ${HOME_DIR}/pokemon_student.hef"
    ls -lh "${HOME_DIR}/pokemon_student.hef"
else
    echo "❌ Compilation failed - HEF file not found"
    exit 1
fi

echo ""
echo "======================================"
echo "✅ COMPILATION COMPLETE"
echo "======================================"
echo "Next: Download HEF and deploy to Raspberry Pi"
