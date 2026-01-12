#!/bin/bash
#
# Compile EfficientNet-Lite0 to Hailo HEF on EC2
#
# This script:
# 1. Uploads ONNX model + calibration images to EC2
# 2. Runs Hailo compilation in Docker on EC2
# 3. Downloads the resulting HEF file
#
# Prerequisites:
# - EC2 instance with Hailo Docker image running
# - SSH key configured (~/.ssh/your-key.pem)
# - EC2 instance has 100GB+ storage for calibration images

set -e

# Configuration
EC2_HOST="${EC2_HOST:-ec2-user@your-ec2-instance.compute.amazonaws.com}"
EC2_KEY="${EC2_KEY:-~/.ssh/your-key.pem}"
PROJECT_ROOT="/Users/marcos/dev/raspberry-pi/pokemon-card-recognition"

echo "======================================================================"
echo "HAILO COMPILATION ON EC2"
echo "======================================================================"
echo ""
echo "EC2 Host: $EC2_HOST"
echo "SSH Key: $EC2_KEY"
echo ""

# Check if EC2_HOST is configured
if [[ "$EC2_HOST" == *"your-ec2-instance"* ]]; then
    echo "❌ ERROR: EC2_HOST not configured"
    echo ""
    echo "Please set EC2_HOST environment variable:"
    echo "  export EC2_HOST=ec2-user@ec2-XX-XX-XX-XX.us-east-2.compute.amazonaws.com"
    echo ""
    echo "Or find your EC2 instance:"
    echo "  aws ec2 describe-instances --region us-east-2 --filters 'Name=instance-state-name,Values=running' --query 'Reservations[].Instances[].[InstanceId,PublicDnsName,State.Name]' --output table"
    exit 1
fi

# Test SSH connection
echo "[1/6] Testing SSH connection to EC2..."
if ssh -i "$EC2_KEY" -o ConnectTimeout=5 "$EC2_HOST" "echo 'SSH connection successful'" 2>/dev/null; then
    echo "   ✅ SSH connection OK"
else
    echo "   ❌ ERROR: Cannot connect to EC2"
    echo "   Check EC2_HOST and EC2_KEY settings"
    exit 1
fi
echo ""

# Create working directory on EC2
echo "[2/6] Setting up EC2 working directory..."
ssh -i "$EC2_KEY" "$EC2_HOST" "mkdir -p ~/hailo_compilation/{models,calibration,scripts}"
echo "   ✅ Directory created"
echo ""

# Upload ONNX model
echo "[3/6] Uploading ONNX model to EC2..."
ONNX_MODEL="$PROJECT_ROOT/models/onnx/pokemon_student_stage2_final.onnx"
if [ ! -f "$ONNX_MODEL" ]; then
    echo "   ❌ ERROR: ONNX model not found: $ONNX_MODEL"
    exit 1
fi

MODEL_SIZE=$(du -h "$ONNX_MODEL" | cut -f1)
echo "   Model: pokemon_student_stage2_final.onnx ($MODEL_SIZE)"

scp -i "$EC2_KEY" "$ONNX_MODEL" "$EC2_HOST:~/hailo_compilation/models/"
echo "   ✅ Model uploaded"
echo ""

# Upload calibration images
echo "[4/6] Uploading 1,024 calibration images to EC2..."
CALIB_DIR="$PROJECT_ROOT/data/calibration"
CALIB_COUNT=$(ls "$CALIB_DIR"/*.png 2>/dev/null | wc -l | tr -d ' ')
CALIB_SIZE=$(du -sh "$CALIB_DIR" | cut -f1)

echo "   Images: $CALIB_COUNT Pokemon cards ($CALIB_SIZE)"
echo "   This will take 2-5 minutes..."

rsync -avz --progress -e "ssh -i $EC2_KEY" \
    "$CALIB_DIR/" \
    "$EC2_HOST:~/hailo_compilation/calibration/"

echo "   ✅ Calibration images uploaded"
echo ""

# Upload compilation script
echo "[5/6] Uploading compilation script..."
scp -i "$EC2_KEY" "$PROJECT_ROOT/scripts/hailo_compile.py" "$EC2_HOST:~/hailo_compilation/scripts/"
echo "   ✅ Script uploaded"
echo ""

# Run compilation on EC2
echo "[6/6] Running Hailo compilation on EC2..."
echo "   This will take 10-15 minutes..."
echo ""

ssh -i "$EC2_KEY" "$EC2_HOST" << 'ENDSSH'
cd ~/hailo_compilation

echo "Starting Docker compilation..."
docker run --rm \
    -v "$(pwd)/models:/workspace" \
    -v "$(pwd)/calibration:/workspace/calibration_images" \
    -v "$(pwd)/scripts:/workspace/scripts" \
    -e CALIB_DATA_PATH=/workspace/calibration_images \
    -e ONNX_MODEL=pokemon_student_stage2_final.onnx \
    hailo8_ai_sw_suite_2025-10:1 \
    python3 /workspace/scripts/hailo_compile.py

echo ""
echo "Compilation complete!"
ls -lh models/*.hef
ENDSSH

# Download HEF file
echo ""
echo "Downloading HEF file from EC2..."
scp -i "$EC2_KEY" "$EC2_HOST:~/hailo_compilation/models/*.hef" "$PROJECT_ROOT/models/embedding/"

HEF_FILE=$(ls "$PROJECT_ROOT/models/embedding/"*.hef 2>/dev/null | tail -n 1)
if [ -f "$HEF_FILE" ]; then
    HEF_SIZE=$(du -h "$HEF_FILE" | cut -f1)
    echo "   ✅ Downloaded: $(basename $HEF_FILE) ($HEF_SIZE)"
else
    echo "   ❌ ERROR: HEF file not found"
    exit 1
fi

echo ""
echo "======================================================================"
echo "✅ HAILO COMPILATION COMPLETE"
echo "======================================================================"
echo ""
echo "HEF Model: $(basename $HEF_FILE)"
echo "Location: $HEF_FILE"
echo "Size: $HEF_SIZE"
echo ""
echo "Next steps:"
echo "  1. Build reference database: python scripts/build_reference_database.py"
echo "  2. Deploy to Raspberry Pi: rsync -avz deploy/ pi@raspberrypi.local:~/"
echo "  3. Test pipeline: python main.py --embedding-model $HEF_FILE"
echo ""
