#!/bin/bash
#
# Deploy Pokemon Card Recognition to Raspberry Pi 5
#
# This script:
# 1. Copies models to Raspberry Pi
# 2. Compiles YOLO11 ONNX to IMX500 RPK
# 3. Installs dependencies
# 4. Tests the pipeline
#

set -e  # Exit on error

# Configuration
PI_HOST="${PI_HOST:-raspberrypi.local}"
PI_USER="${PI_USER:-pi}"
PI_DIR="${PI_DIR:-~/pokemon-card-recognition}"

echo "================================================================================"
echo "Pokemon Card Recognition - Raspberry Pi Deployment"
echo "================================================================================"
echo ""
echo "Target: $PI_USER@$PI_HOST"
echo "Remote directory: $PI_DIR"
echo ""

# Check if we can reach the Pi
echo "[1/6] Checking Raspberry Pi connectivity..."
if ! ping -c 1 "$PI_HOST" &> /dev/null; then
    echo "❌ Cannot reach $PI_HOST"
    echo "   Make sure the Raspberry Pi is on and connected to the network"
    exit 1
fi
echo "✅ Raspberry Pi is reachable"
echo ""

# Copy models to Pi
echo "[2/6] Copying models to Raspberry Pi..."
echo "  - YOLO11-OBB ONNX (10.3 MB)"
echo "  - Hailo HEF (will be copied after compilation completes)"
echo "  - Reference database (17,592 embeddings)"
echo ""

ssh "$PI_USER@$PI_HOST" "mkdir -p $PI_DIR/models/detection $PI_DIR/models/embedding $PI_DIR/models/reference"

# Copy YOLO ONNX
scp models/detection/yolo11n-obb-imx500.onnx "$PI_USER@$PI_HOST:$PI_DIR/models/detection/"
echo "✅ YOLO ONNX copied"

# Copy reference database (if exists)
if [ -d "models/reference" ]; then
    scp -r models/reference/* "$PI_USER@$PI_HOST:$PI_DIR/models/reference/"
    echo "✅ Reference database copied"
else
    echo "⚠️  Reference database not found (will be built on Pi)"
fi

echo ""

# Install IMX500 SDK on Pi
echo "[3/6] Installing IMX500 SDK on Raspberry Pi..."
ssh "$PI_USER@$PI_HOST" << 'EOF'
    set -e

    echo "  Checking for IMX500 SDK..."
    if ! command -v imx500-convert-model &> /dev/null; then
        echo "  Installing IMX500 SDK..."
        sudo apt update
        sudo apt install -y imx500-all
        echo "  ✅ IMX500 SDK installed"
    else
        echo "  ✅ IMX500 SDK already installed"
    fi
EOF
echo ""

# Compile YOLO ONNX to IMX500 RPK
echo "[4/6] Compiling YOLO11 to IMX500 RPK format..."
echo "  This takes ~2-3 minutes..."
echo ""

ssh "$PI_USER@$PI_HOST" << EOF
    set -e
    cd $PI_DIR

    echo "  Converting ONNX to RPK..."
    sudo imx500-convert-model \
        --input models/detection/yolo11n-obb-imx500.onnx \
        --output models/detection/yolo11n-obb-imx500.rpk \
        2>&1 | tee imx500_conversion.log

    if [ -f models/detection/yolo11n-obb-imx500.rpk ]; then
        SIZE=\$(ls -lh models/detection/yolo11n-obb-imx500.rpk | awk '{print \$5}')
        echo "  ✅ RPK compiled: \$SIZE"
    else
        echo "  ❌ RPK compilation failed"
        exit 1
    fi
EOF
echo ""

# Install Python dependencies
echo "[5/6] Installing Python dependencies..."
ssh "$PI_USER@$PI_HOST" << 'EOF'
    set -e
    cd ~/pokemon-card-recognition

    # Install picamera2 (usually pre-installed on Raspberry Pi OS)
    sudo apt install -y python3-picamera2

    # Install hailo-platform SDK (for Hailo-8L)
    if ! python3 -c "import hailo_platform" 2>/dev/null; then
        echo "  Installing Hailo Python SDK..."
        wget https://hailo.ai/downloads/hailo_platform-4.18.0-cp39-cp39-linux_aarch64.whl
        pip3 install hailo_platform-4.18.0-cp39-cp39-linux_aarch64.whl
    fi

    # Install other dependencies
    pip3 install numpy pillow usearch tqdm

    echo "  ✅ Dependencies installed"
EOF
echo ""

# Test IMX500 model loading
echo "[6/6] Testing IMX500 model loading..."
ssh "$PI_USER@$PI_HOST" << EOF
    cd $PI_DIR

    python3 << 'PYTHON'
try:
    from picamera2.devices import IMX500
    imx500 = IMX500('models/detection/yolo11n-obb-imx500.rpk')
    print("  ✅ IMX500 model loaded successfully")
    print(f"     Model: {imx500.network_intrinsics.network_name if imx500.network_intrinsics else 'Unknown'}")
except Exception as e:
    print(f"  ❌ Failed to load IMX500 model: {e}")
    exit(1)
PYTHON
EOF
echo ""

echo "================================================================================"
echo "✅ Deployment Complete"
echo "================================================================================"
echo ""
echo "Models deployed to $PI_USER@$PI_HOST:$PI_DIR"
echo ""
echo "Next steps:"
echo ""
echo "1. Copy Hailo HEF file (once compilation finishes):"
echo "   scp models/embedding/pokemon_student_efficientnet_lite0_stage2.hef \\"
echo "       $PI_USER@$PI_HOST:$PI_DIR/models/embedding/"
echo ""
echo "2. Build reference database (if not copied):"
echo "   ssh $PI_USER@$PI_HOST"
echo "   cd $PI_DIR"
echo "   python3 scripts/build_reference_database.py"
echo ""
echo "3. Test the full pipeline:"
echo "   python3 main.py \\"
echo "       --detection-model models/detection/yolo11n-obb-imx500.rpk \\"
echo "       --embedding-model models/embedding/pokemon_student_efficientnet_lite0_stage2.hef \\"
echo "       --database-dir models/reference"
echo ""
