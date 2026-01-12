#!/bin/bash
set -e

# Pokemon Card Recognition - Raspberry Pi Deployment Script
# Deploys models, reference database, and inference app to Pi

PI_USER="grailseeker"
PI_HOST="raspberrypi.local"
PI_IP="192.168.141.152"
PI_DIR="~/pokemon-card-recognition"

echo "🚀 Pokemon Card Recognition - Pi Deployment"
echo "============================================"
echo ""

# Try hostname first, fallback to IP
PI_TARGET="${PI_USER}@${PI_HOST}"
if ! ping -c 1 -W 1 "${PI_HOST}" &>/dev/null; then
    echo "⚠️  Hostname not reachable, using IP: ${PI_IP}"
    PI_TARGET="${PI_USER}@${PI_IP}"
fi

echo "📡 Target: ${PI_TARGET}"
echo ""

# Create directory structure on Pi
echo "📁 Creating directory structure on Pi..."
ssh "${PI_TARGET}" "mkdir -p ${PI_DIR}/{models/{detection,embedding},data/reference,src/inference}"

# Deploy detection model (YOLO for IMX500)
echo ""
echo "📦 Deploying YOLO detection model (IMX500)..."
scp models/detection/yolo11n-obb-imx500.onnx "${PI_TARGET}:${PI_DIR}/models/detection/"
echo "   ✅ yolo11n-obb-imx500.onnx (10 MB)"

# Deploy embedding model (Hailo HEF)
echo ""
echo "🧠 Deploying EfficientNet-Lite0 model (Hailo 8L)..."
scp models/embedding/pokemon_student_efficientnet_lite0_stage2.hef "${PI_TARGET}:${PI_DIR}/models/embedding/"
echo "   ✅ pokemon_student_efficientnet_lite0_stage2.hef (14 MB)"

# Deploy reference database
echo ""
echo "💾 Deploying reference database (106 MB)..."
echo "   - embeddings.npy (52 MB)"
echo "   - usearch.index (54 MB)"
echo "   - index.json (653 KB)"
echo "   - metadata.json (544 KB)"
rsync -avz --progress \
    --include="embeddings.npy" \
    --include="usearch.index" \
    --include="index.json" \
    --include="metadata.json" \
    --exclude="*" \
    data/reference/ "${PI_TARGET}:${PI_DIR}/data/reference/"

# Deploy inference application
echo ""
echo "🎮 Deploying inference application..."
rsync -avz --progress \
    src/ "${PI_TARGET}:${PI_DIR}/src/" \
    --exclude="__pycache__" \
    --exclude="*.pyc"

# Deploy requirements
echo ""
echo "📋 Deploying requirements.txt..."
scp requirements.txt "${PI_TARGET}:${PI_DIR}/"

# Install dependencies on Pi
echo ""
echo "📦 Installing dependencies on Pi..."
ssh "${PI_TARGET}" "cd ${PI_DIR} && python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt"

echo ""
echo "✅ Deployment Complete!"
echo ""
echo "📊 Deployed Components:"
echo "   ✅ YOLO11n-OBB detection (IMX500)"
echo "   ✅ EfficientNet-Lite0 HEF (Hailo 8L)"
echo "   ✅ Reference database (17,592 card embeddings)"
echo "   ✅ Inference application"
echo ""
echo "🎯 Next Steps:"
echo "   1. SSH to Pi: ssh ${PI_TARGET}"
echo "   2. Test Hailo: cd ${PI_DIR} && python3 -c 'import hailo_platform; print(\"Hailo OK\")'"
echo "   3. Run inference: python3 src/inference/recognize_card.py --image test.jpg"
echo ""
