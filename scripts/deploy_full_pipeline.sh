#!/bin/bash
#
# Full Pipeline Deployment Script
#
# This script:
# 1. Prepares calibration dataset (1024 Pokemon card images)
# 2. Compiles EfficientNet-Lite0 ONNX → Hailo HEF for card recognition
# 3. Builds reference database with all 17,592 card embeddings
# 4. Packages everything for Raspberry Pi deployment
#
# Prerequisites:
# - ONNX model exported: models/onnx/pokemon_student_stage2_final.onnx
# - YOLO model trained: models/detection/yolo11n-obb.pt
# - Card images in: data/raw/card_images/ (17,592 images)

set -e  # Exit on error

PROJECT_ROOT="/Users/marcos/dev/raspberry-pi/pokemon-card-recognition"
cd "$PROJECT_ROOT"

echo "======================================================================"
echo "FULL PIPELINE DEPLOYMENT"
echo "======================================================================"
echo ""

# ===================================================================
# STEP 1: Prepare Calibration Dataset
# ===================================================================
echo "[1/5] Preparing calibration dataset (1024 images)..."
echo ""

CALIB_DIR="$PROJECT_ROOT/data/calibration"
CARD_IMAGES_DIR="$PROJECT_ROOT/data/raw/card_images"

if [ ! -d "$CARD_IMAGES_DIR" ]; then
    echo "❌ ERROR: Card images directory not found: $CARD_IMAGES_DIR"
    exit 1
fi

# Count available images
TOTAL_IMAGES=$(find "$CARD_IMAGES_DIR" -name "*.png" | wc -l | tr -d ' ')
echo "   Found $TOTAL_IMAGES card images"

if [ "$TOTAL_IMAGES" -lt 1024 ]; then
    echo "   ⚠️  WARNING: Less than 1024 images available"
    CALIB_COUNT=$TOTAL_IMAGES
else
    CALIB_COUNT=1024
fi

# Create calibration directory
mkdir -p "$CALIB_DIR"

# Sample random images (macOS compatible)
echo "   Sampling $CALIB_COUNT random images for calibration..."
find "$CARD_IMAGES_DIR" -name "*.png" | \
    sort -R | head -n "$CALIB_COUNT" | \
    while read img; do
        cp "$img" "$CALIB_DIR/"
    done

ACTUAL_COUNT=$(ls "$CALIB_DIR"/*.png 2>/dev/null | wc -l | tr -d ' ')
echo "   ✅ Created calibration dataset: $ACTUAL_COUNT images"
echo ""

# ===================================================================
# STEP 2: Compile EfficientNet-Lite0 to Hailo HEF
# ===================================================================
echo "[2/5] Compiling EfficientNet-Lite0 to Hailo HEF..."
echo ""

ONNX_MODEL="$PROJECT_ROOT/models/onnx/pokemon_student_stage2_final.onnx"
if [ ! -f "$ONNX_MODEL" ]; then
    echo "   ❌ ERROR: ONNX model not found: $ONNX_MODEL"
    echo "   Run: python scripts/export_student_to_onnx.py first"
    exit 1
fi

echo "   ONNX model: $(basename $ONNX_MODEL) ($(du -h "$ONNX_MODEL" | cut -f1))"
echo "   Target: Hailo-8L on Raspberry Pi 5"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "   ❌ ERROR: Docker is not running"
    echo "   Start Docker Desktop and try again"
    exit 1
fi

# Compile using Hailo Docker
echo "   Launching Hailo SDK Docker container..."
echo "   This will take 10-15 minutes..."
echo ""

docker run --rm \
    -v "$PROJECT_ROOT/models/onnx:/workspace" \
    -v "$CALIB_DIR:/workspace/calibration_images" \
    -v "$PROJECT_ROOT/scripts:/workspace/scripts" \
    -e CALIB_DATA_PATH=/workspace/calibration_images \
    -e ONNX_MODEL=pokemon_student_stage2_final.onnx \
    hailo8_ai_sw_suite_2025-10:1 \
    python3 /workspace/scripts/hailo_compile.py

# Move HEF to models directory
HEF_FILE="$PROJECT_ROOT/models/onnx/pokemon_student_efficientnet_lite0_stage2.hef"
if [ -f "$HEF_FILE" ]; then
    mv "$HEF_FILE" "$PROJECT_ROOT/models/embedding/"
    echo ""
    echo "   ✅ HEF compiled successfully"
    echo "   Location: models/embedding/pokemon_student_efficientnet_lite0_stage2.hef"
    echo "   Size: $(du -h "$PROJECT_ROOT/models/embedding/pokemon_student_efficientnet_lite0_stage2.hef" | cut -f1)"
else
    echo "   ❌ ERROR: HEF compilation failed"
    exit 1
fi
echo ""

# ===================================================================
# STEP 3: Build Reference Database
# ===================================================================
echo "[3/5] Building reference database with 17,592 card embeddings..."
echo ""

# This step will use the HEF model to generate embeddings for all cards
# Requires Hailo-8L hardware (run on Pi) or use PyTorch version temporarily

if [ -f "$PROJECT_ROOT/scripts/build_reference_database.py" ]; then
    echo "   Building database with PyTorch model (for now)..."
    echo "   Full HEF-based database will be built on Raspberry Pi"
    echo ""

    python3 "$PROJECT_ROOT/scripts/build_reference_database.py" \
        --model-type pytorch \
        --checkpoint "s3://pokemon-card-training-us-east-2/models/embedding/student/pytorch-training-2026-01-11-23-31-10-757/output/model.tar.gz" \
        --output-dir "$PROJECT_ROOT/data/reference" \
        --card-images "$CARD_IMAGES_DIR"

    echo "   ✅ Reference database created"
    echo "   Location: data/reference/"
else
    echo "   ⚠️  Skipping database build (script not found)"
    echo "   Will build on Raspberry Pi after deployment"
fi
echo ""

# ===================================================================
# STEP 4: Export YOLO for IMX500 (if needed)
# ===================================================================
echo "[4/5] Checking YOLO model for card detection..."
echo ""

YOLO_MODEL="$PROJECT_ROOT/models/detection/yolo11n-obb.pt"
YOLO_ONNX="$PROJECT_ROOT/models/detection/yolo11n-obb.onnx"

if [ ! -f "$YOLO_MODEL" ]; then
    echo "   ❌ ERROR: YOLO model not found: $YOLO_MODEL"
    exit 1
fi

echo "   ✅ YOLO model found: $(basename $YOLO_MODEL)"

# Export to ONNX if not already done
if [ ! -f "$YOLO_ONNX" ]; then
    echo "   Exporting to ONNX for IMX500..."
    python3 -c "
from ultralytics import YOLO
model = YOLO('$YOLO_MODEL')
model.export(format='onnx', imgsz=640)
print('   ✅ ONNX export complete')
"
fi

echo "   ✅ YOLO ready for IMX500"
echo "   Note: IMX500 compilation requires Sony SDK (on Raspberry Pi)"
echo ""

# ===================================================================
# STEP 5: Create Deployment Package
# ===================================================================
echo "[5/5] Creating deployment package for Raspberry Pi..."
echo ""

DEPLOY_DIR="$PROJECT_ROOT/deploy"
mkdir -p "$DEPLOY_DIR"

# Copy models
mkdir -p "$DEPLOY_DIR/models/detection"
mkdir -p "$DEPLOY_DIR/models/embedding"
mkdir -p "$DEPLOY_DIR/models/reference"

echo "   Packaging models..."
cp "$PROJECT_ROOT/models/detection/yolo11n-obb.pt" "$DEPLOY_DIR/models/detection/" 2>/dev/null || true
cp "$PROJECT_ROOT/models/detection/yolo11n-obb.onnx" "$DEPLOY_DIR/models/detection/" 2>/dev/null || true
cp "$PROJECT_ROOT/models/embedding/pokemon_student_efficientnet_lite0_stage2.hef" "$DEPLOY_DIR/models/embedding/"

# Copy reference database if it exists
if [ -d "$PROJECT_ROOT/data/reference" ]; then
    cp -r "$PROJECT_ROOT/data/reference/"* "$DEPLOY_DIR/models/reference/" 2>/dev/null || true
fi

# Copy source code
echo "   Packaging source code..."
mkdir -p "$DEPLOY_DIR/src"
cp -r "$PROJECT_ROOT/src/"* "$DEPLOY_DIR/src/" 2>/dev/null || true

# Create deployment README
cat > "$DEPLOY_DIR/README.md" << 'EOF'
# Pokemon Card Recognition - Raspberry Pi Deployment

## Models Included

### 1. Card Detection (IMX500)
- **Model:** yolo11n-obb.pt / yolo11n-obb.onnx
- **Purpose:** Detect card in frame and extract 4 corner keypoints
- **Runs on:** IMX500 AI Camera (17 TOPS)
- **Performance:** ~30 FPS

### 2. Card Recognition (Hailo-8L)
- **Model:** pokemon_student_efficientnet_lite0_stage2.hef
- **Purpose:** Generate 768-dim embedding for card recognition
- **Runs on:** Hailo-8L AI Accelerator (13 TOPS)
- **Performance:** ~40 FPS (25ms per card)

### 3. Reference Database
- **Embeddings:** 17,592 Pokemon cards × 768 dimensions
- **Search:** USearch vector similarity (cosine distance)
- **Runs on:** CPU (very fast lookup)

## Deployment Steps

### 1. Transfer to Raspberry Pi
```bash
rsync -avz --progress deploy/ grailseeker@raspberrypi.local:~/pokemon-card-recognition/
```

### 2. On Raspberry Pi: Compile YOLO for IMX500
```bash
# Requires Sony IMX500 SDK
sudo imx500-convert-model \
    --input models/detection/yolo11n-obb.onnx \
    --output models/detection/yolo11n-obb.rpk
```

### 3. Build Reference Database (if not included)
```bash
python3 scripts/build_reference_database.py \
    --model-hef models/embedding/pokemon_student_efficientnet_lite0_stage2.hef \
    --output-dir models/reference
```

### 4. Run Full Pipeline
```bash
python3 main.py \
    --detection-model models/detection/yolo11n-obb.rpk \
    --embedding-model models/embedding/pokemon_student_efficientnet_lite0_stage2.hef \
    --database-dir models/reference
```

## Pipeline Architecture

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Camera  │──▶│ Detection│──▶│  Crop &  │──▶│ Embedding│──▶│  Search  │
│  Capture │   │  (IMX500)│   │  Preproc │   │ (Hailo 8)│   │    &     │
│          │   │          │   │  (CPU)   │   │          │   │  Smooth  │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
     │              │              │              │              │
     ▼              ▼              ▼              ▼              ▼
Raw Frame     Detection      Cropped Card   768-dim Vector   CardMatch
2028x1520     + 4 Corners    224x224        (normalized)     or None
```

## Expected Performance

- **End-to-end latency:** <100ms per frame
- **Frame rate:** 10-30 FPS (depending on pipeline stage)
- **Recognition accuracy:** >95% on clean cards
- **Power consumption:** ~15W total (Pi 5 + Hailo + Camera)

## Troubleshooting

### IMX500 not detected
```bash
sudo systemctl status imx500-server
libcamera-hello --list-cameras
```

### Hailo-8L not detected
```bash
hailortcli scan
lsusb | grep Hailo
```

### Database search slow
Rebuild with optimized index:
```bash
python3 scripts/optimize_database.py --database models/reference
```
EOF

echo "   ✅ Deployment package created"
echo ""

# ===================================================================
# Summary
# ===================================================================
echo "======================================================================"
echo "✅ DEPLOYMENT PREPARATION COMPLETE"
echo "======================================================================"
echo ""
echo "📦 Deployment Package: $DEPLOY_DIR/"
echo ""
echo "Contents:"
echo "  • EfficientNet-Lite0 HEF for Hailo-8L"
echo "  • YOLO11 model for IMX500"
echo "  • Reference database (if built)"
echo "  • Source code"
echo "  • Deployment instructions"
echo ""
echo "Next Steps:"
echo ""
echo "1. Transfer to Raspberry Pi:"
echo "   rsync -avz --progress $DEPLOY_DIR/ grailseeker@raspberrypi.local:~/pokemon-card-recognition/"
echo ""
echo "2. On Raspberry Pi:"
echo "   - Compile YOLO for IMX500 (requires Sony SDK)"
echo "   - Build reference database with HEF model"
echo "   - Run full pipeline"
echo ""
echo "======================================================================"
