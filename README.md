# Pokemon Card Recognition System

Real-time Pokemon card detection and recognition on Raspberry Pi 5 with AI Camera (IMX500) and Hailo 8 accelerator.

## ⚠️ Data Not Included in Git

This repository contains **code only**. Large data files (25+ GB) are stored on AWS S3 due to GitHub's 100 MB file size limit:
- **Raw card images**: 12.6 GB
- **Processed training data**: 12.5 GB
- **Reference database**: 128 MB (embeddings for 17,592 cards)
- **Model weights**: 6+ GB (teacher + student models)

See the [Data](#data) section below for S3 download instructions.

---

## 📚 Complete Documentation Wiki

**→ [View Complete Wiki](wiki/Home.md) ←**

This project has comprehensive documentation covering architecture, development, deployment, and infrastructure. Start with the wiki for:
- 🚀 **[Quick Start Guide](wiki/Getting-Started/Quick-Start.md)** - Get running in minutes
- 🏗️ **[System Architecture](wiki/Architecture/System-Overview.md)** - Understand the pipeline
- 💻 **[Development Guide](wiki/Development/Training.md)** - Train your own models
- 🔧 **[Deployment Guide](wiki/Deployment/Raspberry-Pi-Setup.md)** - Deploy to Raspberry Pi
- ☁️ **[AWS Infrastructure](wiki/Infrastructure/AWS-Organization.md)** - Cloud setup and data access
- 📖 **[Project History](wiki/Project-History/Organization-Journey.md)** - How we built this

---

## Architecture

Uses a multi-layer AI architecture:
1. **Detection (IMX500)**: YOLO11n-OBB detects cards with oriented bounding boxes
2. **Preprocessing (CPU)**: Perspective warp, crop, glare mitigation
3. **Embedding (Hailo 8L)**: EfficientNet-Lite0 converts card image to 768-dim vector
4. **Matching (CPU)**: uSearch finds nearest neighbor in 17,592 card reference database
5. **Validation (CPU)**: Temporal smoothing and confidence thresholding

The embedding model was trained using knowledge distillation from DINOv3 ViT-B/14 (86M params) to EfficientNet-Lite0 (4.7M params), achieving 18× compression while maintaining accuracy.

## Project Structure

```
pokemon-card-recognition/
├── data/                       # ⚠️ NOT IN GIT - Download from S3
│   ├── raw/                    # 12.6 GB raw card images
│   ├── processed/              # 12.5 GB training datasets
│   └── reference/              # 128 MB embeddings + index
├── models/
│   ├── detection/              # YOLO models
│   ├── embedding/              # Student models
│   └── onnx/                   # 23 MB ONNX (in git)
├── src/
│   ├── data/                   # Dataset loaders
│   ├── models/                 # Model architectures
│   ├── training/               # Training scripts (SageMaker)
│   ├── inference/              # Inference pipeline
│   └── hardware/               # IMX500 + Hailo integration
├── scripts/                    # Deployment & utilities
├── docs/                       # Technical specifications
└── wiki/                       # Complete documentation
```

**Note**: Large model files and datasets are excluded from git via `.gitignore` and stored on AWS S3. See the [Data](#data) section for download instructions.

## Hardware Requirements

- Raspberry Pi 5 (8GB recommended)
- Raspberry Pi AI Camera (IMX500)
- Hailo 8L AI Accelerator
- microSD card (64GB+)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run inference demo
python -m src.inference.pipeline
```

## Documentation

### Wiki (Recommended)
**→ [Complete Wiki](wiki/Home.md)** - Organized, cross-linked documentation covering everything

### Technical Specifications
See `docs/` for original PRD specifications:
- `PRD_01_OVERVIEW.md` - System overview
- `PRD_02_DETECTION.md` - Detection model specs
- `PRD_03_EMBEDDING.md` - Embedding model specs
- `PRD_04_DATABASE.md` - Reference database design
- `PRD_05_PIPELINE.md` - Matching pipeline
- `FINAL_PLAN.md` - Implementation plan

### Infrastructure & Access
- **[PROJECT_ACCESS.md](PROJECT_ACCESS.md)** - AWS access, S3 structure, SageMaker
- **[ORGANIZATION_COMPLETE.md](ORGANIZATION_COMPLETE.md)** - Current project status
- **[COST_BREAKDOWN.md](COST_BREAKDOWN.md)** - Training and storage costs

## Data

⚠️ **Important**: Large data files are NOT stored in this git repository due to GitHub's file size limits (100 MB max). All data is stored on AWS S3.

### What's on AWS S3 (31.7 GB total)

| Data Type | S3 Location | Size | Description |
|-----------|-------------|------|-------------|
| **Raw card images** | `data/raw/card_images/` | 12.6 GB | 17,592 Pokemon card PNGs + metadata |
| **Processed training data** | `data/processed/classification/` | 12.5 GB | Training/val/test splits for embedding model |
| **Reference database** | `data/reference/` | 128 MB | Pre-computed embeddings + uSearch index |
| **Hailo calibration** | `models/efficientnet-hailo/calibration/` | 734 MB | 1,024 images for Hailo quantization |
| **Teacher model** | `models/dinov3-teacher/v1.0/` | 5.6 GB | DINOv3 ViT-B/14 (86M params) |
| **Student model** | `models/efficientnet-student/stage2/v2.0/` | 97 MB | EfficientNet-Lite0 PyTorch + ONNX |

### Getting the Data

**Prerequisites**: AWS CLI configured with access to `pokemon-card-training-us-east-2` S3 bucket

```bash
# Required for inference - Download reference database (128 MB)
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/ \
  ./data/reference/

# Optional for development - Download raw card images (12.6 GB)
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/raw/ \
  ./data/raw/

# Optional for training - Download processed datasets (12.5 GB)
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/processed/ \
  ./data/processed/

# Optional for Hailo compilation - Download calibration data (734 MB)
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-hailo/calibration/ \
  ./models/efficientnet-hailo/calibration/
```

### What's Deployed on Raspberry Pi

The Raspberry Pi deployment in `~/pokemon-card-recognition/` includes:
- ✅ Reference database (128 MB) - already deployed
- ✅ EfficientNet-Lite0 HEF model (14 MB) - compiled for Hailo 8
- ✅ YOLO11n-OBB ONNX model (10 MB) - for IMX500 camera
- ❌ Raw images NOT needed for inference (only for training)

See `PROJECT_ACCESS.md` for AWS credentials and detailed S3 structure.

## Models

### Edge Deployment (Raspberry Pi)
| Model | Purpose | Format | Size | Hardware |
|-------|---------|--------|------|----------|
| YOLO11n-OBB | Card detection | .pt/.onnx | 5.5MB | IMX500 Camera |
| EfficientNet-Lite0 | Card embedding | .hef | 13.8MB | Hailo 8L NPU |

### Training Models (AWS SageMaker)
| Model | Purpose | Parameters | Size |
|-------|---------|------------|------|
| DINOv3 ViT-B/14 | Teacher (knowledge distillation) | 86M | 5.6GB |
| EfficientNet-Lite0 | Student (deployed model) | 4.7M | 75MB (.pt) |

All models are version-controlled on S3 and in SageMaker Model Registry. See `PROJECT_ACCESS.md` for download instructions.

## License

Private project.
