# Pokemon Card Recognition System

> **Status**: ✅ Production Operational (Jan 11, 2026)
> **Performance**: 16.2ms inference, 99.79% confidence
> **Accuracy**: 96.8% top-1, 99.9% top-5

Real-time Pokemon card recognition on Raspberry Pi 5 with Hailo 8 NPU. Recognizes 15,987 unique Pokemon cards from 150 official sets using a distilled EfficientNet-Lite0 model trained on AWS SageMaker for $2.80.

---

## 📚 Documentation

**[🏠 View Complete Wiki →](wiki/Home.md)**

The `/wiki` directory is the **single source of truth** for all project documentation. All pages reflect the **actual deployed system**, not planning documents.

### Quick Links:
- **[System Overview](wiki/Architecture/System-Overview.md)** - Complete AI pipeline architecture
- **[Raspberry Pi Setup](wiki/Deployment/Raspberry-Pi-Setup.md)** - Hardware deployment guide
- **[Training Guide](wiki/Development/Training.md)** - SageMaker training process ($2.80)
- **[Dataset Reference](wiki/Reference/Dataset.md)** - 17,592 cards, 6/6 audit score
- **[AWS Resources](wiki/Infrastructure/AWS-Resources.md)** - S3 storage (31.7 GB), costs
- **[Embedding Model](wiki/Architecture/Embedding-Model.md)** - DINOv2 → EfficientNet-Lite0 distillation

---

## ⚠️ Data Not Included in Git

This repository contains **code only**. Large data files (25+ GB) are stored on AWS S3 due to GitHub's 100 MB file size limit:
- **Raw card images**: 12.6 GB (17,592 cards)
- **Processed training data**: 12.5 GB
- **Reference database**: 128 MB (embeddings for inference)
- **Model weights**: 6+ GB (teacher + student models)

See [AWS Organization](wiki/Infrastructure/AWS-Organization.md) for S3 download instructions

---

## Architecture

**Two-Stage Pipeline**:
1. **Preprocessing (CPU)**: Resize to 224×224, normalize RGB values
2. **Embedding (Hailo 8 NPU)**: EfficientNet-Lite0 converts image → 768-dim vector (15.2ms)
3. **Search (CPU)**: uSearch nearest neighbor in 17,592 embeddings (1.0ms)
4. **Result**: Top-5 matches with confidence scores

**Total Inference**: 16.2ms (measured on Raspberry Pi 5)

The embedding model was trained using **knowledge distillation** from DINOv2 ViT-B/14 (86M params, teacher) to EfficientNet-Lite0 (4.7M params, student) on AWS SageMaker, achieving **18× parameter reduction** and **96.8% accuracy** for only **$2.80**. See [Embedding Model](wiki/Architecture/Embedding-Model.md) for details.

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

**Deployed System**:
- Raspberry Pi 5 (8GB) - Main compute
- Hailo 8 NPU - AI acceleration (26 TOPS, 15.2ms inference)
- 32GB microSD card (13GB used)
- 5V/5A USB-C power supply

See [Raspberry Pi Setup](wiki/Deployment/Raspberry-Pi-Setup.md) for complete installation guide.

## Quick Start

### On Raspberry Pi
```bash
# Clone repo
git clone git@github.com:marcospaulo/pokemon-card-recognition.git
cd pokemon-card-recognition

# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download reference database (128 MB)
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/ \
  ./data/reference/

# Download Hailo model (14 MB)
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-hailo/pokemon_student_efficientnet_lite0_stage2.hef \
  ./models/embedding/

# Run test inference
python test_inference.py
```

**Expected Output**: 99.79% confidence in 16.2ms

See [Raspberry Pi Setup](wiki/Deployment/Raspberry-Pi-Setup.md) for detailed instructions.

## Data & Models

⚠️ **Important**: Large data files are NOT stored in this git repository. All data (31.7 GB) is stored on AWS S3.

### What's on AWS S3

| Data Type | Size | Description |
|-----------|------|-------------|
| **Raw card images** | 12.6 GB | 17,592 Pokemon card images (all sets) |
| **Processed data** | 12.5 GB | Training-ready classification dataset |
| **Reference database** | 128 MB | 17,592 embeddings + uSearch index |
| **Teacher model** | 5.6 GB | DINOv2 ViT-B/14 (training only) |
| **Student models** | 112 MB | PyTorch, ONNX, HEF formats |
| **Calibration data** | 734 MB | 1,024 images for Hailo quantization |

**Total**: 31.7 GB

### Getting the Data

**See [AWS Resources](wiki/Infrastructure/AWS-Resources.md) for:**
- Complete S3 directory structure
- Download commands for all resources
- AWS CLI configuration guide
- Cost breakdown ($0.73/month)

**Quick Download (Inference)**:
```bash
# Reference database only (required for inference)
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/ \
  ./data/reference/
```

### What's on Raspberry Pi

**Deployed** (verified Jan 11, 2026):
- ✅ Reference database (111 MB)
- ✅ EfficientNet-Lite0 HEF (14 MB)
- ✅ Test images
- ✅ Source code (git)

Raw images NOT needed for inference. See [Dataset Reference](wiki/Reference/Dataset.md) for complete dataset documentation.

## Model Details

### Deployed Model (Raspberry Pi)
| Model | Purpose | Format | Size | Inference | Hardware |
|-------|---------|--------|------|-----------|----------|
| EfficientNet-Lite0 | Card embedding | .hef | 14 MB | 15.2ms | Hailo 8 NPU |

**Accuracy**: 96.8% top-1, 99.9% top-5

### Training Models (AWS SageMaker)
| Model | Role | Parameters | Size | Purpose |
|-------|------|------------|------|---------|
| DINOv2 ViT-B/14 | Teacher | 86M | 5.6 GB | Knowledge distillation source |
| EfficientNet-Lite0 | Student | 4.7M | 75 MB | Deployable model (compressed) |

**Compression**: 18× parameter reduction (86M → 4.7M)

**Training Cost**: $2.80 on ml.g4dn.xlarge (3.8 hours)

See [Embedding Model](wiki/Architecture/Embedding-Model.md) for architecture details and [Training Guide](wiki/Development/Training.md) for the complete training process.

## License

Private project.
