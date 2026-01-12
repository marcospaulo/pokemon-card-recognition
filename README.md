# Pokemon Card Recognition System

Real-time Pokemon card detection and recognition on Raspberry Pi 5 with AI Camera (IMX500) and Hailo 8 accelerator.

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
├── data/
│   ├── raw/                    # Source images and metadata
│   │   ├── card_images/        # 17,592 Pokemon card PNGs
│   │   └── metadata/           # Card metadata JSONs
│   ├── processed/              # Training datasets
│   │   ├── classification/     # Embedding model training
│   │   └── detection/          # YOLO detection training
│   └── reference/              # Inference database
├── models/
│   ├── detection/              # YOLO models (.pt, .onnx)
│   └── embedding/              # LeViT models (.hef, .onnx)
├── src/
│   ├── data/                   # Dataset and augmentations
│   ├── models/                 # Model architectures
│   ├── training/               # Training scripts
│   ├── inference/              # Inference pipeline
│   └── hardware/               # Camera and accelerator
├── scripts/                    # Utility scripts
├── docs/                       # PRD and documentation
├── docker/                     # Docker/Hailo SDK
└── references/                 # Third-party references
```

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

All training data and inference databases are backed up on AWS S3 and available locally:

- **17,592** Pokemon card images (~13 GB raw)
- **17,592** Processed classification images (~13 GB)
- **1,024** Hailo calibration images (734 MB)
- **160** card sets with complete metadata
- **Reference database**: Pre-computed embeddings + uSearch index (106 MB)
  - 768-dimensional embeddings for all 17,592 cards
  - ARM-optimized vector search for real-time matching

### Data Access

All data is stored on S3 and can be downloaded:
```bash
# Download reference database (required for inference)
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/ ./data/reference/

# Download raw card images (optional, for development)
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/raw/ ./data/raw/
```

See `PROJECT_ACCESS.md` for complete S3 access documentation.

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
