# Project Overview

Real-time Pokemon card detection and recognition on Raspberry Pi 5 using edge AI hardware.

---

## What is This?

A complete end-to-end system that can identify any of **17,592 Pokemon cards** in real-time using:
- **AI Camera (IMX500)** for card detection
- **Hailo 8L NPU** for neural network inference
- **uSearch** for vector similarity matching

The system runs entirely on edge hardware with no cloud dependency after deployment.

---

## Key Features

### 🎯 Accuracy
- Recognizes 17,592 unique Pokemon cards across 160 sets
- 768-dimensional embeddings for precise matching
- Knowledge distillation from 86M parameter teacher model

### ⚡ Speed
- ~11ms total inference time on Raspberry Pi 5
- 8ms embedding extraction (Hailo NPU)
- 3ms vector search (CPU)

### 🔬 Model Compression
- **Teacher Model:** DINOv3 ViT-B/14 (86M parameters)
- **Student Model:** EfficientNet-Lite0 (4.7M parameters)
- **Compression Ratio:** 18× smaller while maintaining accuracy

### 💾 Complete Data Pipeline
- 17,592 original card images (13 GB)
- 17,592 processed training images (13 GB)
- 1,024 Hailo calibration images (734 MB)
- Pre-computed reference database (106 MB)
- Everything organized and backed up on AWS S3

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Raspberry Pi 5                           │
│                                                             │
│  ┌───────────────┐      ┌──────────────┐                  │
│  │ IMX500 Camera │      │  Hailo 8L    │                  │
│  │ (On-sensor)   │──────│  (NPU)       │                  │
│  │ YOLO11n-OBB   │      │ EfficientNet │                  │
│  └───────────────┘      └──────────────┘                  │
│         │                      │                           │
│         │ Bounding Box         │ 768-dim Embedding         │
│         ▼                      ▼                           │
│  ┌────────────────────────────────────────┐               │
│  │           CPU (ARM Cortex-A76)         │               │
│  │  ┌─────────────┐    ┌───────────────┐ │               │
│  │  │ Perspective │    │    uSearch    │ │               │
│  │  │   Warp      │────│ Vector Search │ │               │
│  │  └─────────────┘    └───────────────┘ │               │
│  │                            │           │               │
│  │                            ▼           │               │
│  │                    Card Match + Info   │               │
│  └────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### Pipeline Stages

1. **Detection (IMX500)**: YOLO11n-OBB detects cards with oriented bounding boxes
2. **Preprocessing (CPU)**: Perspective correction, cropping, normalization
3. **Embedding (Hailo 8L)**: EfficientNet-Lite0 extracts 768-dimensional features
4. **Matching (CPU)**: uSearch finds nearest neighbor in reference database
5. **Validation (CPU)**: Temporal smoothing and confidence thresholding

See **[System Overview](../Architecture/System-Overview.md)** for detailed architecture documentation.

---

## Technology Stack

### Hardware
- **Raspberry Pi 5** (8GB RAM recommended)
- **IMX500 AI Camera** (Sony sensor with on-chip NPU)
- **Hailo 8L** AI accelerator (26 TOPS)

### Software
- **Detection:** YOLO11n-OBB (Ultralytics)
- **Embedding:** EfficientNet-Lite0 (timm)
- **Training:** PyTorch, AWS SageMaker
- **Compilation:** Hailo Dataflow Compiler
- **Search:** uSearch (ARM-optimized)

### Cloud Infrastructure
- **Storage:** AWS S3 (31.7 GB organized data)
- **Training:** AWS SageMaker (ml.g5.2xlarge)
- **Model Registry:** SageMaker Model Registry
- **Monitoring:** CloudWatch

---

## Project Statistics

### Data
- **17,592** unique Pokemon cards
- **160** card sets (Base Set through modern)
- **31.7 GB** total data on S3
- **51,970** files in organized structure

### Models
- **Teacher Model:** DINOv3 ViT-B/14 - 5.6 GB (86M parameters)
- **Student Model:** EfficientNet-Lite0 - 75 MB PyTorch, 13.8 MB HEF
- **Training:** Two-stage knowledge distillation
- **Versions:** 2 models in SageMaker Registry

### Infrastructure
- **Training Cost:** $11.50 (one-time)
- **Storage Cost:** $0.73/month (~$8.76/year)
- **Region:** us-east-2
- **Account:** marcospaulo (943271038849)

---

## Use Cases

### Primary: Real-Time Card Recognition
Point the camera at a Pokemon card and get instant identification with metadata (name, set, rarity, HP, attacks, etc.)

### Secondary Applications
1. **Card Collection Management:** Scan and catalog your collection
2. **Trading Verification:** Verify card authenticity and edition
3. **Game Assistance:** Quick lookup during gameplay
4. **Educational Tool:** Learn about Pokemon cards and their attributes

---

## Development Timeline

| Date | Milestone |
|------|-----------|
| **2026-01-09** | Initial project structure created |
| **2026-01-10** | DINOv3 teacher model trained on SageMaker |
| **2026-01-10** | Stage 1 distillation (with teacher) |
| **2026-01-11** | Stage 2 distillation (independent) |
| **2026-01-11** | Hailo HEF compilation completed |
| **2026-01-11** | Reference database generated (17,592 embeddings) |
| **2026-01-11** | Complete S3 data migration (31.7 GB) |
| **2026-01-11** | Documentation wiki created |

See **[Training History](../Project-History/Training-History.md)** for detailed timeline.

---

## Quick Links

### Get Started
- **[Quick Start](Quick-Start.md)** - Download and run inference
- **[Hardware Requirements](Hardware-Requirements.md)** - Components needed
- **[Raspberry Pi Setup](../Deployment/Raspberry-Pi-Setup.md)** - Deploy to edge

### Learn More
- **[System Overview](../Architecture/System-Overview.md)** - Detailed architecture
- **[Model Training](../Development/Training.md)** - How models were trained
- **[AWS Organization](../Infrastructure/AWS-Organization.md)** - Cloud infrastructure

### Access Data
- **[Data Management](../Infrastructure/S3-Data-Management.md)** - S3 structure
- **[Access Control](../Infrastructure/Access-Control.md)** - IAM and permissions
- **[Cost Analysis](../Infrastructure/Cost-Analysis.md)** - Pricing breakdown

---

## Project Goals

### Completed ✅
- [x] Train high-accuracy teacher model (DINOv3)
- [x] Distill to efficient student model (EfficientNet-Lite0)
- [x] Compile for Hailo NPU
- [x] Generate reference database (17,592 cards)
- [x] Organize all data on S3
- [x] Document infrastructure and architecture

### In Progress 🚧
- [ ] Deploy to Raspberry Pi 5
- [ ] Integrate IMX500 camera
- [ ] Real-time inference pipeline
- [ ] UI for card display

### Future 🔮
- [ ] Mobile app integration
- [ ] Multi-card detection (multiple cards in frame)
- [ ] Condition assessment (card quality)
- [ ] Price estimation integration

---

## Contributing

This is a private project, but the architecture and techniques are reusable for similar recognition tasks:
- Trading card games (MTG, Yu-Gi-Oh, etc.)
- Product recognition
- Document classification
- Any visual search application

---

**Ready to get started?** Head to the **[Quick Start](Quick-Start.md)** guide!

---

**Last Updated:** 2026-01-11
