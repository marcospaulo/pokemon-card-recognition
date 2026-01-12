# Pokemon Card Recognition - Project Wiki

Welcome to the Pokemon Card Recognition System wiki! This comprehensive documentation covers everything from architecture to deployment.

## 🎯 Project Overview

Real-time Pokemon card detection and recognition on Raspberry Pi 5 using AI Camera (IMX500) and Hailo 8L accelerator. The system can identify any of 17,592 Pokemon cards in real-time using a multi-stage AI pipeline.

**Key Stats:**
- **17,592** Pokemon cards from 160 sets
- **768-dimensional** embeddings for each card
- **~30ms** inference time on Raspberry Pi 5
- **86M → 4.7M params** via knowledge distillation (18× compression)
- **31.7 GB** of organized data on AWS S3

---

## 📚 Documentation Sections

### Getting Started
- **[Overview](Getting-Started/Overview.md)** - High-level system introduction
- **[Quick Start](Getting-Started/Quick-Start.md)** - Get up and running quickly
- **[Hardware Requirements](Getting-Started/Hardware-Requirements.md)** - Required hardware components

### Architecture
- **[System Overview](Architecture/System-Overview.md)** - Multi-stage AI pipeline architecture
- **[Detection Pipeline](Architecture/Detection-Pipeline.md)** - YOLO-based card detection
- **[Embedding Model](Architecture/Embedding-Model.md)** - EfficientNet-Lite0 feature extraction
- **[Reference Database](Architecture/Reference-Database.md)** - uSearch vector database design

### Development
- **[Model Training](Development/Training.md)** - Training pipeline on AWS SageMaker
- **[Knowledge Distillation](Development/Model-Development.md)** - DINOv3 → EfficientNet distillation
- **[SageMaker Setup](Development/SageMaker-Setup.md)** - AWS infrastructure configuration

### Deployment
- **[Raspberry Pi Setup](Deployment/Raspberry-Pi-Setup.md)** - Deploy to edge device
- **[Hardware Integration](Deployment/Hardware-Integration.md)** - IMX500 + Hailo configuration

### Infrastructure
- **[AWS Organization](Infrastructure/AWS-Organization.md)** - Complete S3 and SageMaker setup
- **[Data Management](Infrastructure/S3-Data-Management.md)** - 31.7 GB organized data structure
- **[Access Control](Infrastructure/Access-Control.md)** - IAM roles and permissions
- **[Cost Analysis](Infrastructure/Cost-Analysis.md)** - Training and storage costs

### Project History
- **[Organization Journey](Project-History/Organization-Journey.md)** - How we organized the project
- **[Data Integration](Project-History/Data-Integration.md)** - S3 data migration story
- **[Training History](Project-History/Training-History.md)** - Model training timeline

---

## 🔗 Quick Links

### AWS Resources
- [S3 Project Root](https://s3.console.aws.amazon.com/s3/buckets/pokemon-card-training-us-east-2?region=us-east-2&prefix=project/pokemon-card-recognition/)
- [SageMaker Console](https://console.aws.amazon.com/sagemaker/home?region=us-east-2)
- [Model Registry](https://console.aws.amazon.com/sagemaker/home?region=us-east-2#/model-package-groups/pokemon-card-recognition-models)

### Key Commands
```bash
# Download reference database
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/ ./data/reference/

# Download Hailo model for Pi
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-hailo/v2.1/pokemon_student_efficientnet_lite0_stage2.hef ./

# Run inference
python -m src.inference.pipeline
```

---

## 🏗️ Project Structure

```
pokemon-card-recognition/
├── data/                              # 25.2 GB - All training & inference data
│   ├── raw/                          # 13 GB - 17,592 original cards
│   ├── processed/                    # 13 GB - Processed for training
│   ├── calibration/                  # 734 MB - Hailo calibration
│   └── reference/                    # 106 MB - Embeddings + uSearch index
├── models/
│   ├── detection/                    # YOLO11n-OBB models
│   ├── embedding/                    # EfficientNet-Lite0 models
│   └── efficientnet-hailo/          # Hailo HEF (13.8 MB)
├── src/
│   ├── data/                         # Dataset loaders
│   ├── models/                       # Model architectures
│   ├── training/                     # Training scripts
│   ├── inference/                    # Inference pipeline
│   └── hardware/                     # IMX500 + Hailo integration
├── docs/                             # PRD documents
├── scripts/                          # Utility scripts
└── wiki/                             # This documentation

Backup on S3: s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/
```

---

## 🎓 Learning Path

**New to the project?** Follow this path:
1. Start with [Overview](Getting-Started/Overview.md) to understand the system
2. Review [System Overview](Architecture/System-Overview.md) for architecture details
3. Check [Quick Start](Getting-Started/Quick-Start.md) to run inference
4. Read [AWS Organization](Infrastructure/AWS-Organization.md) for data access
5. Explore [Model Training](Development/Training.md) for development

---

## 📞 Support

- **AWS Account:** marcospaulo (943271038849)
- **Region:** us-east-2
- **Service Role:** `SageMaker-MarcosAdmin-ExecutionRole`
- **Project Status:** ✅ Production Ready

---

**Last Updated:** 2026-01-11
**Version:** 1.0.0
