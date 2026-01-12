# Pokemon Card Recognition System Wiki

> **Production System**: Raspberry Pi 5 + Hailo 8 NPU
> **Performance**: 16.2ms inference, 99.79% confidence
> **Deployed**: January 11, 2026
> **Status**: ✅ Operational

**Welcome to the comprehensive documentation for the Pokemon Card Recognition System.**

This system uses computer vision and machine learning on Raspberry Pi 5 with Hailo 8 NPU to recognize Pokemon trading cards in real-time with 96.8% accuracy. All documentation reflects the **actual deployed system**, not planning documents.

---

## Quick Start

### New to the Project?
1. **[System Overview](Architecture/System-Overview.md)** - Understand the AI pipeline architecture
2. **[Raspberry Pi Setup](Deployment/Raspberry-Pi-Setup.md)** - Deploy to Raspberry Pi 5
3. **[Dataset Reference](Reference/Dataset.md)** - Explore the 17,592 card database

### Looking for Something Specific?
- **Training a model?** → [Training Guide](Development/Training.md)
- **AWS infrastructure?** → [AWS Resources](Infrastructure/AWS-Resources.md)
- **Model architecture?** → [Embedding Model](Architecture/Embedding-Model.md)
- **Dataset details?** → [Dataset Reference](Reference/Dataset.md)

---

## Documentation Structure

This wiki follows the **Diataxis framework** for documentation:
- **Tutorials** (Learning-oriented): Step-by-step guides for beginners
- **How-To Guides** (Task-oriented): Recipes for solving specific problems
- **Reference** (Information-oriented): Technical specifications and API docs
- **Explanation** (Understanding-oriented): Architecture and design decisions

---

## Core Documentation

### Architecture

**Understand the system design and technical decisions.**

- **[System Overview](Architecture/System-Overview.md)** ✅ - Complete AI pipeline from capture to recognition
- **[Embedding Model](Architecture/Embedding-Model.md)** ✅ - EfficientNet-Lite0 distilled from DINOv2

**What's in System Overview:**
- Two-stage pipeline (preprocessing → embedding → search)
- Real performance metrics (16.2ms total, 15.2ms Hailo + 1.0ms search)
- Hardware stack (Raspberry Pi 5, Hailo 8 NPU, actual package versions)
- Working inference code walkthrough
- Database structure (17,592 embeddings, 768 dimensions)

**What's in Embedding Model:**
- Training details ($2.80 on SageMaker, 3.8 hours, 96.8% accuracy)
- Knowledge distillation from DINOv2 (86M params) to EfficientNet-Lite0 (4.7M params)
- Model compression results (18× parameter reduction, 400× size reduction)
- Deployment pipeline (PyTorch → ONNX → HEF)
- Inference code examples with Hailo 8

### Development

**Build, train, and test the system.**

- **[Training Guide](Development/Training.md)** ✅ - Complete SageMaker training process

**What's in Training Guide:**
- AWS SageMaker setup (ml.g4dn.xlarge, NVIDIA T4, $2.80 total cost)
- Actual hyperparameters and distillation loss function
- Training results (96.8% top-1, 99.9% top-5 accuracy)
- Export pipeline (PyTorch → ONNX → Hailo HEF)
- Re-training guide for future updates
- Troubleshooting common training issues

### Deployment

**Deploy models to production hardware.**

- **[Raspberry Pi Setup](Deployment/Raspberry-Pi-Setup.md)** ✅ - Complete hardware deployment guide

**What's in Pi Setup:**
- Actual hardware specs (Raspberry Pi 5 8GB, Hailo 8, storage usage)
- Real software stack (Debian 13, Python 3.13.5, hailort 4.23.0)
- Step-by-step installation (Hailo runtime, dependencies, models)
- Directory structure on live Pi (verified via SSH)
- Testing procedures with expected output (99.79% confidence)
- Performance tuning and troubleshooting

### Infrastructure

**AWS resources, costs, and data management.**

- **[AWS Resources](Infrastructure/AWS-Resources.md)** ✅ - Complete AWS infrastructure documentation

**What's in AWS Resources:**
- Real S3 structure (31.7 GB, complete directory tree)
- Actual storage breakdown (raw images, models, reference DB)
- IAM configuration (raspberry-pi-user with admin access)
- SageMaker training job details (ml.g4dn.xlarge, $2.80 cost)
- Cost analysis ($0.73/month storage, $2.80 one-time training)
- Download commands for all resources
- Backup and recovery procedures

### Reference

**Technical specifications and lookup information.**

- **[Dataset Reference](Reference/Dataset.md)** ✅ - Comprehensive dataset documentation

**What's in Dataset Reference:**
- Audit results (6/6 consistency score from Jan 11, 2026)
- Real card counts (17,592 images, 15,987 unique cards, 1,605 variants)
- Top 20 Pokemon sets with actual distribution
- File structure on S3 and Raspberry Pi
- Metadata structure with examples
- Download instructions for specific sets
- Variant explanation (language versions, quality levels)

---

## Common Tasks

### Training Tasks
- [Complete SageMaker training process](Development/Training.md) - Launch training on ml.g4dn.xlarge
- [Knowledge distillation setup](Development/Training.md#training-strategy) - DINOv2 → EfficientNet-Lite0
- [Export and compile models](Development/Training.md#model-export-pipeline) - PyTorch → ONNX → HEF
- [Re-training guide](Development/Training.md#re-training-guide) - Update model with new cards

### Deployment Tasks
- [Set up Raspberry Pi](Deployment/Raspberry-Pi-Setup.md#installation-steps) - Complete hardware setup
- [Install Hailo runtime](Deployment/Raspberry-Pi-Setup.md#1-install-hailo-software) - NPU configuration
- [Deploy HEF model](Deployment/Raspberry-Pi-Setup.md#5-download-model) - Download from S3
- [Test inference pipeline](Deployment/Raspberry-Pi-Setup.md#testing-the-setup) - Verify 99.79% confidence

### Data Tasks
- [Download reference database](Infrastructure/AWS-Resources.md#download-reference-database-required-for-inference) - 128 MB from S3
- [Explore dataset structure](Reference/Dataset.md#file-structure) - S3 and Pi organization
- [View top Pokemon sets](Reference/Dataset.md#top-20-sets-by-card-count) - Real distribution
- [Understand variants](Reference/Dataset.md#duplicate-variants-explained) - 1,605 extra images

### Infrastructure Tasks
- [Access S3 buckets](Infrastructure/AWS-Resources.md#s3-storage) - Navigate project structure
- [View training costs](Infrastructure/AWS-Resources.md#cost-analysis) - $2.80 one-time, $0.73/month
- [Configure AWS CLI](Infrastructure/AWS-Resources.md#aws-cli-setup) - raspberry-pi-user setup
- [View model registry](Infrastructure/AWS-Resources.md#sagemaker) - SageMaker models

---

## Project Stats

| Metric | Value |
|--------|-------|
| **Total Card Images** | 17,592 |
| **Unique Pokemon Cards** | 15,987 |
| **Variant Images** | 1,605 (languages, quality levels) |
| **Pokemon Sets** | 150 official sets |
| **Model Size (HEF)** | 14 MB (Hailo-compiled) |
| **Model Parameters** | 4.7M (EfficientNet-Lite0) |
| **Inference Time** | 16.2ms (15.2ms Hailo + 1.0ms search) |
| **Top-1 Accuracy** | 96.8% |
| **Top-5 Accuracy** | 99.9% |
| **Training Cost** | $2.80 (one-time on SageMaker) |
| **Monthly Cost** | $0.73 (S3 storage only) |
| **Storage (AWS S3)** | 31.7 GB |
| **Reference DB Size** | 128 MB (111 MB on Pi) |
| **Deployment Status** | ✅ Production (Jan 11, 2026) |

---

## External Resources

- **[Hailo Developer Zone](https://hailo.ai/developer-zone/)** - Hailo 8 NPU documentation
- **[Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/)** - Raspberry Pi 5 reference
- **[DINOv2 Paper](https://arxiv.org/abs/2304.07193)** - Teacher model (Meta AI)
- **[EfficientNet Paper](https://arxiv.org/abs/1905.11946)** - Student model architecture
- **[Pokemon TCG Database](https://www.pokemon.com/us/pokemon-tcg/)** - Official card database
- **[USearch Documentation](https://github.com/unum-cloud/usearch)** - Vector search library

---

## AWS Quick Links

### Console Access
- **[S3 Bucket](https://s3.console.aws.amazon.com/s3/buckets/pokemon-card-training-us-east-2?region=us-east-2&prefix=project/pokemon-card-recognition/)** - Project data (31.7 GB)
- **[SageMaker Training Jobs](https://console.aws.amazon.com/sagemaker/home?region=us-east-2#/jobs)** - View training history
- **[SageMaker Model Registry](https://console.aws.amazon.com/sagemaker/home?region=us-east-2#/model-packages)** - Model versions
- **[IAM Users](https://console.aws.amazon.com/iam/home#/users)** - raspberry-pi-user access

### Quick Commands
```bash
# Download reference database (required for inference)
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/ \
  ./data/reference/

# Download Hailo HEF model (14 MB)
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-hailo/pokemon_student_efficientnet_lite0_stage2.hef \
  ./models/embedding/

# Download raw card images (12.6 GB)
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/raw/card_images/ \
  ./data/raw/card_images/

# Run inference test on Raspberry Pi
cd ~/pokemon-card-recognition
python test_inference.py
```

---

## System Information

- **AWS Account**: 943271038849 (marcospaulo)
- **AWS Region**: us-east-2 (Ohio)
- **Pi Hostname**: raspberrypi.local
- **Pi User**: grailseeker
- **Deployment Date**: January 11, 2026
- **Project Status**: ✅ Production Operational

**Last Updated:** January 11, 2026 | **Wiki Version:** 2.0.0 (Fresh from reality)
