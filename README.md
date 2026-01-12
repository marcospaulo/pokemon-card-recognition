# Pokemon Card Recognition

Real-time Pokemon card detection and recognition using YOLO + EfficientNet on Raspberry Pi 5 with Hailo-8L NPU.

**Status**: Production Operational (16.2ms inference, 96.8% accuracy on 15,987 cards)

---

## 📚 Documentation

**All project documentation is in the [WIKI](wiki/Home.md)**

The `/wiki` directory is the single source of truth for:
- Architecture & system design
- Raspberry Pi setup & deployment
- Training guides (SageMaker)
- Dataset & AWS resources
- Model details & performance
- API documentation

---

## Essential Scripts

Core operational scripts are in `scripts/`:

**Data Management**
- `download_cards.py` - Download Pokemon card images
- `deduplicate_metadata.py` - Fix metadata duplicates
- `build_usearch_index.py` - Build vector search index

**Model Export & Training**
- `export_student_to_onnx.py` - Export EfficientNet to ONNX
- `export_yolo_to_imx500.py` - Export YOLO for IMX500
- `generate_reference_embeddings.py` - Create reference database
- `prepare_training_dataset.py` - Prepare training data
- `launch_student_distillation_8xA100.py` - Train on SageMaker
- `distill_yolo_for_imx500.py` - YOLO distillation

**Hailo Compilation**
- `create_correct_student_calibration.py` - Generate calibration data
- `recompile_student_with_correct_calibration.sh` - Recompile HEF
- `run_ec2_recompilation.sh` - Orchestrate EC2 recompilation
- `compare_hef_vs_onnx.py` - Validate HEF vs ONNX

**Deployment & Testing**
- `deploy_to_pi.sh` - Deploy to Raspberry Pi
- `test_inference_pi.py` - Test inference

**AWS**
- `cleanup_aws_resources.py` - Clean up AWS resources
- `audit_aws_resources.py` - Audit AWS usage

See [WIKI](wiki/Home.md) for detailed usage instructions.

---

## Quick Start

**On Raspberry Pi:**
```bash
# Clone and setup
git clone git@github.com:marcospaulo/pokemon-card-recognition.git
cd pokemon-card-recognition
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download reference database (128 MB)
aws s3 sync s3://pokemon-card-training/data/reference/ ./data/reference/

# Run inference
python scripts/test_inference_pi.py
```

**For training, deployment, and detailed setup**: See [WIKI](wiki/Home.md)

---

## Data Storage

⚠️ Large files (31.7 GB) are stored on AWS S3, not in git:
- Card images: 12.6 GB
- Training data: 12.5 GB
- Models: 6 GB
- Reference database: 128 MB

See [WIKI - AWS Resources](wiki/Infrastructure/AWS-Resources.md) for download instructions.

---

## License

Private project.
