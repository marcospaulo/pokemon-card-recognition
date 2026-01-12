# Pre-Training Checklist
## Pokemon Card Recognition System

**Date:** January 10, 2026
**Purpose:** Final verification before starting training on AWS SageMaker

---

## 1. Environment Setup

### ✅ Python Environment
- [ ] Python 3.10+ installed
- [ ] Create virtual environment: `python -m venv venv`
- [ ] Activate environment: `source venv/bin/activate` (macOS/Linux)

### ✅ Install Latest Packages (January 2026)
```bash
pip install --upgrade pip

# Core packages - latest versions
pip install torch>=2.9.0 torchvision>=0.20.0
pip install transformers>=5.0.0
pip install timm>=1.0.12
pip install albumentations>=1.4.0
pip install opencv-python>=4.11.0
pip install numpy>=2.0.0
pip install Pillow>=11.0.0
pip install tqdm>=4.67.0

# AWS SDK
pip install boto3 sagemaker
```

### ✅ Verify Installations
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import timm; print(f'timm: {timm.__version__}')"
```

**Expected output:**
- PyTorch: 2.9.0 or later
- Transformers: 5.0.0 or later
- timm: 1.0.12 or later

---

## 2. Data Verification

### ✅ Classification Dataset Structure
```bash
data/raw/card_images/
├── train/
│   ├── [card_id_1]/
│   │   └── *.png
│   ├── [card_id_2]/
│   └── ... (17,592 classes)
└── val/
    ├── [card_id_1]/
    └── ...
```

**Verify:**
- [ ] `ls data/raw/card_images/train/ | wc -l` → Should be 17,592 directories
- [ ] `find data/raw/card_images/train/ -name "*.png" | wc -l` → Check total image count
- [ ] Each class has at least 1 image in train and 1 in val

### ✅ Detection Dataset (for future Phase 1)
```bash
data/processed/detection/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml
```

**Verify:**
- [ ] YOLO format labels exist (one .txt per image)
- [ ] `dataset.yaml` has correct paths

---

## 3. AWS Setup

### ✅ AWS Credentials
- [ ] AWS CLI installed: `aws --version`
- [ ] Credentials configured: `aws configure`
- [ ] Test connection: `aws s3 ls`

### ✅ S3 Bucket Setup
```bash
# Create bucket
aws s3 mb s3://pokemon-card-recognition

# Upload classification dataset
aws s3 sync data/raw/card_images/train/ s3://pokemon-card-recognition/datasets/embedding/train/
aws s3 sync data/raw/card_images/val/ s3://pokemon-card-recognition/datasets/embedding/val/

# Verify upload
aws s3 ls s3://pokemon-card-recognition/datasets/embedding/train/ | wc -l
```

**Checklist:**
- [ ] S3 bucket created: `pokemon-card-recognition`
- [ ] Training data uploaded to S3
- [ ] Validation data uploaded to S3
- [ ] Verify file count matches local dataset

### ✅ SageMaker IAM Role
- [ ] IAM role exists with S3 read/write permissions
- [ ] Role has ECR permissions for Docker
- [ ] Role has CloudWatch Logs permissions
- [ ] Note role ARN: `arn:aws:iam::ACCOUNT_ID:role/SageMakerRole`

---

## 4. Model Verification

### ✅ Test DINOv3 Model Loading Locally
```bash
python -c "
from transformers import Dinov3Model
import torch

print('Loading DINOv3 ViT-L/16...')
model = Dinov3Model.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m')
print(f'Model loaded! Hidden size: {model.config.hidden_size}')

# Test forward pass
dummy_input = torch.randn(1, 3, 224, 224)
outputs = model(dummy_input)
print(f'Output shape: {outputs.last_hidden_state.shape}')
print('✓ DINOv3 model works correctly!')
"
```

**Expected output:**
```
Loading DINOv3 ViT-L/16...
Model loaded! Hidden size: 1024
Output shape: torch.Size([1, 257, 1024])
✓ DINOv3 model works correctly!
```

**Checklist:**
- [ ] DINOv3 model downloads successfully
- [ ] No import errors
- [ ] Output shape is correct (batch_size, num_patches+1, hidden_dim)

### ✅ Test Student Model (ConvNeXt-Tiny)
```bash
python -c "
import timm
import torch

print('Loading ConvNeXt-Tiny...')
model = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
print(f'Model loaded! Features: {model.num_features}')

# Test forward pass
dummy_input = torch.randn(1, 3, 224, 224)
features = model(dummy_input)
print(f'Output shape: {features.shape}')
print('✓ ConvNeXt-Tiny works correctly!')
"
```

**Expected output:**
```
Loading ConvNeXt-Tiny...
Model loaded! Features: 768
Output shape: torch.Size([1, 768])
✓ ConvNeXt-Tiny works correctly!
```

---

## 5. Training Scripts Verification

### ✅ File Structure
```bash
# Verify all training scripts exist
ls src/training/train_dinov3_teacher.py
ls src/training/distill_student.py
ls src/models/dinov3_embedding.py
ls scripts/launch_teacher_training.py
```

**Checklist:**
- [ ] `src/models/dinov3_embedding.py` exists
- [ ] `src/training/train_dinov3_teacher.py` exists
- [ ] `src/training/distill_student.py` exists
- [ ] `scripts/launch_teacher_training.py` exists

### ✅ Syntax Check
```bash
python -m py_compile src/models/dinov3_embedding.py
python -m py_compile src/training/train_dinov3_teacher.py
python -m py_compile src/training/distill_student.py
```

**Checklist:**
- [ ] No syntax errors in any training scripts
- [ ] All imports resolve correctly

### ✅ Local Dry Run (Optional but Recommended)
```bash
# Test teacher training with tiny dataset (2 classes, 10 images each)
python src/training/train_dinov3_teacher.py \
  --train-dir ./test_data/train \
  --val-dir ./test_data/val \
  --model-dir ./test_outputs \
  --epochs-frozen 1 \
  --epochs-unfrozen 1 \
  --batch-size 4
```

**Checklist:**
- [ ] Script starts without errors
- [ ] Model loads successfully
- [ ] Training loop runs
- [ ] Checkpoints are saved to `test_outputs/`
- [ ] ONNX export completes

---

## 6. Cost Estimation Review

### Expected Costs (Spot Instances)

| Phase | Instance Type | Duration | Spot Price | Cost |
|-------|--------------|----------|------------|------|
| Teacher Fine-tuning | ml.g5.4xlarge | ~3 hours | ~$0.60/hr | ~$1.80 |
| Hailo Student | ml.g4dn.xlarge | ~1 hour | ~$0.25/hr | ~$0.25 |
| iOS Student | ml.g4dn.xlarge | ~1 hour | ~$0.25/hr | ~$0.25 |
| Android Student | ml.g4dn.xlarge | ~1 hour | ~$0.25/hr | ~$0.25 |
| **Total** | | | | **~$2.55** |

**Checklist:**
- [ ] Budget approved: ~$3 for training
- [ ] Spot instances enabled in SageMaker config
- [ ] CloudWatch alarms set for cost monitoring (optional)

---

## 7. Training Execution Plan

### Phase 1: Teacher Fine-tuning (~3 hours)

**Command:**
```bash
cd scripts/
python launch_teacher_training.py
```

**Expected Output:**
- Training logs in CloudWatch
- Checkpoints saved to S3: `s3://pokemon-card-recognition/models/embedding/teacher/`
- Final files:
  - `best_teacher.pt`
  - `dinov3_teacher.onnx`
  - `metrics.json`

**Checklist:**
- [ ] Job starts successfully
- [ ] Training logs show decreasing loss
- [ ] Validation accuracy > 90%
- [ ] Final ONNX export completes
- [ ] Files uploaded to S3

### Phase 2: Student Distillation (~1 hour each)

**Download teacher checkpoint:**
```bash
aws s3 cp s3://pokemon-card-recognition/models/embedding/teacher/best_teacher.pt ./models/teacher/
```

**Distill to ConvNeXt-Tiny (Hailo):**
```bash
python src/training/distill_student.py \
  --teacher-checkpoint ./models/teacher/best_teacher.pt \
  --student-model convnext_tiny \
  --image-size 224 \
  --train-dir data/raw/card_images/train/ \
  --val-dir data/raw/card_images/val/ \
  --model-dir ./models/students/hailo/ \
  --epochs 30 \
  --batch-size 128
```

**Checklist:**
- [ ] Teacher checkpoint loads successfully
- [ ] Student training starts
- [ ] Validation loss decreases
- [ ] Best model saved: `convnext_tiny_best.pt`
- [ ] ONNX export completes: `convnext_tiny.onnx`

**Repeat for other students:**
- [ ] iOS: `convnext_base` (image size 384)
- [ ] Android: `vit_small_patch16_224` (image size 224)

---

## 8. Post-Training Verification

### ✅ Model Files
```bash
# Verify all models were saved
ls models/embedding/teacher/best_teacher.pt
ls models/embedding/teacher/dinov3_teacher.onnx

ls models/students/hailo/convnext_tiny_best.pt
ls models/students/hailo/convnext_tiny.onnx
```

**Checklist:**
- [ ] Teacher `.pt` file exists
- [ ] Teacher `.onnx` file exists
- [ ] Hailo student `.pt` file exists
- [ ] Hailo student `.onnx` file exists
- [ ] iOS student files exist
- [ ] Android student files exist

### ✅ Model Size Verification
```bash
ls -lh models/embedding/teacher/best_teacher.pt
ls -lh models/students/hailo/convnext_tiny_best.pt
```

**Expected sizes:**
- Teacher: ~1.2GB (304M params)
- ConvNeXt-Tiny: ~115MB (29M params)
- ConvNeXt-Base: ~350MB (89M params)

**Checklist:**
- [ ] File sizes are reasonable
- [ ] No corrupted files (can load with `torch.load()`)

### ✅ Generate Reference Embeddings
```bash
python scripts/build_reference_db.py \
  --model models/students/hailo/convnext_tiny_best.pt \
  --data data/raw/card_images/train/ \
  --output data/reference/hailo/
```

**Checklist:**
- [ ] Embeddings generated: `embeddings.npy` (17592, 768)
- [ ] Index created: `usearch.index`
- [ ] Metadata saved: `metadata.json`

---

## 9. Final Checklist Summary

### Before Training
- [ ] All packages installed and verified (torch 2.9.0+, transformers 5.0.0+)
- [ ] Dataset structure correct (17,592 classes)
- [ ] AWS credentials configured
- [ ] S3 bucket created and data uploaded
- [ ] IAM role configured
- [ ] DINOv3 model loads successfully locally
- [ ] Training scripts syntax check passes

### During Training
- [ ] Monitor CloudWatch logs for errors
- [ ] Check validation accuracy after each epoch
- [ ] Verify checkpoints are being saved
- [ ] Watch for spot instance interruptions

### After Training
- [ ] All model files exist and are correct size
- [ ] Teacher Top-1 accuracy > 90%
- [ ] Student validation loss decreases consistently
- [ ] ONNX exports complete without errors
- [ ] Reference embeddings generated successfully

---

## 10. Troubleshooting

### Issue: DINOv3 model not found
**Solution:** Verify transformers version >= 5.0.0
```bash
pip install --upgrade transformers>=5.0.0
```

### Issue: Out of memory during training
**Solutions:**
- Reduce batch size: `--batch-size 16` (instead of 32)
- Use gradient accumulation (add to script)
- Use larger instance: `ml.g5.12xlarge`

### Issue: Spot instance interrupted
**Solution:** SageMaker auto-resumes from checkpoints. Check S3 for latest checkpoint.

### Issue: Model checkpoint not saving
**Solution:** Verify the updated distillation script includes best model tracking (lines 889-893 in PRD_06)

### Issue: S3 upload fails
**Solutions:**
- Check IAM permissions
- Verify bucket name is correct
- Test with: `aws s3 ls s3://pokemon-card-recognition/`

---

## Ready to Train?

If all checkboxes are completed, you're ready to start training!

**Start command:**
```bash
cd scripts/
python launch_teacher_training.py
```

**Estimated completion:** ~3 hours (teacher) + ~3 hours (3 students) = **~6 hours total**
**Estimated cost:** **~$2.55** (with spot instances)

---

## Contact

If you encounter issues, check:
1. CloudWatch Logs: AWS Console → CloudWatch → Log Groups → `/aws/sagemaker/TrainingJobs`
2. SageMaker Console: AWS Console → SageMaker → Training Jobs
3. S3 Outputs: `aws s3 ls s3://pokemon-card-recognition/models/ --recursive`

Good luck with training! 🚀
