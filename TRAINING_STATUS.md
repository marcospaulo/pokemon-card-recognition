# Pre-Training Checklist Status
**Date:** January 10, 2026
**Project:** Pokemon Card Recognition System

---

## ✅ READY

### 1. Directory Structure
✅ **Status:** All directories exist and properly organized
```
pokemon-card-recognition/
├── docs/          ✅ (12 files including PRDs and checklist)
├── data/          ✅ (raw, processed, reference)
├── models/        ✅
├── src/           ✅
├── scripts/       ✅
├── tests/         ✅
├── configs/       ✅
├── docker/        ✅
├── notebooks/     ✅
└── references/    ✅
```

### 2. Classification Dataset
✅ **Status:** Dataset prepared and split
- **Train:** 14,073 classes
- **Val:** 2,638 classes
- **Test:** 881 classes
- **Total:** 17,592 unique Pokemon cards ✅

**Location:** `/Users/marcos/dev/raspberry-pi/pokemon-card-recognition/data/processed/classification/`

**Files:**
- `train/` - Each subdirectory is a card class
- `val/` - Validation split
- `test/` - Test split
- `card_metadata.json` - Card information
- `class_index.json` - Class mappings

### 3. AWS Infrastructure
✅ **AWS CLI:** Installed (v2.32.9)
✅ **Credentials:** Configured (SSO profile: awscli)
✅ **Region:** us-east-2
✅ **S3 Access:** Working

**Existing S3 Bucket:** `s3://pokemon-card-training-us-east-2`
- Already has `classification_dataset/` uploaded
- Already has some training outputs from previous runs

---

## ⚠️ NEEDS ATTENTION

### 1. Python Version
❌ **Current:** Python 3.9.6
✅ **Required:** Python 3.10+

**Issue:** System Python is 3.9.6. PyTorch 2.9.0 and newer packages require Python 3.10+.

**Solutions:**
```bash
# Option 1: Install Python 3.10+ via Homebrew (recommended)
brew install python@3.10

# Option 2: Use pyenv
brew install pyenv
pyenv install 3.10
pyenv local 3.10
```

### 2. Package Versions
❌ **Current requirements.txt:** Outdated versions
```
torch>=2.0.0    (need 2.9.0+)
timm>=1.0.0     (need 1.0.12+)
NO transformers (need 5.0.0+ for DINOv3)
```

✅ **Updated requirements in PRD_06_TRAINING.md**

**Action needed:** Update `requirements.txt` file

### 3. Virtual Environment
❓ **Status:** Unknown if venv exists

**Action needed:**
```bash
cd /Users/marcos/dev/raspberry-pi/pokemon-card-recognition
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 4. Package Installation
❓ **Status:** Packages not verified

**Action needed:** Install latest packages after creating venv

---

## 📋 TODO BEFORE TRAINING

### Immediate Actions

1. **Upgrade Python to 3.10+**
   ```bash
   brew install python@3.10
   # or use pyenv
   ```

2. **Update requirements.txt**
   - Copy updated versions from `docs/PRD_06_TRAINING.md` lines 213-223
   - Add transformers>=5.0.0

3. **Create virtual environment**
   ```bash
   cd /Users/marcos/dev/raspberry-pi/pokemon-card-recognition
   python3.10 -m venv venv
   source venv/bin/activate
   ```

4. **Install packages**
   ```bash
   pip install --upgrade pip
   pip install torch>=2.9.0 torchvision>=0.20.0
   pip install transformers>=5.0.0
   pip install timm>=1.0.12
   pip install albumentations>=1.4.0
   pip install opencv-python>=4.11.0
   pip install numpy>=2.0.0
   pip install Pillow>=11.0.0
   pip install tqdm>=4.67.0
   pip install boto3 sagemaker
   ```

5. **Test DINOv3 model loading**
   ```python
   python -c "
   from transformers import Dinov3Model
   import torch

   print('Loading DINOv3...')
   model = Dinov3Model.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m')
   print(f'✅ Model loaded! Hidden size: {model.config.hidden_size}')

   dummy = torch.randn(1, 3, 224, 224)
   outputs = model(dummy)
   print(f'✅ Forward pass works: {outputs.last_hidden_state.shape}')
   "
   ```

6. **Verify dataset is already in S3**
   ```bash
   aws s3 ls s3://pokemon-card-training-us-east-2/classification_dataset/
   ```
   - If empty, upload: `aws s3 sync data/processed/classification/ s3://pokemon-card-training-us-east-2/datasets/embedding/`

7. **Create or verify SageMaker IAM role**
   - Check AWS Console → IAM → Roles
   - Look for "SageMakerExecutionRole" or similar
   - Needs permissions: S3 read/write, ECR, CloudWatch Logs

---

## 🎯 NEXT STEPS (After TODO Complete)

### Training Timeline (8x A100)

Once Python/packages are ready, you can start training immediately:

1. **Teacher Fine-tuning** (~15-20 minutes)
   ```bash
   cd scripts/
   python launch_teacher_training_8xA100.py
   ```
   - Instance: ml.p4d.24xlarge (8x A100 80GB)
   - Batch size: 256 (optimized for multi-GPU)
   - Epochs: 13 total (3 frozen + 10 unfrozen)
   - Cost: ~$1.20-1.50 (spot)

2. **Student Distillation** (~5-10 minutes each)
   - Hailo (ConvNeXt-Tiny): 5-7 mins
   - iOS (ConvNeXt-Base): 8-10 mins
   - Android (ViT-Small): 5-7 mins
   - Total: ~20-25 minutes for all 3 students

3. **Generate Reference Embeddings** (~2-3 minutes)

**TOTAL TIME: ~40-50 minutes** (vs the original 6 hour estimate for single GPU)
**TOTAL COST: ~$2-3** (spot instances)

---

## Summary

**Good news:**
- ✅ Project structure is perfect
- ✅ Dataset is ready (17,592 cards properly split)
- ✅ AWS is configured and S3 bucket exists
- ✅ Training scripts are bug-free and ready

**Needs fixing:**
- ❌ Python version (3.9.6 → 3.10+)
- ❌ Package installation (need latest versions)
- ❌ requirements.txt update

**Estimated time to ready:** ~30 minutes (Python upgrade + package install)

---

## Quick Start Commands

```bash
# 1. Upgrade Python
brew install python@3.10

# 2. Create venv
cd /Users/marcos/dev/raspberry-pi/pokemon-card-recognition
/opt/homebrew/bin/python3.10 -m venv venv
source venv/bin/activate

# 3. Install packages (copy from PRD_06_TRAINING.md)
pip install --upgrade pip
pip install torch>=2.9.0 torchvision>=0.20.0
pip install transformers>=5.0.0 timm>=1.0.12
pip install albumentations>=1.4.0 opencv-python>=4.11.0
pip install numpy>=2.0.0 Pillow>=11.0.0 tqdm>=4.67.0
pip install boto3 sagemaker

# 4. Test DINOv3
python -c "from transformers import Dinov3Model; print('✅ DINOv3 ready!')"

# 5. You're ready to train!
```
