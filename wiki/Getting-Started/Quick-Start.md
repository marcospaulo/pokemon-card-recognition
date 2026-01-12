# Quick Start Guide

Get the Pokemon card recognition system running quickly.

## Prerequisites

- AWS CLI configured with access to `pokemon-card-training-us-east-2`
- Python 3.9+ installed locally
- (Optional) Raspberry Pi 5 for edge deployment

---

## Step 1: Download Reference Database (5 minutes)

The reference database contains pre-computed embeddings for all 17,592 cards:

```bash
# Create project directory
mkdir -p pokemon-card-recognition/data/reference
cd pokemon-card-recognition

# Download reference database (106 MB)
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/ \
  ./data/reference/
```

**What you get:**
- `embeddings.npy` - 17,592 x 768 embedding vectors
- `usearch.index` - ARM-optimized vector search index
- `index.json` - Row-to-card-ID mapping
- `metadata.json` - Card names, sets, rarity, etc.

---

## Step 2: Download Models (2 minutes)

### For Development (PyTorch)
```bash
# Download PyTorch weights (75 MB)
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-student/stage2/v2.0/student_stage2_final.pt \
  ./models/embedding/
```

### For Raspberry Pi (Hailo HEF)
```bash
# Download compiled HEF model (13.8 MB)
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-hailo/v2.1/pokemon_student_efficientnet_lite0_stage2.hef \
  ./models/embedding/
```

---

## Step 3: Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Key packages installed:
# - torch, torchvision, timm (for EfficientNet)
# - usearch (ARM-optimized vector search)
# - numpy, pillow (data handling)
```

---

## Step 4: Test Inference

### Option A: Local Testing (PyTorch)

```python
from src.inference.recognizer import CardRecognizer
from PIL import Image

# Initialize recognizer
recognizer = CardRecognizer(
    model_path='models/embedding/student_stage2_final.pt',
    reference_db='data/reference/'
)

# Recognize a card
image = Image.open('test_card.jpg')
result = recognizer.recognize(image)

print(f"Card: {result['name']}")
print(f"Set: {result['set']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Option B: Raspberry Pi Deployment

See **[Raspberry Pi Setup](../Deployment/Raspberry-Pi-Setup.md)** for complete deployment guide.

---

## Step 5: Test with Real Card Images

Download sample card images:
```bash
# Download a few test cards
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/raw/card_images/ \
  ./test_images/ \
  --exclude "*" \
  --include "Charizard*.png" \
  --include "Pikachu*.png"
```

Run batch recognition:
```bash
python scripts/test_recognition.py --input test_images/ --output results.json
```

---

## Expected Performance

| Environment | Embedding Time | Search Time | Total |
|-------------|---------------|-------------|-------|
| **MacBook Pro (M1)** | 15ms | 2ms | ~17ms |
| **Raspberry Pi 5 + Hailo** | 8ms | 3ms | ~11ms |
| **AWS SageMaker (ml.c5.xlarge)** | 5ms | 1ms | ~6ms |

---

## Next Steps

1. **Explore Architecture**: Read [System Overview](../Architecture/System-Overview.md) to understand the pipeline
2. **Train Your Own Model**: See [Model Training](../Development/Training.md) for custom training
3. **Deploy to Edge**: Follow [Raspberry Pi Setup](../Deployment/Raspberry-Pi-Setup.md) for hardware deployment
4. **Access Full Dataset**: Check [Data Management](../Infrastructure/S3-Data-Management.md) for complete data access

---

## Troubleshooting

### AWS Access Denied
```bash
# Verify AWS credentials
aws sts get-caller-identity

# Ensure you're using us-east-2 region
aws configure get region
```

### Missing Dependencies
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt

# For uSearch on ARM (Raspberry Pi)
pip install usearch --no-binary usearch
```

### Model Loading Errors
```python
# Remove torch.compile prefixes if needed
import torch
state_dict = torch.load('model.pt')
new_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
torch.save(new_dict, 'model_clean.pt')
```

---

## Additional Resources

- **[Overview](Overview.md)** - Project introduction
- **[Hardware Requirements](Hardware-Requirements.md)** - Required components
- **[AWS Organization](../Infrastructure/AWS-Organization.md)** - S3 and SageMaker setup
- **[Project History](../Project-History/Organization-Journey.md)** - How we built this

---

**Need help?** Check the [Home](../Home.md) page for more documentation links.

---

**Last Updated:** 2026-01-11
