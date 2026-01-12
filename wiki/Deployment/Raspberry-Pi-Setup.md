# Raspberry Pi Setup

> **Deployed**: January 11, 2026
> **Hardware**: Raspberry Pi 5 (8GB) + Hailo 8 NPU
> **OS**: Debian 13 (Trixie) 64-bit
> **Status**: ✅ Operational

[← Back to Wiki Home](../Home.md)

---

## Current Deployment

This documents the **actual Raspberry Pi setup** currently running the Pokemon card recognition system.

**Hostname**: `raspberrypi.local`
**User**: `grailseeker`
**Location**: `~/pokemon-card-recognition/`

---

## Hardware Specifications

### Actual Hardware Deployed

| Component | Specification | Status |
|-----------|--------------|--------|
| **Board** | Raspberry Pi 5 (8GB RAM) | ✅ Operational |
| **CPU** | 4× Cortex-A76 @ 2.4GHz | ✅ Active |
| **AI Accelerator** | Hailo 8 (26 TOPS) | ✅ Connected |
| **Storage** | 32GB microSD (29GB usable) | ✅ 47% used (13GB) |
| **Power** | 5V/5A USB-C | ✅ Stable |
| **Cooling** | Passive heatsink | ✅ Adequate |

### Storage Usage

```
Filesystem      Size  Used Avail Use%
/dev/mmcblk0p2   29G   13G   15G  47%
/dev/mmcblk0p1  510M   78M  433M  16%  (boot)
```

**Breakdown**:
- OS + packages: ~8 GB
- Project files: ~5 GB
  - Reference database: 111 MB
  - Model (HEF): 14 MB
  - Source code: ~10 MB
  - Test images: ~50 MB
  - Virtual env: ~500 MB

---

## Software Stack

### Operating System

```
OS: Debian GNU/Linux 13 (Trixie)
Kernel: Linux 6.x (64-bit)
Architecture: aarch64
Python: 3.13.5
```

### Key Packages (Deployed)

| Package | Version | Purpose |
|---------|---------|---------|
| **hailort** | 4.23.0 | Hailo runtime library |
| **hailo-tappas-core** | 5.1.0 | Hailo Python bindings |
| **usearch** | 2.x | Vector search (ARM-optimized) |
| **numpy** | 2.2.4 | Array operations |
| **opencv-python** | 4.12.0.88 | Image processing |
| **opencv** | 4.10.0 | CV operations |

### Full Requirements

Located at `~/pokemon-card-recognition/requirements.txt`:

```txt
numpy>=2.0.0
opencv-python>=4.8.0
usearch>=2.0.0
hailort>=4.23.0
```

---

## Directory Structure

### Actual Layout on Pi

```
~/pokemon-card-recognition/
├── data/
│   └── reference/              # Reference database (111 MB)
│       ├── embeddings.npy      # 52 MB - 17,592 × 768 embeddings
│       ├── usearch.index       # 55 MB - HNSW index
│       ├── index.json          # 374 KB - row → card_id mapping
│       └── metadata.json       # 4.8 MB - card details
│
├── models/
│   └── embedding/              # Embedding models
│       └── pokemon_student_efficientnet_lite0_stage2.hef  # 14 MB
│
├── src/
│   ├── inference/
│   │   └── recognize_card.py   # Main inference script
│   └── ...
│
├── test_images/                # Sample card images for testing
├── test_inference.py           # Quick test script
├── venv/                       # Python virtual environment
├── requirements.txt            # Python dependencies
└── hailort.log                # Hailo runtime logs
```

---

## Installation Steps

### Prerequisites

**Hardware**:
- Raspberry Pi 5 (4GB+ RAM recommended, 8GB for this deployment)
- Hailo 8 AI accelerator (M.2 HAT+ or USB)
- 32GB+ microSD card
- 5V/5A USB-C power supply

**Software**:
- Raspberry Pi OS 64-bit (Debian 13 recommended)
- Internet connection for initial setup

### 1. Install Hailo Software

```bash
# Install Hailo runtime
sudo apt update
sudo apt install hailort hailo-tappas-core-python-binding

# Verify installation
hailortcli fw-control identify
```

**Expected output**:
```
Identifying board
Control Protocol Version: 2
Firmware Version: 4.23.0 (release,app,extended context switch buffer)
Device Architecture: HAILO8
Board Name: Hailo-8
```

### 2. Clone Repository

```bash
# Create project directory
mkdir -p ~/pokemon-card-recognition
cd ~/pokemon-card-recognition

# Clone from GitHub (or download release)
git clone git@github.com:marcospaulo/pokemon-card-recognition.git .
```

### 3. Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 4. Download Reference Database

```bash
# Create data directory
mkdir -p data/reference

# Download from AWS S3 (requires AWS CLI configured)
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/ \
  data/reference/
```

**Files downloaded**:
- `embeddings.npy` (52 MB)
- `usearch.index` (55 MB)
- `index.json` (374 KB)
- `metadata.json` (4.8 MB)

### 5. Download Model

```bash
# Create models directory
mkdir -p models/embedding

# Download HEF model from S3
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-hailo/pokemon_student_efficientnet_lite0_stage2.hef \
  models/embedding/
```

**Model file**: 14 MB

---

## Testing the Setup

### Quick Test

```bash
cd ~/pokemon-card-recognition
source venv/bin/activate

# Run test inference
python test_inference.py
```

**Expected output**:
```
🚀 Initializing Pokemon Card Recognizer...
📚 Loading reference database...
   ✅ Loaded 17592 embeddings (768D)
   ✅ Loaded uSearch index
   ✅ Loaded metadata for 15987 cards
🧠 Initializing Hailo 8 NPU...
   ✅ Connected to Hailo device
   ✅ Loaded HEF: pokemon_student_efficientnet_lite0_stage2.hef
   ✅ Network configured
✅ Recognizer ready!

📸 Running inference...

#1: Technical Machine: Evolution (sv4)
   Confidence: 99.79%

#2: Technical Machine: Blindside (sv4)
   Confidence: 99.78%

#3: Technical Machine: Crisis Punch (sv4pt5)
   Confidence: 99.78%

⏱️  Performance:
   Embedding (Hailo): 15.2 ms
   Search (uSearch):  1.0 ms
   Total:             16.2 ms

✅ Inference pipeline working!
```

### Manual Test

```bash
# Run inference on specific image
python -m src.inference.recognize_card \
  --image test_images/pikachu.png \
  --model models/embedding/pokemon_student_efficientnet_lite0_stage2.hef \
  --reference data/reference/
```

---

## Performance Tuning

### CPU Governor

Set CPU to performance mode for lower latency:

```bash
# Check current governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Set to performance (requires root)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### Hailo Configuration

Ensure Hailo device is properly configured:

```bash
# Check device status
hailortcli fw-control identify

# Monitor during inference
hailortcli monitor
```

### Memory

Monitor memory usage:

```bash
# Check available memory
free -h

# Monitor during inference
watch -n 1 free -h
```

**Expected usage**:
- Idle: ~1.5 GB RAM used
- During inference: ~1.8 GB RAM used
- Peak: ~2.0 GB RAM used

---

## Network Configuration

### SSH Access

```bash
# From local machine
ssh grailseeker@raspberrypi.local

# Or using IP
ssh grailseeker@192.168.x.x
```

### GitHub SSH Key

Already configured (generated January 11, 2026):

```bash
# Location
~/.ssh/id_rsa
~/.ssh/id_rsa.pub

# Added to GitHub account
# Can push/pull without password
```

### AWS CLI

Already configured with IAM user `raspberry-pi-user`:

```bash
# Verify
aws sts get-caller-identity

# Expected output:
# {
#   "UserId": "AIDA5XH2VZ6AR3ZLAQYZF",
#   "Account": "943271038849",
#   "Arn": "arn:aws:iam::943271038849:user/raspberry-pi-user"
# }
```

**Access**:
- S3 buckets: Full access
- SageMaker: Read access
- IAM: Limited (can't create users)

---

## Troubleshooting

### Hailo Device Not Found

```bash
# Check if device is detected
lsusb | grep Hailo
# or
lspci | grep Hailo

# Reinstall runtime
sudo apt reinstall hailort
```

### Import Error: hailort

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall hailort Python bindings
pip install --upgrade hailort
```

### Out of Memory

```bash
# Check memory usage
free -h

# Kill other processes
sudo systemctl stop [service]

# Reduce batch size or use smaller model
```

### Slow Inference

```bash
# Check CPU governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Should be "performance" not "powersave"
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

---

## Maintenance

### Update Code

```bash
cd ~/pokemon-card-recognition
git pull origin main
```

### Update Dependencies

```bash
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

### Update Model

```bash
# Download new model from S3
aws s3 cp s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-hailo/[NEW_MODEL].hef \
  models/embedding/

# Update path in scripts
```

### Update Database

```bash
# Sync latest reference database
aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/reference/ \
  data/reference/ \
  --delete
```

---

## System Monitoring

### Check Hailo Status

```bash
# Device info
hailortcli fw-control identify

# Monitor in real-time
hailortcli monitor
```

### Check Logs

```bash
# Hailo runtime logs
tail -f ~/pokemon-card-recognition/hailort.log

# System logs
journalctl -f
```

### Resource Usage

```bash
# CPU and memory
htop

# Disk usage
df -h

# GPU (Hailo) usage
hailortcli monitor
```

---

## Backup and Recovery

### Backup Configuration

```bash
# Backup entire project
tar -czf pokemon-card-backup-$(date +%Y%m%d).tar.gz \
  ~/pokemon-card-recognition/

# Backup to S3
aws s3 cp pokemon-card-backup-$(date +%Y%m%d).tar.gz \
  s3://pokemon-card-training-us-east-2/backups/raspberry-pi/
```

### Restore from Backup

```bash
# Download backup
aws s3 cp s3://pokemon-card-training-us-east-2/backups/raspberry-pi/[BACKUP].tar.gz .

# Extract
tar -xzf [BACKUP].tar.gz

# Restore
mv pokemon-card-recognition ~/
```

---

## Security

### Firewall

```bash
# Check firewall status
sudo ufw status

# Allow SSH
sudo ufw allow ssh

# Enable firewall
sudo ufw enable
```

### Updates

```bash
# Keep system updated
sudo apt update && sudo apt upgrade -y

# Reboot if needed
sudo reboot
```

---

## Related Documentation

- **[System Overview](../Architecture/System-Overview.md)** - Architecture details
- **[Performance Tuning](Performance.md)** - Optimization guide
- **[AWS Resources](../Infrastructure/AWS-Resources.md)** - Data download

---

*Deployed and verified: January 11, 2026*
*Hostname: raspberrypi.local*
*User: grailseeker*
