# Hardware Requirements

Complete hardware specifications for running the Pokemon card recognition system.

---

## For Development (Local Machine)

### Minimum Requirements
- **CPU:** Modern quad-core (Intel i5/AMD Ryzen 5 or better)
- **RAM:** 8 GB
- **Storage:** 50 GB free space
- **OS:** macOS, Linux, or Windows 10/11
- **Python:** 3.9 or later

### Recommended for Training
- **GPU:** NVIDIA GPU with 16+ GB VRAM (RTX 3090, A5000, or better)
- **RAM:** 32 GB
- **Storage:** 100 GB SSD
- **OS:** Linux (Ubuntu 22.04) or macOS

### Tested Configurations
✅ **MacBook Pro M1 Max** (32GB RAM) - Excellent for inference testing
✅ **AWS SageMaker ml.g5.2xlarge** - Used for model training
✅ **Ubuntu 22.04** with NVIDIA RTX 3090 - Fast local development

---

## For Edge Deployment (Raspberry Pi)

### Required Hardware

#### 1. Raspberry Pi 5
- **Model:** Raspberry Pi 5 (8GB RAM recommended)
- **Cost:** ~$80
- **CPU:** Quad-core ARM Cortex-A76 @ 2.4 GHz
- **RAM:** 8 GB LPDDR4X (4GB works but limited)
- **I/O:** PCIe 2.0 x1 for Hailo accelerator

**Why Pi 5?**
- PCIe slot for Hailo NPU
- Faster CPU than Pi 4
- Better thermal management
- USB 3.0 for peripherals

#### 2. Hailo 8L AI Accelerator
- **Model:** Hailo-8L M.2 module
- **Cost:** ~$70
- **Performance:** 13 TOPS (tera operations per second)
- **Power:** 2.5W typical
- **Interface:** M.2 B+M key (PCIe 2.0 x1)

**Why Hailo-8L?**
- 18× faster than CPU inference
- Low power consumption
- Native PyTorch/ONNX support
- Excellent ARM optimization

**Purchase:** [Hailo Store](https://hailo.ai/products/hailo-8l-ai-accelerator-for-raspberry-pi-5/)

#### 3. Raspberry Pi AI Camera (IMX500)
- **Model:** Sony IMX500 + Raspberry Pi adapter
- **Cost:** ~$70
- **Resolution:** 12.3 MP (4056 x 3040)
- **On-sensor AI:** Built-in NPU for YOLO inference
- **Interface:** CSI-2 (Camera Serial Interface)

**Why IMX500?**
- On-sensor inference (detection runs on camera)
- Reduces latency and CPU load
- Optimized for real-time detection
- Official Raspberry Pi support

**Purchase:** [Raspberry Pi Store](https://www.raspberrypi.com/products/ai-camera/)

#### 4. Storage
- **microSD Card:** 64 GB Class 10 (minimum)
- **Recommended:** 128 GB UHS-I for better performance
- **Cost:** ~$15-25

**Note:** Models and reference database need ~500 MB, but OS and dependencies require 32+ GB.

#### 5. Power Supply
- **Official:** Raspberry Pi 5 27W USB-C Power Supply
- **Cost:** ~$12
- **Requirements:** 5V @ 5A (27W) for stable operation with Hailo

**Important:** Inadequate power causes throttling and instability with Hailo!

#### 6. Cooling (Highly Recommended)
- **Active Cooler:** Raspberry Pi Active Cooler
- **Cost:** ~$5
- **Why:** Pi 5 + Hailo generate significant heat under load

---

### Optional Hardware

#### Display
- **HDMI Monitor** - For visual feedback during development
- **Cost:** Varies ($50-200)

#### Case
- **Raspberry Pi 5 Case** with Hailo cutout
- **Cost:** ~$10-20
- **Options:**
  - Official Raspberry Pi Case for Pi 5
  - Argon ONE M.2 case (includes NVMe support)

#### Ethernet
- **Cat 6 Cable** - For stable SSH/file transfer
- **Cost:** ~$5

---

## Complete Hardware Shopping List

### Edge Deployment Setup

| Component | Cost | Notes |
|-----------|------|-------|
| Raspberry Pi 5 (8GB) | $80 | Core computer |
| Hailo 8L M.2 Accelerator | $70 | AI inference |
| IMX500 AI Camera | $70 | Detection + capture |
| 128GB microSD Card | $20 | Storage |
| 27W USB-C Power Supply | $12 | Reliable power |
| Active Cooler | $5 | Thermal management |
| Case (optional) | $15 | Protection |
| **Total** | **~$272** | One-time hardware cost |

---

## System Specifications

### Raspberry Pi 5 + Hailo 8L Performance

| Metric | Without Hailo | With Hailo |
|--------|---------------|-----------|
| **Embedding Inference** | 180ms | 8ms |
| **Power Draw** | 3.5W | 6.0W |
| **Temp (idle)** | 45°C | 50°C |
| **Temp (load)** | 70°C | 65°C |
| **Total Latency** | ~200ms | ~11ms |

**18× speedup** with Hailo acceleration!

---

## AWS Infrastructure (For Training)

If you want to train your own models:

### SageMaker Training Instance
- **Instance Type:** ml.g5.2xlarge
- **GPU:** NVIDIA A10G (24 GB VRAM)
- **vCPUs:** 8
- **RAM:** 32 GB
- **Storage:** 250 GB EBS
- **Cost:** $1.52/hour
- **Training Duration:** ~7.5 hours total
- **Total Cost:** ~$11.50

### SageMaker Compilation (Hailo)
- **Instance Type:** m5.2xlarge
- **vCPUs:** 8
- **RAM:** 32 GB
- **Cost:** $0.384/hour
- **Compilation Time:** ~30 minutes
- **Total Cost:** ~$0.19

### S3 Storage
- **Bucket:** pokemon-card-training-us-east-2
- **Total Size:** 31.7 GB
- **Cost:** $0.73/month (~$8.76/year)

**One-time training cost:** ~$11.70
**Ongoing storage cost:** ~$0.73/month

---

## Network Requirements

### For Development
- **Bandwidth:** 10+ Mbps (for AWS S3 downloads)
- **Stability:** Required for large downloads (13 GB datasets)

### For Raspberry Pi
- **Initial Setup:** WiFi or Ethernet (for OS updates, model download)
- **Runtime:** No internet required (inference is local)
- **SSH Access:** WiFi/Ethernet for development and debugging

---

## Software Requirements

### Development Machine
- **OS:** macOS 11+, Ubuntu 20.04+, Windows 10+
- **Python:** 3.9, 3.10, or 3.11
- **CUDA:** 11.8 or 12.1 (if using NVIDIA GPU)
- **AWS CLI:** v2.x for S3 access

### Raspberry Pi
- **OS:** Raspberry Pi OS Bookworm (64-bit)
- **Python:** 3.11 (included)
- **Hailo Runtime:** hailort 4.17+
- **Camera Stack:** libcamera (for IMX500)

See **[Raspberry Pi Setup](../Deployment/Raspberry-Pi-Setup.md)** for detailed software installation.

---

## Compatibility Notes

### Tested ✅
- Raspberry Pi 5 (8GB) + Hailo 8L + IMX500
- MacBook Pro M1/M2/M3 (native ARM)
- AWS SageMaker ml.g5.2xlarge
- Ubuntu 22.04 with NVIDIA RTX 3090

### Not Tested ❓
- Raspberry Pi 4 (lacks PCIe for Hailo)
- Raspberry Pi 5 (4GB) - may work but limited RAM
- Other Hailo models (8, 15)

### Not Supported ❌
- Raspberry Pi 3 or earlier (insufficient compute)
- Jetson Nano (different NPU ecosystem)
- Google Coral (different model format)

---

## Where to Buy

### Official Stores
- **Raspberry Pi:** [raspberrypi.com](https://www.raspberrypi.com)
- **Hailo:** [hailo.ai/shop](https://hailo.ai/shop)

### Distributors (US)
- **Adafruit:** [adafruit.com](https://www.adafruit.com)
- **SparkFun:** [sparkfun.com](https://www.sparkfun.com)
- **CanaKit:** [canakit.com](https://www.canakit.com)

### Distributors (International)
- **PiHut (UK):** [thepihut.com](https://thepihut.com)
- **PiShop (EU):** [pishop.eu](https://www.pishop.eu)

---

## Next Steps

1. **Order Hardware:** Use shopping list above
2. **Setup AWS:** See [AWS Organization](../Infrastructure/AWS-Organization.md)
3. **Download Models:** Follow [Quick Start](Quick-Start.md)
4. **Deploy to Pi:** See [Raspberry Pi Setup](../Deployment/Raspberry-Pi-Setup.md)

---

**Questions about hardware?** Check the [Overview](Overview.md) or [System Overview](../Architecture/System-Overview.md) for more context.

---

**Last Updated:** 2026-01-11
