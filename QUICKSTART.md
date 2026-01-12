# 🚀 Quick Start: Deploy to Raspberry Pi

**Goal**: Get your distilled Pokemon card recognition model running on Raspberry Pi in 30-60 minutes.

## ✅ What You Have

- ✅ Trained distilled model (Stage 2 complete)
- ✅ Model artifact: `s3://sagemaker-us-east-2-943271038849/pytorch-training-2026-01-11-09-27-50-349/output/model.tar.gz`
- ✅ All deployment scripts ready
- ✅ Raspberry Pi: `grailseeker@raspberrypi.local`

---

## 🎯 Three-Step Deployment

### Step 1: Export to ONNX (5 minutes)

```bash
cd pokemon-card-recognition

# Export model
python scripts/export_student_to_onnx.py
```

✅ **Output**: `models/onnx/pokemon_student_convnext_tiny.onnx`

---

### Step 2: Compile with Hailo (30 minutes)

#### 2a. Launch EC2 Instance

```bash
# From AWS Console or CLI
# Instance: m5.2xlarge, Ubuntu 22.04, 100GB storage
```

#### 2b. Setup & Compile

```bash
# SSH to EC2
ssh -i your-key.pem ubuntu@ec2-xxx

# Setup environment
wget https://raw.githubusercontent.com/.../setup_hailo_ec2.sh
bash setup_hailo_ec2.sh

# Download Hailo Dataflow Compiler from:
# https://hailo.ai/developer-zone/software-downloads/
pip3 install hailo_dataflow_compiler-*.whl

# Copy ONNX model
# (From local machine)
scp models/onnx/pokemon_student_convnext_tiny.onnx ubuntu@ec2-xxx:~/

# Compile
python3 compile_hailo.py --onnx pokemon_student_convnext_tiny.onnx
```

✅ **Output**: `hailo_output/pokemon_student.hef`

---

### Step 3: Deploy & Test (10 minutes)

```bash
# From EC2 or local machine, deploy to Pi
./scripts/deploy_to_pi.sh \
    hailo_output/pokemon_student.hef \
    grailseeker@raspberrypi.local

# SSH to Pi and test
ssh grailseeker@raspberrypi.local
cd ~/pokemon-card-detection
python3 test_inference_pi.py --model models/pokemon_student.hef
```

✅ **Expected**: ~10ms inference time, 768-dim embeddings

---

## 🎮 Next Actions

Once deployed:

1. **Test with real cards**:
   ```bash
   python3 test_inference_pi.py --model models/pokemon_student.hef --image card.jpg
   ```

2. **Benchmark performance**:
   ```bash
   python3 test_inference_pi.py --model models/pokemon_student.hef --benchmark
   ```

3. **Build reference database**: Generate embeddings for all 17,592 Pokemon cards

4. **Integrate camera**: Connect IMX500 for real-time capture

---

## 📚 Full Documentation

See `DEPLOYMENT_GUIDE.md` for complete details, troubleshooting, and advanced options.

---

## 💡 Tips

- **Hailo DFC**: Requires free registration at hailo.ai
- **Calibration**: Use 100-1000 Pokemon card images for optimal quantization
- **Pi Connection**: If `.local` doesn't work, use IP address
- **Cost**: EC2 m5.2xlarge = ~$0.38/hour (compile takes ~30 min = $0.19)

---

## 🆘 Quick Troubleshooting

**Can't reach Pi**:
```bash
nmap -sn 192.168.1.0/24 | grep -i raspberry
ssh grailseeker@[IP_ADDRESS]
```

**Hailo not found**:
```bash
lspci | grep Hailo  # On Pi
pip3 install hailort-*.whl
```

**ONNX export fails**:
```bash
pip install --upgrade torch torchvision timm
```

---

**Ready?** Start with Step 1! 🎉
