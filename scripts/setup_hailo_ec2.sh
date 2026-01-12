#!/bin/bash
# Setup EC2 instance for Hailo model compilation
#
# Instance requirements:
#   - Ubuntu 22.04 or 20.04
#   - m5.2xlarge or larger (8 vCPUs, 32 GB RAM minimum)
#   - 100 GB EBS storage
#
# Usage:
#   1. Launch EC2 instance
#   2. SSH to instance
#   3. Run: bash setup_hailo_ec2.sh

set -e

echo "======================================"
echo "Hailo Dataflow Compiler Setup"
echo "======================================"

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install prerequisites
echo "Installing prerequisites..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install \
    numpy \
    onnx \
    onnxruntime \
    pillow \
    tqdm

# Download Hailo Dataflow Compiler
# Note: You need to get this from Hailo's website after registering
# https://hailo.ai/developer-zone/software-downloads/
echo ""
echo "======================================"
echo "⚠️  MANUAL STEP REQUIRED"
echo "======================================"
echo ""
echo "You need to download the Hailo Dataflow Compiler from:"
echo "https://hailo.ai/developer-zone/software-downloads/"
echo ""
echo "Download: 'Hailo Dataflow Compiler v3.28.0' (or latest)"
echo "File: hailo_dataflow_compiler-3.28.0-py3-none-linux_x86_64.whl"
echo ""
echo "Then install with:"
echo "  pip3 install hailo_dataflow_compiler-3.28.0-py3-none-linux_x86_64.whl"
echo ""
echo "Or if you have it in S3:"
echo "  aws s3 cp s3://YOUR_BUCKET/hailo_dataflow_compiler-*.whl ."
echo "  pip3 install hailo_dataflow_compiler-*.whl"
echo ""
echo "======================================"

# Create working directory
mkdir -p ~/hailo-workspace
cd ~/hailo-workspace

echo ""
echo "✅ System setup complete!"
echo ""
echo "Next steps:"
echo "1. Install Hailo Dataflow Compiler (see instructions above)"
echo "2. Copy your ONNX model to this instance"
echo "3. Run the compilation script"
