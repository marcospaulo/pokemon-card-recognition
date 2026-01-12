#!/bin/bash
#
# Run Hailo HEF Recompilation on EC2
#
# This script connects to EC2 and runs the recompilation with corrected calibration data.
#

set -e

EC2_HOST="ubuntu@18.118.102.134"
WORKSPACE="hailo_workspace"

echo "======================================================================"
echo "Running Hailo HEF Recompilation on EC2"
echo "======================================================================"

echo ""
echo "Step 1: Verifying files on EC2..."
ssh "$EC2_HOST" << 'EOSSH'
cd ~/hailo_workspace

echo "Checking required files..."

if [ ! -f "pokemon_student_stage2_final.onnx" ]; then
    echo "❌ ONNX model not found"
    exit 1
fi
echo "✓ ONNX model found"

if [ ! -f "student_pokemon_calib_correct.npy" ]; then
    echo "❌ Corrected calibration not found"
    echo ""
    echo "Please upload it first:"
    echo "  scp data/hailo_calibration/student_pokemon_calib_correct.npy ubuntu@18.118.102.134:~/hailo_workspace/"
    exit 1
fi
echo "✓ Corrected calibration found"

# Check calibration data has correct normalization
python3 << 'PYEOF'
import numpy as np

calib = np.load('student_pokemon_calib_correct.npy')
print(f"\nCalibration data:")
print(f"  Shape: {calib.shape}")
print(f"  Range: [{calib.min():.3f}, {calib.max():.3f}]")

if calib.min() < -2.0 and calib.max() > 2.0:
    print(f"  ✓ ImageNet normalization detected")
else:
    print(f"  ⚠️  WARNING: Normalization may be incorrect")
    print(f"     Expected range: [-2.5, 2.5]")
PYEOF

EOSSH

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Pre-flight check failed"
    exit 1
fi

echo ""
echo "Step 2: Uploading recompilation script..."
scp scripts/recompile_student_with_correct_calibration.sh "$EC2_HOST:~/$WORKSPACE/"

echo ""
echo "Step 3: Running recompilation (this will take 5-10 minutes)..."
echo ""

ssh -t "$EC2_HOST" << 'EOSSH'
cd ~/hailo_workspace

# Make script executable
chmod +x recompile_student_with_correct_calibration.sh

# Run recompilation
./recompile_student_with_correct_calibration.sh \
    pokemon_student_stage2_final.onnx \
    student_pokemon_calib_correct.npy

EOSSH

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✅ Recompilation Complete!"
    echo "======================================================================"
    echo ""
    echo "The new HEF file is on EC2: ~/hailo_workspace/pokemon_student_stage2_final_FIXED.hef"
    echo ""
    echo "Next steps:"
    echo "  1. Download the new HEF:"
    echo "     scp ubuntu@18.118.102.134:~/hailo_workspace/pokemon_student_stage2_final_FIXED.hef models/onnx/"
    echo ""
    echo "  2. Test it locally or on Raspberry Pi to verify it matches ONNX"
    echo ""
else
    echo ""
    echo "❌ Recompilation failed"
    echo ""
    echo "Check the EC2 logs for errors"
    exit 1
fi
