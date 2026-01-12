#!/bin/bash
#
# Clean up and organize EC2 Hailo workspace
#
# Creates organized structure:
# hailo_workspace/
# ├── models/           # ONNX models
# ├── compiled/         # HEF files (output)
# ├── calibration/      # Calibration data (.npy)
# ├── scripts/          # Compilation scripts
# └── archive/          # Old files (for safety)
#

set -e

EC2_HOST="ubuntu@18.118.102.134"
WORKSPACE="hailo_workspace"

echo "======================================================================"
echo "EC2 Workspace Cleanup"
echo "======================================================================"

ssh "$EC2_HOST" << 'EOSSH'
cd ~/hailo_workspace

echo ""
echo "[1/5] Current state:"
du -sh . 2>/dev/null
echo ""
ls -lh | grep -E "^d|\.onnx$|\.hef$|\.har$|\.npy$" | head -20

echo ""
echo "[2/5] Creating organized structure..."
mkdir -p models compiled calibration scripts archive logs

echo ""
echo "[3/5] Organizing files..."

# Move ONNX models
echo "  Moving ONNX models..."
mv -f *.onnx models/ 2>/dev/null || true

# Move calibration data
echo "  Moving calibration data..."
mv -f *.npy calibration/ 2>/dev/null || true
mv -f calibration_* archive/ 2>/dev/null || true

# Move compilation scripts
echo "  Moving scripts..."
mv -f *.sh scripts/ 2>/dev/null || true
mv -f *.py scripts/ 2>/dev/null || true

# Move logs
echo "  Moving logs..."
mv -f *.log logs/ 2>/dev/null || true

# Archive old compilations
echo "  Archiving old compilations..."
mv -f calibration_set.npy archive/ 2>/dev/null || true
mv -f parsed_model archive/ 2>/dev/null || true
mv -f optimized_model archive/ 2>/dev/null || true
mv -f compiled_model archive/ 2>/dev/null || true
mv -f compiled_h8 archive/ 2>/dev/null || true

# Keep student_h8 as is (latest compilation)
if [ -d "student_h8" ]; then
    echo "  Keeping student_h8/ (latest compilation)"
    # Move logs from student_h8 to logs/
    mv -f student_h8/*.log logs/ 2>/dev/null || true

    # Copy HEF to compiled/
    if [ -f "student_h8/pokemon_student_stage2_final.hef" ]; then
        cp -f student_h8/pokemon_student_stage2_final.hef compiled/pokemon_student_stage2_final.hef
        echo "  ✓ Copied HEF to compiled/"
    fi

    # Archive HAR files
    mv -f student_h8/*.har archive/ 2>/dev/null || true
fi

# Archive old model files
mv -f levit384.* archive/ 2>/dev/null || true
mv -f pokemon_tcgp_trained.onnx archive/ 2>/dev/null || true
mv -f pokemon_model_script.alls archive/ 2>/dev/null || true

echo ""
echo "[4/5] Cleaning up empty directories..."
find . -maxdepth 1 -type d -empty -delete 2>/dev/null || true

echo ""
echo "[5/5] New structure:"
echo ""
echo "hailo_workspace/"
for dir in models compiled calibration scripts logs archive student_h8; do
    if [ -d "$dir" ]; then
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        count=$(find "$dir" -maxdepth 1 -type f 2>/dev/null | wc -l)
        printf "├── %-15s %6s (%2d files)\n" "$dir/" "$size" "$count"
    fi
done

echo ""
echo "======================================================================"
echo "✅ Cleanup Complete"
echo "======================================================================"
echo ""
echo "Active files:"
echo "  Models: $(ls models/*.onnx 2>/dev/null | wc -l) ONNX"
echo "  Compiled: $(ls compiled/*.hef 2>/dev/null | wc -l) HEF"
echo "  Calibration: $(ls calibration/*.npy 2>/dev/null | wc -l) .npy"
echo "  Scripts: $(ls scripts/*.{sh,py} 2>/dev/null | wc -l) scripts"
echo ""
echo "Archived: $(find archive -type f 2>/dev/null | wc -l) files (old compilations)"
echo ""
echo "Current workspace size: $(du -sh . 2>/dev/null | cut -f1)"
echo ""

EOSSH

if [ $? -eq 0 ]; then
    echo "✅ EC2 workspace cleaned up and organized!"
else
    echo "❌ Cleanup failed"
    exit 1
fi
