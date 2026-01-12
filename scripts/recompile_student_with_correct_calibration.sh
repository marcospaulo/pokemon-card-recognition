#!/bin/bash
#
# Recompile EfficientNet-Lite0 Student Model with CORRECTED Calibration Data
#
# This script recompiles the Hailo HEF with properly normalized calibration data
# that includes ImageNet mean/std normalization (fixing the original bug).
#

set -e  # Exit on any error

echo "======================================================================"
echo "Recompiling Student Model HEF with Corrected Calibration"
echo "======================================================================"

# Configuration
ONNX_MODEL="${1:-pokemon_student_stage2_final.onnx}"
CALIB_NPY="${2:-student_pokemon_calib_correct.npy}"
OUTPUT_HEF="pokemon_student_stage2_final_FIXED.hef"
OUTPUT_HAR="pokemon_student_stage2_final_FIXED.har"

echo ""
echo "Configuration:"
echo "  ONNX Model: $ONNX_MODEL"
echo "  Calibration: $CALIB_NPY"
echo "  Output HEF: $OUTPUT_HEF"
echo ""

# Verify files exist
if [ ! -f "$ONNX_MODEL" ]; then
    echo "❌ ERROR: ONNX model not found: $ONNX_MODEL"
    exit 1
fi

if [ ! -f "$CALIB_NPY" ]; then
    echo "❌ ERROR: Calibration data not found: $CALIB_NPY"
    echo ""
    echo "💡 Upload the corrected calibration file:"
    echo "   scp data/hailo_calibration/student_pokemon_calib_correct.npy ubuntu@18.118.102.134:~/hailo_workspace/"
    exit 1
fi

echo "✓ All input files verified"

# Create Python compilation script
echo ""
echo "[1/4] Creating compilation script..."

cat > /tmp/hailo_recompile.py << 'PYEOF'
#!/usr/bin/env python3
"""Recompile with corrected calibration data"""

from hailo_sdk_client import ClientRunner
import numpy as np
import sys
import os

def main():
    onnx_model = sys.argv[1]
    calib_npy = sys.argv[2]
    output_hef = sys.argv[3]
    output_har = sys.argv[4]

    print("\n[2/4] Loading calibration data...")
    calib_data = np.load(calib_npy)
    print(f"   Shape: {calib_data.shape}")
    print(f"   Dtype: {calib_data.dtype}")
    print(f"   Range: [{calib_data.min():.3f}, {calib_data.max():.3f}]")

    # Verify normalization looks correct
    if calib_data.min() < -2.0 and calib_data.max() > 2.0:
        print("   ✓ ImageNet normalization detected (correct!)")
    else:
        print(f"   ⚠️  WARNING: Calibration may not have ImageNet normalization!")
        print(f"      Expected range: [-2.5, 2.5]")
        print(f"      Actual range: [{calib_data.min():.2f}, {calib_data.max():.2f}]")

    print("\n[3/4] Initializing Hailo SDK...")
    runner = ClientRunner(hw_arch='hailo8l')  # Target Hailo-8L

    print("   Parsing ONNX model...")
    hn, npz = runner.translate_onnx_model(
        onnx_model,
        'pokemon_student_corrected'
    )
    print("   ✓ ONNX parsed successfully")

    print("\n[4/4] Optimizing with corrected calibration...")
    print("   This step performs INT8 quantization using the calibration data")
    print("   Expected duration: 5-10 minutes")

    # Save calibration array to temporary directory for Hailo SDK
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp(prefix='hailo_calib_')
    print(f"   Using temp dir: {temp_dir}")

    try:
        # Hailo SDK expects individual .npy files or a directory structure
        # Since we have a batch array, we'll split it into individual files
        print(f"   Splitting {calib_data.shape[0]} images into individual files...")

        for i in range(calib_data.shape[0]):
            img_path = os.path.join(temp_dir, f'calib_{i:04d}.npy')
            np.save(img_path, calib_data[i])

        print(f"   ✓ Created {calib_data.shape[0]} calibration files")

        # Run optimization
        runner.optimize(temp_dir)
        print("   ✓ Optimization complete")

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        print(f"   Cleaned up temp dir")

    print("\n[5/5] Compiling to HEF...")
    hef_bytes = runner.compile()

    # Save HEF
    with open(output_hef, 'wb') as f:
        f.write(hef_bytes)

    # Save HAR for debugging
    runner.save_har(output_har)

    print(f"   ✓ HEF saved: {output_hef}")
    print(f"   ✓ HAR saved: {output_har}")

    # Verify
    hef_size = os.path.getsize(output_hef)
    print(f"\n✅ Compilation complete!")
    print(f"   HEF size: {hef_size:,} bytes ({hef_size/1024/1024:.2f} MB)")

    return 0

if __name__ == '__main__':
    sys.exit(main())
PYEOF

chmod +x /tmp/hailo_recompile.py

echo "✓ Compilation script created"

# Run compilation
echo ""
python3 /tmp/hailo_recompile.py "$ONNX_MODEL" "$CALIB_NPY" "$OUTPUT_HEF" "$OUTPUT_HAR"

# Final summary
echo ""
echo "======================================================================"
echo "✅ RECOMPILATION COMPLETE"
echo "======================================================================"
echo ""
echo "New HEF file: $OUTPUT_HEF"
echo ""
echo "Next steps:"
echo "  1. Test the new HEF vs ONNX to verify embeddings match"
echo "  2. If validation passes, replace the old HEF"
echo "  3. Deploy to Raspberry Pi"
echo ""
echo "Validation command:"
echo "  python3 scripts/compare_hef_vs_onnx.py \\"
echo "    --hef $OUTPUT_HEF \\"
echo "    --onnx $ONNX_MODEL \\"
echo "    --test-images data/test_cards/"
echo ""
