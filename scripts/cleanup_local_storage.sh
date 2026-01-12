#!/bin/bash
# Local Storage Cleanup - Pokemon Card Recognition
# Removes files that are safely backed up in S3

set -e

echo "════════════════════════════════════════════════════════════════════════════════"
echo "LOCAL STORAGE CLEANUP"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "⚠️  This will delete ~36 GB of files that are backed up in S3"
echo ""
echo "Files to DELETE:"
echo "  - data/processed/ (25 GB)"
echo "  - data/calibration/ (734 MB)"
echo "  - docker/hailo/hailo8_ai_sw_suite_2025-10.tar.gz (8.7 GB)"
echo "  - Old model files (2 GB)"
echo ""
echo "Files to KEEP:"
echo "  - data/reference/ (128 MB) ← Needed for Raspberry Pi"
echo "  - Final Hailo model (14 MB) ← Needed for deployment"
echo "  - Scripts and documentation"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Dry run mode by default
DRY_RUN=true
if [ "$1" == "--execute" ]; then
    DRY_RUN=false
    echo "⚠️  EXECUTE MODE: Files WILL be deleted!"
    read -p "Type 'DELETE' to confirm: " confirm
    if [ "$confirm" != "DELETE" ]; then
        echo "❌ Cleanup cancelled"
        exit 1
    fi
else
    echo "ℹ️  DRY RUN MODE (add --execute to actually delete)"
fi

echo ""

# Function to delete with confirmation
delete_item() {
    local path=$1
    local size=$2

    if [ ! -e "$path" ]; then
        echo "⏭️  Skip: $path (not found)"
        return
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would delete: $path ($size)"
    else
        echo "🗑️  Deleting: $path ($size)"
        rm -rf "$path"
        echo "   ✅ Deleted"
    fi
}

# Delete training data (backed up in S3)
echo "───────────────────────────────────────────────────────────────────────────────"
echo "TRAINING DATA"
echo "───────────────────────────────────────────────────────────────────────────────"
delete_item "data/processed" "25 GB"
delete_item "data/calibration" "734 MB"
# data/raw already deleted by user

# Delete Hailo software suite (if installed)
echo ""
echo "───────────────────────────────────────────────────────────────────────────────"
echo "HAILO SOFTWARE"
echo "───────────────────────────────────────────────────────────────────────────────"
echo "⚠️  Only delete if Hailo SDK is already installed or not needed"
delete_item "docker/hailo/hailo8_ai_sw_suite_2025-10.tar.gz" "8.7 GB"

# Delete old model files (keeping only the final Hailo model)
echo ""
echo "───────────────────────────────────────────────────────────────────────────────"
echo "OLD MODEL FILES (all backed up in S3)"
echo "───────────────────────────────────────────────────────────────────────────────"
delete_item "models/embedding/pytorch_weights/model.tar.gz" "413 MB"
delete_item "models/embedding/pokemon_vit_embedding.onnx" "328 MB"
delete_item "models/embedding/pytorch_weights/student_stage2_checkpoint.pt" "298 MB"
delete_item "models/embedding/pytorch_weights/student_stage2.pt" "75 MB"
delete_item "models/embedding/pytorch_weights" "directory"
delete_item "models/onnx/pokemon_student_convnext_tiny.onnx" "111 MB"
delete_item "models/onnx/pokemon_student_convnext_tiny_opset13.onnx" "111 MB"
delete_item "models/embedding/levit384.hef" "41 MB"

# Summary
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "SUMMARY"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN completed. To actually delete files, run:"
    echo "  bash scripts/cleanup_local_storage.sh --execute"
else
    echo "✅ Cleanup complete!"
    echo ""
    echo "Space freed: ~36 GB"
    echo ""
    echo "Remaining local files:"
    echo "  ✓ data/reference/ (128 MB) - production database"
    echo "  ✓ models/embedding/pokemon_student_efficientnet_lite0_stage2.hef (14 MB)"
    echo "  ✓ Scripts and documentation"
    echo ""
    echo "All deleted files can be re-downloaded from S3 if needed:"
    echo "  aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/ ./"
fi

echo ""
