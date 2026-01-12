#!/bin/bash
# Upload all model files to the organized SageMaker project structure

PROJECT_PREFIX="project/pokemon-card-recognition"
BUCKET="pokemon-card-training-us-east-2"

echo "=============================================================="
echo "Uploading Models to Unified SageMaker Project"
echo "=============================================================="
echo ""
echo "Project: $PROJECT_PREFIX"
echo "Bucket: $BUCKET"
echo ""

# 1. Teacher model (use S3 copy for large file with multipart)
echo "[1/4] Copying teacher model (5.3 GB - may take a few minutes)..."
aws s3 cp \
  s3://$BUCKET/models/embedding/teacher/pokemon-card-dinov3-teacher-2026-01-10-13-31-34-937/output/model.tar.gz \
  s3://$BUCKET/$PROJECT_PREFIX/models/dinov3-teacher/v1.0/model.tar.gz \
  --only-show-errors

if [ $? -eq 0 ]; then
    echo "✓ Teacher model copied"
else
    echo "⚠ Teacher model copy failed"
fi
echo ""

# 2. Student Stage 2 PyTorch (upload from local)
echo "[2/4] Uploading student Stage 2 PyTorch checkpoint..."
if [ -f "models/embedding/pytorch_weights/student_stage2_final.pt" ]; then
    aws s3 cp \
      models/embedding/pytorch_weights/student_stage2_final.pt \
      s3://$BUCKET/$PROJECT_PREFIX/models/efficientnet-student/stage2/v2.0/student_stage2_final.pt \
      --only-show-errors

    if [ $? -eq 0 ]; then
        echo "✓ Student PyTorch model uploaded"
    else
        echo "⚠ Student PyTorch upload failed"
    fi
else
    echo "⚠ Local file not found: models/embedding/pytorch_weights/student_stage2_final.pt"
fi
echo ""

# 3. Student Stage 2 ONNX (upload from local)
echo "[3/4] Uploading student Stage 2 ONNX model..."
if [ -f "models/onnx/pokemon_student_stage2_final.onnx" ]; then
    aws s3 cp \
      models/onnx/pokemon_student_stage2_final.onnx \
      s3://$BUCKET/$PROJECT_PREFIX/models/efficientnet-student/stage2/v2.0/student_stage2_final.onnx \
      --only-show-errors

    if [ $? -eq 0 ]; then
        echo "✓ Student ONNX model uploaded"
    else
        echo "⚠ Student ONNX upload failed"
    fi
else
    echo "⚠ Local file not found: models/onnx/pokemon_student_stage2_final.onnx"
fi
echo ""

# 4. Hailo HEF (upload from local)
echo "[4/4] Uploading Hailo optimized model (HEF)..."
if [ -f "models/embedding/pokemon_student_efficientnet_lite0_stage2.hef" ]; then
    aws s3 cp \
      models/embedding/pokemon_student_efficientnet_lite0_stage2.hef \
      s3://$BUCKET/$PROJECT_PREFIX/models/efficientnet-hailo/v2.1/pokemon_student_efficientnet_lite0_stage2.hef \
      --only-show-errors

    if [ $? -eq 0 ]; then
        echo "✓ Hailo HEF model uploaded"
    else
        echo "⚠ Hailo HEF upload failed"
    fi
else
    echo "⚠ Local file not found: models/embedding/pokemon_student_efficientnet_lite0_stage2.hef"
fi
echo ""

echo "=============================================================="
echo "Upload Summary"
echo "=============================================================="
echo ""
echo "Verifying uploaded files..."
aws s3 ls s3://$BUCKET/$PROJECT_PREFIX/models/ --recursive --human-readable | grep -E '\.(tar\.gz|pt|onnx|hef)$'
echo ""
echo "✓ Model upload process complete!"
echo ""
echo "Next steps:"
echo "  1. Register models to Model Registry"
echo "  2. Verify organization with: aws s3 ls s3://$BUCKET/$PROJECT_PREFIX/ --recursive"
