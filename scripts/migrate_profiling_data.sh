#!/bin/bash
# Migrate profiling data from training jobs to organized project structure

BUCKET="pokemon-card-training-us-east-2"
PROJECT_PREFIX="project/pokemon-card-recognition"

echo "=============================================================="
echo "SageMaker Profiling Data Migration"
echo "=============================================================="
echo ""
echo "Source: Training job outputs"
echo "Destination: $PROJECT_PREFIX/profiling/"
echo ""

# Teacher profiling data
echo "[1/2] Migrating teacher profiling data..."
echo "  Source: pokemon-card-dinov3-teacher-2026-01-10-13-31-34-937"
echo "  Destination: profiling/teacher/2026-01-10/"

aws s3 sync \
  s3://$BUCKET/models/embedding/teacher/pokemon-card-dinov3-teacher-2026-01-10-13-31-34-937/profiler-output/ \
  s3://$BUCKET/$PROJECT_PREFIX/profiling/teacher/2026-01-10/ \
  --only-show-errors

if [ $? -eq 0 ]; then
    # Get size of migrated data
    SIZE=$(aws s3 ls s3://$BUCKET/$PROJECT_PREFIX/profiling/teacher/2026-01-10/ --recursive --summarize | grep "Total Size" | awk '{print $3}')
    echo "  ✓ Teacher profiling data migrated ($SIZE bytes)"
else
    echo "  ✗ Teacher profiling migration failed"
fi
echo ""

# Student Stage 2 profiling data (latest successful training job)
echo "[2/2] Migrating student Stage 2 profiling data..."
echo "  Source: pytorch-training-2026-01-11-23-31-10-757"
echo "  Destination: profiling/student_stage2/2026-01-11/"

aws s3 sync \
  s3://$BUCKET/models/embedding/student/pytorch-training-2026-01-11-23-31-10-757/profiler-output/ \
  s3://$BUCKET/$PROJECT_PREFIX/profiling/student_stage2/2026-01-11/ \
  --only-show-errors

if [ $? -eq 0 ]; then
    SIZE=$(aws s3 ls s3://$BUCKET/$PROJECT_PREFIX/profiling/student_stage2/2026-01-11/ --recursive --summarize | grep "Total Size" | awk '{print $3}')
    echo "  ✓ Student Stage 2 profiling data migrated ($SIZE bytes)"
else
    echo "  ✗ Student profiling migration failed"
fi
echo ""

echo "=============================================================="
echo "Migration Summary"
echo "=============================================================="
echo ""
echo "Verifying migrated files..."
aws s3 ls s3://$BUCKET/$PROJECT_PREFIX/profiling/ --recursive --human-readable | grep -E '\.(json|ts)$' | tail -20
echo ""
echo "✓ Profiling data migration complete!"
echo ""
echo "Next steps:"
echo "  1. View profiling data: aws s3 ls s3://$BUCKET/$PROJECT_PREFIX/profiling/ --recursive"
echo "  2. Generate analytics from profiling metrics"
echo "  3. Set up CloudWatch dashboard"
