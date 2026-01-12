#!/usr/bin/env python3
"""
Launch Stage 2 Fine-Tuning on SageMaker with 8xA100 GPUs

Continues from Stage 1 distilled model with task-specific fine-tuning:
- Loads Stage 1 EfficientNet-Lite0 distilled weights
- Fine-tunes with ArcFace on 17,592 Pokemon card classes
- 20 epochs with lower learning rate
"""

import sagemaker
from sagemaker.pytorch import PyTorch
import boto3
import os
from pathlib import Path

session = sagemaker.Session()

# Get account ID and use SageMaker execution role
sts = boto3.client('sts')
account_id = sts.get_caller_identity()['Account']
role = f'arn:aws:iam::{account_id}:role/SageMaker-ExecutionRole'

# Load HuggingFace token for gated models (DINOv3)
hf_token = os.environ.get('HUGGING_FACE_HUB_TOKEN')
if not hf_token:
    hf_token_file = Path.home() / '.huggingface' / 'token'
    if hf_token_file.exists():
        hf_token = hf_token_file.read_text().strip()
        print(f"✅ Loaded HuggingFace token from {hf_token_file}")
    else:
        raise ValueError("HuggingFace token not found. Set HUGGING_FACE_HUB_TOKEN or login with `huggingface-cli login`")

# Use existing bucket
bucket = 'pokemon-card-training-us-east-2'

print(f"Using S3 bucket: {bucket}")
print(f"Using role: {role}")
print(f"Region: {session.boto_region_name}")

# Stage 1 model path (completed training)
STAGE1_MODEL_PATH = 's3://pokemon-card-training-us-east-2/models/embedding/student/pytorch-training-2026-01-11-22-26-25-713/output/model.tar.gz'

# ========== STAGE 2: Task-Specific Fine-Tuning ==========
print("\n" + "="*70)
print(" " * 15 + "STAGE 2: Task-Specific Fine-Tuning")
print("="*70)

stage2_estimator = PyTorch(
    entry_point='train_student_distillation.py',
    source_dir='src/training',
    role=role,

    # Same instance configuration as Stage 1
    instance_count=1,
    instance_type='ml.p4d.24xlarge',  # 8x A100 80GB
    framework_version='2.8.0',
    py_version='py312',

    # HuggingFace authentication
    environment={
        'HUGGING_FACE_HUB_TOKEN': hf_token,
    },

    hyperparameters={
        'student-model': 'efficientnet_lite0',
        'teacher-model': 'dinov3_vitl16',
        'embedding-dim': 768,
        'stage': 'stage2',

        # Stage 2 training
        'epochs-stage2': 20,
        'lr-stage2': 1e-5,  # Lower learning rate for fine-tuning

        'batch-size': 512,  # 64 per GPU

        # Loss weights (same as Stage 1)
        'alpha-feature': 0.35,
        'alpha-kl': 0.25,
        'alpha-attention': 0.25,
        'alpha-highfreq': 0.15,
    },

    # Outputs
    output_path=f's3://{bucket}/models/embedding/student/',
    checkpoint_s3_uri=f's3://{bucket}/checkpoints/student/stage2/',

    # Download data to local disk
    input_mode='File',

    # Distributed training for 8 GPUs
    distribution={
        'torch_distributed': {
            'enabled': True
        }
    },

    use_spot_instances=False,
    max_run=1800,  # 30 minutes max
)

print(f"Instance: ml.p4d.24xlarge (8x A100 80GB)")
print(f"Batch size: 512 (64 per GPU)")
print(f"Epochs: 20")
print(f"Learning rate: 1e-5")
print(f"Stage 1 model: {STAGE1_MODEL_PATH}")
print(f"\n✅ TensorBoard enabled (/opt/ml/output/tensorboard/stage2)")
print(f"✅ MLflow enabled (experiment tracking)")
print("="*70 + "\n")

# Train Stage 2
print("🚀 Starting Stage 2 training...")
stage2_estimator.fit({
    'train': f's3://{bucket}/classification_dataset/train/',
    'teacher': 's3://pokemon-card-training-us-east-2/models/embedding/teacher/pokemon-card-dinov3-teacher-2026-01-10-13-31-34-937/output/model.tar.gz',
    'stage1': STAGE1_MODEL_PATH,  # Load Stage 1 model
})

print("\n" + "="*70)
print(" " * 20 + "🎉 STAGE 2 COMPLETE!")
print("="*70)
print(f"Final model: s3://{bucket}/models/embedding/student/")
print(f"Stage 2 checkpoints: s3://{bucket}/checkpoints/student/stage2/")
print("\nModel: EfficientNet-Lite0 (4.7M parameters)")
print("Ready for ONNX export and Hailo compilation!")
print("="*70)
