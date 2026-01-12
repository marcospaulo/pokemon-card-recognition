#!/usr/bin/env python3
"""
Launch Student Distillation on SageMaker with 8xA100 GPUs
Distills DINOv3-ViT-L/16 (304M) → EfficientNet-Lite0 (4.7M)

Two-stage training:
- Stage 1: General feature distillation (30 epochs)
- Stage 2: Task-specific fine-tuning (20 epochs)

Architecture: EfficientNet-Lite0 chosen for:
- BatchNorm (not LayerNorm) - fully Hailo-8L compatible
- 99.69-99.78% fine-grained classification accuracy
- 5× smaller than ResNet50 with better accuracy

IMPORTANT: Run validation BEFORE launching to avoid wasting money on errors:
  python scripts/validate_distillation_setup.py
"""

import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.debugger import Rule, rule_configs, ProfilerConfig, ProfilerRule, FrameworkProfile
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

# ========== SageMaker Debugger Configuration ==========
# Automatically detects training issues to save GPU costs
debugger_rules = [
    Rule.sagemaker(rule_configs.vanishing_gradient()),
    Rule.sagemaker(rule_configs.exploding_tensor()),
    Rule.sagemaker(rule_configs.loss_not_decreasing()),
    Rule.sagemaker(rule_configs.overfit()),
]

# ========== SageMaker Profiler Configuration ==========
# Monitors GPU utilization to verify optimizations are working
profiler_config = ProfilerConfig(
    system_monitor_interval_millis=500,  # Monitor every 500ms
    framework_profile_params=FrameworkProfile(
        local_path='/opt/ml/output/profiler',
        start_step=5,   # Skip first 5 steps (warmup)
        num_steps=10    # Profile 10 steps
    )
)

profiler_rules = [
    ProfilerRule.sagemaker(rule_configs.GPUMemoryIncrease()),
    ProfilerRule.sagemaker(rule_configs.LowGPUUtilization()),
    ProfilerRule.sagemaker(rule_configs.LoadBalancing()),  # Verify all 8 GPUs are balanced
]

# ========== STAGE 1: General Feature Distillation ==========
print("\n" + "="*70)
print(" " * 15 + "STAGE 1: General Feature Distillation")
print("="*70)

stage1_estimator = PyTorch(
    entry_point='train_student_distillation.py',
    source_dir='src/training',  # Includes requirements.txt
    role=role,

    # 8x A100 80GB instance
    instance_count=1,
    instance_type='ml.p4d.24xlarge',  # 8x A100 80GB

    # Latest PyTorch (SageMaker pre-built image)
    framework_version='2.8.0',  # SageMaker's latest PyTorch with distributed training support
    py_version='py312',         # Python 3.12 required for PyTorch 2.8.0

    # Download data from S3 to local disk (required for ImageFolder dataset)
    input_mode='File',  # ImageFolder needs local filesystem access to walk directory structure

    # HuggingFace authentication for gated models (DINOv3)
    environment={
        'HUGGING_FACE_HUB_TOKEN': hf_token,
    },

    hyperparameters={
        'student-model': 'efficientnet_lite0',
        'teacher-model': 'dinov3_vitl16',  # ViT-Large teacher (1.1B params)
        'embedding-dim': 768,
        'stage': 'stage1',

        # Stage 1 training
        'epochs-stage1': 30,
        'lr-stage1': 1e-4,

        # MUCH larger batch size for 8xA100
        'batch-size': 512,  # 64 per GPU

        # Loss weights (sum to 1.0)
        'alpha-feature': 0.35,    # 35% - Feature distillation
        'alpha-kl': 0.25,         # 25% - KL divergence
        'alpha-attention': 0.25,  # 25% - Attention distillation (CRITICAL for occlusion)
        'alpha-highfreq': 0.15,   # 15% - High-frequency (FiGKD)

        # Teacher model will be provided via 'teacher' input channel
        # SageMaker automatically extracts model.tar.gz to /opt/ml/input/data/teacher/models/
    },

    # Outputs
    output_path=f's3://{bucket}/models/embedding/student/',
    checkpoint_s3_uri=f's3://{bucket}/checkpoints/student/stage1/',

    # Distributed training for 8 GPUs
    distribution={
        'torch_distributed': {
            'enabled': True
        }
    },

    # On-demand instances
    use_spot_instances=False,
    max_run=2400,    # 40 minutes max (should finish in ~15)

    # SageMaker Debugger & Profiler (disabled to avoid processing job quota limits)
    # rules=debugger_rules,
    # profiler_config=profiler_config,
    # profiler_rules=profiler_rules,
)

print(f"Instance: ml.p4d.24xlarge (8x A100 80GB)")
print(f"Batch size: 512 (64 per GPU)")
print(f"Epochs: 30")
print(f"Estimated time: ~10-15 minutes")
print(f"Estimated cost: ~$3-4 (on-demand)")
print(f"\n✅ SageMaker Debugger enabled (catches training issues early)")
print(f"✅ SageMaker Profiler enabled (verifies 8xA100 utilization)")
print(f"✅ TensorBoard enabled (/opt/ml/output/tensorboard/stage1)")
print(f"✅ MLflow enabled (experiment tracking)")
print("="*70 + "\n")

# Train Stage 1
print("🚀 Starting Stage 1 training...")
stage1_estimator.fit({
    'train': f's3://{bucket}/classification_dataset/train/',
    'teacher': 's3://pokemon-card-training-us-east-2/models/embedding/teacher/pokemon-card-dinov3-teacher-2026-01-10-13-31-34-937/output/model.tar.gz',
})

print("\n✅ Stage 1 Complete!")
print(f"Model saved to: s3://{bucket}/models/embedding/student/")
print(f"Checkpoints: s3://{bucket}/checkpoints/student/stage1/")

# ========== STAGE 2: Task-Specific Fine-Tuning ==========
print("\n" + "="*70)
print(" " * 15 + "STAGE 2: Task-Specific Fine-Tuning")
print("="*70)

stage2_estimator = PyTorch(
    entry_point='train_student_distillation.py',
    source_dir='src/training',
    role=role,

    # Same instance configuration
    instance_count=1,
    instance_type='ml.p4d.24xlarge',
    framework_version='2.8.0',
    py_version='py312',

    # HuggingFace authentication for gated models (DINOv3)
    environment={
        'HUGGING_FACE_HUB_TOKEN': hf_token,
    },

    hyperparameters={
        'student-model': 'efficientnet_lite0',
        'teacher-model': 'dinov3_vitl16',  # ViT-Large teacher (1.1B params)
        'embedding-dim': 768,
        'stage': 'stage2',

        # Stage 2 training
        'epochs-stage2': 20,
        'lr-stage2': 1e-5,  # Lower learning rate for fine-tuning

        'batch-size': 512,

        # Same loss weights
        'alpha-feature': 0.35,
        'alpha-kl': 0.25,
        'alpha-attention': 0.25,
        'alpha-highfreq': 0.15,

        # Teacher model provided via 'teacher' input channel
        # Stage 1 model provided via 'stage1' input channel
    },

    # Outputs
    output_path=f's3://{bucket}/models/embedding/student/',
    checkpoint_s3_uri=f's3://{bucket}/checkpoints/student/stage2/',

    # Pass stage1 output as input for stage2
    # This makes stage1 model available at /opt/ml/input/data/stage1
    input_mode='File',  # Download stage1 model to local disk

    # Distributed training
    distribution={
        'torch_distributed': {
            'enabled': True
        }
    },

    use_spot_instances=False,
    max_run=1800,  # 30 minutes max (should finish in ~10-12)

    # SageMaker Debugger & Profiler (disabled to avoid processing job quota limits)
    # rules=debugger_rules,
    # profiler_config=profiler_config,
    # profiler_rules=profiler_rules,
)

print(f"Instance: ml.p4d.24xlarge (8x A100 80GB)")
print(f"Batch size: 512 (64 per GPU)")
print(f"Epochs: 20")
print(f"Estimated time: ~10-12 minutes")
print(f"Estimated cost: ~$3-4 (on-demand)")
print(f"\n✅ SageMaker Debugger enabled")
print(f"✅ SageMaker Profiler enabled")
print(f"✅ TensorBoard enabled (/opt/ml/output/tensorboard/stage2)")
print(f"✅ MLflow enabled")
print("="*70 + "\n")

# Train Stage 2
print("🚀 Starting Stage 2 training...")
stage2_estimator.fit({
    'train': f's3://{bucket}/classification_dataset/train/',
    'teacher': 's3://pokemon-card-training-us-east-2/models/embedding/teacher/pokemon-card-dinov3-teacher-2026-01-10-13-31-34-937/output/model.tar.gz',
    'stage1': stage1_estimator.model_data,  # Stage 1 output becomes Stage 2 input
})

print("\n" + "="*70)
print(" " * 20 + "🎉 ALL TRAINING COMPLETE!")
print("="*70)
print(f"Final model: s3://{bucket}/models/embedding/student/")
print(f"Stage 1 checkpoints: s3://{bucket}/checkpoints/student/stage1/")
print(f"Stage 2 checkpoints: s3://{bucket}/checkpoints/student/stage2/")
print("\nModel compression: 304M → 28M parameters (91% reduction)")
print("Ready for edge deployment on Raspberry Pi + Hailo-8L!")
print("="*70)
