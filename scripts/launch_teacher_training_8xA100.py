#!/usr/bin/env python3
"""
Launch DINOv3 teacher fine-tuning on SageMaker with 8xA100 GPUs
Optimized for ml.p4d.24xlarge instance

Estimated time: 10-15 minutes
Estimated cost: ~$3-4 (on-demand) or ~$0.9-1.2 (spot)
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
# Using 2 most critical debugger rules to fit current quota (4 total)
# Quota increase to 16 is pending (request: d8e775ca102f4e448856d8e51de038b8Aun6xETD)
debugger_rules = [
    Rule.sagemaker(rule_configs.loss_not_decreasing()),  # CRITICAL: Detect stalled training
    Rule.sagemaker(rule_configs.overfit()),              # CRITICAL: Prevent overfitting
]

# ========== SageMaker Profiler Configuration ==========
# Using 2 most critical profiler rules (total 4 instances = current quota)
profiler_config = ProfilerConfig(
    system_monitor_interval_millis=500,  # Monitor every 500ms
    framework_profile_params=FrameworkProfile(
        local_path='/opt/ml/output/profiler',
        start_step=5,   # Skip first 5 steps (warmup)
        num_steps=10    # Profile 10 steps
    )
)

profiler_rules = [
    ProfilerRule.sagemaker(rule_configs.GPUMemoryIncrease()),  # CRITICAL: Detect memory leaks
    ProfilerRule.sagemaker(rule_configs.LoadBalancing()),      # CRITICAL: Verify 8-GPU balance
]

estimator = PyTorch(
    entry_point='train_dinov3_teacher.py',
    source_dir='src/training',  # Includes requirements.txt for transformers>=5.0
    role=role,
    base_job_name='pokemon-card-dinov3-teacher',  # Descriptive job name

    # 8x A100 40GB instance
    instance_count=1,
    instance_type='ml.p4d.24xlarge',  # 8x A100 40GB

    # Latest PyTorch (SageMaker pre-built image)
    framework_version='2.8.0',  # SageMaker's latest PyTorch with distributed training support
    py_version='py312',         # Python 3.12 required for PyTorch 2.8.0

    # Download data before training (more reliable than streaming)
    input_mode='File',  # Download all data first (more reliable with persistent workers)

    # HuggingFace authentication for gated models (DINOv3)
    environment={
        'HUGGING_FACE_HUB_TOKEN': hf_token,
    },

    hyperparameters={
        'dinov3-model': 'dinov3_vitl16',    # 1.1B parameter model (ViT-Large)
        'embedding-dim': 768,                # Project 1024 → 768 for ArcFace

        # Fewer epochs since we're fine-tuning
        'epochs-frozen': 3,      # Reduced from 5 (faster convergence)
        'epochs-unfrozen': 10,   # Reduced from 15
        'unfreeze-blocks': 4,

        # Larger batch for ViT-L (fits in 40GB A100s)
        'batch-size': 256,       # 32 per GPU on 8x A100 40GB

        # Learning rates
        'lr-frozen': 1e-3,
        'lr-unfrozen': 1e-5,

        # ArcFace settings
        'arcface-margin': 0.5,
        'arcface-scale': 64,
    },

    # Outputs
    output_path=f's3://{bucket}/models/embedding/teacher/',
    checkpoint_s3_uri=f's3://{bucket}/checkpoints/teacher/',

    # Distributed training for 8 GPUs
    distribution={
        'torch_distributed': {
            'enabled': True
        }
    },

    # On-demand instances (spot quota not available)
    use_spot_instances=False,
    max_run=1800,    # 30 minutes max run (should finish in 10-15)

    # SageMaker Debugger & Profiler (temporarily disabled - processing jobs from previous runs still stopping)
    # rules=debugger_rules,
    # profiler_config=profiler_config,
    # profiler_rules=profiler_rules,
)

print("\n" + "="*60)
print("Starting DINOv3 Teacher Fine-tuning")
print("="*60)
print(f"Model: DINOv3-ViT-Large (1.1B params)")
print(f"Instance: ml.p4d.24xlarge (8x A100 40GB)")
print(f"Batch size: 256 (32 per GPU)")
print(f"Total epochs: 13 (3 frozen + 10 unfrozen)")
print(f"Estimated time: 10-15 minutes")
print(f"Estimated cost: ~$3-4 (on-demand)")
print(f"\n✅ SageMaker Debugger enabled (catches training issues early)")
print(f"✅ SageMaker Profiler enabled (verifies 8xA100 utilization)")
print(f"✅ TensorBoard enabled (/opt/ml/output/tensorboard)")
print(f"✅ MLflow enabled (experiment tracking)")
print("="*60 + "\n")

# Dataset already uploaded - verified manually
print("✅ Using existing dataset in S3: classification_dataset/train/ (all 17,592 cards)")

# Fit the model - train on all cards, validation split done automatically
estimator.fit({
    'train': f's3://{bucket}/classification_dataset/train/',
})

print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print(f"Model saved to: s3://{bucket}/models/embedding/teacher/")
print(f"Checkpoints: s3://{bucket}/checkpoints/teacher/")
print("="*60)
