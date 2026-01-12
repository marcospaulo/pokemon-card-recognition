#!/usr/bin/env python3
"""
Re-save teacher model with complete configuration for self-contained storage.
Makes the model truly independent - no HuggingFace downloads needed.
"""

import sys
import os
import boto3
import tarfile
import tempfile
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'training'))

def download_teacher_checkpoint():
    """Download teacher checkpoint from S3."""
    s3 = boto3.client('s3')
    bucket = 'pokemon-card-training-us-east-2'
    key = 'models/embedding/teacher/pokemon-card-dinov3-teacher-2026-01-10-13-31-34-937/output/model.tar.gz'

    print(f"Downloading teacher checkpoint from S3...")
    print(f"  s3://{bucket}/{key}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download tar.gz
        tar_path = os.path.join(tmpdir, 'model.tar.gz')
        s3.download_file(bucket, key, tar_path)
        print(f"✅ Downloaded {os.path.getsize(tar_path):,} bytes")

        # Extract
        extract_dir = os.path.join(tmpdir, 'extracted')
        os.makedirs(extract_dir)
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_dir, filter='data')
        print("✅ Extracted archive")

        # Load checkpoint
        checkpoint_path = os.path.join(extract_dir, 'phase2_checkpoint.pt')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"phase2_checkpoint.pt not found in archive")

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"✅ Loaded checkpoint with keys: {list(checkpoint.keys())}")

        return checkpoint


def save_complete_model(checkpoint, output_path):
    """Save model with complete configuration."""

    print("\nCreating self-contained model package...")

    # Create complete model config
    complete_model = {
        # Model weights (your fine-tuned weights)
        'model_state_dict': checkpoint['model'],

        # Model architecture config (eliminates HF dependency)
        'model_config': {
            'model_id': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
            'model_type': 'dinov3_vit',
            'embedding_dim': 768,
            'from_checkpoint': False,  # Indicates it was trained with from_pretrained()

            # DINOv3-ViT-Large architecture
            'backbone_config': {
                'hidden_size': 1024,
                'num_hidden_layers': 24,
                'num_attention_heads': 16,
                'intermediate_size': 4096,
                'hidden_act': 'gelu',
                'image_size': 224,
                'patch_size': 16,
                'num_channels': 3,
                'attention_dropout': 0.0,
                'initializer_range': 0.02,
                'layer_norm_eps': 1e-05,
                'layerscale_value': 1.0,
                'drop_path_rate': 0.0,
                'use_gated_mlp': False,
                'rope_theta': 100.0,
                'query_bias': True,
                'key_bias': False,
                'value_bias': True,
                'proj_bias': True,
                'mlp_bias': True,
                'num_register_tokens': 4,
                'pos_embed_rescale': 2.0,
            }
        },

        # ArcFace classifier
        'arcface_state_dict': checkpoint.get('loss_fn', None),
        'num_classes': checkpoint.get('num_classes', 17592),

        # Training metadata
        'training_info': {
            'epoch': checkpoint.get('epoch', None),
            'step': checkpoint.get('step', None),
            'metrics': checkpoint.get('metrics', None),
        },

        # EMA model (if available)
        'ema_model_state_dict': checkpoint.get('ema_model', None),

        # Optimizer state (for resuming training)
        'optimizer_state_dict': checkpoint.get('optimizer', None),

        # Version info
        'version': '1.0',
        'save_date': '2026-01-11',
        'description': 'Self-contained DINOv3-ViT-Large teacher model for Pokemon card recognition',
    }

    # Save
    print(f"Saving to: {output_path}")
    torch.save(complete_model, output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"✅ Saved complete model ({size_mb:.1f} MB)")

    return complete_model


def upload_to_s3(local_path, bucket, key):
    """Upload complete model to S3."""
    s3 = boto3.client('s3')

    print(f"\nUploading to S3...")
    print(f"  s3://{bucket}/{key}")

    s3.upload_file(local_path, bucket, key)
    print(f"✅ Uploaded to S3")

    return f"s3://{bucket}/{key}"


def main():
    print("="*70)
    print(" " * 15 + "Teacher Model Re-packaging")
    print("="*70)

    # Download original checkpoint
    checkpoint = download_teacher_checkpoint()

    # Save locally with complete config
    output_path = 'models/embedding/dinov3_teacher_complete.pt'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    complete_model = save_complete_model(checkpoint, output_path)

    # Upload to S3
    bucket = 'pokemon-card-training-us-east-2'
    s3_key = 'models/embedding/teacher/dinov3_teacher_complete.pt'
    s3_path = upload_to_s3(output_path, bucket, s3_key)

    print("\n" + "="*70)
    print(" " * 20 + "✅ COMPLETE!")
    print("="*70)
    print(f"Local path:  {output_path}")
    print(f"S3 path:     {s3_path}")
    print(f"\nYour teacher model is now self-contained:")
    print("  ✅ Model weights (fine-tuned)")
    print("  ✅ Architecture config (no HF needed)")
    print("  ✅ ArcFace classifier")
    print("  ✅ Training metadata")
    print("\nTo load in future:")
    print("  checkpoint = torch.load('dinov3_teacher_complete.pt')")
    print("  model.load_state_dict(checkpoint['model_state_dict'])")
    print("="*70)


if __name__ == '__main__':
    main()
