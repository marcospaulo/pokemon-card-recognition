#!/usr/bin/env python3
"""
Export the distilled EfficientNet-Lite0 student model to ONNX format.

This prepares the model for Hailo compilation on EC2.
EfficientNet-Lite0 uses BatchNorm (not LayerNorm) - fully compatible with Hailo-8L.
"""

import torch
import torch.nn as nn
import timm
import boto3
import tarfile
import os
from pathlib import Path
import argparse


class StudentModel(nn.Module):
    """EfficientNet-Lite0 student model for Pokemon card recognition"""

    def __init__(self, model_name='efficientnet_lite0', embedding_dim=768, num_classes=17592):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        backbone_dim = self.backbone.num_features  # 1280 for efficientnet_lite0

        # Projection head (matches training architecture exactly)
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(backbone_dim, embedding_dim),
        )

        # Classification head (not used in inference, but needed for loading checkpoint)
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

    def forward(self, x):
        """Forward pass - returns normalized embeddings"""
        features = self.backbone(x)
        embeddings = self.projection(features)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


def download_model_from_s3(s3_path, local_dir):
    """Download and extract model from S3"""
    print(f"Downloading model from {s3_path}...")

    s3 = boto3.client('s3')
    bucket = s3_path.split('/')[2]
    key = '/'.join(s3_path.split('/')[3:])

    local_path = os.path.join(local_dir, 'model.tar.gz')
    s3.download_file(bucket, key, local_path)

    print(f"Extracting to {local_dir}...")
    with tarfile.open(local_path, 'r:gz') as tar:
        tar.extractall(local_dir)

    # Find the model weights file
    model_files = ['student_stage2.pt', 'student_stage2_final.pt']
    for f in model_files:
        model_path = os.path.join(local_dir, f)
        if os.path.exists(model_path):
            print(f"✅ Found model: {f}")
            return model_path

    raise FileNotFoundError(f"No model weights found in {local_dir}")


def export_to_onnx(model, onnx_path, input_shape=(1, 3, 224, 224)):
    """Export PyTorch model to ONNX format"""
    print(f"\nExporting to ONNX: {onnx_path}")

    model.eval()
    dummy_input = torch.randn(*input_shape)

    # Export with optimization
    # NOTE: Hailo SDK requires fixed dimensions (no dynamic axes)
    # Using opset 13 for better LayerNorm support
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,  # Opset 13 has better LayerNorm support
        do_constant_folding=True,
        input_names=['input'],
        output_names=['embeddings'],
        # No dynamic_axes for Hailo - use fixed batch size
    )

    file_size_mb = os.path.getsize(onnx_path) / (1024**2)
    print(f"✅ ONNX model saved: {file_size_mb:.1f} MB")

    return onnx_path


def verify_onnx(onnx_path):
    """Verify ONNX model can be loaded and run"""
    try:
        import onnxruntime as ort

        print("\nVerifying ONNX model...")
        session = ort.InferenceSession(onnx_path)

        # Check inputs/outputs
        print(f"Inputs: {[i.name for i in session.get_inputs()]}")
        print(f"Outputs: {[o.name for o in session.get_outputs()]}")

        # Test inference
        dummy_input = torch.randn(1, 3, 224, 224).numpy()
        outputs = session.run(None, {'input': dummy_input})

        print(f"Output shape: {outputs[0].shape}")
        print("✅ ONNX model verified!")

    except ImportError:
        print("⚠️ onnxruntime not installed - skipping verification")
        print("   Install with: pip install onnxruntime")


def main():
    parser = argparse.ArgumentParser(description='Export student model to ONNX')
    parser.add_argument('--s3-model', type=str,
                       default='s3://pokemon-card-training-us-east-2/models/embedding/student/pytorch-training-2026-01-11-23-31-10-757/output/model.tar.gz',
                       help='S3 path to model.tar.gz (Stage 2 final model)')
    parser.add_argument('--output-dir', type=str,
                       default='./models/onnx',
                       help='Output directory for ONNX model')
    parser.add_argument('--output-name', type=str,
                       default='pokemon_student_efficientnet_lite0.onnx',
                       help='Output ONNX filename')
    args = parser.parse_args()

    # Create directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = Path('/tmp/student_model_export')
    work_dir.mkdir(parents=True, exist_ok=True)

    # Download model
    model_path = download_model_from_s3(args.s3_model, str(work_dir))

    # Load model
    print("\nLoading PyTorch model...")
    model = StudentModel(
        model_name='efficientnet_lite0',
        embedding_dim=768,
        num_classes=17592
    )

    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)

    # Strip _orig_mod. prefix if present
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        print("Stripped _orig_mod. prefix from state dict")

    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded: {total_params:,} parameters ({total_params/1e6:.1f}M)")

    # Export to ONNX
    onnx_path = output_dir / args.output_name
    export_to_onnx(model, str(onnx_path))

    # Verify
    verify_onnx(str(onnx_path))

    print(f"\n{'='*60}")
    print("✅ EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"ONNX Model: {onnx_path}")
    print(f"Next step: Compile to Hailo HEF on EC2")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
