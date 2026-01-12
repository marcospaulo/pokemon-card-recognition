#!/usr/bin/env python3
"""
Export YOLO11-OBB to ONNX format compatible with Sony IMX500

The ONNX model can then be compiled to RPK on the Raspberry Pi using:
    sudo imx500-convert-model --input yolo11n-obb.onnx --output yolo11n-obb.rpk
"""

import torch
from ultralytics import YOLO
import argparse
from pathlib import Path

def export_yolo_to_imx500_onnx(
    model_path: str,
    output_path: str,
    imgsz: int = 640,
    simplify: bool = True
):
    """
    Export YOLO model to IMX500-compatible ONNX format

    Args:
        model_path: Path to YOLO .pt model
        output_path: Output .onnx file path
        imgsz: Input image size (IMX500 works best with 416 or 640)
        simplify: Use onnx-simplifier to optimize the graph
    """

    print("=" * 80)
    print("YOLO11 → IMX500 ONNX Export")
    print("=" * 80)
    print()

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Model loaded successfully")
    print(f"  Task: {model.task}")
    if hasattr(model, 'names'):
        print(f"  Classes: {len(model.names)}")
    print()

    print(f"Exporting to ONNX (imgsz={imgsz})...")
    print(f"  Output: {output_path}")
    print()

    # Export with IMX500-specific settings
    success = model.export(
        format='onnx',
        imgsz=imgsz,
        simplify=simplify,
        opset=11,  # IMX500 supports ONNX opset 11
        dynamic=False,  # IMX500 requires fixed input size
    )

    # Move exported file to desired location
    export_path = Path(model_path).with_suffix('.onnx')
    if export_path.exists() and export_path != Path(output_path):
        import shutil
        shutil.move(str(export_path), output_path)

    if Path(output_path).exists():
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"\n✅ Export successful!")
        print(f"   File: {output_path}")
        print(f"   Size: {size_mb:.1f} MB")
        print()

        print("=" * 80)
        print("Next Steps: Compile to RPK on Raspberry Pi")
        print("=" * 80)
        print()
        print("1. Copy ONNX to Raspberry Pi:")
        print(f"   scp {output_path} pi@raspberrypi:~/models/")
        print()
        print("2. On Raspberry Pi, install IMX500 SDK (if not already):")
        print("   sudo apt update")
        print("   sudo apt install imx500-all")
        print()
        print("3. Convert ONNX to RPK:")
        onnx_name = Path(output_path).name
        rpk_name = Path(output_path).stem + '.rpk'
        print(f"   sudo imx500-convert-model \\")
        print(f"       --input ~/models/{onnx_name} \\")
        print(f"       --output ~/models/{rpk_name}")
        print()
        print("4. Verify RPK file:")
        print(f"   ls -lh ~/models/{rpk_name}")
        print()
        print("5. Test with camera:")
        print("   python3 -c \"from picamera2.devices import IMX500; ")
        print(f"imx500 = IMX500('~/models/{rpk_name}'); print('✅ Model loaded')\"")
        print()

        return True
    else:
        print(f"\n❌ Export failed - {output_path} not found")
        return False


def main():
    parser = argparse.ArgumentParser(description='Export YOLO11 to IMX500 ONNX format')
    parser.add_argument('--model', type=str,
                       default='models/detection/yolo11n-obb.pt',
                       help='Path to YOLO .pt model')
    parser.add_argument('--output', type=str,
                       default='models/detection/yolo11n-obb-imx500.onnx',
                       help='Output ONNX file path')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size (416 or 640 recommended for IMX500)')
    parser.add_argument('--no-simplify', action='store_true',
                       help='Disable ONNX simplification')

    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    success = export_yolo_to_imx500_onnx(
        model_path=args.model,
        output_path=args.output,
        imgsz=args.imgsz,
        simplify=not args.no_simplify
    )

    return 0 if success else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
