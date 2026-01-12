#!/usr/bin/env python3
"""Compile ONNX model to Hailo HEF format using Pokemon card images"""

from hailo_sdk_client import ClientRunner
import sys
import os
import numpy as np
from PIL import Image
import glob

def preprocess_images_to_npy(png_dir, npy_dir, size=(224, 224)):
    """Convert PNG images to preprocessed NPY format for calibration"""
    print(f'\n[Pre-processing] Converting PNG images to NPY format...')
    os.makedirs(npy_dir, exist_ok=True)

    # Find all PNG images
    image_paths = glob.glob(os.path.join(png_dir, '*.png'))
    if not image_paths:
        print(f'  ERROR: No PNG images found in {png_dir}')
        return None

    print(f'  Found {len(image_paths)} PNG images')

    # Convert each image
    for i, img_path in enumerate(image_paths):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(npy_dir, f'{basename}.npy')

        # Load image
        img = Image.open(img_path).convert('RGB')
        img = img.resize(size, Image.BILINEAR)
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Hailo expects HWC format (height, width, channels) without batch dimension
        # img_array is already in HWC format from PIL
        # Just normalize to [0, 1] range

        # Save as NPY
        np.save(output_path, img_array)

        if (i + 1) % 20 == 0:
            print(f'    Processed {i + 1}/{len(image_paths)}')

    # Verify output
    npy_files = glob.glob(os.path.join(npy_dir, '*.npy'))
    print(f'  ✓ Created {len(npy_files)} NPY files in {npy_dir}')

    return npy_dir

def main():
    print('=' * 50)
    print('Hailo Model Compilation')
    print('=' * 50)

    # Check for Pokemon card calibration data
    calib_data_path = os.environ.get('CALIB_DATA_PATH')

    if calib_data_path and os.path.exists(calib_data_path):
        image_count = len([f for f in os.listdir(calib_data_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f'\n✓ Found {image_count} Pokemon card images for calibration')
        print(f'  Path: {calib_data_path}')

        # Convert PNG images to NPY format (required by Hailo SDK)
        npy_dir = '/tmp/calibration_npy'
        calib_data = preprocess_images_to_npy(calib_data_path, npy_dir)

        if not calib_data:
            print('  ! Preprocessing failed, using SDK default calibration')
            calib_data = None
    else:
        print('\n! No Pokemon card images found, using SDK default calibration')
        calib_data = None

    print('\n[1/4] Initializing Hailo runner...')
    runner = ClientRunner(hw_arch='hailo8l')

    print('[2/4] Loading ONNX model...')
    onnx_model = os.environ.get('ONNX_MODEL', 'pokemon_student_stage2_final.onnx')
    print(f'      Model: {onnx_model}')
    try:
        hn, npz = runner.translate_onnx_model(
            onnx_model,
            'pokemon_student_efficientnet_lite0'
        )
        print('      ✓ Translation successful')
    except Exception as e:
        print(f'      ✗ Translation failed: {e}')
        return 1

    print('[3/4] Optimizing model with quantization...')
    try:
        if calib_data:
            print(f'      Using Pokemon card images from {calib_data}')
            runner.optimize(calib_data)
        else:
            print('      Using SDK default calibration')
            runner.optimize(None)
        print('      ✓ Optimization successful')
    except Exception as e:
        import traceback
        print(f'      ✗ Optimization failed: {type(e).__name__}: {e}')
        print('      Full traceback:')
        traceback.print_exc()
        print('      This is a critical error - model requires quantization')
        return 1

    print('[4/4] Compiling and saving HEF...')
    # Write to /output/ if it exists (Docker volume mount), otherwise current dir
    output_dir = '/output' if os.path.exists('/output') and os.access('/output', os.W_OK) else '.'
    output_hef = os.path.join(output_dir, 'pokemon_student_efficientnet_lite0_stage2.hef')
    output_har = os.path.join(output_dir, 'pokemon_student_efficientnet_lite0_stage2.har')
    try:
        hef_bytes = runner.compile()

        # Try multiple methods to save the HEF
        try:
            # Method 1: Direct HEF bytes (newest SDK)
            with open(output_hef, 'wb') as f:
                f.write(hef_bytes)
            print('      ✓ HEF saved via direct bytes')
        except (TypeError, AttributeError):
            try:
                # Method 2: Save HAR then convert to HEF
                runner.save_har(output_har)
                # HEF is automatically created during compile()
                # Just need to find it
                import glob
                hef_files = glob.glob('*.hef')
                if hef_files:
                    os.rename(hef_files[0], output_hef)
                    print(f'      ✓ HEF saved via HAR conversion: {hef_files[0]} -> {output_hef}')
                else:
                    print('      ! HAR saved but no HEF found')
            except Exception as e2:
                print(f'      ⚠️  HEF save methods failed: {e2}')
                print('      Checking for any .hef files created...')
                hef_files = glob.glob('*.hef')
                if hef_files:
                    os.rename(hef_files[0], output_hef)
                    print(f'      ✓ Found and renamed: {hef_files[0]} -> {output_hef}')

        print('      ✓ Compilation complete')
    except Exception as e:
        print(f'      ✗ Compilation failed: {e}')
        import traceback
        traceback.print_exc()
        return 1

    print('\n' + '=' * 50)
    print('✅ COMPILATION COMPLETE')
    print('=' * 50)
    print(f'Output: {output_hef}')
    print('Architecture: EfficientNet-Lite0 (Stage 2 final)')
    print('Purpose: Card recognition embeddings (768-dim)')
    print('Target: Hailo-8L on Raspberry Pi 5')

    # Verify file exists and has reasonable size
    if os.path.exists(output_hef):
        size = os.path.getsize(output_hef)
        print(f'File size: {size:,} bytes ({size/1024/1024:.1f} MB)')
        print(f'✅ HEF file verified at: {os.path.abspath(output_hef)}')
        return 0
    else:
        print(f'❌ ERROR: HEF file not found at {output_hef}')
        print(f'   Checked: {os.path.abspath(output_hef)}')
        return 1

if __name__ == '__main__':
    sys.exit(main())
