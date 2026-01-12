#!/usr/bin/env python3
"""Compile ONNX to Hailo HEF with checkpoint support via HAR files"""

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

    image_paths = glob.glob(os.path.join(png_dir, '*.png'))
    if not image_paths:
        print(f'  ERROR: No PNG images found in {png_dir}')
        return None

    print(f'  Found {len(image_paths)} PNG images')

    for i, img_path in enumerate(image_paths):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(npy_dir, f'{basename}.npy')

        img = Image.open(img_path).convert('RGB')
        img = img.resize(size, Image.BILINEAR)
        img_array = np.array(img, dtype=np.float32) / 255.0

        np.save(output_path, img_array)

        if (i + 1) % 20 == 0:
            print(f'    Processed {i + 1}/{len(image_paths)}')

    npy_files = glob.glob(os.path.join(npy_dir, '*.npy'))
    print(f'  ✓ Created {len(npy_files)} NPY files in {npy_dir}')

    return npy_dir

def main():
    print('=' * 50)
    print('Hailo Model Compilation (with checkpoints)')
    print('=' * 50)

    # Output directory (must be writable)
    output_dir = '/output' if os.path.exists('/output') and os.access('/output', os.W_OK) else '.'
    print(f'\n✓ Output directory: {output_dir}')

    # Check write permissions
    test_file = os.path.join(output_dir, '.write_test')
    try:
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print(f'✓ Output directory is writable')
    except Exception as e:
        print(f'❌ Output directory NOT writable: {e}')
        return 1

    # Checkpoint files
    har_checkpoint = os.path.join(output_dir, 'pokemon_student_efficientnet_lite0_stage2.har')
    output_hef = os.path.join(output_dir, 'pokemon_student_efficientnet_lite0_stage2.hef')

    # Check for Pokemon card calibration data
    calib_data_path = os.environ.get('CALIB_DATA_PATH')

    if calib_data_path and os.path.exists(calib_data_path):
        image_count = len([f for f in os.listdir(calib_data_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f'\n✓ Found {image_count} Pokemon card images for calibration')
        print(f'  Path: {calib_data_path}')

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
        traceback.print_exc()
        return 1

    # SAVE HAR CHECKPOINT after optimization (before lengthy compilation)
    print('\n[Checkpoint] Saving HAR file (allows resuming compilation)...')
    try:
        runner.save_har(har_checkpoint)
        har_size = os.path.getsize(har_checkpoint)
        print(f'      ✓ HAR checkpoint saved: {har_size:,} bytes ({har_size/1024/1024:.1f} MB)')
        print(f'      Location: {har_checkpoint}')
    except Exception as e:
        print(f'      ⚠️  HAR save failed (non-critical): {e}')

    print('\n[4/4] Compiling and saving HEF...')
    print('      (This step takes ~18 minutes)')
    try:
        hef_bytes = runner.compile()

        # Save HEF file
        with open(output_hef, 'wb') as f:
            f.write(hef_bytes)

        # Verify it was written
        if os.path.exists(output_hef):
            size = os.path.getsize(output_hef)
            print(f'      ✓ HEF saved: {size:,} bytes ({size/1024/1024:.1f} MB)')
            print(f'      ✓ Location: {output_hef}')
        else:
            print(f'      ❌ HEF write reported success but file not found')
            return 1

    except Exception as e:
        print(f'      ✗ Compilation failed: {e}')
        import traceback
        traceback.print_exc()
        return 1

    print('\n' + '=' * 50)
    print('✅ COMPILATION COMPLETE')
    print('=' * 50)
    print(f'HAR checkpoint: {har_checkpoint}')
    print(f'HEF output: {output_hef}')
    print('Architecture: EfficientNet-Lite0 (Stage 2 final)')
    print('Target: Hailo-8L on Raspberry Pi 5')
    print()
    print('The HAR file can be used to resume/recompile without redoing optimization.')

    return 0

if __name__ == '__main__':
    sys.exit(main())
