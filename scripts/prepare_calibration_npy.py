#!/usr/bin/env python3
"""Convert Pokemon card PNG images to NPY format for Hailo calibration"""

import os
import sys
import numpy as np
from PIL import Image
import glob

def preprocess_image(image_path, size=(224, 224)):
    """
    Load and preprocess image for model calibration
    - Resize to 224x224
    - Convert to RGB
    - Normalize to [0, 1]
    - Convert to float32
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size, Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Model expects CHW format (channels, height, width)
    img_array = np.transpose(img_array, (2, 0, 1))

    return img_array

def main():
    if len(sys.argv) < 3:
        print("Usage: prepare_calibration_npy.py <input_png_dir> <output_npy_dir>")
        return 1

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    print(f'Converting Pokemon card images to NPY format')
    print(f'Input:  {input_dir}')
    print(f'Output: {output_dir}')

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all PNG images
    image_paths = glob.glob(os.path.join(input_dir, '*.png'))
    print(f'\nFound {len(image_paths)} images')

    if len(image_paths) == 0:
        print('ERROR: No PNG images found')
        return 1

    # Convert each image
    for i, img_path in enumerate(image_paths):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(output_dir, f'{basename}.npy')

        # Preprocess and save
        img_array = preprocess_image(img_path)
        np.save(output_path, img_array)

        if (i + 1) % 10 == 0:
            print(f'  Processed {i + 1}/{len(image_paths)} images')

    print(f'\n✓ Converted {len(image_paths)} images to NPY format')
    print(f'  Output directory: {output_dir}')

    # Verify output
    npy_files = glob.glob(os.path.join(output_dir, '*.npy'))
    print(f'  Verification: {len(npy_files)} NPY files created')

    return 0

if __name__ == '__main__':
    sys.exit(main())
