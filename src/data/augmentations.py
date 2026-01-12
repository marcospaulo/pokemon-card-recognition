#!/usr/bin/env python3
"""
Card-specific augmentation transforms for Pokemon card training
Designed to handle real-world card detection scenarios
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random
from PIL import Image, ImageFilter
import numpy as np


class RandomPerspective:
    """Apply random perspective transform to simulate viewing angles"""

    def __init__(self, distortion_scale=0.3, p=0.5):
        self.distortion_scale = distortion_scale
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            width, height = img.size

            # Generate random perspective points
            startpoints = [[0, 0], [width, 0], [width, height], [0, height]]
            endpoints = []

            for point in startpoints:
                x = point[0] + random.randint(-int(width * self.distortion_scale),
                                               int(width * self.distortion_scale))
                y = point[1] + random.randint(-int(height * self.distortion_scale),
                                               int(height * self.distortion_scale))
                endpoints.append([x, y])

            return F.perspective(img, startpoints, endpoints)
        return img


class RandomGlare:
    """Add random glare effect to simulate holographic card reflections"""

    def __init__(self, p=0.3, intensity_range=(0.1, 0.4)):
        self.p = p
        self.intensity_range = intensity_range

    def __call__(self, img):
        if random.random() < self.p:
            img_array = np.array(img).astype(np.float32)

            # Random glare intensity
            intensity = random.uniform(*self.intensity_range)

            # Random glare position
            width, height = img.size
            x = random.randint(0, width)
            y = random.randint(0, height)

            # Create radial gradient for glare
            Y, X = np.ogrid[:height, :width]
            dist = np.sqrt((X - x)**2 + (Y - y)**2)
            max_dist = np.sqrt(width**2 + height**2) / 2

            glare = np.maximum(0, 1 - dist / max_dist) * intensity * 255
            glare = glare[:, :, np.newaxis]

            # Add glare
            img_array = np.clip(img_array + glare, 0, 255).astype(np.uint8)

            return Image.fromarray(img_array)
        return img


class RandomMotionBlur:
    """Add motion blur to simulate camera shake"""

    def __init__(self, p=0.2, kernel_size_range=(3, 7)):
        self.p = p
        self.kernel_size_range = kernel_size_range

    def __call__(self, img):
        if random.random() < self.p:
            kernel_size = random.choice(range(self.kernel_size_range[0],
                                              self.kernel_size_range[1], 2))

            # Create motion blur kernel
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size

            # Random rotation
            angle = random.uniform(0, 360)
            kernel_img = Image.fromarray(kernel)
            kernel_img = kernel_img.rotate(angle)

            # Apply blur
            return img.filter(ImageFilter.BLUR)
        return img


class RandomShadow:
    """Add random shadow to simulate lighting variations"""

    def __init__(self, p=0.3, intensity_range=(0.2, 0.5)):
        self.p = p
        self.intensity_range = intensity_range

    def __call__(self, img):
        if random.random() < self.p:
            img_array = np.array(img).astype(np.float32)

            # Random shadow intensity
            intensity = random.uniform(*self.intensity_range)

            # Random shadow direction (gradient)
            width, height = img.size
            direction = random.choice(['left', 'right', 'top', 'bottom'])

            if direction == 'left':
                gradient = np.linspace(1 - intensity, 1, width)
                shadow = np.tile(gradient, (height, 1))
            elif direction == 'right':
                gradient = np.linspace(1, 1 - intensity, width)
                shadow = np.tile(gradient, (height, 1))
            elif direction == 'top':
                gradient = np.linspace(1 - intensity, 1, height)
                shadow = np.tile(gradient[:, np.newaxis], (1, width))
            else:  # bottom
                gradient = np.linspace(1, 1 - intensity, height)
                shadow = np.tile(gradient[:, np.newaxis], (1, width))

            shadow = shadow[:, :, np.newaxis]

            # Apply shadow
            img_array = np.clip(img_array * shadow, 0, 255).astype(np.uint8)

            return Image.fromarray(img_array)
        return img


def get_card_train_transforms(img_size=224):
    """Get training transforms optimized for Pokemon cards

    Args:
        img_size: Target image size (default: 224 for ViT-B/16)

    Returns:
        transforms.Compose object with card-specific augmentations
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),

        # Geometric augmentations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        RandomPerspective(distortion_scale=0.3, p=0.4),

        # Color/lighting augmentations
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1
        ),
        RandomGlare(p=0.3),
        RandomShadow(p=0.3),

        # Quality augmentations
        RandomMotionBlur(p=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),

        # Random erasing (simulates occlusion)
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),

        # Normalization (ImageNet stats)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_card_val_transforms(img_size=224):
    """Get validation transforms (no augmentation, only preprocessing)

    Args:
        img_size: Target image size (default: 224 for ViT-B/16)

    Returns:
        transforms.Compose object for validation
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# Example usage
if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt

    # Load sample image
    img_path = "/Users/marcos/dev/raspberry-pi/PokeTCG_downloader/assets/card_images/base1-1_Alakazam_high.png"
    img = Image.open(img_path)

    # Apply augmentations
    transform = get_card_train_transforms()

    # Show multiple augmented versions
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        # Apply transform (returns tensor)
        augmented = transform(img)

        # Convert back to displayable image
        augmented = augmented.permute(1, 2, 0).numpy()
        augmented = augmented * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        augmented = np.clip(augmented, 0, 1)

        ax.imshow(augmented)
        ax.axis('off')
        ax.set_title(f'Augmentation {i+1}')

    plt.tight_layout()
    plt.savefig('/Users/marcos/dev/raspberry-pi/training_prep/augmentation_examples.png', dpi=150)
    print("✓ Saved augmentation examples to augmentation_examples.png")
