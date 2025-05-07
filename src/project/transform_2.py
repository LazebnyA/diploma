import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from typing import Tuple, Optional, Union
from PIL import ImageEnhance, Image, ImageOps, ImageFilter
from PIL.Image import Resampling

# Constants with customization options
DEFAULT_TARGET_WIDTH = 300  # Common width of 600px for HTR systems
DEFAULT_TARGET_HEIGHT = 64  # Standard height for many HTR models
DEFAULT_PADDING_RANGE = (15, 30)  # Range for random padding
DEFAULT_BACKGROUND_COLOR = 255  # White background


def resize_without_padding(
        img: Image.Image,
        target_height: int = DEFAULT_TARGET_HEIGHT,
        target_width: int = DEFAULT_TARGET_WIDTH,
        padding_range: Tuple[int, int] = DEFAULT_PADDING_RANGE,
        background_color: int = DEFAULT_BACKGROUND_COLOR,
        random_padding: bool = True
):
    # Конвертуємо до grayscale, якщо потрібно
    if img.mode != 'L':
        img = img.convert('L')

    original_width, original_height = img.size
    ratio = target_height / original_height
    new_width = int(original_width * ratio)
    img = img.resize((new_width, target_height), Resampling.LANCZOS)

    if new_width < target_width:
        # Додаємо паддинг справа для досягнення цільової ширини
        right_pad = target_width - new_width
        img = ImageOps.expand(img, border=(0, 0, right_pad, 0), fill=background_color)
    elif new_width > target_width:
        # Центруємо зображення та обрізаємо до цільової ширини
        # Це краще ніж обрізати справа, щоб не втратити текст
        left_margin = (new_width - target_width) // 2
        img = img.crop((left_margin, 0, left_margin + target_width, target_height))

    return img


def enhance_document(
        img: Image.Image,
        contrast_factor: float = 1.5,
        brightness_factor: float = 1.2,
        sharpness_factor: float = 1.3
) -> Image.Image:
    """
    Enhance document quality by adjusting contrast, brightness, and sharpness.

    Args:
        img: Input image
        contrast_factor: Contrast factor (>1 increases contrast)
        brightness_factor: Brightness factor (>1 increases brightness)
        sharpness_factor: Sharpness factor (>1 increases sharpness)

    Returns:
        Enhanced image
    """
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    img = ImageEnhance.Sharpness(img).enhance(sharpness_factor)
    return img


def remove_noise(
        img: Image.Image,
        median_kernel_size: int = 3,
        gaussian_kernel_size: int = 3,
        gaussian_sigma: float = 0.5
) -> Image.Image:
    """
    Remove noise from image using a combination of filters.

    Args:
        img: Input image
        median_kernel_size: Kernel size for median filter
        gaussian_kernel_size: Kernel size for Gaussian filter
        gaussian_sigma: Sigma for Gaussian filter

    Returns:
        Denoised image
    """
    # Ensure kernel sizes are odd
    if median_kernel_size % 2 == 0:
        median_kernel_size += 1
    if gaussian_kernel_size % 2 == 0:
        gaussian_kernel_size += 1

    # Convert to numpy array
    img_np = np.array(img.convert('L'))

    # Apply median filter (for salt-and-pepper noise removal)
    img_denoised = cv2.medianBlur(img_np, median_kernel_size)

    # Apply Gaussian filter (for Gaussian noise reduction)
    img_denoised = cv2.GaussianBlur(
        img_denoised,
        (gaussian_kernel_size, gaussian_kernel_size),
        gaussian_sigma
    )

    return Image.fromarray(img_denoised)


def adaptive_binarization(
        img: Image.Image,
        block_size: int = 11,
        constant: int = 2
) -> Image.Image:
    """
    Perform adaptive binarization on the image.

    Args:
        img: Input image
        block_size: Block size for adaptive binarization
        constant: Constant for adaptive binarization

    Returns:
        Binarized image
    """
    # Ensure block size is odd
    if block_size % 2 == 0:
        block_size += 1

    # Convert to numpy array in grayscale
    img_np = np.array(img.convert('L'))

    # Perform adaptive binarization
    binary = cv2.adaptiveThreshold(
        img_np,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        constant
    )

    return Image.fromarray(binary)


def get_validation_transform() -> transforms.Compose:
    """
    Transformation for validation with image quality enhancement.
    """
    return transforms.Compose([
        # transforms.Lambda(deskew_image),  # Skew correction
        # transforms.Lambda(remove_ruled_lines),  # Line removal
        transforms.Lambda(lambda img: remove_noise(img, median_kernel_size=3, gaussian_kernel_size=3)),
        transforms.Lambda(lambda img: enhance_document(img, contrast_factor=1.5, brightness_factor=1.2)),
        transforms.Lambda(adaptive_binarization),  # Adaptive binarization is better for handwritten text
        # transforms.Lambda(resize_with_padding),
        transforms.ToTensor()
    ])
