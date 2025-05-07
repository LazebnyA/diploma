import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

from PIL import ImageEnhance, Image, ImageOps
from PIL.Image import Resampling

TARGET_WIDTH = 300


def resize_aspect_ratio_add_padding(img, img_height=64):
    # 2. Add random padding (15-30px) to all sides
    pad_left = random.randint(15, 30)
    pad_right = random.randint(15, 30)
    pad_top = random.randint(15, 30)
    pad_bottom = random.randint(15, 30)

    img = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=255)

    # 3. Resize maintaining aspect ratio based on height
    original_width, original_height = img.size
    target_height = img_height
    ratio = target_height / original_height
    new_width = int(original_width * ratio)
    new_height = target_height
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 4. Add right padding to reach 600px width
    if new_width < TARGET_WIDTH:
        right_pad = TARGET_WIDTH - new_width
        img = ImageOps.expand(img, border=(0, 0, right_pad, 0), fill=255)
    elif new_width > TARGET_WIDTH:
        img = img.crop((0, 0, TARGET_WIDTH, new_height))

    return img

def resize_height_aspect_ratio(img, img_height):
    original_width, original_height = img.size
    target_height = img_height
    ratio = target_height / original_height
    new_width = int(original_width * ratio)
    new_height = target_height
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img


def add_gaussian_noise(tensor, mean=0, std=0.05):
    """
    Додає гаусів шум до тензора зображення.
    """
    noise = torch.randn_like(tensor) * std + mean
    return torch.clamp(tensor + noise, 0, 1)


def random_distortion(image):
    """
    Випадковим чином застосовує перспективне спотворення з користувацькими параметрами.
    """
    if random.random() < 0.5:
        width, height = image.size
        # Стартові точки – кути зображення
        startpoints = [(0, 0), (width, 0), (0, height), (width, height)]
        # Визначення максимально допустимих зсувів (до 10% від розміру)
        max_shift_w = int(0.1 * width)
        max_shift_h = int(0.1 * height)
        endpoints = [
            (random.randint(0, max_shift_w), random.randint(0, max_shift_h)),
            (width - random.randint(0, max_shift_w), random.randint(0, max_shift_h)),
            (random.randint(0, max_shift_w), height - random.randint(0, max_shift_h)),
            (width - random.randint(0, max_shift_w), height - random.randint(0, max_shift_h))
        ]
        image = TF.perspective(image, startpoints, endpoints)
    return image


def get_simple_train_transform_v0(img_height):
    transform = transforms.Compose([
        # transforms.Lambda(lambda img: resize_height_aspect_ratio(img, img_height)),
        transforms.ToTensor()
    ])
    return transform


def get_simple_recognize_transform():
    transform = transforms.Compose([
        transforms.Lambda(lambda img: resize_aspect_ratio_add_padding(img)),
        transforms.ToTensor()
    ])
    return transform


# Повна трансформаційна послідовність
def get_augment_transform():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Lambda(lambda img: resize_aspect_ratio_add_padding(img)),
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Lambda(lambda img: random_distortion(img)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: add_gaussian_noise(t, std=0.02))
    ])
    return transform


# ----- Image preprocessing modifications ----- #

# 1. Adjust contrast and brightness
def adjust_contrast_brightness_sharpness(
    img: Image.Image,
    contrast_factor: float = 1.5,
    brightness_factor: float = 1.2,
    sharpness_factor: float = 1.3
) -> Image.Image:
    """
    Покращує контраст і яскравість зображення.

    :param img: Вхідне зображення PIL.
    :param contrast_factor: Фактор контрасту (1.0 — без змін).
    :param brightness_factor: Фактор яскравості (1.0 — без змін).
    :return: Модифіковане зображення.
    """
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    img = ImageEnhance.Sharpness(img).enhance(sharpness_factor)
    return img


# 2. Noise removal: median filter then Gaussian blur.
def remove_noise(img: Image.Image, median_kernel_size=3, gaussian_kernel_size=3) -> Image.Image:
    img_np = np.array(img)
    # Apply median filter (for salt and pepper noise)
    img_denoised = cv2.medianBlur(img_np, median_kernel_size)
    # Apply Gaussian blur (to reduce Gaussian noise)
    img_denoised = cv2.GaussianBlur(img_denoised, (gaussian_kernel_size, gaussian_kernel_size), 0)
    return Image.fromarray(img_denoised)


# 3. Otsu's binarization.
def otsu_binarization(img: Image.Image) -> Image.Image:
    img_np = np.array(img.convert('L'))  # Convert to grayscale
    _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)


###############################################
# Define transforms as functions for testing
###############################################

def get_contrast_brightness_transform():
    return transforms.Compose([
        transforms.Lambda(lambda img: resize_aspect_ratio_add_padding(img)),
        transforms.Lambda(adjust_contrast_brightness),
        transforms.ToTensor()
    ])


def get_noise_removal_transform():
    return transforms.Compose([
        transforms.Lambda(lambda img: resize_aspect_ratio_add_padding(img)),
        transforms.Lambda(remove_noise),
        transforms.ToTensor()
    ])


def get_contrast_brightness_noise_removal_transform():
    return transforms.Compose([
        transforms.Lambda(lambda img: resize_aspect_ratio_add_padding(img)),
        transforms.Lambda(remove_noise),
        transforms.Lambda(adjust_contrast_brightness),
        transforms.ToTensor()
    ])


def get_contrast_brightness_otsu_transform():
    return transforms.Compose([
        transforms.Lambda(lambda img: resize_aspect_ratio_add_padding(img)),
        transforms.Lambda(remove_noise),
        transforms.Lambda(adjust_contrast_brightness),
        transforms.ToTensor()
    ])

def get_otsu_noise_removal():
    return transforms.Compose([
        transforms.Lambda(lambda img: resize_aspect_ratio_add_padding(img)),
        transforms.Lambda(remove_noise),
        transforms.Lambda(otsu_binarization),
        transforms.ToTensor()
    ])

def get_otsu_binarization_transform():
    return transforms.Compose([
        transforms.Lambda(lambda img: resize_aspect_ratio_add_padding(img)),
        transforms.Lambda(otsu_binarization),
        transforms.ToTensor()
    ])


def get_augmented_and_preprocessed_training_transform():
    return transforms.Compose([
        transforms.Lambda(remove_noise),  # легке згладження
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Lambda(lambda img: random_distortion(img)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: add_gaussian_noise(t, std=0.02))
    ])


def get_validation_transform():
    return transforms.Compose([
        transforms.Lambda(lambda img: resize_aspect_ratio_add_padding(img)),
        transforms.Lambda(remove_noise),                 # зменшення шуму
        transforms.Lambda(adjust_contrast_brightness),  # покращення видимості тексту
        transforms.Lambda(otsu_binarization),           # бінаризація зображення
        transforms.ToTensor()
    ])
