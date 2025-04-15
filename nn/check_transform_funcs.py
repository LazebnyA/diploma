# import os
# from PIL import Image
# import matplotlib.pyplot as plt
# from nn.transform import (
#     get_contrast_brightness_transform,
#     get_clahe_transform,
#     get_noise_removal_transform,
#     get_otsu_binarization_transform,
#     get_full_transform,
# )
#
# # Функція для відображення зображень після кожної трансформації
# import torchvision.transforms.functional as TF
#
# def display_images(img, img_height):
#     # Окремі трансформації
#     transforms_dict = {
#         "Original": img,
#         "Contrast & Brightness Adjusted": get_contrast_brightness_transform(img_height)(img),
#         "CLAHE Adjusted": get_clahe_transform(img_height)(img),
#         "Noise Removal": get_noise_removal_transform(img_height)(img),
#         "Otsu Binarization": get_otsu_binarization_transform(img_height)(img),
#         "All Transformations Combined": get_full_transform(img_height)(img)
#     }
#
#     # Створюємо підграфіки для порівняння
#     import matplotlib.pyplot as plt
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     axes = axes.ravel()
#
#     for idx, (title, transformed_img) in enumerate(transforms_dict.items()):
#         # Якщо transformed_img є PIL Image, перетворимо його у тензор
#         if not hasattr(transformed_img, "permute"):
#             transformed_img = TF.to_tensor(transformed_img)
#         # Для зображень у форматі [C x H x W] робимо перетворення для matplotlib
#         axes[idx].imshow(transformed_img.permute(1, 2, 0).numpy(), cmap="gray")
#         axes[idx].set_title(title)
#         axes[idx].axis('off')
#
#     plt.tight_layout()
#     plt.show()
#
#
# # Читання зображення з директорії
# img_path = 'img.png'
# if os.path.exists(img_path):
#     img = Image.open(img_path)
# else:
#     print(f"Image not found at {img_path}. Please ensure the image is in the directory.")
#
# # Висота для трансформацій
# img_height = 256
#
# # Відображення результатів
# display_images(img, img_height)


import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


###############################################
# Preprocessing functions and transforms
###############################################

# Resize image while maintaining aspect ratio.
def resize_with_aspect(img, target_height):
    w, h = img.size
    new_w = int(target_height * w / h)
    # Use a high-quality resizing method
    return img.resize((new_w, target_height), Image.Resampling.LANCZOS)


# 1. Adjust contrast and brightness
def adjust_contrast_brightness(img: Image.Image, contrast_factor=2, brightness_factor=2) -> Image.Image:
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    return img


# CLAHE adjustment (Contrast Limited Adaptive Histogram Equalization)
def clahe_contrast_adjustment(img: Image.Image, clip_limit=2.0, tile_grid_size=(8, 8)) -> Image.Image:
    img_np = np.array(img.convert('L'))  # convert to grayscale
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_img = clahe.apply(img_np)
    return Image.fromarray(clahe_img)


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


# 4. Hough-based straightening to correct skew.
def straighten_image_with_hough(img: Image.Image) -> Image.Image:
    img_np = np.array(img.convert('L'))  # convert to grayscale

    # Edge detection (Canny)
    edges = cv2.Canny(img_np, 50, 150, apertureSize=3)

    # Hough transform to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is not None:
        # Calculate the average angle (in radians)
        angles = [line[0][1] for line in lines]
        avg_angle = np.mean(angles)

        # Convert to degrees. Adjust angle if necessary.
        angle_deg = np.degrees(avg_angle)
        if angle_deg < -45:
            angle_deg += 90

        # Get rotation parameters and rotate image
        (h, w) = img_np.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        rotated_img = cv2.warpAffine(img_np, rotation_matrix, (w, h),
                                     flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return Image.fromarray(rotated_img)
    else:
        return img  # if no lines detected, return the original image


###############################################
# Define transforms as functions for testing
###############################################

def get_contrast_brightness_transform(img_height):
    return transforms.Compose([
        transforms.Lambda(lambda img: resize_with_aspect(img, img_height)),
        transforms.Lambda(adjust_contrast_brightness),
        transforms.ToTensor()
    ])


def get_clahe_transform(img_height):
    return transforms.Compose([
        transforms.Lambda(lambda img: resize_with_aspect(img, img_height)),
        transforms.Lambda(clahe_contrast_adjustment),
        transforms.ToTensor()
    ])


def get_noise_removal_transform(img_height):
    return transforms.Compose([
        transforms.Lambda(lambda img: resize_with_aspect(img, img_height)),
        transforms.Lambda(remove_noise),
        transforms.ToTensor()
    ])


def get_otsu_binarization_transform(img_height):
    return transforms.Compose([
        transforms.Lambda(lambda img: resize_with_aspect(img, img_height)),
        transforms.Lambda(otsu_binarization),
        transforms.ToTensor()
    ])


def get_full_transform(img_height):
    return transforms.Compose([
        transforms.Lambda(lambda img: resize_with_aspect(img, img_height)),
        transforms.Lambda(remove_noise),
        transforms.Lambda(adjust_contrast_brightness),
        transforms.Lambda(otsu_binarization),
        transforms.ToTensor()
    ])


###############################################
# Script to display images in a vertical layout
###############################################

def display_images_vertical(img, img_height):
    # Create a dictionary with various techniques and combinations.
    transforms_dict = {}

    # Original image (resized for display)
    transforms_dict["Оригінал"] = resize_with_aspect(img, img_height)

    # Apply individual transformations:
    transforms_dict[
        "Покращення Контрасту & Яскравості \n(PIL.ImageEnhance, cf=1.5, bf=1.2)"] = get_contrast_brightness_transform(
        img_height)(img)
    transforms_dict["CLAHE"] = get_clahe_transform(img_height)(img)
    transforms_dict["Видалення шуму \n (медіанним та фільтром Гауса)"] = get_noise_removal_transform(img_height)(img)
    transforms_dict["Бінаризація Отсу"] = get_otsu_binarization_transform(img_height)(img)

    # Combined transforms:
    full = get_full_transform(img_height)(img)
    transforms_dict["Все разом"] = full

    # Create vertical subplots (one column) with a reduced width for better fit.
    n = len(transforms_dict)
    fig, axes = plt.subplots(n, 1, figsize=(5, 3 * n))
    if n == 1:
        axes = [axes]
    for ax, (title, result_img) in zip(axes, transforms_dict.items()):
        # Convert PIL Image to tensor if necessary.
        if not hasattr(result_img, "permute"):
            result_img = TF.to_tensor(result_img)
        # The tensor shape is [C, H, W]; permute it for matplotlib as [H, W, C].
        ax.imshow(result_img.permute(1, 2, 0).numpy(), cmap="gray")
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


###############################################
# Main script to load image and display all transforms.
###############################################
if __name__ == "__main__":
    img_path = "img_2.png"
    if not os.path.exists(img_path):
        print(f"Image not found at {img_path}. Please ensure the image file 'img.png' is in the current directory.")
    else:
        img = Image.open(img_path)
        # Set a smaller height for display so the whole vertical column fits on your screen
        display_images_vertical(img, img_height=128)
