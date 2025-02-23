import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


# Function to add random Gaussian noise
def add_gaussian_noise(image, mean=0, std=0.05):
    """Applies Gaussian noise to an image."""
    noise = torch.randn_like(image) * std + mean
    return torch.clamp(image + noise, 0, 1)  # Keep values in [0,1]


# Function to randomly apply distortions (Elastic Transform, Perspective, etc.)
def random_distortion(image):
    """Applies a random perspective transformation or elastic transformation."""
    if random.random() < 0.5:
        image = TF.perspective(image, startpoints=[(0, 0), (10, 0), (0, 20), (10, 20)],
                               endpoints=[(2, 3), (8, 1), (2, 18), (8, 19)])
    return image


# Define a transform that resizes the image while preserving aspect ratio and applies augmentations
def resize_with_aspect(image, target_height=32):
    w, h = image.size
    new_w = int(w * (target_height / h))
    return image.resize((new_w, target_height))


# Augmented Transform Pipeline
transform = transforms.Compose([
    transforms.Lambda(lambda img: resize_with_aspect(img)),  # Resize
    transforms.RandomRotation(degrees=(-5, 5)),  # Small rotations (-5 to 5 degrees)
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.2)),  # Small shifts
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Perspective distortion
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color variations
    transforms.Lambda(lambda img: random_distortion(img)),  # Custom distortions
    transforms.ToTensor(),  # Convert to tensor
    transforms.Lambda(lambda img: add_gaussian_noise(img)),  # Apply Gaussian noise
])
