import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


def resize_with_aspect(image, target_height=32):
    """
    Змінює розмір зображення до заданої висоти з збереженням аспектного співвідношення.
    """
    w, h = image.size
    if h == 0:
        raise ValueError(f"Invalid image with height=0, original size: {image.size}")
    new_w = max(1, int(w * (target_height / h)))  # забезпечує мінімум 1 піксель
    return image.resize((new_w, target_height))


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

def get_simple_transform(img_height):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: resize_with_aspect(img, target_height=img_height)),
        transforms.ToTensor()
    ])
    return transform

# Повна трансформаційна послідовність
def get_augment_transform(img_height):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: resize_with_aspect(img, target_height=img_height)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Lambda(lambda img: random_distortion(img)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: add_gaussian_noise(t, std=0.02))
    ])
    return transform
