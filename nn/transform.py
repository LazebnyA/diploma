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

# Константи з можливістю налаштування
DEFAULT_TARGET_WIDTH = 300  # Частіше використовується ширина 600px для HTR систем
DEFAULT_TARGET_HEIGHT = 64  # Стандартна висота для багатьох HTR моделей
DEFAULT_PADDING_RANGE = (15, 30)  # Діапазон для випадкового паддингу
DEFAULT_BACKGROUND_COLOR = 255  # Білий фон


def resize_with_padding(
        img: Image.Image,
        target_height: int = DEFAULT_TARGET_HEIGHT,
        target_width: int = DEFAULT_TARGET_WIDTH,
        padding_range: Tuple[int, int] = DEFAULT_PADDING_RANGE,
        background_color: int = DEFAULT_BACKGROUND_COLOR,
        random_padding: bool = True
) -> Image.Image:
    """
    Змінює розмір зображення, зберігаючи співвідношення сторін, та додає паддинг.

    Args:
        img: Вхідне зображення
        target_height: Цільова висота
        target_width: Цільова ширина
        padding_range: Діапазон для випадкового паддингу (min, max)
        background_color: Колір фону (0-255)
        random_padding: Чи додавати випадковий паддинг

    Returns:
        Оброблене зображення
    """
    # Конвертуємо до grayscale, якщо потрібно
    if img.mode != 'L':
        img = img.convert('L')

    # Додаємо початковий паддинг
    if random_padding:
        pad_left = random.randint(*padding_range)
        pad_right = random.randint(*padding_range)
        pad_top = random.randint(*padding_range)
        pad_bottom = random.randint(*padding_range)
        img = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom),
                              fill=background_color)

    # Змінюємо розмір зберігаючи співвідношення сторін
    original_width, original_height = img.size
    ratio = target_height / original_height
    new_width = int(original_width * ratio)
    img = img.resize((new_width, target_height), Resampling.LANCZOS)

    # Обробляємо випадок коли нова ширина відрізняється від цільової
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
    Покращує якість документу шляхом регулювання контрасту, яскравості та різкості.

    Args:
        img: Вхідне зображення
        contrast_factor: Фактор контрасту (>1 збільшує контраст)
        brightness_factor: Фактор яскравості (>1 збільшує яскравість)
        sharpness_factor: Фактор різкості (>1 збільшує різкість)

    Returns:
        Покращене зображення
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
    Видаляє шум із зображення за допомогою комбінації фільтрів.

    Args:
        img: Вхідне зображення
        median_kernel_size: Розмір ядра для медіанного фільтру
        gaussian_kernel_size: Розмір ядра для гаусового фільтру
        gaussian_sigma: Сигма для гаусового фільтру
    Returns:
        Зображення без шуму
    """
    # Перевіряємо, що розмір ядра непарний
    if median_kernel_size % 2 == 0:
        median_kernel_size += 1
    if gaussian_kernel_size % 2 == 0:
        gaussian_kernel_size += 1

    # Конвертуємо до numpy array
    img_np = np.array(img)

    # Медіанний фільтр (для видалення salt-and-pepper шуму)
    img_denoised = cv2.medianBlur(img_np, median_kernel_size)

    # Гаусовий фільтр (для зменшення гаусового шуму)
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
    Виконує адаптивну бінаризацію зображення.

    Args:
        img: Вхідне зображення
        block_size: Розмір блоку для адаптивної бінаризації
        constant: Константа для адаптивної бінаризації

    Returns:
        Бінаризоване зображення
    """
    # Перевіряємо, що розмір блоку непарний
    if block_size % 2 == 0:
        block_size += 1

    # Конвертуємо до numpy array у відтінках сірого
    img_np = np.array(img.convert('L'))

    # Виконуємо адаптивну бінаризацію
    binary = cv2.adaptiveThreshold(
        img_np,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        constant
    )

    return Image.fromarray(binary)


def add_gaussian_noise(tensor: torch.Tensor, mean: float = 0, std: float = 0.05) -> torch.Tensor:
    """
    Додає гаусів шум до тензора зображення.

    Args:
        tensor: Вхідний тензор
        mean: Середнє значення шуму
        std: Стандартне відхилення шуму

    Returns:
        Тензор з доданим шумом
    """
    noise = torch.randn_like(tensor) * std + mean
    return torch.clamp(tensor + noise, 0, 1)


def random_distortion(img: Image.Image, distortion_scale: float = 0.1) -> Image.Image:
    """
    Застосовує випадкове перспективне спотворення до зображення.

    Args:
        img: Вхідне зображення
        distortion_scale: Масштаб спотворення (0-1)

    Returns:
        Спотворене зображення
    """
    if random.random() < 0.5:
        width, height = img.size

        # Початкові точки - кути зображення
        startpoints = [(0, 0), (width, 0), (0, height), (width, height)]

        # Максимальні зсуви залежать від масштабу спотворення
        max_shift_w = int(distortion_scale * width)
        max_shift_h = int(distortion_scale * height)

        # Кінцеві точки з невеликими зсувами
        endpoints = [
            (random.randint(0, max_shift_w), random.randint(0, max_shift_h)),
            (width - random.randint(0, max_shift_w), random.randint(0, max_shift_h)),
            (random.randint(0, max_shift_w), height - random.randint(0, max_shift_h)),
            (width - random.randint(0, max_shift_w), height - random.randint(0, max_shift_h))
        ]

        img = TF.perspective(img, startpoints, endpoints)

    return img


def random_elasticity(img: Image.Image, alpha: float = 1.0, sigma: float = 50.0) -> Image.Image:
    """
    Застосовує еластичну деформацію до зображення.
    Це особливо корисно для аугментації рукописного тексту.

    Args:
        img: Вхідне зображення
        alpha: Фактор інтенсивності деформації
        sigma: Гладкість деформації

    Returns:
        Деформоване зображення
    """
    if random.random() < 0.5:
        img_np = np.array(img)
        shape = img_np.shape

        # Випадкові зсуви
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

        # Створюємо сітку індексів
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

        # Додаємо зсуви
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        # Застосовуємо деформацію
        distorted_image = map_coordinates(img_np, indices, order=1, mode='reflect')
        distorted_image = distorted_image.reshape(shape)

        return Image.fromarray(distorted_image)

    return img


# Імпортуємо додаткові необхідні функції
try:
    from scipy.ndimage import gaussian_filter, map_coordinates
except ImportError:
    print("Для використання еластичної деформації встановіть scipy: pip install scipy")


    # Реалізуємо альтернативну функцію без використання scipy
    def random_elasticity(img: Image.Image, *args, **kwargs) -> Image.Image:
        """Заглушка для випадку коли scipy не встановлено"""
        return img


#################################################
# Трансформаційні піпелайни для різних сценаріїв
#################################################

def get_base_transform() -> transforms.Compose:
    """
    Базова трансформація без аугментацій.
    """
    return transforms.Compose([
        transforms.Lambda(lambda img: resize_with_padding(img)),
        transforms.ToTensor()
    ])


def get_validation_transform() -> transforms.Compose:
    """
    Трансформація для валідації з покращенням якості зображення.
    """
    return transforms.Compose([
        # transforms.Lambda(deskew_image),  # Виправлення нахилу
        # transforms.Lambda(remove_ruled_lines),  # Видалення ліній
        transforms.Lambda(lambda img: remove_noise(img, median_kernel_size=3, gaussian_kernel_size=3)),
        transforms.Lambda(lambda img: enhance_document(img, contrast_factor=1.5, brightness_factor=1.2)),
        transforms.Lambda(adaptive_binarization),  # Адаптивна бінаризація краще для рукописного тексту
        transforms.Lambda(resize_with_padding),
        transforms.ToTensor()
    ])


def get_training_transform() -> transforms.Compose:
    """
    Розширена трансформація для навчання з аугментаціями.
    """
    return transforms.Compose([
        # transforms.Lambda(deskew_image),  # Виправлення нахилу
        # transforms.Lambda(remove_ruled_lines),  # Видалення ліній
        transforms.Lambda(lambda img: remove_noise(img,
                                                   median_kernel_size=random.choice([3, 5]),
                                                   gaussian_kernel_size=random.choice([3, 5]))),
        # Випадкові аугментації
        transforms.RandomApply([
            transforms.Lambda(lambda img: random_distortion(img, distortion_scale=0.15))
        ], p=0.5),
        transforms.RandomApply([
            transforms.Lambda(random_elasticity)
        ], p=0.3),
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=(-5, 5)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        # Покращення якості
        transforms.Lambda(lambda img: enhance_document(img,
                                                       contrast_factor=random.uniform(1.3, 1.7),
                                                       brightness_factor=random.uniform(1.0, 1.3))),
        # Бінаризація
        transforms.RandomApply([
            transforms.Lambda(adaptive_binarization)
        ], p=0.7),
        transforms.ToTensor(),
        # Додавання шуму після конвертації в тензор
        transforms.RandomApply([
            transforms.Lambda(lambda t: add_gaussian_noise(t, std=random.uniform(0.01, 0.03)))
        ], p=0.3)
    ])


def get_transform_pipeline(transform_type: str):
    """
    Повертає композицію трансформацій вказаного типу.

    Args:
        transform_type: Тип трансформації ('resize', 'no_lines', 'denoise', 'enhance', 'binarize', 'full')
        img_height: Висота зображення для змінення розміру

    Returns:
        Композиція трансформацій
    """
    if transform_type == 'resize':
        return transforms.Compose([
            transforms.Lambda(lambda img: resize_with_padding(img)),
            transforms.ToTensor()
        ])

    elif transform_type == 'denoise':
        return transforms.Compose([
            transforms.Lambda(lambda img: resize_with_padding(img)),
            transforms.Lambda(remove_noise),
            transforms.ToTensor()
        ])

    elif transform_type == 'enhance':
        return transforms.Compose([
            transforms.Lambda(lambda img: resize_with_padding(img)),
            transforms.Lambda(enhance_document),
            transforms.ToTensor()
        ])

    elif transform_type == 'binarize':
        return transforms.Compose([
            transforms.Lambda(lambda img: resize_with_padding(img)),
            transforms.Lambda(adaptive_binarization),
            transforms.ToTensor()
        ])

    elif transform_type == 'full':
        return transforms.Compose([
            transforms.Lambda(resize_with_padding),
            transforms.Lambda(remove_noise),
            transforms.Lambda(enhance_document),
            transforms.Lambda(adaptive_binarization),
            transforms.ToTensor()
        ])

    else:
        raise ValueError(f"Unknown transform type: {transform_type}")


def apply_transform_to_image(img: Image.Image, transform: transforms.Compose) -> Image.Image:
    """
    Застосовує трансформацію до зображення і повертає результат як PIL Image.

    Args:
        img: Вхідне зображення
        transform: Трансформація для застосування

    Returns:
        Трансформоване зображення
    """
    # Застосовуємо трансформацію
    tensor = transform(img)

    # Конвертуємо тензор назад до зображення
    if isinstance(tensor, torch.Tensor):
        if len(tensor.shape) == 3:
            tensor = tensor.squeeze(0) if tensor.shape[0] == 1 else tensor
        elif len(tensor.shape) == 4:
            tensor = tensor.squeeze(0).squeeze(0)

        # Переводимо в діапазон 0-255
        tensor = (tensor * 255).byte()

        # Конвертуємо до numpy і створюємо PIL Image
        img_np = tensor.cpu().numpy()
        if len(img_np.shape) == 3 and img_np.shape[0] == 3:
            img_np = img_np.transpose(1, 2, 0)  # CHW -> HWC

        return Image.fromarray(img_np)

    return img  # Повертаємо оригінал, якщо щось пішло не так


def display_images_vertical(img: Image.Image, save_path: Optional[str] = None) -> None:
    """
    Відображає результати різних трансформацій зображення у вертикальному розташуванні.

    Args:
        img: Вхідне зображення
        img_height: Висота для відображення зображень
        save_path: Шлях для збереження результату (опціонально)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Для візуалізації потрібен matplotlib: pip install matplotlib")
        return

    # Створюємо словник з різними трансформаціями
    transforms_dict = {}

    # Оригінальне зображення
    transforms_dict["Оригінал"] = apply_transform_to_image(img, get_transform_pipeline('resize'))

    transforms_dict["Видалення шуму"] = apply_transform_to_image(img, get_transform_pipeline('denoise'))
    transforms_dict["Покращення контрасту"] = apply_transform_to_image(img,
                                                                       get_transform_pipeline('enhance'))
    transforms_dict["Адаптивна бінаризація"] = apply_transform_to_image(img,
                                                                        get_transform_pipeline('binarize'))

    # Повна трансформація
    transforms_dict["Все разом"] = apply_transform_to_image(img, get_transform_pipeline('full'))

    # Створюємо вертикальні підграфіки (один стовпець)
    n = len(transforms_dict)
    fig, axes = plt.subplots(n, 1, figsize=(5, 3 * n))
    if n == 1:
        axes = [axes]

    for ax, (title, result_img) in zip(axes, transforms_dict.items()):
        if isinstance(result_img, torch.Tensor):
            # Тензор форми [C, H, W]; перестановка для matplotlib як [H, W, C]
            ax.imshow(result_img.permute(1, 2, 0).numpy(), cmap="gray")
        else:
            # Якщо це PIL Image
            ax.imshow(np.array(result_img), cmap="gray")
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()

    # Зберігаємо або показуємо
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    img_path = "img_6.png"
    if not os.path.exists(img_path):
        print(f"Image not found at {img_path}. Please ensure the image file 'img.png' is in the current directory.")
    else:
        img = Image.open(img_path)
        # Використовуємо нову функцію для вертикального відображення
        display_images_vertical(img)
        # Також можна використовувати оновлену версію оригінальної функції
        # visualize_transformations(img, img_height=128)
