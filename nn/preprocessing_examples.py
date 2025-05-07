# Імпорт необхідних бібліотек
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
import os  # Для роботи зі шляхами


# --- Функції для додавання шуму ---

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    """
    Додає шум 'сіль та перець' до зображення.

    Args:
        image (np.ndarray): Вхідне зображення (очікується у градаціях сірого або BGR).
        salt_prob (float): Ймовірність появи 'солі' (білий піксель).
        pepper_prob (float): Ймовірність появи 'перцю' (чорний піксель).

    Returns:
        np.ndarray: Зашумлене зображення.
    """
    noisy_image = np.copy(image)
    # Визначаємо загальну кількість пікселів залежно від кількості каналів
    total_pixels = image.shape[0] * image.shape[1]

    # Додавання солі
    num_salt = np.ceil(salt_prob * total_pixels).astype(int)
    # Генеруємо випадкові координати для солі
    rows_salt = np.random.randint(0, image.shape[0] - 1, num_salt)
    cols_salt = np.random.randint(0, image.shape[1] - 1, num_salt)

    if len(image.shape) == 3:  # Кольорове зображення
        # Встановлюємо білий колір для всіх каналів у вибраних координатах
        noisy_image[rows_salt, cols_salt, :] = 255
    else:  # Градації сірого
        # Встановлюємо білий колір (255) у вибраних координатах
        noisy_image[rows_salt, cols_salt] = 255

    # Додавання перцю
    num_pepper = np.ceil(pepper_prob * total_pixels).astype(int)
    # Генеруємо випадкові координати для перцю
    rows_pepper = np.random.randint(0, image.shape[0] - 1, num_pepper)
    cols_pepper = np.random.randint(0, image.shape[1] - 1, num_pepper)

    if len(image.shape) == 3:  # Кольорове зображення
         # Встановлюємо чорний колір для всіх каналів у вибраних координатах
        noisy_image[rows_pepper, cols_pepper, :] = 0
    else: # Градації сірого
        # Встановлюємо чорний колір (0) у вибраних координатах
        noisy_image[rows_pepper, cols_pepper] = 0


    return noisy_image


def add_gaussian_noise(image, mean=0, sigma=25):
    """
    Додає гаусівський шум до зображення.

    Args:
        image (np.ndarray): Вхідне зображення.
        mean (float): Середнє значення шуму.
        sigma (float): Стандартне відхилення шуму.

    Returns:
        np.ndarray: Зашумлене зображення.
    """
    # Генеруємо гаусівський шум з такими ж розмірами, як у зображення
    # Переконуємось, що тип даних шуму відповідає зображенню, але дозволяє від'ємні значення перед додаванням
    if len(image.shape) == 3:
        row, col, ch = image.shape
        gauss = np.random.normal(mean, sigma, (row, col, ch)).astype(np.float32)
    else: # Градації сірого
        row, col = image.shape
        gauss = np.random.normal(mean, sigma, (row, col)).astype(np.float32)

    # Додаємо шум до зображення
    # Переконуємось, що зображення має тип float32 перед додаванням шуму
    noisy_image = cv2.add(image.astype(np.float32), gauss)

    # Обрізаємо значення, щоб вони залишалися в діапазоні [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)

    # Повертаємо до оригінального типу даних (зазвичай uint8)
    noisy_image = noisy_image.astype(image.dtype)

    return noisy_image


# --- Параметри ---
image_url = "img_6.png"  # URL або локальний шлях до зображення
output_dir = "preprocessing_output"  # Директорія для збереження результатів

# Параметри шуму "сіль та перець" (буде застосовано до кольорового зображення)
salt_probability = 0.02  # 2% пікселів стануть білими
pepper_probability = 0.02  # 2% пікселів стануть чорними

# Параметри медіанного фільтра (буде застосовано до кольорового зображення з шумом "сіль та перець")
median_kernel_size = 5

# Параметри гаусового шуму (буде застосовано до зображення в градаціях сірого)
gaussian_mean = 0
gaussian_sigma = 30  # Стандартне відхилення (інтенсивність шуму)

# Параметри гаусового фільтра (буде застосовано до зображення в градаціях сірого з гаусовим шумом)
gaussian_kernel_size = 5
gaussian_filter_sigma = 1.5  # Sigma для гаусового ядра (не плутати з sigma шуму)

# --- Основний процес ---

# Створення директорії для виводу, якщо її немає
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Завантаження зображення
img_bgr = None
try:
    # Спробувати завантажити з URL
    print(f"Спроба завантажити зображення з URL: {image_url}")
    response = requests.get(image_url)
    response.raise_for_status()  # Перевірка на помилки HTTP
    img_pil = Image.open(BytesIO(response.content)).convert('RGB')  # Конвертувати в RGB
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)  # Конвертувати PIL RGB в OpenCV BGR
    print(f"Зображення успішно завантажено з URL.")
except requests.exceptions.RequestException as e:
    print(f"Помилка завантаження зображення з URL: {e}")
    # Спробувати завантажити як локальний файл, якщо URL не спрацював
    try:
        print(f"Спроба завантажити зображення з локального шляху: {image_url}")
        img_bgr = cv2.imread(image_url)
        if img_bgr is None:
            raise FileNotFoundError(f"Не вдалося знайти або відкрити локальний файл: {image_url}")
        print(f"Зображення успішно завантажено з локального шляху.")
    except Exception as e_local:
        print(f"Помилка завантаження локального файлу: {e_local}")
        print("Будь ласка, перевірте URL або шлях до файлу.")
        exit()  # Вихід, якщо зображення не завантажено

img_original_color = img_bgr.copy()  # Зберігаємо копію оригіналу (кольорового)

# Конвертація в градації сірого для операцій з гаусовим шумом
img_original_gray = cv2.cvtColor(img_original_color, cv2.COLOR_BGR2GRAY)
print("Створено версію зображення в градаціях сірого.")

# --- Обробка кольорового зображення (зберігаємо логіку з S&P та медіанним фільтром) ---

# 1. Додавання шуму "сіль та перець" до КОЛЬОРОВОГО зображення
img_salt_pepper_color = add_salt_and_pepper_noise(img_original_color, salt_probability, pepper_probability)
print("Додано шум 'сіль та перець' до кольорового зображення.")

# 2. Застосування медіанного фільтра до КОЛЬОРОВОГО зображення з S&P шумом
img_median_denoised_color = cv2.medianBlur(img_salt_pepper_color, median_kernel_size)
print(f"Застосовано медіанний фільтр з ядром {median_kernel_size}x{median_kernel_size} до кольорового зображення.")

# --- Обробка зображення в градаціях сірого (додавання та фільтрація гаусового шуму) ---

# 3. Додавання гаусового шуму до зображення в ГРАДАЦІЯХ СІРОГО
img_gaussian_added_gray = add_gaussian_noise(img_original_gray, gaussian_mean, gaussian_sigma)
print(f"Додано гаусівський шум (sigma={gaussian_sigma}) до зображення в градаціях сірого.")

# 4. Застосування гаусового фільтра до зашумленого зображення в ГРАДАЦІЯХ СІРОГО
img_gaussian_denoised_gray = cv2.GaussianBlur(
    img_gaussian_added_gray,
    (gaussian_kernel_size, gaussian_kernel_size),
    gaussian_filter_sigma  # Використовуємо sigma для ядра фільтра
)
print(
    f"Застосовано гаусівський фільтр з ядром {gaussian_kernel_size}x{gaussian_kernel_size} та sigma={gaussian_filter_sigma} до зображення в градаціях сірого.")


# --- Візуалізація та збереження результатів ---

# Створення фігури для відображення
plt.figure(figsize=(18, 12)) # Збільшуємо розмір фігури, щоб вмістити більше зображень
plt.style.use('seaborn-v0_8-darkgrid')  # Використання стилю для кращого вигляду

titles = [
    'Оригінал (Колір)',
    'Шум "Сіль та Перець" (Колір)',
    'Після Медіанного фільтра (Колір)',
    'Оригінал (Grayscale)',
    'Grayscale + Гаусівський шум',
    'Grayscale + Гаусівський фільтр'
]
images = [
    cv2.cvtColor(img_original_color, cv2.COLOR_BGR2RGB),  # Конвертуємо BGR в RGB для Matplotlib
    cv2.cvtColor(img_salt_pepper_color, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(img_median_denoised_color, cv2.COLOR_BGR2RGB),
    img_original_gray, # Grayscale зображення не потребує конвертації в RGB для відображення як grayscale
    img_gaussian_added_gray,
    img_gaussian_denoised_gray
]

# Відображення зображень
for i in range(len(images)):
    plt.subplot(2, 3, i + 1)  # Створюємо сітку 2x3 для 6 зображень
    if len(images[i].shape) == 2: # Якщо зображення в градаціях сірого
        plt.imshow(images[i], cmap='gray') # Вказуємо колірну карту 'gray'
    else: # Якщо зображення кольорове
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])  # Прибираємо осі

plt.tight_layout()  # Автоматичне налаштування відступів
plt.suptitle('Демонстрація додавання та фільтрації шумів', fontsize=16, y=1.02)  # Загальний заголовок

# Збереження фігури
plot_filename = os.path.join(output_dir, 'noise_filtering_comparison_grayscale_gaussian.png')
plt.savefig(plot_filename)
print(f"Порівняльний графік збережено як: {plot_filename}")

# Збереження окремих зображень
cv2.imwrite(os.path.join(output_dir, '01_original_color.png'), img_original_color)
cv2.imwrite(os.path.join(output_dir, '02_salt_pepper_noise_color.png'), img_salt_pepper_color)
cv2.imwrite(os.path.join(output_dir, '03_median_denoised_color.png'), img_median_denoised_color)
cv2.imwrite(os.path.join(output_dir, '04_original_gray.png'), img_original_gray)
cv2.imwrite(os.path.join(output_dir, '05_gaussian_added_gray.png'), img_gaussian_added_gray)
cv2.imwrite(os.path.join(output_dir, '06_gaussian_denoised_gray.png'), img_gaussian_denoised_gray)
print(f"Окремі оброблені зображення збережено в директорії: {output_dir}")

# Показати графік (якщо запускається не в середовищі без GUI)
plt.show()

print("Роботу завершено.")