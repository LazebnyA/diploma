import os
import random
from PIL import Image, ImageOps
from tqdm import tqdm

# Base directory for original dataset
BASE_DIR = r'C:\uni\Diploma'

# Output base directory for processed data_preprocessed
OUTPUT_BASE = os.path.join(BASE_DIR, 'iam_words', 'data_preprocessed_imH32')

# Mapping files
mapping_files = [
    'dataset/writer_independent_word_splits/train_word_mappings.txt',
    'dataset/writer_independent_word_splits/val_word_mappings.txt',
    'dataset/writer_independent_word_splits/test_word_mappings.txt'
]

# Target width
TARGET_WIDTH = 150


def process_image(input_path, output_path):
    try:
        # 1. Open in grayscale
        img = Image.open(input_path).convert('L')

        # 2. Add random padding (15-30px) to all sides
        pad_left = random.randint(15, 30)
        pad_right = random.randint(15, 30)
        pad_top = random.randint(15, 30)
        pad_bottom = random.randint(15, 30)

        img = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=255)

        # 3. Resize maintaining aspect ratio based on height
        original_width, original_height = img.size
        target_height = 32
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

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save processed image
        img.save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


# Collect all image paths
image_paths = set()

for mapping_file in mapping_files:
    mapping_path = os.path.join(BASE_DIR, mapping_file)
    if not os.path.isfile(mapping_path):
        print(f"Mapping file not found: {mapping_path}")
        continue

    with open(mapping_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                relative_path, _ = line.split('\t')
                abs_input_path = os.path.join(BASE_DIR, relative_path.replace('/', os.sep))
                if os.path.isfile(abs_input_path):
                    # Generate corresponding output path
                    relative_subpath = os.path.relpath(abs_input_path, os.path.join(BASE_DIR, 'iam_words'))
                    abs_output_path = os.path.join(OUTPUT_BASE, relative_subpath)
                    image_paths.add((abs_input_path, abs_output_path))
            except Exception as e:
                print(f"Error processing line: {line} -> {e}")

# Process all images
print(f"Processing {len(image_paths)} images...")
success_count = 0

for input_path, output_path in tqdm(image_paths):
    if process_image(input_path, output_path):
        success_count += 1

print(f"\nDone! Successfully processed {success_count}/{len(image_paths)} images to '{OUTPUT_BASE}'.")
