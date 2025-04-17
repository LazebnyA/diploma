import os
from PIL import Image

# Base directory for image paths
BASE_DIR = r'C:\uni\Diploma'

# Mapping files
mapping_files = [
    'writer_independent_word_splits/train_word_mappings.txt',
    'writer_independent_word_splits/val_word_mappings.txt',
    'writer_independent_word_splits/test_word_mappings.txt'
]

# Store (abs_path, word, width) tuples
word_images = []

for mapping_file in mapping_files:
    if not os.path.isfile(mapping_file):
        print(f"Mapping file not found: {mapping_file}")
        continue

    with open(mapping_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                relative_path, word = line.split('\t')
                abs_path = os.path.join(BASE_DIR, relative_path.replace('/', os.sep))

                if not os.path.isfile(abs_path):
                    continue

                with Image.open(abs_path) as img:
                    width, _ = img.size
                    word_images.append((abs_path, word, width))
            except Exception as e:
                print(f"Error processing line: {line} -> {e}")

# Sort by width in descending order
top_longest = sorted(word_images, key=lambda x: x[2], reverse=True)[:100]

# Print the top 20
print("Top 20 longest words by pixel width:\n")
for idx, (path, word, width) in enumerate(top_longest, 1):
    print(f"{idx:2d}. Width: {width:4d} px | Word: '{word}' | Path: {path}")
