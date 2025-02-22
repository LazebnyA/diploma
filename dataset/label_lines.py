import xml.etree.ElementTree as ET
import os
from pathlib import Path


def process_form_lines(xml_path):
    """Process a single XML form file and return line text with image mappings."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    form_id = os.path.splitext(os.path.basename(xml_path))[0]

    results = []
    for line in root.findall('.//line'):
        line_id = line.get('id')
        text = line.get('text')
        png_filename = f"{line_id}.png"
        png_path = f"data/{form_id[:3]}/{form_id}/{png_filename}"
        results.append((png_path, text))

    return results


def process_form_words(xml_path):
    """Process a single XML form file and return word text with image mappings."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    form_id = os.path.splitext(os.path.basename(xml_path))[0]

    results = []
    for word in root.findall('.//word'):
        word_id = word.get('id')
        text = word.get('text')
        if text:  # Перевіряємо чи є текст
            png_filename = f"{word_id}.png"
            png_path = f"dataset/iam_words/data/{form_id[:3]}/{form_id}/{png_filename}"
            results.append((png_path, text))

    return results


def generate_mapping_files(metadata_dir):
    """Generate text files with PNG to text mappings for both lines and words."""
    line_mappings = []
    word_mappings = []

    for xml_file in Path(metadata_dir).glob('*.xml'):
        try:
            # Обробка рядків
            line_maps = process_form_lines(xml_file)
            line_mappings.extend(line_maps)

            # Обробка слів
            word_maps = process_form_words(xml_file)
            word_mappings.extend(word_maps)
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")

    # Записуємо розмітку рядків
    with open('line_mappings.txt', 'w', encoding='utf-8') as f:
        for png_path, text in line_mappings:
            f.write(f"{png_path}\t{text}\n")

    # Записуємо розмітку слів
    with open('word_mappings.txt', 'w', encoding='utf-8') as f:
        for png_path, text in word_mappings:
            f.write(f"{png_path}\t{text}\n")


# Використання
metadata_dir = "metadata"
generate_mapping_files(metadata_dir)
