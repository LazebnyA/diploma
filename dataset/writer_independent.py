import os
import xml.etree.ElementTree as ET
import random


def get_form_id_from_image_path(image_path):
    """
    Extracts form ID from image filename (e.g. "a01-000u" from "a01-000u-00-00.png").
    """
    filename = os.path.basename(image_path)
    parts = filename.split('-')
    if len(parts) < 2:
        raise ValueError(f"Unexpected filename format: {filename}")
    return '-'.join(parts[:2])


def build_form_to_writer_map(metadata_folder):
    """
    Creates a mapping from form ID to writer ID by scanning XML metadata files.
    """
    form_to_writer = {}
    for fname in os.listdir(metadata_folder):
        if not fname.endswith('.xml'):
            continue
        xml_path = os.path.join(metadata_folder, fname)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            form_id = root.attrib.get('id')
            writer_id = root.attrib.get('writer-id')
            if form_id and writer_id:
                form_to_writer[form_id] = writer_id
            else:
                print(f"Warning: Missing 'id' or 'writer-id' in {fname}")
        except Exception as e:
            print(f"Error parsing {fname}: {e}")
    return form_to_writer


def writer_independent_split(word_mappings_file, metadata_folder, output_dir, ratios=(0.7, 0.15, 0.15)):
    """
    Creates writer-independent train, validation, and test splits.

    Args:
        word_mappings_file: Path to the file mapping image paths to words
        metadata_folder: Folder containing XML metadata files
        output_dir: Directory to write the output mapping files
        ratios: Tuple of (train, validation, test) ratios, must sum to 1.0
    """
    if sum(ratios) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Build mapping from form id to writer id
    form_to_writer = build_form_to_writer_map(metadata_folder)
    if not form_to_writer:
        print("No valid XML files found in the metadata folder.")
        return

    # Read the word mappings file
    entries = []  # list of tuples: (image_path, word, writer_id)
    with open(word_mappings_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"Skipping malformed line: {line}")
                continue
            image_path, word = parts
            form_id = get_form_id_from_image_path(image_path)
            writer_id = form_to_writer.get(form_id)
            if writer_id is None:
                print(f"Warning: No writer ID found for form '{form_id}' from image {image_path}")
                continue
            entries.append((image_path, word, writer_id))

    if not entries:
        print("No entries found. Please check your mapping and metadata files.")
        return

    # Group entries by writer
    writer_to_entries = {}
    for entry in entries:
        writer = entry[2]
        writer_to_entries.setdefault(writer, []).append(entry)

    # Create a list of unique writers and shuffle them
    unique_writers = list(writer_to_entries.keys())
    random.shuffle(unique_writers)

    # Calculate split indices
    train_end = int(ratios[0] * len(unique_writers))
    val_end = train_end + int(ratios[1] * len(unique_writers))

    # Split writers into train, validation, and test sets
    train_writers = set(unique_writers[:train_end])
    val_writers = set(unique_writers[train_end:val_end])
    test_writers = set(unique_writers[val_end:])

    # Verify we have the right number of writers in each set
    print(f"Writers split: {len(train_writers)} train, {len(val_writers)} validation, {len(test_writers)} test")

    # Partition entries based on writer split
    train_entries = []
    val_entries = []
    test_entries = []

    for writer, writer_entries in writer_to_entries.items():
        if writer in train_writers:
            train_entries.extend(writer_entries)
        elif writer in val_writers:
            val_entries.extend(writer_entries)
        else:  # writer in test_writers
            test_entries.extend(writer_entries)

    # Output file paths
    train_output = os.path.join(output_dir, "train_word_mappings.txt")
    val_output = os.path.join(output_dir, "val_word_mappings.txt")
    test_output = os.path.join(output_dir, "test_word_mappings.txt")

    # Write out the new word mapping files
    with open(train_output, 'w', encoding='utf-8') as f_train:
        for img_path, word, _ in train_entries:
            f_train.write(f"{img_path}\t{word}\n")

    with open(val_output, 'w', encoding='utf-8') as f_val:
        for img_path, word, _ in val_entries:
            f_val.write(f"{img_path}\t{word}\n")

    with open(test_output, 'w', encoding='utf-8') as f_test:
        for img_path, word, _ in test_entries:
            f_test.write(f"{img_path}\t{word}\n")

    print(f"Created training mapping with {len(train_entries)} entries")
    print(f"Created validation mapping with {len(val_entries)} entries")
    print(f"Created test mapping with {len(test_entries)} entries")

    # Create a summary file with statistics
    with open(os.path.join(output_dir, "split_summary.txt"), 'w', encoding='utf-8') as f_summary:
        f_summary.write(f"Total writers: {len(unique_writers)}\n")
        f_summary.write(f"Train writers: {len(train_writers)} ({len(train_entries)} samples)\n")
        f_summary.write(f"Validation writers: {len(val_writers)} ({len(val_entries)} samples)\n")
        f_summary.write(f"Test writers: {len(test_writers)} ({len(test_entries)} samples)\n")
        f_summary.write("\nSplit ratios:\n")
        f_summary.write(f"Train: {len(train_entries) / len(entries):.2%}\n")
        f_summary.write(f"Validation: {len(val_entries) / len(entries):.2%}\n")
        f_summary.write(f"Test: {len(test_entries) / len(entries):.2%}\n")


if __name__ == "__main__":
    # Specify your paths here
    WORD_MAPPINGS_FILE = "word_mappings.txt"
    METADATA_FOLDER = "metadata"
    OUTPUT_DIR = "writer_independent_word_splits"

    # Default ratio: 70% train, 15% validation, 15% test
    writer_independent_split(WORD_MAPPINGS_FILE, METADATA_FOLDER, OUTPUT_DIR)
