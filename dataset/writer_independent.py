import os
import xml.etree.ElementTree as ET
import random


def get_form_id_from_image_path(image_path):
    """
    Assumes the filename is of the format: a01-000u-00-00.png
    and that the form id is the first two hyphen-separated parts (e.g. "a01-000u").
    """
    filename = os.path.basename(image_path)
    parts = filename.split('-')
    if len(parts) < 2:
        raise ValueError(f"Unexpected filename format: {filename}")
    # Join the first two parts to get form id
    return '-'.join(parts[:2])


def build_form_to_writer_map(metadata_folder):
    """
    Scan all XML files in the metadata folder and build a mapping
    from form id (extracted from the XML's <form id="..."> attribute)
    to the writer id (<form writer-id="..."> attribute).
    Assumes each XML file corresponds to one form.
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


def writer_independent_split(word_mappings_file, metadata_folder, train_output, test_output):
    # Build mapping from form id to writer id using metadata XML files.
    form_to_writer = build_form_to_writer_map(metadata_folder)
    if not form_to_writer:
        print("No valid XML files found in the metadata folder.")
        return

    # Read the word mappings file.
    entries = []  # list of tuples: (image_path, word, writer_id)
    with open(word_mappings_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Assumes tab-separated file: image_path \t word
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"Skipping malformed line: {line}")
                continue
            image_path, word = parts
            form_id = get_form_id_from_image_path(image_path)
            writer_id = form_to_writer.get(form_id)
            if writer_id is None:
                print(f"Warning: No writer id found for form '{form_id}' from image {image_path}")
                continue
            entries.append((image_path, word, writer_id))

    if not entries:
        print("No entries found. Please check your mapping and metadata files.")
        return

    # Group entries by writer.
    writer_to_entries = {}
    for entry in entries:
        writer = entry[2]
        writer_to_entries.setdefault(writer, []).append(entry)

    # Create a list of unique writers and shuffle them.
    unique_writers = list(writer_to_entries.keys())
    random.shuffle(unique_writers)
    num_train = int(0.8 * len(unique_writers))
    train_writers = set(unique_writers[:num_train])
    test_writers = set(unique_writers[num_train:])

    # Partition entries based on writer split.
    train_entries = []
    test_entries = []
    for writer, writer_entries in writer_to_entries.items():
        if writer in train_writers:
            train_entries.extend(writer_entries)
        else:
            test_entries.extend(writer_entries)

    # Write out the new word mapping files.
    with open(train_output, 'w', encoding='utf-8') as f_train:
        for img_path, word, _ in train_entries:
            f_train.write(f"{img_path}\t{word}\n")
    with open(test_output, 'w', encoding='utf-8') as f_test:
        for img_path, word, _ in test_entries:
            f_test.write(f"{img_path}\t{word}\n")

    print(
        f"Created training mapping with {len(train_entries)} entries and test mapping with {len(test_entries)} entries.")


if __name__ == "__main__":
    # Specify your paths here.
    WORD_MAPPINGS_FILE = "word_mappings.txt"  # e.g., "C:/data/iam/word_mappings.txt"
    METADATA_FOLDER = "metadata"  # e.g., "C:/data/iam/metadata"
    TRAIN_OUTPUT = "writer_independent_mappings/train_word_mappings.txt"
    TEST_OUTPUT = "writer_independent_mappings/test_word_mappings.txt"

    writer_independent_split(WORD_MAPPINGS_FILE, METADATA_FOLDER, TRAIN_OUTPUT, TEST_OUTPUT)
