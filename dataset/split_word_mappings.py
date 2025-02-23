import random


def split_word_mappings(input_file, train_file, test_file, split_ratio=0.8):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    random.shuffle(lines)

    split_index = int(len(lines) * split_ratio)
    train_lines = lines[:split_index]
    test_lines = lines[split_index:]

    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)

    with open(test_file, 'w', encoding='utf-8') as f:
        f.writelines(test_lines)

    print(f"Training set size: {len(train_lines)}")
    print(f"Testing set size: {len(test_lines)}")

split_word_mappings('word_mappings.txt', 'train_word_mappings.txt', 'test_word_mappings.txt')
