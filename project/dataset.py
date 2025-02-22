from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class ProjectPaths:
    def __init__(self):
        # Get the project root directory (assuming this file is in the project)
        self.PROJECT_ROOT = Path(__file__).parent.parent.resolve()

        # Define common paths relative to project root
        self.DATASET_DIR = self.PROJECT_ROOT / "dataset"
        self.IAM_WORDS_DIR = self.DATASET_DIR / "iam_words"
        self.MAPPINGS_DIR = self.DATASET_DIR / "mappings"

    def get_path(self, relative_path: str) -> Path:
        """Convert a relative path string to absolute path based on project root"""
        return (self.PROJECT_ROOT / relative_path).resolve()


class LabelConverter:
    def __init__(self, mapping_file: str, paths: ProjectPaths):
        # Convert relative path to absolute using ProjectPaths
        mapping_path = paths.get_path(mapping_file)
        if not mapping_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

        # Build vocabulary from the labels in the mapping file
        vocab = set()
        with mapping_path.open('r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) != 2:
                    continue
                _, text = parts
                vocab.update(list(text))

        # Sort the vocabulary so that the mapping is consistent
        self.chars = sorted(list(vocab))

        # Reserve index 0 for CTC blank token
        self.char_to_index = {char: i + 1 for i, char in enumerate(self.chars)}
        self.index_to_char = {i + 1: char for i, char in enumerate(self.chars)}
        self.blank = 0

    def encode(self, text):
        """Converts a text string into a list of label indices"""
        return [self.char_to_index[char] for char in text]

    def decode(self, preds):
        """
        Decodes a sequence of predictions (indices) into text.
        It collapses repeated characters and removes blanks.
        """
        decoded = []
        prev = None
        for idx in preds:
            if idx != self.blank and idx != prev:
                decoded.append(self.index_to_char.get(idx, ''))
            prev = idx
        return ''.join(decoded)


class IAMDataset(Dataset):
    def __init__(self, mapping_file: str, paths: ProjectPaths, transform=None, label_converter=None):
        """
        Args:
            mapping_file (str): relative path from project root to mapping file
            paths (ProjectPaths): instance of ProjectPaths for path handling
            transform: torchvision transforms for image pre-processing
            label_converter: instance of LabelConverter class
        """
        self.paths = paths
        self.mapping_path = paths.get_path(mapping_file)

        if not self.mapping_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {self.mapping_path}")

        self.samples = []
        with self.mapping_path.open('r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) != 2:
                    continue

                img_path, text = parts
                # Convert relative path to absolute using ProjectPaths
                full_img_path = paths.get_path(img_path)

                if not full_img_path.exists():
                    print(f"Warning: Image file not found: {full_img_path}")
                    continue

                self.samples.append((full_img_path, text))

        self.transform = transform
        self.label_converter = label_converter

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]

        # Load image in grayscale
        image = Image.open(img_path).convert('L')
        if self.transform is not None:
            image = self.transform(image)

        # Encode the text label into a list of indices
        label = self.label_converter.encode(text)
        return image, label
