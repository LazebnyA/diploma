import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


##############################################
# 1. Label Converter (builds character vocab)
##############################################

class LabelConverter:
    def __init__(self, mapping_file):
        # Build vocabulary from the labels in the mapping file
        vocab = set()
        with open(mapping_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) != 2:
                    continue
                _, text = parts
                vocab.update(list(text))
        # sort the vocabulary so that the mapping is consistent
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


##############################################
# 2. Dataset for IAM (single-word images)
##############################################

class IAMDataset(Dataset):
    def __init__(self, mapping_file, transform=None, label_converter=None):
        """
        mapping_file: path to the word_mappings.txt file.
        transform: torchvision transforms for image pre-processing.
        label_converter: instance of `LabelConverter` class
        """

        self.samples = []
        with open(mapping_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) != 2:
                    continue

                img_pth, text = parts
                img_pth = '../../dataset/iam_words/' + img_pth
                self.samples.append((img_pth, text))

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


##############################################
# 3. Collate function to pad variable widths
##############################################

def collate_fn(batch):
    """
    batch: list of (image, label) tuples.
    Pads images to the same width and concatenates labels.
    Also computes target lengths and input lengths (after CNN downsampling).
    """
    images, labels = zip(*batch)

    # images are tensors of shape (C - channels, H - height, W - width)
    widths = [img.size(2) for img in images]
    max_width = max(widths)

    padded_images = []
    for img in images:
        pad_width = max_width - img.size(2)
        if pad_width > 0:
            # pad the right side of the width dimension
            img = F.pad(img, (0, pad_width), value=0)
        padded_images.append(img)
    images_tensor = torch.stack(padded_images, dim=0)

    # Concatenate labels into one long tensor & record individual lengths
    targets = []
    target_lenghts = []
    for label in labels:
        targets.extend(label)
        target_lenghts.append(len(label))
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    targets_lengths_tensor = torch.tensor(target_lenghts, dtype=torch.long)

    # Assuming our CNN downsamples the width by a factor of 4
    input_lengths = [w // 4 for w in widths]
    input_lengths_tensor = torch.tensor(input_lengths, dtype=torch.long)

    return images_tensor, targets_tensor, targets_lengths_tensor, input_lengths_tensor


##############################################
# 4. The CRNN Model: CNN + BiLSTM + FC (CTC)
##############################################

class CNN_LSTM_CTC_V0(nn.Module):
    def __init__(self, img_height, num_channels, n_classes, n_h):
        """
        img_height: image height (after resize)
        num_channels: number of input channels (1 for grayscale)
        n_classes: number of output classes (vocab size + 1 for blank)
        n_h: number of hidden units in the LSTM
        """
        super().__init__()

        # CNN Module
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # downsample height & width by 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        # After CNN, height becomes img_height // 4
        self.lstm_input_size = 128 * (img_height // 4)

        # One directional LSTM
        self.lstm = nn.LSTM(self.lstm_input_size, n_h, num_layers=1, batch_first=True)

        # Final classifier: maps LSTM output to character classes
        self.fc = nn.Linear(n_h, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: input images with shape (batch, channels, height, width)
        Returns output in shape (T, batch, n_classes) required by CTCLoss
        """

        conv = self.cnn(x)  # shape: (batch, 128, H', W')
        b, c, h, w = conv.size()

        # Permute and flatten so that each column (width) is a time step
        conv = conv.permute(0, 3, 1, 2)  # now (batch, width, channels, height)
        conv = conv.contiguous().view(b, w, c * h)  # (batch, width, feature_size)

        # LSTM expects input shape (batch, seq_len, input_size)
        recurrent, _ = self.lstm(conv)  # (batch, seq_len, n_h)
        output = self.fc(recurrent)  # (batch, seq_len, n_classes)

        # For CTCLoss, we need (seq_len, batch, n_classes)
        output = output.permute(1, 0, 2)
        return output


##############################################
# 5. Main Training Loop
##############################################

def evaluate_execution_time(func, *args, **kwargs):
    start_time = time.time()  # Record start time
    result = func(*args, **kwargs)  # Execute the function
    end_time = time.time()  # Record end time
    execution_time = end_time - start_time  # Calculate execution time

    return result, start_time, end_time, execution_time


def main():
    # Path to mapping file
    print(f"Workdir: {os.getcwd()}")
    mapping_file = '../../dataset/word_mappings.txt'

    # Define a transform that resizes the image to a fixed height (32)
    # while preserving the aspect ratio
    def resize_with_aspect(image, target_height=32):
        w, h = image.size
        new_w = int(w * (target_height / h))
        return image.resize((new_w, target_height))

    transform = transforms.Compose([
        transforms.Lambda(lambda img: resize_with_aspect(img)),
        transforms.ToTensor()
    ])

    # Initialize the label converter and dataset
    label_converter = LabelConverter(mapping_file)
    dataset = IAMDataset(mapping_file,
                         transform=transform,
                         label_converter=label_converter)

    # Create DataLoader with the custom collate_fn
    dataloader = DataLoader(dataset,
                            batch_size=8,
                            shuffle=True,
                            collate_fn=collate_fn)

    # Define model parameters
    n_classes = len(label_converter.chars) + 1  # +1 for CTC blank char
    model = CNN_LSTM_CTC_V0(
        img_height=32,
        num_channels=1,
        n_classes=n_classes,
        n_h=256
    )

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"Device: {device}")

    # Define the CTCLoss and optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 75
    model.train()
    avg_loss = 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        desc_line = f"\tEpoch [{epoch + 1}/{num_epochs}], Prev Loss: {avg_loss:.4f}" \
            if avg_loss != 0 else f"\tEpoch [{epoch + 1}/{num_epochs}]"
        for images, targets, target_lengths, input_lengths in tqdm(dataloader, desc=desc_line):
            images, targets = images.to(device), targets.to(device)  # (batch, 1, H, W)
            target_lengths = target_lengths.to(device)
            input_lengths = input_lengths.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # (T, batch, n_classes)

            # Apply log_softmax for CTCloss
            outputs = F.log_softmax(outputs, dim=2)

            # Compute CTC Loss. Note: outputs' time dimension T can be different per sample,
            # so we pass the precomputed input_lengths (from the CNN down-sampling)
            loss = criterion(outputs, targets, input_lengths, target_lengths)
            loss.backward()

            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
    # Save the trained model
    torch.save(model.state_dict(), "cnn_lstm_ctc_handwritten_v0_75ep.pth")


if __name__ == '__main__':
    result, start_time, end_time, execution_time = evaluate_execution_time(main)
    print(f"Time elapsed: {execution_time}")
    print(f"Start time: {start_time}\nEnd time: {end_time}")
