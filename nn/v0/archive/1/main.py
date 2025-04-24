import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from nn.archive.dataset import ProjectPaths, LabelConverter, IAMDataset
from nn.logger import logger_model_training
from nn.v0.models import CNN_LSTM_CTC_V0


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


##############################################
# 5. Main Training Loop
##############################################

def evaluate_execution_time(func, *args, **kwargs):
    start_time = time.time()  # Record start time
    result = func(*args, **kwargs)  # Execute the function
    end_time = time.time()  # Record end time
    execution_time = end_time - start_time  # Calculate execution time

    return result, start_time, end_time, execution_time


@logger_model_training(version="0")
def main():
    # Initialize nn paths
    paths = ProjectPaths()

    # Use relative paths from nn root
    mapping_file = "dataset/train_word_mappings.txt"

    # Define a transform that resizes the image to a fixed height (32) while preserving aspect ratio.
    def resize_with_aspect(image, target_height=32):
        w, h = image.size
        new_w = int(w * (target_height / h))
        return image.resize((new_w, target_height))

    transform = transforms.Compose([
        transforms.Lambda(lambda img: resize_with_aspect(img)),
        transforms.ToTensor()
    ])

    # Initialize converter and dataset
    label_converter = LabelConverter(mapping_file, paths)

    dataset = IAMDataset(
        mapping_file=mapping_file,
        paths=paths,
        transform=transform,
        label_converter=label_converter
    )

    batch_size = 8

    # Create DataLoader with the custom collate_fn.
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn)

    validation_mapping_file = "dataset/words_writing_styles_recognition_demo.txt"

    validation_dataset = IAMDataset(
        mapping_file=validation_mapping_file,
        paths=paths,
        transform=transform,
        label_converter=label_converter
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    print(next(iter(validation_loader)))

    # Define model_params parameters.
    n_classes = len(label_converter.chars) + 1  # +1 for CTC blank char
    img_height = 32
    num_channels = 1
    n_h = 256

    model = CNN_LSTM_CTC_V0(
        img_height=img_height,
        num_channels=num_channels,
        n_classes=n_classes,
        n_h=n_h
    )

    # Device configuration.
    # Move the model_params to the configured device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Device: {device}")

    # Define the CTCLoss and optimizer.
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_epochs = 50

    model.train()

    avg_loss = 0
    completed_epochs = 0  # Local counter for completed epochs

    # Print the neural network architecture to the console
    print("\nNeural Network Architecture:")
    print(model)

    # Create a dictionary of hyperparameters
    hyperparams = {
        "img_height": img_height,
        "num_channels": num_channels,
        "n_classes": n_classes,
        "n_h": n_h,
        "optimizer": str(optimizer),
        "learning_rate": lr,
        "criterion": str(criterion),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "transform": "Resize with aspect ratio and ToTensor"
    }

    # Print hyperparameters to the console
    print("\nHyperparameters:")
    for key, value in hyperparams.items():
        print(f"{key}: {value}")

    print("Starting training process: \n")
    try:
        for epoch in range(num_epochs):
            completed_epochs = epoch  # Update counter at the start of each epoch.
            epoch_loss = 0
            desc_line = (f"Epoch [{epoch + 1}/{num_epochs}], Prev Loss: {avg_loss:.4f}"
                         if avg_loss != 0 else f"Epoch [{epoch + 1}/{num_epochs}]")

            # ---- TRAINING LOOP ----
            model.train()
            for images, targets, target_lengths, input_lengths in tqdm(dataloader,
                                                                       desc=desc_line,
                                                                       file=sys.__stdout__):
                images, targets = images.to(device), targets.to(device)  # (batch, 1, H, W)
                target_lengths = target_lengths.to(device)
                input_lengths = input_lengths.to(device)

                optimizer.zero_grad()
                outputs = model(images)  # (T, batch, n_classes)

                # Apply log_softmax for CTCLoss
                outputs = F.log_softmax(outputs, dim=2)

                # Compute CTC Loss
                loss = criterion(outputs, targets, input_lengths, target_lengths)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)

            # ---- VALIDATION LOOP ----
            model.eval()  # switch to eval mode
            val_loss = 0
            with torch.no_grad():
                for images, targets, target_lengths, input_lengths in validation_loader:
                    images, targets = images.to(device), targets.to(device)
                    target_lengths = target_lengths.to(device)
                    input_lengths = input_lengths.to(device)

                    outputs = model(images)  # (T, batch, n_classes)
                    outputs = F.log_softmax(outputs, dim=2)

                    loss = criterion(outputs, targets, input_lengths, target_lengths)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(validation_loader)

            print(f"Epoch {epoch + 1} completed. "
                  f"Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # Save the model_params using the number of epochs actually completed.
        base_filename = f"cnn_lstm_ctc_handwritten_v0_{completed_epochs + 1}ep"
        model_filename = f"{base_filename}.pth"
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved as {model_filename}")

    return {"completed_epochs": completed_epochs}


if __name__ == '__main__':
    result, start_time, end_time, execution_time = evaluate_execution_time(main)
    print(f"Time elapsed: {execution_time}")
    print(f"Start time: {start_time}\nEnd time: {end_time}")
