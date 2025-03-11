import sys

import torch
from torch import nn as nn, optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from project.dataset import ProjectPaths, LabelConverter, IAMDataset, collate_fn
from project.logger import logger_model_training
from project.transform import get_transform
from project.utils import execution_time_decorator
from project.v6.models import CNN_BiLSTM_CTC_V5_3ConvBlocks
from project.v7.models import CNNBiLSTMResBlocks


@logger_model_training(version="5", additional="2-Layered-BiLSTM-3-CNN-Blocks")
@execution_time_decorator
def main(version, additional):
    # Initialize project paths
    paths = ProjectPaths()

    # Use relative paths from project root
    mapping_file = "dataset/writer_independent_mappings/train_word_mappings.txt"

    # Initialize converter and dataset
    label_converter = LabelConverter(mapping_file, paths)

    img_height = 32

    dataset = IAMDataset(
        mapping_file=mapping_file,
        paths=paths,
        transform=get_transform(img_height),
        label_converter=label_converter
    )

    batch_size = 8

    # Create DataLoader with the custom collate_fn.
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn)

    validation_mapping_file = "dataset/validation_words_mappings.txt"

    validation_dataset = IAMDataset(
        mapping_file=validation_mapping_file,
        paths=paths,
        transform=get_transform(img_height),
        label_converter=label_converter
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Define model parameters.
    n_classes = len(label_converter.chars) + 1  # +1 for CTC blank char

    num_channels = 1
    n_h = 256

    model = CNN_BiLSTM_CTC_V5_3ConvBlocks(
        img_height=img_height,
        num_channels=num_channels,
        n_classes=n_classes,
        n_h=n_h
    )

    # Device configuration.
    # Move the model to the configured device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Device: {device}")

    # Define the CTCLoss and optimizer.
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_epochs = 75

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
        # Save the model using the number of epochs actually completed.
        base_filename = f"cnn_lstm_ctc_handwritten_v{version}_{completed_epochs + 1}ep_{additional}"
        model_filename = f"{base_filename}.pth"
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved as {model_filename}")

    return {"completed_epochs": completed_epochs}


if __name__ == '__main__':
    main()
