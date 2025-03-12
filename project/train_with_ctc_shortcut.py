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
from project.v8.models import CNNBiLSTMResBlocksNoDenseBetweenCNNCtcShortcut


@logger_model_training(version="8", additional="2-Layered-BiLSTM-ResNet-CNN-Shortcut")
@execution_time_decorator
def main(version, additional):
    paths = ProjectPaths()
    mapping_file = "dataset/writer_independent_mappings/train_word_mappings.txt"
    label_converter = LabelConverter(mapping_file, paths)
    img_height = 32

    dataset = IAMDataset(
        mapping_file=mapping_file,
        paths=paths,
        transform=get_transform(img_height),
        label_converter=label_converter
    )

    batch_size = 8
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    validation_mapping_file = "dataset/validation_words_mappings.txt"
    validation_dataset = IAMDataset(
        mapping_file=validation_mapping_file,
        paths=paths,
        transform=get_transform(img_height),
        label_converter=label_converter
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    n_classes = len(label_converter.chars) + 1
    num_channels = 1
    n_h = 256

    model = CNNBiLSTMResBlocksNoDenseBetweenCNNCtcShortcut(
        img_height=img_height,
        num_channels=num_channels,
        n_classes=n_classes,
        n_h=n_h
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Device: {device}")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_epochs = 75

    model.train()
    avg_loss = 0
    completed_epochs = 0

    print("\nNeural Network Architecture:")
    print(model)

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

    print("\nHyperparameters:")
    for key, value in hyperparams.items():
        print(f"{key}: {value}")

    print("Starting training process: \n")
    try:
        for epoch in range(num_epochs):
            completed_epochs = epoch
            epoch_loss = 0
            desc_line = (f"Epoch [{epoch + 1}/{num_epochs}], Prev Loss: {avg_loss:.4f}"
                         if avg_loss != 0 else f"Epoch [{epoch + 1}/{num_epochs}]")

            model.train()
            for images, targets, target_lengths, input_lengths in tqdm(dataloader, desc=desc_line, file=sys.__stdout__):
                images, targets = images.to(device), targets.to(device)
                target_lengths = target_lengths.to(device)
                input_lengths = input_lengths.to(device)

                optimizer.zero_grad()
                main_outputs, shortcut_outputs = model(images)

                main_outputs = F.log_softmax(main_outputs, dim=2)
                shortcut_outputs = F.log_softmax(shortcut_outputs, dim=2)

                main_loss = criterion(main_outputs, targets, input_lengths, target_lengths)
                shortcut_loss = criterion(shortcut_outputs, targets, input_lengths, target_lengths)

                loss = main_loss + 0.1 * shortcut_loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, targets, target_lengths, input_lengths in validation_loader:
                    images, targets = images.to(device), targets.to(device)
                    target_lengths = target_lengths.to(device)
                    input_lengths = input_lengths.to(device)

                    main_outputs, shortcut_outputs = model(images)
                    main_outputs = F.log_softmax(main_outputs, dim=2)
                    shortcut_outputs = F.log_softmax(shortcut_outputs, dim=2)

                    main_loss = criterion(main_outputs, targets, input_lengths, target_lengths)
                    shortcut_loss = criterion(shortcut_outputs, targets, input_lengths, target_lengths)

                    loss = main_loss + 0.1 * shortcut_loss
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(validation_loader)

            print(f"Epoch {epoch + 1} completed. Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        base_filename = f"cnn_lstm_ctc_handwritten_v{version}_{completed_epochs + 1}ep_{additional}"
        model_filename = f"{base_filename}.pth"
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved as {model_filename}")

    return {"completed_epochs": completed_epochs}


if __name__ == '__main__':
    main()
