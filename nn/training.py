import sys
import json
from pathlib import Path

import torch
from torch import nn as nn, optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from nn.dataset_2 import ProjectPaths, LabelConverter, IAMDataset, collate_fn
from nn.logger import logger_model_training
from nn.transform import get_augment_transform, get_simple_train_transform, get_simple_train_transform
from nn.utils import execution_time_decorator
from nn.v0.models import CNN_LSTM_CTC_V2_CNN_more_filters_batch_norm_more_imH
from nn.v1.models import CNN_LSTM_CTC_V2_CNN_more_filters_batch_norm_deeper_vgg16like

torch.manual_seed(42)


def greedy_decoder(output, label_converter):
    """
    Greedy decoder for CTC output.
    Args:
        output (Tensor): Log probabilities with shape (T, batch, n_classes)
        label_converter (LabelConverter): Instance to decode indices to text
    Returns:
        List of decoded strings (one per sample in batch)
    """
    # Change shape to (batch, T, n_classes)
    output = output.permute(1, 0, 2)
    arg_maxes = torch.argmax(output, dim=2)
    decoded_preds = []
    for pred in arg_maxes:
        pred = pred.cpu().numpy().tolist()
        decoded = label_converter.decode(pred)
        decoded_preds.append(decoded)
    return decoded_preds


def calculate_metrics(predictions, ground_truths):
    """
    Calculate CER and WER metrics for a batch of predictions.
    """
    from jiwer import wer as calculate_wer
    from jiwer import cer as calculate_cer

    total_cer = calculate_cer(ground_truths, predictions)
    total_wer = calculate_wer(ground_truths, predictions)

    return total_cer, total_wer


@logger_model_training(version="0", additional="CNN-BiLSTM-CTC_CNN-VGG16_BiLSTM-1dim")
@execution_time_decorator
def main(version, additional):
    # Initialize nn paths
    paths = ProjectPaths()

    # Use relative paths from nn root
    mapping_file = "dataset/writer_independent_word_splits/v2/train_word_mappings.txt"

    # Initialize converter and dataset
    label_converter = LabelConverter(mapping_file, paths)

    dataset = IAMDataset(
        mapping_file=mapping_file,
        paths=paths,
        transform=get_simple_train_transform(),
        label_converter=label_converter
    )

    batch_size = 8

    # Create DataLoader with the custom collate_fn.
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn,
                            )

    validation_mapping_file = "dataset/writer_independent_word_splits/v2/val_word_mappings.txt"

    validation_dataset = IAMDataset(
        mapping_file=validation_mapping_file,
        paths=paths,
        transform=get_simple_train_transform(),
        label_converter=label_converter
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Define model_params parameters.
    n_classes = len(label_converter.chars) + 1  # +1 for CTC blank char

    num_channels = 1
    n_h = 256
    img_height = 64

    model = CNN_LSTM_CTC_V2_CNN_more_filters_batch_norm_more_imH(
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

    # base_filename = f"cnn_lstm_ctc_handwritten_v{version}_initial_imH{img_height}"
    # model_filename = f"{base_filename}.pth"
    # torch.save(model.state_dict(), model_filename)

    # Load initial random weights (hardcoded path)
    weights_path = "cnn_lstm_ctc_handwritten_v0_initial_imH64.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Loaded initial random weights from {weights_path}")

    # Define the CTCLoss and optimizer.
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_epochs = 30

    model.train()

    # Initialize training history
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_cer': [],
        'val_cer': [],
        'train_wer': [],
        'val_wer': []
    }

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
        "transform": "Resize with aspect ratio. Contrast/Brightness Transform, Otsu-binarization",
        "dataset": "IAM Lines Dataset (writer-independent split)"
    }

    # Print hyperparameters to the console
    print("\nHyperparameters:")
    for key, value in hyperparams.items():
        print(f"{key}: {value}")

    print("Starting training process: \n")
    try:
        for epoch in range(num_epochs):
            # ---- TRAINING LOOP ----
            model.train()
            train_loss = 0
            train_predictions = []
            train_ground_truths = []

            for images, targets, target_lengths, input_lengths in tqdm(dataloader,
                                                                       desc=f"Training Epoch [{epoch + 1}/{num_epochs}]",
                                                                       file=sys.__stdout__):
                images, targets = images.to(device), targets.to(device)
                target_lengths = target_lengths.to(device)
                input_lengths = input_lengths.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                outputs = F.log_softmax(outputs, dim=2)

                loss = criterion(outputs, targets, input_lengths, target_lengths)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Get predictions for metrics calculation
                preds = greedy_decoder(outputs, label_converter)
                train_predictions.extend(preds)
                start = 0
                for length in target_lengths:
                    gt = targets[start:start + length].cpu().tolist()
                    decoded = label_converter.decode(gt)
                    train_ground_truths.append(decoded)
                    start += length

            avg_train_loss = train_loss / len(dataloader)
            train_cer, train_wer = calculate_metrics(train_predictions, train_ground_truths)

            # ---- VALIDATION LOOP ----
            model.eval()
            val_loss = 0
            val_predictions = []
            val_ground_truths = []

            with torch.no_grad():
                for images, targets, target_lengths, input_lengths in tqdm(validation_loader,
                                                                           desc=f"Validation Epoch [{epoch + 1}/{num_epochs}]",
                                                                           file=sys.__stdout__):
                    images, targets = images.to(device), targets.to(device)
                    target_lengths = target_lengths.to(device)
                    input_lengths = input_lengths.to(device)

                    outputs = model(images)
                    outputs = F.log_softmax(outputs, dim=2)

                    loss = criterion(outputs, targets, input_lengths, target_lengths)
                    val_loss += loss.item()

                    # Get predictions for metrics calculation
                    preds = greedy_decoder(outputs, label_converter)
                    val_predictions.extend(preds)
                    start = 0
                    for length in target_lengths:
                        gt = targets[start:start + length].cpu().tolist()
                        decoded = label_converter.decode(gt)
                        val_ground_truths.append(decoded)
                        start += length

            avg_val_loss = val_loss / len(validation_loader)
            val_cer, val_wer = calculate_metrics(val_predictions, val_ground_truths)

            # Update training history
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(avg_val_loss)
            training_history['train_cer'].append(train_cer)
            training_history['val_cer'].append(val_cer)
            training_history['train_wer'].append(train_wer)
            training_history['val_wer'].append(val_wer)

            # Print epoch results
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"Training - Loss: {avg_train_loss:.4f}, CER: {train_cer:.4f}, WER: {train_wer:.4f}")
            print(f"Validation - Loss: {avg_val_loss:.4f}, CER: {val_cer:.4f}, WER: {val_wer:.4f}")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # Save training history after each epoch
        base_filename = f"cnn_lstm_ctc_handwritten_v{version}_word_{epoch + 1}ep_{additional}"
        history_file = f"{base_filename}.json"
        with open(history_file, 'w') as f:
            json.dump(training_history, f, indent=4)

        # Save the model_params using the number of epochs actually completed.
        base_filename = f"cnn_lstm_ctc_handwritten_v{version}_word_{epoch + 1}ep_{additional}"
        model_filename = f"{base_filename}.pth"
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved as {model_filename}")

    return {"completed_epochs": epoch + 1}


if __name__ == '__main__':
    main()
