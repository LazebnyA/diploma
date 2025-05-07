import os
import sys
import json

import torch
from torch import nn as nn, optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from nn.dataset import ProjectPaths, LabelConverter, IAMDataset, collate_fn
from nn.logger import logger_model_training
from nn.transform_2 import get_training_transform, get_validation_transform
from nn.utils import execution_time_decorator, greedy_decoder, calculate_metrics
from nn.v0.models import CNN_LSTM_CTC_V0
from nn.v1.models import CNN_LSTM_CTC_V1_CNN_deeper_vgg16like
from nn.v2.models import resnet18_htr_sequential, resnet18_htr_sequential_v2

torch.manual_seed(42)


@logger_model_training(version="0", additional="CNN-BiLSTM-CTC_V0")
def main(version, additional):
    # Initialize nn paths
    paths = ProjectPaths()

    # Use relative paths from nn root
    mapping_file = "dataset/writer_independent_word_splits/preprocessed/train_word_mappings.txt"

    # Initialize converter and dataset
    label_converter = LabelConverter(mapping_file, paths)

    img_height = 64

    dataset = IAMDataset(
        mapping_file=mapping_file,
        paths=paths,
        transform=get_training_transform(),
        label_converter=label_converter
    )

    batch_size = 8

    # Create DataLoader with the custom collate_fn.
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn,
                            )

    validation_mapping_file = "dataset/writer_independent_word_splits/preprocessed_80_10_10/val_word_mappings.txt"

    validation_dataset = IAMDataset(
        mapping_file=validation_mapping_file,
        paths=paths,
        transform=get_validation_transform(),
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
    n_h = 512

    model = CNN_LSTM_CTC_V0(
        img_height=img_height,
        num_channels=num_channels,
        n_classes=n_classes,
        n_h=n_h,
        out_channels=48,
        lstm_layers=1
    )

    # Device configuration.
    # Move the model_params to the configured device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Device: {device}")

    base_filename = f"{additional}_initial_weights"
    model_filename = f"{base_filename}.pth"
    torch.save(model.state_dict(), model_filename)

    # Load initial random weights (hardcoded path)
    weights_path = "parameters/CNN-BiLSTM-CTC_V0_initial_weights.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Loaded initial random weights from {weights_path}")

    # Define the CTCLoss and optimizer.
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    lr = 0.0001
    optimizer = optim.RMSprop(model.parameters(), lr=lr)

    start_epoch = 0
    checkpoint_path = 'checkpoint.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1  # Resume from next epoch
        print(f"Resuming training from checkpoint at epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    num_epochs = 100

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
        "transform": "Resize with aspect ratio. Data Preprocessing + Augmentation",
        "dataset": "IAM Lines Dataset (writer-independent split). Cleaned dataset"
    }

    # Print hyperparameters to the console
    print("\nHyperparameters:")
    for key, value in hyperparams.items():
        print(f"{key}: {value}")

    print("Starting training process: \n")

    # Early stopping variables
    patience = 10  # Number of epochs to wait for improvement
    no_improvement_count = 0
    best_val_cer = float('inf')
    best_epoch = 0
    early_stopping = False

    try:
        for epoch in range(start_epoch, num_epochs):
            # ---- TRAINING LOOP ----
            model.train()
            train_loss = 0
            train_predictions = []
            train_ground_truths = []

            # batch_counter = 0
            for images, targets, target_lengths, input_lengths in tqdm(dataloader,
                                                                       desc=f"Training Epoch [{epoch + 1}/{num_epochs}]",
                                                                       file=sys.__stdout__):
                images, targets = images.to(device), targets.to(device)
                target_lengths = target_lengths.to(device)
                input_lengths = input_lengths.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                outputs = F.log_softmax(outputs, dim=2)

                # print('-'*50)
                # print(batch_counter)
                # print("Model output shape:", outputs.shape)
                # print("Target length:", target_lengths)
                # print("Input length:", input_lengths)
                # batch_counter += 1

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
                    decoded = label_converter.decode_gt(gt)
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
                        decoded = label_converter.decode_gt(gt)
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

            # Early stopping check
            if val_cer < best_val_cer:
                best_val_cer = val_cer
                best_epoch = epoch
                no_improvement_count = 0

                # Save the best model
                best_model_filename = f"cnn_lstm_ctc_handwritten_v{version}_word_best_{additional}.pth"
                torch.save(model.state_dict(), best_model_filename)
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_train_loss
                }
                torch.save(checkpoint, 'checkpoint.pth')
                print(f"New best model saved with validation CER: {best_val_cer:.4f}")
            else:
                no_improvement_count += 1
                print(
                    f"No improvement in validation CER for {no_improvement_count} epochs. Best CER: {best_val_cer:.4f} at epoch {best_epoch + 1}")

            if no_improvement_count >= patience:
                print(
                    f"\nEarly stopping triggered! No improvement in validation CER for {patience} consecutive epochs.")
                print(f"Best validation CER: {best_val_cer:.4f} achieved at epoch {best_epoch + 1}")
                early_stopping = True
                break

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # Save training history
        completed_epochs = epoch + 1 if 'epoch' in locals() else 0
        base_filename = f"cnn_lstm_ctc_handwritten_v{version}_word_{completed_epochs}ep_{additional}"
        history_file = f"{base_filename}.json"
        with open(history_file, 'w') as f:
            json.dump(training_history, f, indent=4)

        # Save the model using the number of epochs actually completed
        model_filename = f"{base_filename}.pth"
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved as {model_filename}")

        # Load and return the best model if early stopping was triggered
        if early_stopping and best_epoch < completed_epochs - 1:
            best_model_filename = f"cnn_lstm_ctc_handwritten_v{version}_word_best_{additional}.pth"
            model.load_state_dict(torch.load(best_model_filename))
            print(f"Loaded best model from epoch {best_epoch + 1}")

    result = {
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch + 1 if 'best_epoch' in locals() else None,
        "best_val_cer": best_val_cer if 'best_val_cer' in locals() else None,
        "early_stopping_triggered": early_stopping
    }

    return result


if __name__ == '__main__':
    main()