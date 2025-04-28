import sys
import json
import os
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn as nn, optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from nn.dataset import ProjectPaths, LabelConverter, IAMDataset, collate_fn
from nn.logger import logger_model_training, logger_hyperparameters_tuning
from nn.transform import get_simple_train_transform_v0
from nn.v0.models import CNN_LSTM_CTC_V0
from nn.v1.models import CNN_LSTM_CTC_V1_CNN_deeper_vgg16like
from nn.v2.models import resnet18_htr_sequential

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


def train_model(model, train_loader, val_loader, optimizer, criterion,
                device, label_converter, epochs=5):
    """
    Train model for given number of epochs and return history
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_cer': [],
        'val_cer': [],
        'train_wer': [],
        'val_wer': [],
        'epoch_times': []  # Track time per epoch
    }

    for epoch in range(epochs):
        epoch_start_time = time.time()  # Start time measurement

        # Training
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_ground_truths = []

        for images, targets, target_lengths, input_lengths in tqdm(
                train_loader, desc=f"Training Epoch [{epoch + 1}/{epochs}]", file=sys.__stdout__):
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            input_lengths = input_lengths.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            outputs = F.log_softmax(outputs, dim=2)

            loss = criterion(outputs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Get predictions for metrics
            with torch.no_grad():
                preds = greedy_decoder(outputs.detach(), label_converter)
                batch_ground_truths = []
                start = 0
                for length in target_lengths:
                    gt = targets[start:start + length].cpu().tolist()
                    decoded = label_converter.decode(gt)
                    batch_ground_truths.append(decoded)
                    start += length

                train_predictions.extend(preds)
                train_ground_truths.extend(batch_ground_truths)

        avg_train_loss = train_loss / len(train_loader)
        train_cer, train_wer = calculate_metrics(train_predictions, train_ground_truths)

        # Validation
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_ground_truths = []

        with torch.no_grad():
            for images, targets, target_lengths, input_lengths in tqdm(
                    val_loader, desc=f"Validation Epoch [{epoch + 1}/{epochs}]"):
                images = images.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)
                input_lengths = input_lengths.to(device)

                outputs = model(images)
                outputs = F.log_softmax(outputs, dim=2)

                loss = criterion(outputs, targets, input_lengths, target_lengths)
                val_loss += loss.item()

                # Get predictions for metrics
                preds = greedy_decoder(outputs, label_converter)
                batch_ground_truths = []
                start = 0
                for length in target_lengths:
                    gt = targets[start:start + length].cpu().tolist()
                    decoded = label_converter.decode(gt)
                    batch_ground_truths.append(decoded)
                    start += length

                val_predictions.extend(preds)
                val_ground_truths.extend(batch_ground_truths)

        avg_val_loss = val_loss / len(val_loader)
        val_cer, val_wer = calculate_metrics(val_predictions, val_ground_truths)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        history['epoch_times'].append(epoch_time)

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_cer'].append(train_cer)
        history['val_cer'].append(val_cer)
        history['train_wer'].append(train_wer)
        history['val_wer'].append(val_wer)

        # Print results
        print(f"Epoch {epoch + 1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, CER: {train_cer:.4f}, WER: {train_wer:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, CER: {val_cer:.4f}, WER: {val_wer:.4f}")
        print(f"  Time: {epoch_time:.2f} seconds")

    # Add total training time to history
    history['total_time'] = sum(history['epoch_times'])

    return history


def plot_parameter_comparison(results_dict, param_name, output_path):
    """
    Create comparison plots for different parameter values
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot Loss
    ax = axes[0]
    for param_value, history in results_dict.items():
        ax.plot(history['train_loss'], label=f"{param_value} (train)")
        ax.plot(history['val_loss'], linestyle='--', label=f"{param_value} (val)")
    ax.set_xlabel("Епоха")
    ax.set_ylabel("Функція втрат")
    ax.legend(loc='upper right', bbox_to_anchor=(1.6, 1))
    ax.grid(True)

    # Plot CER
    ax = axes[1]
    for param_value, history in results_dict.items():
        ax.plot(history['train_cer'], label=f"{param_value} (train)")
        ax.plot(history['val_cer'], linestyle='--', label=f"{param_value} (val)")
    ax.set_xlabel("Епоха")
    ax.set_ylabel("CER")
    ax.legend(loc='upper right', bbox_to_anchor=(1.6, 1))
    ax.grid(True)

    # Plot WER
    ax = axes[2]
    for param_value, history in results_dict.items():
        ax.plot(history['train_wer'], label=f"{param_value} (train)")
        ax.plot(history['val_wer'], linestyle='--', label=f"{param_value} (val)")
    ax.set_xlabel("Епоха")
    ax.set_ylabel("WER")
    ax.legend(loc='upper right', bbox_to_anchor=(1.6, 1))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def create_datasets(paths, mapping_files, img_height, label_converter, batch_size):
    """
    Create and return train and validation datasets and dataloaders
    """
    train_mapping_file, validation_mapping_file = mapping_files

    train_dataset = IAMDataset(
        mapping_file=train_mapping_file,
        paths=paths,
        transform=get_simple_train_transform_v0(img_height),
        label_converter=label_converter
    )

    val_dataset = IAMDataset(
        mapping_file=validation_mapping_file,
        paths=paths,
        transform=get_simple_train_transform_v0(img_height),
        label_converter=label_converter
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, val_loader


def create_model(img_height, num_channels, n_classes, n_h, device, start_filters=24):
    """
    Create and return a model instance with preloaded initial random weights.
    """
    model = resnet18_htr_sequential(
        img_height=img_height,
        num_channels=num_channels,
        n_classes=n_classes,
        n_h=n_h,
        lstm_layers=1,
        out_channels=start_filters
    )
    model.to(device)

    # weights_path = f"cnn_lstm_ctc_handwritten_v0_initial_imH{img_height}.pth"
    # model.load_state_dict(torch.load(weights_path, map_location=device))
    # print(f"Loaded initial random weights from {weights_path}")

    return model


def create_optimizer(model, opt_name, learning_rate=0.0001):
    """
    Create and return optimizer based on name
    """
    if opt_name == "Adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif opt_name == "SGD":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif opt_name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")


def setup_environment():
    """Setup environment and return common objects"""
    # Create output directory
    output_dir = Path("comparison_results")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize project paths
    paths = ProjectPaths()

    # Use relative paths from nn root
    train_mapping_file = "dataset/writer_independent_word_splits/preprocessed/train_word_mappings.txt"
    validation_mapping_file = "dataset/writer_independent_word_splits/preprocessed/val_word_mappings.txt"
    test_mapping_file = "dataset/writer_independent_word_splits/preprocessed/test_word_mappings.txt"

    mapping_files = (train_mapping_file, validation_mapping_file)

    # Initialize converter
    label_converter = LabelConverter(train_mapping_file, paths)

    return paths, label_converter, mapping_files, output_dir, test_mapping_file


def train_and_evaluate_config(config, param_name, param_value, paths, label_converter,
                              mapping_files, device, criterion, num_epochs):
    """
    Train and evaluate a model with a specific configuration

    Args:
        config: Dictionary with all hyperparameters
        param_name: Name of the parameter being varied
        param_value: Value of the parameter being varied
        paths: Project paths object
        label_converter: Label converter object
        mapping_files: Tuple of mapping files
        device: Device to run on
        criterion: Loss function
        num_epochs: Number of epochs to train for

    Returns:
        History of training metrics
    """
    print(f"\nTraining with {param_name}={param_value}")

    # Create datasets
    train_loader, val_loader = create_datasets(
        paths,
        mapping_files,
        config['img_height'],
        label_converter,
        config['batch_size']
    )

    # Create model
    n_classes = len(label_converter.chars) + 1  # +1 for CTC blank char
    model = create_model(
        img_height=config['img_height'],
        num_channels=1,
        n_classes=n_classes,
        n_h=config['n_h'],
        device=device,
        start_filters=config.get('num_filters', 24)
    )

    # Create optimizer
    optimizer = create_optimizer(
        model,
        config['optimizer'],
        config.get('learning_rate', 0.0001)
    )

    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        label_converter=label_converter,
        epochs=num_epochs
    )

    return history


def tune_parameter(paths, label_converter, mapping_files, output_dir,
                   param_name, param_values, base_config, num_epochs=5):
    """
    Generic function to tune any hyperparameter

    Args:
        param_name: Name of the parameter to tune
        param_values: List of values to try for the parameter
        base_config: Dictionary with base configuration (fixed parameters)
        num_epochs: Number of epochs to train for each configuration
    """
    print(f"\n=== Comparing different {param_name} values ===")

    # Initialize results dictionary
    results = {}

    # Set up common variables
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # Special case for optimizer + learning rate combinations
    if param_name == "optimizer_with_lr":
        optimizers = param_values[0]  # First element is list of optimizers
        learning_rates = param_values[1]  # Second element is list of learning rates

        for opt_name in optimizers:
            for lr in learning_rates:
                config_key = f"{opt_name}_lr_{lr}"
                current_config = base_config.copy()
                current_config['optimizer'] = opt_name
                current_config['learning_rate'] = lr

                history = train_and_evaluate_config(
                    current_config, f"{param_name} ({opt_name}, lr={lr})", config_key,
                    paths, label_converter, mapping_files, device, criterion, num_epochs
                )

                results[config_key] = history

    # New special case for num_filters + hidden_size combinations
    elif param_name == "num_filters_with_hidden_size":
        filter_hidden_pairs = param_values  # List of (num_filters, hidden_size) tuples

        for num_filters, hidden_size in filter_hidden_pairs:
            config_key = f"filters_{num_filters}_hidden_{hidden_size}"
            current_config = base_config.copy()
            current_config['num_filters'] = num_filters
            current_config['n_h'] = hidden_size

            history = train_and_evaluate_config(
                current_config, f"{param_name} (filters={num_filters}, hidden={hidden_size})", config_key,
                paths, label_converter, mapping_files, device, criterion, num_epochs
            )

            results[config_key] = history
    else:
        # Handle all other parameters
        for value in param_values:
            current_config = base_config.copy()
            current_config[param_name] = value

            history = train_and_evaluate_config(
                current_config, param_name, value,
                paths, label_converter, mapping_files, device, criterion, num_epochs
            )

            results[value] = history

    # Plot and save results
    plot_parameter_comparison(
        results,
        param_name,
        output_dir / f"{param_name}_comparison.png"
    )

    return results


@logger_hyperparameters_tuning("CNN-LSTM-CTC")
def run_hyperparameter_tuning(fixed_params, params_to_tune=None, num_epochs=5):
    """
    Run hyperparameter tuning experiment

    Args:
        fixed_params: Dictionary with fixed hyperparameters
        params_to_tune: List of parameter names to tune. Options are:
                       'img_height', 'n_h', 'optimizer', 'batch_size', 'learning_rate', 'num_filters',
                       'num_filters_with_hidden_size'
                       If None, tune all parameters.
        num_epochs: Number of epochs to train for each configuration
    """
    if params_to_tune is None:
        params_to_tune = ['img_height', 'n_h', 'optimizer', 'batch_size', 'learning_rate', 'num_filters']

    paths, label_converter, mapping_files, output_dir, test_mapping_file = setup_environment()

    # Parameter values to try
    param_configs = {
        'img_height': [64, 72, 96],
        'n_h': [128, 256, 512, 1024],
        'optimizer': ["Adam", "SGD", "RMSprop"],
        'batch_size': [4, 8, 16, 32],
        'learning_rate': [0.0001, 0.001],
        'num_filters': [24, 36, 48, 64],
        'optimizer_with_lr': [["Adam", "SGD", "RMSprop"], [0.0001, 0.001]],  # Special case
        'num_filters_with_hidden_size': [(24, 128), (36, 256), (48, 512), (64, 1024)]  # New combined parameter
    }

    # Store all results
    all_results = {}

    # Iterate through parameters to tune
    for param in params_to_tune:
        # Skip learning_rate if optimizer_with_lr is being tuned
        if param == 'learning_rate' and 'optimizer' in params_to_tune:
            continue

        # Skip n_h and num_filters if num_filters_with_hidden_size is being tuned
        if (param == 'n_h' or param == 'num_filters') and 'num_filters_with_hidden_size' in params_to_tune:
            continue

        # Convert 'hidden_size' to 'n_h' if present (for backward compatibility)
        if param == 'hidden_size':
            param = 'n_h'

        # Get parameter values
        param_values = param_configs.get(param)
        if param_values is None:
            print(f"Warning: No values defined for parameter '{param}'. Skipping.")
            continue

        # Tune the parameter
        results = tune_parameter(
            paths, label_converter, mapping_files, output_dir,
            param, param_values, fixed_params, num_epochs
        )

        all_results[param] = results

    # Save all results to a single JSON file
    with open(output_dir / "all_hyperparameter_results.json", "w") as f:
        # Convert non-serializable objects to strings
        serializable_results = {k: {str(k2): v2 for k2, v2 in v.items()} for k, v in all_results.items()}
        json.dump(serializable_results, f, indent=4)

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    fixed_params = {
        'img_height': 64,
        'n_h': 256,
        'optimizer': 'RMSprop',
        'learning_rate': 0.0001,
        'batch_size': 16,
        'num_filters': 48
    }

    # Run tuning for the combined parameter num_filters_with_hidden_size
    run_hyperparameter_tuning(fixed_params, ['num_filters_with_hidden_size'], num_epochs=5)
