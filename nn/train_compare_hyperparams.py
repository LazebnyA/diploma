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
from nn.transform import get_simple_transform
from nn.v0.models import CNN_LSTM_CTC_V0

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
        'val_wer': []
    }

    for epoch in range(epochs):
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

    return history


def plot_parameter_comparison(results_dict, param_name, output_path):
    """
    Create comparison plots for different parameter values
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot Loss
    ax = axes[0]
    for param_value, history in results_dict.items():
        ax.plot(history['train_loss'], label=f"{param_value} (train)")
        ax.plot(history['val_loss'], linestyle='--', label=f"{param_value} (val)")
    ax.set_xlabel("Епоха")
    ax.set_ylabel("Функція втрат")
    ax.legend()
    ax.grid(True)

    # Plot CER
    ax = axes[1]
    for param_value, history in results_dict.items():
        ax.plot(history['train_cer'], label=f"{param_value} (train)")
        ax.plot(history['val_cer'], linestyle='--', label=f"{param_value} (val)")
    ax.set_xlabel("Епоха")
    ax.set_ylabel("CER")
    ax.legend()
    ax.grid(True)

    # Plot WER
    ax = axes[2]
    for param_value, history in results_dict.items():
        ax.plot(history['train_wer'], label=f"{param_value} (train)")
        ax.plot(history['val_wer'], linestyle='--', label=f"{param_value} (val)")
    ax.set_xlabel("Епоха")
    ax.set_ylabel("WER")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def create_datasets(paths, mapping_files, img_height, label_converter, batch_size):
    """
    Create and return train and validation datasets and dataloaders
    """
    train_mapping_file, validation_mapping_file = mapping_files

    train_dataset = IAMDataset(
        mapping_file=train_mapping_file,
        paths=paths,
        transform=get_simple_transform(img_height),
        label_converter=label_converter
    )

    val_dataset = IAMDataset(
        mapping_file=validation_mapping_file,
        paths=paths,
        transform=get_simple_transform(img_height),
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


def create_model(img_height, num_channels, n_classes, n_h, device):
    """
    Create and return a model instance with preloaded initial random weights.
    """
    model = CNN_LSTM_CTC_V0(
        img_height=img_height,
        num_channels=num_channels,
        n_classes=n_classes,
        n_h=n_h
    )
    model.to(device)

    # weights_path = f"cnn_lstm_ctc_handwritten_v0_initial_imH{img_height}.pth"
    #
    # model.load_state_dict(torch.load(weights_path, map_location=device))
    # print(f"Loaded initial random weights from {weights_path}")

    return model


def create_optimizer(model, opt_name, learning_rate=0.001):
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

def tune_learning_rate(paths, label_converter, mapping_files, output_dir,
                      learning_rates, num_epochs=5, fixed_params=None):
    """Run learning rate hyperparameter tuning"""
    if fixed_params is None:
        fixed_params = {
            'img_height': 32,
            'n_h': 256,
            'batch_size': 8,
            'optimizer': 'Adam'
        }

    print("\n=== Порівняння для різних швидкостей навчання (learning rates) ===")
    learning_rate_results = {}

    n_classes = len(label_converter.chars) + 1  # +1 for CTC blank char
    num_channels = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    for lr in learning_rates:
        print(f"\nНавчання з learning_rate={lr}")

        # Create datasets
        train_loader, val_loader = create_datasets(
            paths,
            mapping_files,
            fixed_params['img_height'],
            label_converter,
            fixed_params['batch_size']
        )

        # Create model
        model = create_model(
            img_height=fixed_params['img_height'],
            num_channels=num_channels,
            n_classes=n_classes,
            n_h=fixed_params['n_h'],
            device=device
        )

        # Create optimizer with current learning rate
        if fixed_params['optimizer'] == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif fixed_params['optimizer'] == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif fixed_params['optimizer'] == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {fixed_params['optimizer']}")

        # Train the model and get history
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

        # Save history
        learning_rate_results[lr] = history

    # Plot and save learning rate comparison
    plot_parameter_comparison(
        learning_rate_results,
        "Learning Rate",
        output_dir / "learning_rate_comparison.png"
    )

    return learning_rate_results

def tune_img_height(paths, label_converter, mapping_files, output_dir,
                    img_heights, num_epochs=5, fixed_params=None):
    """Run image height hyperparameter tuning"""
    if fixed_params is None:
        fixed_params = {
            'n_h': 256,
            'batch_size': 8,
            'optimizer': 'Adam'
        }

    print("\n=== Порівняння різної висоти зображень ===")
    img_height_results = {}

    n_classes = len(label_converter.chars) + 1  # +1 for CTC blank char
    num_channels = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    for img_height in img_heights:
        print(f"\nНавчання з img_height={img_height}")

        # Create datasets with current img_height
        train_loader, val_loader = create_datasets(
            paths,
            mapping_files,
            img_height,
            label_converter,
            fixed_params['batch_size']
        )

        # Create model with current img_height
        model = create_model(
            img_height=img_height,
            num_channels=num_channels,
            n_classes=n_classes,
            n_h=fixed_params['n_h'],
            device=device
        )

        # Create optimizer
        optimizer = create_optimizer(model, fixed_params['optimizer'])

        # Train the model and get history
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

        # Save history
        img_height_results[img_height] = history

    # Plot and save img_height comparison
    plot_parameter_comparison(
        img_height_results,
        "Image Height",
        output_dir / "img_height_comparison.png"
    )

    return img_height_results


def tune_hidden_size(paths, label_converter, mapping_files, output_dir,
                     hidden_sizes, num_epochs=5, fixed_params=None):
    """Run hidden size hyperparameter tuning"""
    if fixed_params is None:
        fixed_params = {
            'img_height': 32,
            'batch_size': 8,
            'optimizer': 'Adam'
        }

    print("\n=== Порівняння різних розмірностей вектору схованого стану h - n_h ===")
    hidden_size_results = {}

    n_classes = len(label_converter.chars) + 1  # +1 for CTC blank char
    num_channels = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    for n_h in hidden_sizes:
        print(f"\nНавчання з hidden_size={n_h}")

        # Create datasets
        train_loader, val_loader = create_datasets(
            paths,
            mapping_files,
            fixed_params['img_height'],
            label_converter,
            fixed_params['batch_size']
        )

        # Create model with current hidden size
        model = create_model(
            img_height=fixed_params['img_height'],
            num_channels=num_channels,
            n_classes=n_classes,
            n_h=n_h,
            device=device
        )

        # Create optimizer
        optimizer = create_optimizer(model, fixed_params['optimizer'])

        # Train the model and get history
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

        # Save history
        hidden_size_results[n_h] = history

    # Plot and save hidden_size comparison
    plot_parameter_comparison(
        hidden_size_results,
        "Hidden Size",
        output_dir / "hidden_size_comparison.png"
    )

    return hidden_size_results


def tune_optimizer(paths, label_converter, mapping_files, output_dir,
                   optimizers, num_epochs=5, fixed_params=None):
    """Run optimizer hyperparameter tuning"""
    if fixed_params is None:
        fixed_params = {
            'img_height': 32,
            'n_h': 256,
            'batch_size': 8
        }

    print("\n=== Порівняння різних алгоритмів оптимізації ===")
    optimizer_results = {}

    n_classes = len(label_converter.chars) + 1  # +1 for CTC blank char
    num_channels = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    for opt_name in optimizers:
        print(f"\nНавчання з оптимізатором {opt_name}")

        # Create datasets
        train_loader, val_loader = create_datasets(
            paths,
            mapping_files,
            fixed_params['img_height'],
            label_converter,
            fixed_params['batch_size']
        )

        # Create model
        model = create_model(
            img_height=fixed_params['img_height'],
            num_channels=num_channels,
            n_classes=n_classes,
            n_h=fixed_params['n_h'],
            device=device
        )

        # Create optimizer based on name
        optimizer = create_optimizer(model, opt_name)

        # Train the model and get history
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

        # Save history
        optimizer_results[opt_name] = history

    # Plot and save optimizer comparison
    plot_parameter_comparison(
        optimizer_results,
        "Optimizer",
        output_dir / "optimizer_comparison.png"
    )

    return optimizer_results


def tune_batch_size(paths, label_converter, mapping_files, output_dir,
                    batch_sizes, num_epochs=5, fixed_params=None):
    """Run batch size hyperparameter tuning"""
    if fixed_params is None:
        fixed_params = {
            'img_height': 32,
            'n_h': 256,
            'optimizer': 'Adam'
        }

    print("\n=== Порівняння для різних розмірів батчів ===")
    batch_size_results = {}

    n_classes = len(label_converter.chars) + 1  # +1 for CTC blank char
    num_channels = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    for batch_size in batch_sizes:
        print(f"\nНавчання з batch_size={batch_size}")

        # Create datasets with current batch size
        train_loader, val_loader = create_datasets(
            paths,
            mapping_files,
            fixed_params['img_height'],
            label_converter,
            batch_size
        )

        # Create model
        model = create_model(
            img_height=fixed_params['img_height'],
            num_channels=num_channels,
            n_classes=n_classes,
            n_h=fixed_params['n_h'],
            device=device
        )

        # Create optimizer
        optimizer = create_optimizer(model, fixed_params['optimizer'])

        # Train the model and get history
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

        # Save history
        batch_size_results[batch_size] = history

    # Plot and save batch_size comparison
    plot_parameter_comparison(
        batch_size_results,
        "Batch Size",
        output_dir / "batch_size_comparison.png"
    )

    return batch_size_results


def setup_environment():
    """Setup environment and return common objects"""
    # Create output directory
    output_dir = Path("comparison_results")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize project paths
    paths = ProjectPaths()

    # Use relative paths from nn root
    train_mapping_file = "dataset/writer_independent_word_splits/train_word_mappings.txt"
    validation_mapping_file = "dataset/writer_independent_word_splits/val_word_mappings.txt"
    test_mapping_file = "dataset/writer_independent_word_splits/test_word_mappings.txt"

    mapping_files = (train_mapping_file, validation_mapping_file)

    # Initialize converter
    label_converter = LabelConverter(train_mapping_file, paths)

    return paths, label_converter, mapping_files, output_dir, test_mapping_file


@logger_hyperparameters_tuning("CNN-LSTM-CTC")
def run_hyperparameter_tuning(fixed_params, params_to_tune=None, num_epochs=5):
    """
    Run hyperparameter tuning experiment

    Args:
        params_to_tune: List of parameter names to tune. Options are:
                       'img_height', 'hidden_size', 'optimizer', 'batch_size', 'learning_rate'
                       If None, tune all parameters.
        num_epochs: Number of epochs to train for each configuration
    """
    if params_to_tune is None:
        params_to_tune = ['img_height', 'hidden_size', 'optimizer', 'batch_size', 'learning_rate']

    paths, label_converter, mapping_files, output_dir, test_mapping_file = setup_environment()

    # Parameters to tune with default values
    img_heights = [16, 32, 64, 96]
    hidden_sizes = [128, 256, 512]
    optimizers = ["Adam", "SGD", "RMSprop"]
    batch_sizes = [4, 8, 16, 32]
    learning_rates = [0.0001, 0.001, 0.01]

    # Store all results
    all_results = {}

    # Tune image height
    if 'img_height' in params_to_tune:
        img_height_results = tune_img_height(
            paths, label_converter, mapping_files, output_dir,
            img_heights, num_epochs, fixed_params=fixed_params
        )
        all_results["img_height"] = img_height_results

    # Tune hidden size
    if 'hidden_size' in params_to_tune:
        hidden_size_results = tune_hidden_size(
            paths, label_converter, mapping_files, output_dir,
            hidden_sizes, num_epochs, fixed_params=fixed_params
        )
        all_results["hidden_size"] = hidden_size_results

    # Tune optimizer
    if 'optimizer' in params_to_tune:
        optimizer_results = tune_optimizer(
            paths, label_converter, mapping_files, output_dir,
            optimizers, num_epochs, fixed_params=fixed_params
        )
        all_results["optimizer"] = optimizer_results

    # Tune batch size
    if 'batch_size' in params_to_tune:
        batch_size_results = tune_batch_size(
            paths, label_converter, mapping_files, output_dir,
            batch_sizes, num_epochs, fixed_params=fixed_params
        )
        all_results["batch_size"] = batch_size_results

    # Tune learning rate
    if 'learning_rate' in params_to_tune:
        learning_rate_results = tune_learning_rate(
            paths, label_converter, mapping_files, output_dir,
            learning_rates, num_epochs, fixed_params=fixed_params
        )
        all_results["learning_rate"] = learning_rate_results

    # Save all results to a single JSON file
    with open(output_dir / "all_hyperparameter_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nAll results saved to {output_dir}")

def evaluate_best_model(best_params=None):
    """
    Evaluate the best model on the test set

    Args:
        best_params: Dictionary with best parameters. If None, use default values.
    """
    if best_params is None:
        best_params = {
            'img_height': 64,
            'n_h': 256,
            'batch_size': 8,
            'optimizer': 'Adam'
        }

    paths, label_converter, mapping_files, output_dir, test_mapping_file = setup_environment()
    train_mapping_file = mapping_files[0]

    # Define model with best parameters
    n_classes = len(label_converter.chars) + 1
    num_channels = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_model(
        img_height=best_params['img_height'],
        num_channels=num_channels,
        n_classes=n_classes,
        n_h=best_params['n_h'],
        device=device
    )

    # Create datasets
    train_dataset = IAMDataset(
        mapping_file=train_mapping_file,
        paths=paths,
        transform=get_simple_transform(best_params['img_height']),
        label_converter=label_converter
    )

    test_dataset = IAMDataset(
        mapping_file=test_mapping_file,
        paths=paths,
        transform=get_simple_transform(best_params['img_height']),
        label_converter=label_converter
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=best_params['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )

    # Create test loader with batch_size=1 for per-example metrics
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Define optimizer and criterion
    optimizer = create_optimizer(model, best_params['optimizer'])
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # Train model for 5 epochs
    print("Training best model on full training set...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,  # Use test set as validation for final evaluation
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        label_converter=label_converter,
        epochs=5
    )

    # Save model
    torch.save(model.state_dict(), output_dir / "best_model.pth")

    # Evaluate on test set with batch_size=1
    print("Evaluating on test set...")
    model.eval()

    test_cers = []
    test_wers = []
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for i, (images, targets, target_lengths, input_lengths) in enumerate(
                tqdm(test_loader, desc="Testing")):
            images = images.to(device)

            outputs = model(images)
            outputs = F.log_softmax(outputs, dim=2)

            pred = greedy_decoder(outputs, label_converter)[0]
            gt = test_loader.dataset.samples[i][1]

            predictions.append(pred)
            ground_truths.append(gt)

            # Calculate metrics for this example
            cer, wer = calculate_metrics([pred], [gt])
            test_cers.append(cer)
            test_wers.append(wer)

    # Calculate average metrics
    avg_cer = sum(test_cers) / len(test_cers)
    avg_wer = sum(test_wers) / len(test_wers)

    # Plot per-example metrics
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(test_cers)
    plt.title(f"CER per Example (Avg: {avg_cer:.4f})")
    plt.xlabel("Example Index")
    plt.ylabel("CER")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(test_wers)
    plt.title(f"WER per Example (Avg: {avg_wer:.4f})")
    plt.xlabel("Example Index")
    plt.ylabel("WER")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "test_metrics_per_example.png")
    plt.close()

    # Save test results
    test_results = {
        "avg_cer": avg_cer,
        "avg_wer": avg_wer,
        "per_example_cer": test_cers,
        "per_example_wer": test_wers,
        "predictions": predictions,
        "ground_truths": ground_truths
    }

    with open(output_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=4)

    # Print summary
    print("\nTest Set Evaluation Results:")
    print(f"Average CER: {avg_cer:.4f}")
    print(f"Average WER: {avg_wer:.4f}")
    print(f"Results saved to {output_dir / 'test_results.json'}")


if __name__ == "__main__":
    # Виберіть, який гіперпараметр налаштовувати
    # Можливі варіанти: 'img_height', 'hidden_size', 'optimizer', 'batch_size'
    # Або можна налаштовувати всі параметри разом, передавши список усіх параметрів

    # Приклад 1: Налаштування тільки висоти зображення
    # run_hyperparameter_tuning(['img_height'], num_epochs=3)

    # Приклад 2: Налаштування тільки розміру прихованого стану
    # run_hyperparameter_tuning(['hidden_size'], num_epochs=3)

    # Приклад 3: Налаштування оптимізатора
    # run_hyperparameter_tuning(['optimizer'], num_epochs=3)

    # Приклад 4: Налаштування розміру батчу
    # run_hyperparameter_tuning(['batch_size'], num_epochs=3)

    # Приклад 5: Налаштування всіх гіперпараметрів (займає більше часу)
    # run_hyperparameter_tuning()

    # Приклад 6: Оцінка моделі з найкращими параметрами
    # evaluate_best_model({
    #     'img_height': 64,
    #     'n_h': 256,
    #     'batch_size': 8,
    #     'optimizer': 'Adam'
    # })

    fixed_params = {
        'img_height': 32,
        'n_h': 256,
        'optimizer': 'Adam',
        'batch_size': 8
    }

    # За замовчуванням запускаємо налаштування тільки оптимізатора
    run_hyperparameter_tuning(fixed_params, ['batch_size'], num_epochs=5)
