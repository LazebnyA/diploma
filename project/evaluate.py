import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys
import json

import seaborn as sns
import pandas as pd
from pathlib import Path

from project.dataset import ProjectPaths, LabelConverter, IAMDataset, collate_fn
from project.utils import execution_time_decorator
from project.logger import evaluation_logger  # Assuming you have this decorator defined
from project.v1.models import CNN_LSTM_CTC_V1_4ConvBlocks
from project.v6.models import CNN_BiLSTM_CTC_V5_3ConvBlocks

# from project.v7.models import CNNBiLSTMResBlocks

MODEL_NAME = "cnn_lstm_ctc_handwritten_v5_75ep_2-Layered-BiLSTM-3-CNN-Blocks"
MODEL_PATH = f"v6/img_augment/{MODEL_NAME}.pth"
OUTPUT_DIR = "v6/img_augment/evaluation_results"
Model = CNN_BiLSTM_CTC_V5_3ConvBlocks

# Hyperparameters
IMG_HEIGHT = 32
NUM_CHANNELS = 1
N_H = 256


def resize_with_aspect(image, target_height=32):
    w, h = image.size
    new_w = int(w * (target_height / h))
    return image.resize((new_w, target_height))


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


def levenshtein_distance(seq1, seq2):
    """
    Compute the Levenshtein distance between two sequences.
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[m][n]


def load_training_history(model_path):
    """
    Loads the training history from a JSON file if available.
    The JSON should contain epoch-wise loss values.
    """
    history_path = Path(model_path).with_suffix('.json')
    if not history_path.exists():
        print(f"Training history not found at {history_path}")
        return None

    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        return history
    except Exception as e:
        print(f"Error loading training history: {e}")
        return None


def plot_loss_per_epoch(history, save_path):
    """
    Plot the training and validation loss per epoch.
    """
    if not history or 'train_loss' not in history:
        print("Training history not available for plotting")
        return

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history['train_loss']) + 1)

    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')

    plt.title('Loss per Epoch', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Add smoothed trend line for training loss
    if len(history['train_loss']) > 5:
        from scipy.signal import savgol_filter
        smooth_loss = savgol_filter(history['train_loss'],
                                    min(11,
                                        len(history['train_loss']) - 2 if len(history['train_loss']) % 2 == 0 else len(
                                            history['train_loss']) - 1),
                                    3)
        plt.plot(epochs, smooth_loss, 'g--', label='Smoothed Training Loss', alpha=0.7)
        plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved loss per epoch graph to {save_path}")


def plot_error_distributions(cer_list, wer_list, save_path):
    """
    Plot improved distribution of CER and WER with better readability.
    """
    plt.figure(figsize=(14, 8))

    # Create subplots for separate error rate displays
    plt.subplot(1, 2, 1)
    n, bins, _ = plt.hist(cer_list, bins=10, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Character Error Rate (CER) Distribution', fontsize=14)
    plt.xlabel('Error Rate', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add mean line
    plt.axvline(np.mean(cer_list), color='red', linestyle='dashed', linewidth=2)
    plt.text(np.mean(cer_list) + 0.01, max(n) / 2,
             f'Mean: {np.mean(cer_list):.4f}',
             color='red', fontsize=12)

    plt.subplot(1, 2, 2)
    n, bins, _ = plt.hist(wer_list, bins=10, alpha=0.7, color='green', edgecolor='black')
    plt.title('Word Error Rate (WER) Distribution', fontsize=14)
    plt.xlabel('Error Rate', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add mean line
    plt.axvline(np.mean(wer_list), color='red', linestyle='dashed', linewidth=2)
    plt.text(np.mean(wer_list) + 0.01, max(n) / 2,
             f'Mean: {np.mean(wer_list):.4f}',
             color='red', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved improved error rates distribution graph to {save_path}")


def plot_error_scatter(cer_list, wer_list, ground_truths, save_path):
    """
    Plot scatter of error rates vs text length to analyze if length affects error rates.
    """
    text_lengths = [len(gt) for gt in ground_truths]

    plt.figure(figsize=(10, 6))
    plt.scatter(text_lengths, cer_list, alpha=0.5, label='CER', color='blue')
    plt.scatter(text_lengths, wer_list, alpha=0.5, label='WER', color='green')

    # Add trend lines
    from scipy import stats
    slope_cer, intercept_cer, r_value_cer, p_value_cer, std_err_cer = stats.linregress(text_lengths, cer_list)
    slope_wer, intercept_wer, r_value_wer, p_value_wer, std_err_wer = stats.linregress(text_lengths, wer_list)

    line_cer = [slope_cer * x + intercept_cer for x in text_lengths]
    line_wer = [slope_wer * x + intercept_wer for x in text_lengths]

    plt.plot(text_lengths, line_cer, 'b--', alpha=0.7, label=f'CER trend (r={r_value_cer:.2f})')
    plt.plot(text_lengths, line_wer, 'g--', alpha=0.7, label=f'WER trend (r={r_value_wer:.2f})')

    plt.title('Error Rates vs Text Length', fontsize=14)
    plt.xlabel('Ground Truth Text Length (chars)', fontsize=12)
    plt.ylabel('Error Rate', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved error vs text length scatter plot to {save_path}")


def plot_error_distributions_boxplot(cer_list, wer_list, save_path):
    """
    Plots a boxplot for CER and WER distributions.

    Args:
        cer_list (list): List of Character Error Rates.
        wer_list (list): List of Word Error Rates.
        save_path (str or Path): Path to save the plot.
    """
    # Convert data into a DataFrame for easier plotting
    data = pd.DataFrame({"CER": cer_list, "WER": wer_list})

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data, palette="Set2", width=0.6)

    plt.title("CER and WER Distributions", fontsize=14)
    plt.ylabel("Error Rate", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved CER/WER boxplot to {save_path}")


@evaluation_logger(model_description=MODEL_NAME)
@execution_time_decorator
def evaluate(output_dir: str):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Initialize paths, transformation, and label converter
    img_height = IMG_HEIGHT

    paths = ProjectPaths()
    transform = transforms.Compose([
        transforms.Lambda(lambda img: resize_with_aspect(img, img_height)),
        transforms.ToTensor()
    ])
    converter_mapping = "dataset/train_word_mappings.txt"
    label_converter = LabelConverter(converter_mapping, paths)

    # Create test dataset and DataLoader (order preserved by shuffle=False)
    test_mapping_file = "dataset/test_word_mappings.txt"
    test_dataset = IAMDataset(
        mapping_file=test_mapping_file,
        paths=paths,
        transform=transform,
        label_converter=label_converter
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Model parameters (should match training settings)
    n_classes = len(label_converter.chars) + 1  # +1 for the CTC blank token

    num_channels = NUM_CHANNELS
    n_h = N_H

    # Initialize the model and load trained weights
    model = Model(img_height, num_channels, n_classes, n_h)
    # Update as needed
    model_path = MODEL_PATH

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Define the CTC loss (used here only to report loss on the test set)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    total_loss = 0.0
    num_batches = 0
    batch_losses = []  # For plotting loss per batch
    all_predictions = []

    with torch.no_grad():
        for images, targets, target_lengths, input_lengths in tqdm(test_loader,
                                                                   desc=f"Evaluating model {model_path}",
                                                                   file=sys.__stdout__):
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            input_lengths = input_lengths.to(device)

            outputs = model(images)  # shape: (T, batch, n_classes)
            outputs = F.log_softmax(outputs, dim=2)

            # Compute the CTC loss for the batch
            loss = criterion(outputs, targets, input_lengths, target_lengths)
            loss_value = loss.item()
            total_loss += loss_value
            num_batches += 1
            batch_losses.append(loss_value)

            # Decode predictions from the model output
            preds = greedy_decoder(outputs, label_converter)
            all_predictions.extend(preds)

    avg_loss = total_loss / num_batches
    print(f"Test CTC Loss: {avg_loss:.4f}")

    # Retrieve ground truth texts in order from the dataset.
    ground_truths = [sample[1] for sample in test_dataset.samples]
    if len(ground_truths) != len(all_predictions):
        print("The number of predictions does not match the number of ground truth texts.")

    total_cer = 0.0
    total_wer = 0.0
    total_ld = 0.0
    n_samples = len(ground_truths)

    # Lists to store per-sample error rates for plotting
    cer_list = []
    wer_list = []
    ld_list = []

    for gt, pred in zip(ground_truths, all_predictions):
        # Compute Levenshtein distance at character level.
        ld = levenshtein_distance(pred, gt)
        total_ld += ld
        cer = ld / max(len(gt), 1)
        total_cer += cer
        cer_list.append(cer)

        # Compute WER by splitting texts into words.
        gt_words = gt.split()
        pred_words = pred.split()
        ld_words = levenshtein_distance(pred_words, gt_words)
        wer = ld_words / max(len(gt_words), 1)
        total_wer += wer
        wer_list.append(wer)
        ld_list.append(ld)

    avg_cer = total_cer / n_samples
    avg_wer = total_wer / n_samples
    avg_ld = total_ld / n_samples

    print("Evaluation Metrics:")
    print(f"Average Levenshtein Distance (character level): {avg_ld:.4f}")
    print(f"Average Character Error Rate (CER): {avg_cer * 100:.2f}%")
    print(f"Average Word Error Rate (WER): {avg_wer * 100:.2f}%")

    # Log a few sample predictions versus ground truths.
    print("Sample Predictions vs. Ground Truths:")
    for i, (pred, gt) in enumerate(zip(all_predictions[:10], ground_truths[:10])):
        print(f"{i + 1}: Prediction: {pred} | Ground Truth: {gt}")

    # --- Graph Generation ---
    # Graph 1: CTC Loss per Batch
    batch_loss_graph_path = output_dir / "batch_loss_graph.png"
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_batches + 1), batch_losses, marker='o', markersize=3, linestyle='-')
    plt.title('CTC Loss per Batch on Test Set', fontsize=14)
    plt.xlabel('Batch Index', fontsize=12)
    plt.ylabel('CTC Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(batch_loss_graph_path, dpi=300)
    plt.close()
    print(f"Saved batch loss graph to {batch_loss_graph_path}")

    # Graph 2: Improved Distribution of CER and WER
    error_graph_path = output_dir / "error_rates_distribution.png"
    plot_error_distributions_boxplot(cer_list, wer_list, error_graph_path)

    # Graph 3: Error Rates vs Text Length
    scatter_graph_path = output_dir / "error_vs_length.png"
    plot_error_scatter(cer_list, wer_list, ground_truths, scatter_graph_path)

    # Graph 4: Loss per Epoch (from training history)
    training_history = load_training_history(model_path)
    if training_history:
        epoch_loss_graph_path = output_dir / "epoch_loss_graph.png"
        plot_loss_per_epoch(training_history, epoch_loss_graph_path)
    else:
        # Create a mock epoch loss graph if history isn't available
        mock_epoch_loss_graph_path = output_dir / "epoch_loss_graph_mock.png"
        plt.figure(figsize=(10, 6))
        plt.title('Loss per Epoch (Mock - No Training History Found)', fontsize=14)
        plt.text(0.5, 0.5,
                 'Training history not available.\nCreate a .json file with the same name as your model '
                 'file\ncontaining "train_loss" and "val_loss" arrays.',
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)
        plt.tight_layout()
        plt.savefig(mock_epoch_loss_graph_path, dpi=300)
        plt.close()
        print(f"Saved mock epoch loss graph to {mock_epoch_loss_graph_path} (training history not found)")

    return {
        "avg_loss": avg_loss,
        "avg_cer": avg_cer,
        "avg_wer": avg_wer,
        "avg_ld": avg_ld,
        "batch_loss_graph": str(batch_loss_graph_path),
        "error_graph": str(error_graph_path),
        "error_vs_length_graph": str(scatter_graph_path),
        "epoch_loss_graph": str(epoch_loss_graph_path) if training_history else str(mock_epoch_loss_graph_path)
    }


if __name__ == "__main__":
    # Save results to JSON
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    results = evaluate(output_dir)

    results_file = output_dir / "evaluation_results.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print("Evaluation completed.")
    print(f"Results saved to {results_file}")
