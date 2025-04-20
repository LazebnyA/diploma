import os

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

from nn.dataset_2 import ProjectPaths, LabelConverter, IAMDataset, collate_fn
from nn.transform import get_otsu_binarization_transform
from nn.utils import execution_time_decorator
from nn.logger import evaluation_logger  # Assuming you have this
from nn.v0.models import CNN_LSTM_CTC_V0, CNN_LSTM_CTC_V2_CNN_more_filters_batch_norm_more_imH

MODEL_NAME = "cnn_lstm_ctc_handwritten_v8_75ep_2-Layered-BiLSTM-ResNet-CNN-Shortcut"
MODEL_PATH = f"archive/v5/better_features/{MODEL_NAME}.pth"
OUTPUT_DIR = "v5/better_features/evaluation_results"
Model = CNN_LSTM_CTC_V0

# Hyperparameters
IMG_HEIGHT = 64
NUM_CHANNELS = 1
N_H = 256
BATCH_SIZE = 1

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


def calculate_individual_metrics(predictions, ground_truths):
    """
    Calculate CER and WER metrics for each individual example.
    Returns lists of CER and WER values.
    """
    from jiwer import wer as calculate_wer
    from jiwer import cer as calculate_cer

    individual_cer = []
    individual_wer = []

    for pred, gt in zip(predictions, ground_truths):
        cer = calculate_cer([gt], [pred])
        wer = calculate_wer([gt], [pred])
        individual_cer.append(cer)
        individual_wer.append(wer)

    return individual_cer, individual_wer


def plot_metrics_distribution(metrics, title, save_path):
    """
    Plot the distribution of metrics (CER or WER) and save the figure.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(metrics, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(metrics), color='red', linestyle='dashed', linewidth=2,
                label=f'Mean: {np.mean(metrics):.4f}')
    plt.axvline(np.median(metrics), color='green', linestyle='dashed', linewidth=2,
                label=f'Median: {np.median(metrics):.4f}')
    plt.title(title)
    plt.xlabel('Error Rate')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_individual_metrics(data_indices, cer_values, wer_values, save_dir):
    """
    Plot CER and WER for each test example and save the figures.
    """
    # Create a dataframe for easier plotting
    df = pd.DataFrame({
        'Index': data_indices,
        'CER': cer_values,
        'WER': wer_values
    })

    # Sort by index
    df = df.sort_values('Index')

    # Plot CER
    plt.figure(figsize=(12, 6))
    plt.plot(df['Index'], df['CER'], marker='o', linestyle='-', alpha=0.7)
    plt.title('Character Error Rate (CER) for Each Test Example')
    plt.xlabel('Example Index')
    plt.ylabel('CER')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'individual_cer.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot WER
    plt.figure(figsize=(12, 6))
    plt.plot(df['Index'], df['WER'], marker='o', linestyle='-', alpha=0.7)
    plt.title('Word Error Rate (WER) for Each Test Example')
    plt.xlabel('Example Index')
    plt.ylabel('WER')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'individual_wer.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_boxplots(cer_values, wer_values, save_dir):
    """
    Create box plots for CER and WER distributions.
    """
    # Prepare data_preprocessed
    data = pd.DataFrame({
        'CER': cer_values,
        'WER': wer_values
    })

    # Plot boxplots
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data)
    plt.title('Distribution of CER and WER')
    plt.ylabel('Error Rate')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'cer_wer_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Individual boxplots with more details
    for metric in ['CER', 'WER']:
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=data[metric])
        sns.stripplot(y=data[metric], color='black', alpha=0.5, size=4)
        plt.title(f'Distribution of {metric}')
        plt.ylabel('Error Rate')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f'{metric.lower()}_boxplot.png'), dpi=300, bbox_inches='tight')
        plt.close()


def plot_metrics_by_word_length(ground_truths, cer_values, wer_values, save_dir):
    """
    Plot CER and WER distributions grouped by word length.
    """
    # Calculate word lengths
    word_lengths = [len(gt) for gt in ground_truths]

    # Create dataframe with word lengths and metrics
    df = pd.DataFrame({
        'Word_Length': word_lengths,
        'CER': cer_values,
        'WER': wer_values
    })

    # Group by word length and calculate mean metrics
    grouped_df = df.groupby('Word_Length').agg({
        'CER': ['mean', 'count'],
        'WER': ['mean', 'count']
    }).reset_index()

    # Flatten column names
    grouped_df.columns = ['Word_Length', 'CER_Mean', 'CER_Count', 'WER_Mean', 'WER_Count']

    # Filter out word lengths with too few examples (less than 5)
    grouped_df = grouped_df[grouped_df['CER_Count'] >= 5]

    # Plot mean CER by word length
    plt.figure(figsize=(12, 6))
    plt.bar(grouped_df['Word_Length'], grouped_df['CER_Mean'], alpha=0.7, color='steelblue')
    plt.title('Mean Character Error Rate (CER) by Word Length')
    plt.xlabel('Word Length (characters)')
    plt.ylabel('Mean CER')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(grouped_df['Word_Length'])

    # Add count labels above bars
    for i, (length, mean_cer, count) in enumerate(zip(grouped_df['Word_Length'],
                                                      grouped_df['CER_Mean'],
                                                      grouped_df['CER_Count'])):
        plt.text(length, mean_cer + 0.02, f'n={count}', ha='center')

    plt.savefig(os.path.join(save_dir, 'cer_by_word_length.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot mean WER by word length
    plt.figure(figsize=(12, 6))
    plt.bar(grouped_df['Word_Length'], grouped_df['WER_Mean'], alpha=0.7, color='darkorange')
    plt.title('Mean Word Error Rate (WER) by Word Length')
    plt.xlabel('Word Length (characters)')
    plt.ylabel('Mean WER')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(grouped_df['Word_Length'])

    # Add count labels above bars
    for i, (length, mean_wer, count) in enumerate(zip(grouped_df['Word_Length'],
                                                      grouped_df['WER_Mean'],
                                                      grouped_df['WER_Count'])):
        plt.text(length, mean_wer + 0.02, f'n={count}', ha='center')

    plt.savefig(os.path.join(save_dir, 'wer_by_word_length.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create detailed histograms for word length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(word_lengths, bins=range(min(word_lengths), max(word_lengths) + 2),
             alpha=0.7, color='teal', edgecolor='black')
    plt.title('Distribution of Word Lengths in Test Set')
    plt.xlabel('Word Length (characters)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(range(min(word_lengths), max(word_lengths) + 1))
    plt.savefig(os.path.join(save_dir, 'word_length_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create heatmap plots for CER and WER by word length
    for metric_name, metric_values in [('CER', cer_values), ('WER', wer_values)]:
        # Create bins for word lengths
        length_bins = range(1, max(word_lengths) + 2)

        # Create bins for metric values (0.0 to 1.0 with 0.1 steps)
        metric_bins = np.linspace(0, 1, 11)

        # Create 2D histogram
        hist, x_edges, y_edges = np.histogram2d(
            word_lengths,
            metric_values,
            bins=[length_bins, metric_bins]
        )

        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(hist.T, cmap='viridis',
                    xticklabels=length_bins[:-1],
                    yticklabels=[f'{b:.1f}' for b in metric_bins[:-1]],
                    cbar_kws={'label': 'Frequency'})
        plt.title(f'Distribution of {metric_name} by Word Length')
        plt.xlabel('Word Length (characters)')
        plt.ylabel(f'{metric_name} Value')
        plt.savefig(os.path.join(save_dir, f'{metric_name.lower()}_word_length_heatmap.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Create box plots of CER and WER grouped by word length
    for metric_name, column_name in [('Character Error Rate', 'CER'), ('Word Error Rate', 'WER')]:
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='Word_Length', y=column_name, data=df)
        plt.title(f'{metric_name} Distribution by Word Length')
        plt.xlabel('Word Length (characters)')
        plt.ylabel(metric_name)
        plt.grid(True, alpha=0.3, axis='y')

        # Add sample count below x-axis labels
        counts = df['Word_Length'].value_counts().sort_index()
        ticks = plt.gca().get_xticks()
        labels = [item.get_text() for item in plt.gca().get_xticklabels()]
        new_labels = []

        for label in labels:
            if label in counts.index.astype(str):
                count = counts[int(label)]
                new_labels.append(f'{label}\nn={count}')
            else:
                new_labels.append(label)

        plt.gca().set_xticklabels(new_labels)

        plt.savefig(os.path.join(save_dir, f'{column_name.lower()}_boxplot_by_length.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


@evaluation_logger(model_description=MODEL_NAME)
@execution_time_decorator
def evaluate():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Initialize paths, transformation, and label converter
    img_height = IMG_HEIGHT
    batch_size = BATCH_SIZE
    num_channels = NUM_CHANNELS
    n_h = N_H

    paths = ProjectPaths()

    # ===== TEST SET EVALUATION =====
    print("\n===== Starting Test Set Evaluation =====")

    # Create evaluation results directory if it doesn't exist
    eval_dir = "v0/evaluation_results"
    os.makedirs(eval_dir, exist_ok=True)

    # Load test dataset
    test_mapping_file = "tests/test_self_wr_base/dataset/dataset.txt"

    label_converter = LabelConverter(test_mapping_file, paths)
    n_classes = 80

    test_dataset = IAMDataset(
        mapping_file=test_mapping_file,
        paths=paths,
        transform=get_otsu_binarization_transform(),
        label_converter=label_converter
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = CNN_LSTM_CTC_V2_CNN_more_filters_batch_norm_more_imH(
        img_height=img_height,
        num_channels=num_channels,
        n_classes=n_classes,
        n_h=n_h
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Device: {device}")

    weights_path = ("./cnn_lstm_ctc_handwritten_v0_word_22ep_CNN-BiLSTM-CTC_CNN-VGG16_BiLSTM-1dim"
                    ".pth")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Loaded initial random weights from {weights_path}")

    # Ensure the model is in evaluation mode
    model.eval()

    # Lists to store individual predictions and ground truths
    all_predictions = []
    all_ground_truths = []
    all_indices = []

    # Current index in the dataset
    current_idx = 0

    with torch.no_grad():
        for images, targets, target_lengths, input_lengths in tqdm(test_loader,
                                                                   desc="Testing Model",
                                                                   file=sys.__stdout__):
            images, targets = images.to(device), targets.to(device)
            target_lengths = target_lengths.to(device)
            input_lengths = input_lengths.to(device)

            # Forward pass
            outputs = model(images)
            outputs = F.log_softmax(outputs, dim=2)

            # Get predictions
            preds = greedy_decoder(outputs, label_converter)
            all_predictions.extend(preds)

            # Get ground truths
            batch_ground_truths = []
            start = 0
            for length in target_lengths:
                gt = targets[start:start + length].cpu().tolist()
                decoded = label_converter.decode(gt)
                batch_ground_truths.append(decoded)
                start += length

            all_ground_truths.extend(batch_ground_truths)

            # Store indices for each example
            batch_indices = list(range(current_idx, current_idx + len(preds)))
            all_indices.extend(batch_indices)
            current_idx += len(preds)

    # Calculate overall metrics
    overall_cer, overall_wer = calculate_metrics(all_predictions, all_ground_truths)
    print(f"\nTest Set Overall Metrics:")
    print(f"CER: {overall_cer:.4f}, WER: {overall_wer:.4f}")

    # Calculate individual metrics
    individual_cer, individual_wer = calculate_individual_metrics(all_predictions, all_ground_truths)

    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Index': all_indices,
        'Prediction': all_predictions,
        'Ground_Truth': all_ground_truths,
        'CER': individual_cer,
        'WER': individual_wer
    })

    metrics_df.to_csv(os.path.join(eval_dir, 'test_metrics.csv'), index=False)

    # Save overall metrics
    overall_metrics = {
        'overall_cer': overall_cer,
        'overall_wer': overall_wer,
        'mean_cer': np.mean(individual_cer),
        'mean_wer': np.mean(individual_wer),
        'median_cer': np.median(individual_cer),
        'median_wer': np.median(individual_wer),
        'min_cer': np.min(individual_cer),
        'min_wer': np.min(individual_wer),
        'max_cer': np.max(individual_cer),
        'max_wer': np.max(individual_wer),
        'std_cer': np.std(individual_cer),
        'std_wer': np.std(individual_wer)
    }

    with open(os.path.join(eval_dir, 'overall_metrics.json'), 'w') as f:
        json.dump(overall_metrics, f, indent=4)

    # Generate plots
    # 1. Histogram of CER and WER
    plot_metrics_distribution(individual_cer, 'Character Error Rate (CER) Distribution',
                              os.path.join(eval_dir, 'cer_distribution.png'))
    plot_metrics_distribution(individual_wer, 'Word Error Rate (WER) Distribution',
                              os.path.join(eval_dir, 'wer_distribution.png'))

    # 2. Individual metrics for each example
    plot_individual_metrics(all_indices, individual_cer, individual_wer, eval_dir)

    # 3. Box plots
    plot_boxplots(individual_cer, individual_wer, eval_dir)

    # 4. Word length analysis
    plot_metrics_by_word_length(all_ground_truths, individual_cer, individual_wer, eval_dir)

    print(f"\nEvaluation complete. Results saved to {eval_dir}/")


if __name__ == '__main__':
    evaluate()
