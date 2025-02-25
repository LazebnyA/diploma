import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

from project.dataset import ProjectPaths, LabelConverter, IAMDataset, collate_fn
from project.v5.main import CNN_BiLSTM_CTC_V5  # Adjust import as needed
from project.logger import evaluation_logger  # Assuming you have this decorator defined


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


@evaluation_logger(model_description="cnn_lstm_ctc_handwritten_v5_75ep_2-Layered-BiLSTM")
def evaluate():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Initialize paths, transformation, and label converter
    paths = ProjectPaths()
    transform = transforms.Compose([
        transforms.Lambda(lambda img: resize_with_aspect(img)),
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
    img_height = 32
    num_channels = 1
    n_h = 256

    # Initialize the model and load trained weights
    model = CNN_BiLSTM_CTC_V5(img_height, num_channels, n_classes, n_h)
    model_path = "cnn_lstm_ctc_handwritten_v5_75ep_2-Layered-BiLSTM.pth"  # Update as needed
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
    plt.figure()
    plt.plot(range(1, num_batches + 1), batch_losses, marker='o', linestyle='-')
    plt.title('CTC Loss per Batch on Test Set')
    plt.xlabel('Batch Index')
    plt.ylabel('CTC Loss')
    plt.grid(True)
    loss_graph_path = "evaluation_loss_graph.png"
    plt.savefig(loss_graph_path)
    plt.close()
    print(f"Saved loss graph to {loss_graph_path}")

    # Graph 2: Distribution of CER and WER
    plt.figure()
    plt.hist(cer_list, bins=20, alpha=0.5, label='CER')
    plt.hist(wer_list, bins=20, alpha=0.5, label='WER')
    plt.title('Distribution of Error Rates on Test Set')
    plt.xlabel('Error Rate')
    plt.ylabel('Frequency')
    plt.legend()
    error_graph_path = "error_rates_distribution.png"
    plt.savefig(error_graph_path)
    plt.close()
    print(f"Saved error rates distribution graph to {error_graph_path}")

    return {
        "avg_loss": avg_loss,
        "avg_cer": avg_cer,
        "avg_wer": avg_wer,
        "avg_ld": avg_ld,
        "loss_graph": loss_graph_path,
        "error_graph": error_graph_path
    }


if __name__ == "__main__":
    results = evaluate()
    print("Evaluation completed.")
    print(f"Results: {results}")
