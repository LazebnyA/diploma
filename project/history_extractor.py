import re
import json
import matplotlib.pyplot as plt
from pathlib import Path


def extract_training_history(log_text):
    pattern = r"Epoch (\d+) completed\. Training Loss: ([\d\.]+), Validation Loss: ([\d\.]+)"
    matches = re.findall(pattern, log_text)

    history = {"train_loss": [], "val_loss": []}
    for epoch, train_loss, val_loss in matches:
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

    return history


def save_history_to_json(history, model_name):
    output_path = Path(f"{model_name}.json")
    with open(output_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"History saved to {output_path}")
    return output_path


def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.plot(epochs, history["train_loss"], 'b-', label='Training Loss')
    plt.plot(epochs, history["val_loss"], 'r-', label='Validation Loss')
    plt.title('Loss per Epoch', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    output_path = "training_history_plot.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved to {output_path}")
    return output_path


def analyze_training_trends(history):
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    min_train_epoch = train_loss.index(min(train_loss)) + 1
    min_val_epoch = val_loss.index(min(val_loss)) + 1
    print("\n===== Training Analysis =====")
    print(f"Lowest training loss: {min(train_loss):.4f} at epoch {min_train_epoch}")
    print(f"Lowest validation loss: {min(val_loss):.4f} at epoch {min_val_epoch}")
    return {
        "min_train_loss": min(train_loss),
        "min_train_epoch": min_train_epoch,
        "min_val_loss": min(val_loss),
        "min_val_epoch": min_val_epoch
    }


if __name__ == "__main__":
    log_file_path = "v5/cnn_lstm_ctc_handwritten_v5_75ep_2-Layered-BiLSTM.txt"  # Change this path if needed

    if Path(log_file_path).is_file():
        with open(log_file_path, 'r') as file:
            log_text = file.read()
    else:
        print(f"Error: File '{log_file_path}' not found.")
        exit(1)

    history = extract_training_history(log_text)
    json_path = save_history_to_json(history, "cnn_lstm_ctc_handwritten_v5_75ep_2-Layered-BiLSTM")
    plot_path = plot_training_history(history)
    analysis = analyze_training_trends(history)

    print(f"\nJSON file created at: {json_path}")
    print(f"Preview plot created at: {plot_path}")
