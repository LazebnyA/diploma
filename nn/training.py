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
from nn.transform import get_training_transform, get_validation_transform
from nn.utils import greedy_decoder, calculate_metrics
from nn.v0.models import CNN_LSTM_CTC_V0
from nn.v2.models import resnet18_htr_sequential_v2

torch.manual_seed(42)

# Не змінювані
IMG_HEIGHT = 64
NUM_CHANNELS = 1

# Змінювані
BATCH_SIZE = 16
N_H = 1024
OUT_CHANNELS = 64
LSTM_LAYERS = 1

NUM_EPOCHS = 100
OPTIMIZER = optim.RMSprop


def initialize_paths_and_converter(mapping_file):
    """Ініціалізує шляхи проекту та конвертер міток."""
    paths = ProjectPaths()
    label_converter = LabelConverter(mapping_file, paths)
    return paths, label_converter


def create_datasets_and_loaders(mapping_file, validation_mapping_file, paths, label_converter, batch_size):
    """Створює набори даних та завантажувачі для тренування та валідації."""
    training_transform = get_training_transform()

    # Тренувальний набір
    dataset = IAMDataset(
        mapping_file=mapping_file,
        paths=paths,
        transform=training_transform,
        label_converter=label_converter
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    validation_transform = get_validation_transform()

    # Валідаційний набір
    validation_dataset = IAMDataset(
        mapping_file=validation_mapping_file,
        paths=paths,
        transform=validation_transform,
        label_converter=label_converter
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return dataloader, validation_loader


def initialize_model(img_height, num_channels, n_classes, n_h, out_channels, lstm_layers, device):
    """Ініціалізує модель та переміщує її на відповідний пристрій."""
    model = resnet18_htr_sequential_v2(
        img_height=img_height,
        num_channels=num_channels,
        n_classes=n_classes,
        n_h=n_h,
        out_channels=out_channels,
        lstm_layers=lstm_layers
    )
    model.to(device)
    return model


def setup_training(model, device, additional):
    """Налаштовує тренування: зберігає початкові ваги, створює оптимізатор, тощо."""
    # Збереження початкових ваг
    base_filename = f"{additional}_initial_weights"
    model_filename = f"{base_filename}.pth"
    torch.save(model.state_dict(), model_filename)

    # Створення функції втрат та оптимізатора
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Перевірка наявності чекпоінта
    start_epoch = 0
    checkpoint_path = 'checkpoint.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming training from checkpoint at epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    return criterion, optimizer, start_epoch


def print_model_info(model, img_height, num_channels, n_classes, n_h, optimizer, lr, criterion, num_epochs, batch_size):
    """Виводить інформацію про модель та гіперпараметри."""
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
        "transform": "Resize with aspect ratio. Data Preprocessing + Augmentation",
        "dataset": "IAM Lines Dataset (writer-independent split). Cleaned dataset"
    }

    print("\nHyperparameters:")
    for key, value in hyperparams.items():
        print(f"{key}: {value}")

    print("Starting training process: \n")


def train_epoch(model, dataloader, optimizer, criterion, device, label_converter):
    """Навчання моделі протягом одного епоху."""
    model.train()
    train_loss = 0
    train_predictions = []
    train_ground_truths = []

    for images, targets, target_lengths, input_lengths in tqdm(dataloader,
                                                               desc="Training",
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

        # Отримання прогнозів для розрахунку метрик
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

    return avg_train_loss, train_cer, train_wer


def validate_model(model, validation_loader, criterion, device, label_converter):
    """Проводить валідацію моделі."""
    model.eval()
    val_loss = 0
    val_predictions = []
    val_ground_truths = []

    with torch.no_grad():
        for images, targets, target_lengths, input_lengths in tqdm(validation_loader,
                                                                   desc="Validation",
                                                                   file=sys.__stdout__):
            images, targets = images.to(device), targets.to(device)
            target_lengths = target_lengths.to(device)
            input_lengths = input_lengths.to(device)

            outputs = model(images)
            outputs = F.log_softmax(outputs, dim=2)

            loss = criterion(outputs, targets, input_lengths, target_lengths)
            val_loss += loss.item()

            # Отримання прогнозів для розрахунку метрик
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

    return avg_val_loss, val_cer, val_wer


def check_early_stopping(val_cer, best_val_cer, no_improvement_count, patience, epoch, model, version, additional,
                         optimizer, avg_train_loss):
    """Перевіряє умови для раннього зупинення та зберігає найкращу модель."""
    early_stopping = False
    best_epoch = None

    if val_cer < best_val_cer:
        best_val_cer = val_cer
        best_epoch = epoch
        no_improvement_count = 0

        # Зберігаємо найкращу модель
        best_model_filename = f"cnn_lstm_ctc_handwritten_v{version}_word_best_{additional}.pth"
        torch.save(model.state_dict(), best_model_filename)

        # Зберігаємо чекпоінт
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
        print(f"\nEarly stopping triggered! No improvement in validation CER for {patience} consecutive epochs.")
        print(f"Best validation CER: {best_val_cer:.4f} achieved at epoch {best_epoch + 1}")
        early_stopping = True

    return best_val_cer, best_epoch, no_improvement_count, early_stopping


def save_results(model, training_history, completed_epochs, best_epoch, early_stopping, version, additional,
                 best_val_cer):
    """Зберігає результати тренування: історію та модель."""
    # Зберігаємо історію тренування
    base_filename = f"cnn_lstm_ctc_handwritten_v{version}_word_{completed_epochs}ep_{additional}"
    history_file = f"{base_filename}.json"
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=4)

    # Зберігаємо модель
    model_filename = f"{base_filename}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as {model_filename}")

    # Завантажуємо найкращу модель, якщо було раннє зупинення
    if early_stopping and best_epoch < completed_epochs - 1:
        best_model_filename = f"cnn_lstm_ctc_handwritten_v{version}_word_best_{additional}.pth"
        model.load_state_dict(torch.load(best_model_filename))
        print(f"Loaded best model from epoch {best_epoch + 1}")

    # Формуємо результати
    result = {
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch + 1 if best_epoch is not None else None,
        "best_val_cer": best_val_cer if best_val_cer != float('inf') else None,
        "early_stopping_triggered": early_stopping
    }

    return result


@logger_model_training(version="0", additional="CNN-BiLSTM-CTC_V0")
def main(version, additional):
    # Ініціалізація шляхів та конвертера
    mapping_file = "dataset/writer_independent_word_splits/preprocessed/train_word_mappings.txt"
    validation_mapping_file = "dataset/writer_independent_word_splits/preprocessed/val_word_mappings.txt"
    paths, label_converter = initialize_paths_and_converter(mapping_file)

    # Створення наборів даних та завантажувачів
    dataloader, validation_loader = create_datasets_and_loaders(
        mapping_file, validation_mapping_file, paths, label_converter, BATCH_SIZE
    )

    # Налаштування параметрів моделі
    n_classes = len(label_converter.chars) + 1  # +1 для CTC blank символу
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Ініціалізація моделі
    model = initialize_model(
        img_height=IMG_HEIGHT,
        num_channels=NUM_CHANNELS,
        n_classes=n_classes,
        n_h=N_H,
        out_channels=OUT_CHANNELS,
        lstm_layers=LSTM_LAYERS,
        device=device
    )

    # Налаштування тренування
    criterion, optimizer, start_epoch = setup_training(model, device, additional)

    # Виведення інформації про модель
    print_model_info(
        model, IMG_HEIGHT, NUM_CHANNELS, n_classes, N_H,
        optimizer, 0.0001, criterion, NUM_EPOCHS, BATCH_SIZE
    )

    # Ініціалізація історії тренування
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_cer': [],
        'val_cer': [],
        'train_wer': [],
        'val_wer': []
    }

    # Параметри для раннього зупинення
    patience = 10
    no_improvement_count = 0
    best_val_cer = float('inf')
    best_epoch = 0
    early_stopping = False

    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            # Тренування на одному епоху
            avg_train_loss, train_cer, train_wer = train_epoch(
                model, dataloader, optimizer, criterion, device, label_converter
            )

            # Валідація моделі
            avg_val_loss, val_cer, val_wer = validate_model(
                model, validation_loader, criterion, device, label_converter
            )

            # Оновлення історії тренування
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(avg_val_loss)
            training_history['train_cer'].append(train_cer)
            training_history['val_cer'].append(val_cer)
            training_history['train_wer'].append(train_wer)
            training_history['val_wer'].append(val_wer)

            # Виведення результатів епоху
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"Training - Loss: {avg_train_loss:.4f}, CER: {train_cer:.4f}, WER: {train_wer:.4f}")
            print(f"Validation - Loss: {avg_val_loss:.4f}, CER: {val_cer:.4f}, WER: {val_wer:.4f}")

            # Перевірка раннього зупинення
            best_val_cer, best_epoch, no_improvement_count, should_stop = check_early_stopping(
                val_cer, best_val_cer, no_improvement_count, patience, epoch,
                model, version, additional, optimizer, avg_train_loss
            )

            if should_stop:
                early_stopping = True
                break

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # Збереження результатів
        completed_epochs = epoch + 1 if 'epoch' in locals() else 0
        return save_results(
            model, training_history, completed_epochs, best_epoch,
            early_stopping, version, additional, best_val_cer
        )


if __name__ == '__main__':
    main()