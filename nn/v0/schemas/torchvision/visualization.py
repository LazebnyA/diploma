import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import io
import os


class CNN_LSTM_CTC_V0(nn.Module):
    def __init__(self, img_height, num_channels, n_classes, n_h):
        """
        img_height: image height (after resize)
        num_channels: number of input channels (1 for grayscale)
        n_classes: number of output classes (vocab size + 1 for blank)
        n_h: number of hidden units in the LSTM
        """
        super().__init__()

        # CNN Module
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # downsample height & width by 2
            nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)),  # downsample height by 4
            nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1))  # downsample height by 8
        )

        # After CNN, height becomes img_height // 8
        self.lstm_input_size = 96 * (img_height // 8)

        # One directional LSTM
        self.lstm = nn.LSTM(self.lstm_input_size, n_h, num_layers=1, batch_first=True, bidirectional=True)

        # Final classifier: maps LSTM output to character classes
        self.fc = nn.Linear(2 * n_h, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: input images with shape (batch, channels, height, width)
        Returns output in shape (T, batch, n_classes) required by CTCLoss
        """

        conv = self.cnn(x)  # shape: (batch, 128, H', W')
        b, c, h, w = conv.size()

        # Permute and flatten so that each column (width) is a time step
        conv = conv.permute(0, 3, 1, 2)  # now (batch, width, channels, height)
        conv = conv.contiguous().view(b, w, c * h)  # (batch, width, feature_size)

        # LSTM expects input shape (batch, seq_len, input_size)
        recurrent, _ = self.lstm(conv)  # (batch, seq_len, n_h)
        output = self.fc(recurrent)  # (batch, seq_len, n_classes)

        # For CTCLoss, we need (seq_len, batch, n_classes)
        output = output.permute(1, 0, 2)
        return output


def visualize_model_flow(model, img_height, num_channels):
    """Візуалізує потік даних через модель з використанням matplotlib"""
    plt.figure(figsize=(15, 10))

    # Вхідний тензор
    x = torch.randn(2, num_channels, img_height, 100)

    # Відстеження форм тензорів на кожному етапі
    shapes = []
    shapes.append(("Input", x.shape))

    # CNN
    conv_output = model.cnn(x)
    shapes.append(("CNN Output", conv_output.shape))

    # Перестановка і зміна форми
    b, c, h, w = conv_output.shape
    reshaped = conv_output.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
    shapes.append(("Reshaped for LSTM", reshaped.shape))

    # LSTM
    lstm_output, _ = model.lstm(reshaped)
    shapes.append(("LSTM Output", lstm_output.shape))

    # Повнозв'язний шар
    fc_output = model.fc(lstm_output)
    shapes.append(("FC Output", fc_output.shape))

    # Кінцевий вихід
    final_output = fc_output.permute(1, 0, 2)
    shapes.append(("Final Output", final_output.shape))

    # Візуалізація потоку даних
    plt.subplot(1, 1, 1)

    # Налаштування для діаграми
    y_positions = range(len(shapes))
    labels = [s[0] for s in shapes]
    shapes_text = [str(s[1]) for s in shapes]

    # Стрілки з'єднання
    for i in range(len(shapes) - 1):
        plt.annotate('', xy=(0.5, i + 0.8), xytext=(0.5, i + 0.2),
                     arrowprops=dict(arrowstyle='->', lw=2, color='blue'))

    # Текстові блоки
    for i, (label, shape) in enumerate(zip(labels, shapes_text)):
        plt.text(0.5, i + 0.5, f"{label}\n{shape}",
                 ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

    plt.ylim(len(shapes), 0)  # Перевернути вісь Y для кращого вигляду потоку
    plt.axis('off')
    plt.tight_layout()

    plt.savefig('cnn_lstm_ctc_data_flow.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Візуалізацію збережено у файл 'cnn_lstm_ctc_data_flow.png'")


def visualize_model_architecture(model):
    """Створює текстову візуалізацію архітектури моделі"""
    # Верхній колонтитул
    print("\n" + "=" * 80)
    print("CNN_LSTM_CTC_V0 АРХІТЕКТУРА МОДЕЛІ".center(80))
    print("=" * 80)

    # Компонент CNN
    print("\nКОМПОНЕНТ CNN:")
    print("-" * 80)
    for i, layer in enumerate(model.cnn):
        print(f"Шар {i + 1}: {layer}")

    # Компонент LSTM
    print("\nКОМПОНЕНТ LSTM:")
    print("-" * 80)
    print(f"LSTM: {model.lstm}")

    # Компонент FC
    print("\nКОМПОНЕНТ ПОВНОЗВ'ЯЗНОГО ШАРУ:")
    print("-" * 80)
    print(f"FC: {model.fc}")

    # Інформація про параметри
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nІНФОРМАЦІЯ ПРО ПАРАМЕТРИ:")
    print("-" * 80)
    print(f"Загальна кількість параметрів: {total_params:,}")
    print(f"Кількість параметрів для навчання: {trainable_params:,}")

    # Обчислення параметрів по компонентах
    cnn_params = sum(p.numel() for p in model.cnn.parameters())
    lstm_params = sum(p.numel() for p in model.lstm.parameters())
    fc_params = sum(p.numel() for p in model.fc.parameters())

    print(f"\nПараметри CNN: {cnn_params:,} ({cnn_params / total_params * 100:.1f}%)")
    print(f"Параметри LSTM: {lstm_params:,} ({lstm_params / total_params * 100:.1f}%)")
    print(f"Параметри FC: {fc_params:,} ({fc_params / total_params * 100:.1f}%)")

    print("\n" + "=" * 80)


def visualize_model():
    # Параметри моделі
    img_height = 32
    num_channels = 1
    n_classes = 95  # ASCII visible chars + blank
    n_h = 256

    # Створення моделі
    model = CNN_LSTM_CTC_V0(img_height, num_channels, n_classes, n_h)

    # Виведення опису моделі
    print("\nМодель CNN_LSTM_CTC_V0:")
    print(model)

    # Спробувати використати TensorBoard для більш детальної візуалізації
    try:
        dummy_input = torch.randn(2, num_channels, img_height, 100)
        writer = SummaryWriter('runs/cnn_lstm_ctc_model')
        writer.add_graph(model, dummy_input)
        print("TensorBoard graph created. Run 'tensorboard --logdir=runs' and open http://localhost:6006 to view")
    except Exception as e:
        print(f"Не вдалося створити граф TensorBoard: {e}")
        print("Продовжуємо з альтернативними методами візуалізації...")

    # Підрахунок параметрів моделі
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nЗагальна кількість параметрів: {total_params:,}")
    print(f"Кількість параметрів для навчання: {trainable_params:,}")

    # Виведення форми вхідних і вихідних тензорів на кожному етапі
    print("\nФорми тензорів у моделі:")

    # Вхідний тензор
    x = torch.randn(2, num_channels, img_height, 100)
    print(f"Вхідне зображення: {x.shape}")

    # CNN
    conv_output = model.cnn(x)
    print(f"Після CNN: {conv_output.shape}")

    # Перестановка і зміна форми
    b, c, h, w = conv_output.shape
    reshaped = conv_output.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
    print(f"Після перестановки і зміни форми: {reshaped.shape}")

    # LSTM
    lstm_output, _ = model.lstm(reshaped)
    print(f"Після LSTM: {lstm_output.shape}")

    # Повнозв'язний шар
    fc_output = model.fc(lstm_output)
    print(f"Після повнозв'язного шару: {fc_output.shape}")

    # Кінцевий вихід
    final_output = fc_output.permute(1, 0, 2)
    print(f"Кінцевий вихід: {final_output.shape}")

    # Візуалізація потоку даних через модель
    try:
        visualize_model_flow(model, img_height, num_channels)
    except Exception as e:
        print(f"Не вдалося створити візуалізацію потоку даних: {e}")

    # Візуалізація архітектури моделі в текстовому форматі
    visualize_model_architecture(model)


if __name__ == "__main__":
    visualize_model()