import torch
import torch.nn as nn


class CNN_LSTM_CTC_V2_CNN_more_filters_batch_norm_deeper_vgg16like(nn.Module):
    def __init__(self, img_height, num_channels, n_classes, n_h):
        """
        img_height: image height (after resize)
        num_channels: number of input channels (1 for grayscale)
        n_classes: number of output classes (vocab size + 1 for blank)
        n_h: number of hidden units in the LSTM

        Adjustments:
            - added LeakyReLu
            - VGG16 like architecture, but with batch normalization
        """
        super().__init__()

        # CNN Module
        self.cnn = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Downsample (H/2, W/2)

            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Downsample (H/4, W/4)

            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # Downsample height only (H/8, W/4)

            # Conv Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d((2, 1))  # Downsample height only (H/16, W/4)
        )

        # After CNN, height becomes img_height // 16
        self.lstm_input_size = 512 * (img_height // 16)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(self.lstm_input_size, n_h, bidirectional=True, num_layers=1, batch_first=True)

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


import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Базовий резидуальний блок згідно з оригінальним документом ResNet 2015 року,
    реалізований з використанням nn.Sequential для прямої частини.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Пряма послідовність з використанням nn.Sequential
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Shortcut connection
        self.shortcut = nn.Identity()  # Default is identity mapping
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv_path(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_BiLSTM_CTC(nn.Module):
    """
    Оригінальна архітектура ResNet, адаптована для HTR з CNN-BiLSTM-CTC,
    з використанням nn.Sequential для CNN частини.
    """

    def __init__(self, img_height, num_channels, n_classes, n_h=256):
        """
        img_height: висота вхідного зображення
        num_channels: кількість вхідних каналів (1 для чорно-білого)
        n_classes: кількість класів на виході (розмір словника + 1 для CTC blank)
        n_h: кількість прихованих нейронів у LSTM
        """
        super(ResNet_BiLSTM_CTC, self).__init__()

        # CNN частина реалізована з використанням nn.Sequential
        self.cnn = nn.Sequential(
            # Початковий шар
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Шар 1
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),  # Зменшуємо тільки висоту

            # Шар 2
            ResidualBlock(64, 128, stride=1),
            ResidualBlock(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),  # Зменшуємо тільки висоту

            # Шар 3
            ResidualBlock(128, 256, stride=1),
            ResidualBlock(256, 256),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Зменшуємо тільки висоту

            # Шар 4
            ResidualBlock(256, 512, stride=1),
            ResidualBlock(512, 512),
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Зменшуємо тільки висоту
        )

        # Розрахунок розміру входу для LSTM
        # Висота зменшується в 16 разів (2*2*2*2), а ширина в 4 рази (2*2 від conv1 та maxpool4)
        self.lstm_input_size = 512 * (img_height // 16)

        # BiLSTM шар
        self.lstm = nn.LSTM(
            self.lstm_input_size,
            n_h,
            bidirectional=True,
            batch_first=True
        )

        # Вихідний класифікатор
        self.fc = nn.Linear(n_h * 2, n_classes)

    def forward(self, x):
        # CNN обробка
        x = self.cnn(x)

        # Підготовка для LSTM
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2)  # [batch, width, channels, height]
        x = x.contiguous().view(batch_size, width, channels * height)  # [batch, width, features]

        # BiLSTM
        x, _ = self.lstm(x)

        # Класифікатор
        x = self.fc(x)  # [batch, width, n_classes]

        # Транспонуємо для CTC Loss: [width, batch, n_classes]
        x = x.permute(1, 0, 2)

        return x


# Додавання додаткових компонентів для глибших моделей
def make_layer(block, in_channels, out_channels, num_blocks, stride=1):
    """
    Допоміжна функція для створення шару з декількох резидуальних блоків.
    """
    layers = []
    # Перший блок може мати stride відмінний від 1
    layers.append(block(in_channels, out_channels, stride))

    # Решта блоків мають stride=1
    for _ in range(1, num_blocks):
        layers.append(block(out_channels, out_channels))

    return nn.Sequential(*layers)


def resnet18_htr_sequential(img_height, num_channels=1, n_classes=80, n_h=256):
    """
    Створює ResNet18 для HTR з послідовною CNN архітектурою.
    """
    model = ResNet_BiLSTM_CTC(img_height, num_channels, n_classes, n_h)
    return model


def resnet34_htr_sequential(img_height, num_channels=1, n_classes=100, n_h=256):
    """
    Створює глибшу версію ResNet34 для HTR з послідовною CNN архітектурою.
    """
    # Створюємо базову модель
    model = ResNet_BiLSTM_CTC(img_height, num_channels, n_classes, n_h)

    # Замінюємо CNN частину на глибшу версію
    model.cnn = nn.Sequential(
        # Початковий шар
        nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1),

        # Шари з більшою кількістю блоків
        *make_layer(ResidualBlock, 64, 64, 3),
        *make_layer(ResidualBlock, 64, 128, 4, stride=1),
        nn.MaxPool2d(kernel_size=2, stride=(2, 1)),

        *make_layer(ResidualBlock, 128, 256, 6, stride=1),
        nn.MaxPool2d(kernel_size=2, stride=(2, 1)),

        *make_layer(ResidualBlock, 256, 512, 3, stride=1),
        nn.MaxPool2d(kernel_size=2, stride=(2, 2))
    )

    return model

# Приклад використання:
# model = resnet18_htr_sequential(img_height=64, num_channels=1, n_classes=100)
# або
# model = resnet34_htr_sequential(img_height=64, num_channels=1, n_classes=100)