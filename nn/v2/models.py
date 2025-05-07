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

    def __init__(self, img_height, num_channels, n_classes, n_h=256, out_channels=24, lstm_layers=2):
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
            nn.Conv2d(num_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # Шар 1
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),  # Зменшуємо тільки висоту

            # Шар 2
            ResidualBlock(out_channels, 2 * out_channels, stride=1),
            ResidualBlock(2 * out_channels, 2 * out_channels),
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),  # Зменшуємо тільки висоту

            # Шар 3
            ResidualBlock(2 * out_channels, 4 * out_channels, stride=1),
            ResidualBlock(4 * out_channels, 4 * out_channels),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Зменшуємо тільки висоту

            # Шар 4
            ResidualBlock(4 * out_channels, 8 * out_channels, stride=1),
            ResidualBlock(8 * out_channels, 8 * out_channels),
            nn.Identity()
        )

        # Розрахунок розміру входу для LSTM
        # Висота зменшується в 16 разів (2*2*2*2), а ширина в 4 рази (2*2 від conv1 та maxpool4)
        self.lstm_input_size = 8 * out_channels * (img_height // 8)

        # BiLSTM шар
        self.lstm = nn.LSTM(
            self.lstm_input_size,
            n_h,
            bidirectional=True,
            batch_first=True,
            num_layers=lstm_layers
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


def resnet18_htr_sequential(img_height, num_channels=1, n_classes=80, n_h=256, out_channels=24, lstm_layers=2):
    """
    Створює ResNet18 для HTR з послідовною CNN архітектурою.
    """
    model = ResNet_BiLSTM_CTC(img_height, num_channels, n_classes, n_h, out_channels=out_channels,
                              lstm_layers=lstm_layers)
    return model


class ResNet_BiLSTM_CTC_v2(nn.Module):
    """
    Оригінальна архітектура ResNet, адаптована для HTR з CNN-BiLSTM-CTC,
    з використанням nn.Sequential для CNN частини.
    """

    def __init__(self, img_height, num_channels, n_classes, n_h=256, out_channels=24, lstm_layers=2):
        """
        img_height: висота вхідного зображення
        num_channels: кількість вхідних каналів (1 для чорно-білого)
        n_classes: кількість класів на виході (розмір словника + 1 для CTC blank)
        n_h: кількість прихованих нейронів у LSTM
        """
        super(ResNet_BiLSTM_CTC_v2, self).__init__()

        # CNN частина реалізована з використанням nn.Sequential
        self.cnn = nn.Sequential(
            # Початковий шар
            nn.Conv2d(num_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # Шар 1
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),  # Зменшуємо тільки висоту

            # Шар 2
            ResidualBlock(out_channels, 2 * out_channels, stride=1),
            ResidualBlock(2 * out_channels, 2 * out_channels),
            ResidualBlock(2 * out_channels, 4 * out_channels, stride=1),
            ResidualBlock(4 * out_channels, 4 * out_channels),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # Зменшуємо тільки висоту

            # Шар 3
            ResidualBlock(4 * out_channels, 8 * out_channels, stride=1),
            ResidualBlock(8 * out_channels, 8 * out_channels),
            nn.Identity()
        )

        # Розрахунок розміру входу для LSTM
        # Висота зменшується в 16 разів (2*2*2*2), а ширина в 4 рази (2*2 від conv1 та maxpool4)
        self.lstm_input_size = 8 * out_channels * (img_height // 4)

        # BiLSTM шар
        self.lstm = nn.LSTM(
            self.lstm_input_size,
            n_h,
            bidirectional=True,
            batch_first=True,
            num_layers=lstm_layers
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


def resnet18_htr_sequential_v2(img_height, num_channels=1, n_classes=80, n_h=256, out_channels=24, lstm_layers=2):
    """
    Створює ResNet18 для HTR з послідовною CNN архітектурою.
    """
    model = ResNet_BiLSTM_CTC_v2(img_height, num_channels, n_classes, n_h, out_channels=out_channels,
                                 lstm_layers=lstm_layers)
    return model


def resnet34_htr_sequential_v2(img_height, num_channels=1, n_classes=80, n_h=256, out_channels=24, lstm_layers=2):
    """
    Створює глибшу версію ResNet34 для HTR з послідовною CNN архітектурою.
    """
    # Створюємо базову модель
    model = ResNet_BiLSTM_CTC_v2(img_height, num_channels, n_classes, n_h, out_channels=out_channels,
                                 lstm_layers=lstm_layers)

    # Замінюємо CNN частину на глибшу версію
    model.cnn = nn.Sequential(
        # Початковий шар
        nn.Conv2d(num_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),

        # Шари з більшою кількістю блоків
        *make_layer(ResidualBlock, out_channels, out_channels, 3),
        *make_layer(ResidualBlock, out_channels, out_channels * 2, 4, stride=1),

        *make_layer(ResidualBlock, out_channels * 2, out_channels * 4, 6, stride=1),
        nn.MaxPool2d(kernel_size=2, stride=(2, 2)),

        *make_layer(ResidualBlock, out_channels * 4, out_channels * 8, 3, stride=1),
    )

    return model

# Приклад використання:
# model = resnet18_htr_sequential(img_height=64, num_channels=1, n_classes=100)
# або
# model = resnet34_htr_sequential(img_height=64, num_channels=1, n_classes=100)
