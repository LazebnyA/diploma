import torch
from torch import nn as nn


class CNN_LSTM_CTC_V0_CNN_36_start_filters(nn.Module):
    def __init__(self, img_height, num_channels, n_classes, n_h, lstm_layers):
        """
        img_height: image height (after resize)
        num_channels: number of input channels (1 for grayscale)
        n_classes: number of output classes (vocab size + 1 for blank)
        n_h: number of hidden units in the LSTM
        """
        super().__init__()

        # CNN Module
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 36, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # downsample height & width by 2
            nn.Conv2d(36, 72, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(72, 72, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # downsample height by 4
            nn.Conv2d(72, 144, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(144, 144, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1))  # downsample height by 8
        )

        # After CNN, height becomes img_height // 8
        self.lstm_input_size = 144 * (img_height // 8)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(self.lstm_input_size, n_h, bidirectional=True, num_layers=lstm_layers, batch_first=True)

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


class CNN_LSTM_CTC_V0_CNN_48_start_filters(nn.Module):
    def __init__(self, img_height, num_channels, n_classes, n_h, lstm_layers):
        """
        img_height: image height (after resize)
        num_channels: number of input channels (1 for grayscale)
        n_classes: number of output classes (vocab size + 1 for blank)
        n_h: number of hidden units in the LSTM
        """
        super().__init__()

        # CNN Module
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # downsample height & width by 2
            nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # downsample height by 4
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1))  # downsample height by 8
        )

        # After CNN, height becomes img_height // 8
        self.lstm_input_size = 192 * (img_height // 8)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(self.lstm_input_size, n_h, bidirectional=True, num_layers=lstm_layers, batch_first=True)

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


class CNN_LSTM_CTC_V0_CNN_64_start_filters(nn.Module):
    def __init__(self, img_height, num_channels, n_classes, n_h, lstm_layers):
        """
        img_height: image height (after resize)
        num_channels: number of input channels (1 for grayscale)
        n_classes: number of output classes (vocab size + 1 for blank)
        n_h: number of hidden units in the LSTM
        """
        super().__init__()

        # CNN Module
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # downsample height & width by 2
            nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # downsample height by 4
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1))  # downsample height by 8
        )

        # After CNN, height becomes img_height // 8
        self.lstm_input_size = 192 * (img_height // 8)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(self.lstm_input_size, n_h, bidirectional=True, num_layers=lstm_layers, batch_first=True)

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


class CNN_LSTM_CTC_V1_CNN_deeper_vgg16like(nn.Module):
    def __init__(self, img_height, num_channels, n_classes, n_h, lstm_layers=1, start_filters=24):
        """
        img_height: image height (after resize)
        num_channels: number of input channels (1 for grayscale)
        n_classes: number of output classes (vocab size + 1 for blank)
        n_h: number of hidden units in the LSTM
        """
        super().__init__()

        # CNN Module
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, start_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(start_filters, start_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # downsample height & width by 2
            nn.Conv2d(start_filters, 2*start_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(2*start_filters, 2*start_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # downsample height by 4
            nn.Conv2d(2*start_filters, 4*start_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(4*start_filters, 4*start_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)),  # downsample height by 8
            nn.Conv2d(4*start_filters, 8*start_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8*start_filters, 8*start_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8 * start_filters, 8 * start_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)),  # downsample height by 16
        )

        # After CNN, height becomes img_height // 16
        self.lstm_input_size = 8*start_filters * (img_height // 16)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(self.lstm_input_size, n_h, bidirectional=True, num_layers=lstm_layers, batch_first=True)

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


class CNN_LSTM_CTC_V1_CNN_deeper_vgg16like_batch_norm(nn.Module):
    def __init__(self, img_height, num_channels, n_classes, n_h, lstm_layers=1, start_filters=24):
        """
        img_height: image height (after resize)
        num_channels: number of input channels (1 for grayscale)
        n_classes: number of output classes (vocab size + 1 for blank)
        n_h: number of hidden units in the LSTM
        """
        super().__init__()

        # CNN Module
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, start_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(start_filters),
            nn.ReLU(True),
            nn.Conv2d(start_filters, start_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(start_filters),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # downsample height & width by 2
            nn.Conv2d(start_filters, 2*start_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2*start_filters),
            nn.ReLU(True),
            nn.Conv2d(2*start_filters, 2*start_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2*start_filters),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # downsample height by 4
            nn.Conv2d(2*start_filters, 4*start_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*start_filters),
            nn.ReLU(True),
            nn.Conv2d(4*start_filters, 4*start_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*start_filters),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)),  # downsample height by 8
            nn.Conv2d(4*start_filters, 8*start_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*start_filters),
            nn.ReLU(True),
            nn.Conv2d(8*start_filters, 8*start_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*start_filters),
            nn.ReLU(True),
            nn.Conv2d(8 * start_filters, 8 * start_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*start_filters),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)),  # downsample height by 16
        )

        # After CNN, height becomes img_height // 16
        self.lstm_input_size = 8*start_filters * (img_height // 16)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(self.lstm_input_size, n_h, bidirectional=True, num_layers=lstm_layers, batch_first=True)

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


class CNN_LSTM_CTC_V1_CNN_deeper_vgg16like_batch_norm_dropout(nn.Module):
    def __init__(self, img_height, num_channels, n_classes, n_h, lstm_layers=1, start_filters=24):
        """
        img_height: image height (after resize)
        num_channels: number of input channels (1 for grayscale)
        n_classes: number of output classes (vocab size + 1 for blank)
        n_h: number of hidden units in the LSTM
        """
        super().__init__()

        # CNN Module
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, start_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(start_filters),
            nn.ReLU(True),
            nn.Conv2d(start_filters, start_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(start_filters),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # downsample height & width by 2
            nn.Conv2d(start_filters, 2*start_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2*start_filters),
            nn.ReLU(True),
            nn.Conv2d(2*start_filters, 2*start_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2*start_filters),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # downsample height by 4
            nn.Conv2d(2*start_filters, 4*start_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*start_filters),
            nn.ReLU(True),
            nn.Conv2d(4*start_filters, 4*start_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*start_filters),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)),  # downsample height by 8
            nn.Conv2d(4*start_filters, 8*start_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*start_filters),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Conv2d(8*start_filters, 8*start_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*start_filters),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Conv2d(8 * start_filters, 8 * start_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*start_filters),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1)),  # downsample height by 16
        )

        # After CNN, height becomes img_height // 16
        self.lstm_input_size = 8*start_filters * (img_height // 16)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(self.lstm_input_size, n_h, bidirectional=True, num_layers=lstm_layers, batch_first=True)

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