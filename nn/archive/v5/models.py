import torch
from torch import nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block for improved gradient flow
    Inspired by ResNet architecture (https://arxiv.org/abs/1512.03385)
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv_block(x)
        out += residual
        out = self.relu(out)
        return out


class CNNBiLSTMResBlocksCtcShortcut(nn.Module):
    def __init__(self, img_height, num_channels, n_classes, n_h, dropout=0.2):
        super().__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            nn.MaxPool2d(2, 2)
        )

        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=1),
            ResidualBlock(128, 128),
            nn.MaxPool2d(2, 2)
        )

        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=1),
            ResidualBlock(256, 256),
            nn.MaxPool2d((2, 1))
        )

        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512, stride=1),
            ResidualBlock(512, 512),
            nn.MaxPool2d((2, 1))
        )

        self.lstm_input_size = 512 * (img_height // 16)

        self.lstm = nn.LSTM(
            self.lstm_input_size, n_h, bidirectional=True, num_layers=2,
            batch_first=True, dropout=dropout if dropout > 0 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * n_h, n_classes)

        # CTC Shortcut: 1D convolutional layer
        self.ctc_shortcut = nn.Conv1d(in_channels=1024, out_channels=n_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        conv_original = self.stage4(x)

        b, c, h, w = conv_original.size()
        conv = conv_original.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)

        recurrent, _ = self.lstm(conv)
        recurrent = self.dropout(recurrent)
        output = self.fc(recurrent)
        output = output.permute(1, 0, 2)

        # CTC Shortcut branch
        shortcut_features = conv.permute(0, 2, 1)
        shortcut_output = self.ctc_shortcut(shortcut_features)
        shortcut_output = shortcut_output.permute(2, 0, 1)

        return output, shortcut_output


class CNNBiLSTMResBlocksCNNCtcShortcutBetterFeatures(nn.Module):
    def __init__(self, img_height, num_channels, n_classes, n_h, dropout=0.2):
        """
        Enhanced CRNN with improved feature extraction capabilities

        Args:
            img_height: image height (after resize)
            num_channels: number of input channels (1 for grayscale)
            n_classes: number of output classes (vocab size + 1 for blank)
            n_h: number of hidden units in the LSTM
            dropout: dropout rate for better regularization
        """
        super().__init__()

        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Stage 1: Residual blocks
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            nn.MaxPool2d(2, 2)  # (H/2, W/2)
        )

        # Stage 2: Residual blocks with channel increase
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=1),
            ResidualBlock(128, 128),
            nn.MaxPool2d(2, 2)  # (H/4, W/4)
        )

        # Stage 3: Residual blocks with channel increase
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 128, stride=1),
            ResidualBlock(128, 128, stride=1),
            ResidualBlock(128, 256, stride=1),
            nn.MaxPool2d((2, 1))  # Downsample height only (H/8, W/4)
        )

        self.stage4 = nn.Sequential(
            ResidualBlock(256, 256, stride=1),
            ResidualBlock(256, 256, stride=1),
            ResidualBlock(256, 512, stride=1),
            ResidualBlock(512, 512),
            nn.MaxPool2d((2, 1))  # Downsample height only (H/8, W/4)
        )

        # After CNN, height becomes img_height // 16
        self.lstm_input_size = 512 * (img_height // 16)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            self.lstm_input_size,
            n_h,
            bidirectional=True,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Final classifier: maps LSTM output to character classes
        self.fc = nn.Linear(2 * n_h, n_classes)

        # CTC Shortcut: 1D convolutional layer
        self.ctc_shortcut = nn.Conv1d(in_channels=1024, out_channels=n_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        conv_original = self.stage4(x)

        b, c, h, w = conv_original.size()
        conv = conv_original.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)

        recurrent, _ = self.lstm(conv)
        recurrent = self.dropout(recurrent)
        output = self.fc(recurrent)
        output = output.permute(1, 0, 2)

        # CTC Shortcut branch
        shortcut_features = conv.permute(0, 2, 1)
        shortcut_output = self.ctc_shortcut(shortcut_features)
        shortcut_output = shortcut_output.permute(2, 0, 1)

        return output, shortcut_output
