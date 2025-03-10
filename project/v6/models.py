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


class DenseBlock(nn.Module):
    """
    Dense block for feature reuse
    Inspired by DenseNet architecture (https://arxiv.org/abs/1608.06993)
    """

    def __init__(self, in_channels, growth_rate, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
            )
            self.layers.append(layer)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)


class Transition(nn.Module):
    """
    Transition layer for reducing feature map size between dense blocks
    """

    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        if downsample:
            self.transition = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2)
            )
        else:
            # No pooling, just feature reduction
            self.transition = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            )

    def forward(self, x):
        return self.transition(x)


class CNNBiLSTMResBlocks(nn.Module):
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

        # Stage 3: DenseBlock for better feature reuse
        self.stage3 = nn.Sequential(
            DenseBlock(128, growth_rate=32, num_layers=3),  # Output: 128 + 3*32 = 224
            Transition(224, 256, downsample=False),  # (H/8, W/4)
            nn.MaxPool2d((1, 1))  # Keep dimensions
        )

        # Stage 4: Final feature extraction
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512, stride=1),
            ResidualBlock(512, 512),
            nn.MaxPool2d((2, 1))  # Downsample height only (H/8, W/4)
        )

        # After CNN, height becomes img_height // 8
        self.lstm_input_size = 512 * (img_height // 8)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network

        Args:
            x: input images with shape (batch, channels, height, width)

        Returns:
            output in shape (T, batch, n_classes) required by CTCLoss
        """
        # Extract CNN features
        x = self.initial_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        conv = self.stage4(x)

        # Shape information
        b, c, h, w = conv.size()

        # Permute and flatten for LSTM
        conv = conv.permute(0, 3, 1, 2)  # (batch, width, channels, height)
        conv = conv.contiguous().view(b, w, c * h)  # (batch, width, feature_size)

        # LSTM sequence modeling
        recurrent, _ = self.lstm(conv)
        recurrent = self.dropout(recurrent)

        # Final classification
        output = self.fc(recurrent)

        # For CTCLoss, we need (seq_len, batch, n_classes)
        output = output.permute(1, 0, 2)
        return output


class CNNBiLSTMResBlocksNoDenseBetweenCNN(nn.Module):
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

        # Stage 4: Final feature extraction
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512, stride=1),
            ResidualBlock(512, 512),
            nn.MaxPool2d((2, 1))  # Downsample height only (H/8, W/4)
        )

        # After CNN, height becomes img_height // 8
        self.lstm_input_size = 512 * (img_height // 8)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network

        Args:
            x: input images with shape (batch, channels, height, width)

        Returns:
            output in shape (T, batch, n_classes) required by CTCLoss
        """
        # Extract CNN features
        x = self.initial_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        conv = self.stage4(x)

        # Shape information
        b, c, h, w = conv.size()

        # Permute and flatten for LSTM
        conv = conv.permute(0, 3, 1, 2)  # (batch, width, channels, height)
        conv = conv.contiguous().view(b, w, c * h)  # (batch, width, feature_size)

        # LSTM sequence modeling
        recurrent, _ = self.lstm(conv)
        recurrent = self.dropout(recurrent)

        # Final classification
        output = self.fc(recurrent)

        # For CTCLoss, we need (seq_len, batch, n_classes)
        output = output.permute(1, 0, 2)
        return output

