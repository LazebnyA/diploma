import torch
from torch import nn as nn

torch.manual_seed(42)


class CNN_LSTM_CTC_V0(nn.Module):
    def __init__(self, img_height, num_channels, n_classes, n_h, lstm_layers=1, out_channels=24):
        """
        img_height: image height (after resize)
        num_channels: number of input channels (1 for grayscale)
        n_classes: number of output classes (vocab size + 1 for blank)
        n_h: number of hidden units in the LSTM
        """
        super().__init__()

        # CNN Module
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # downsample height & width by 2
            nn.Conv2d(out_channels, 2*out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(2*out_channels, 2*out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # downsample height by 4
            nn.Conv2d(2*out_channels, 4*out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(4*out_channels, 4*out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )

        # After CNN, height becomes img_height // 8
        self.lstm_input_size = 4*out_channels * (img_height // 4)

        # One directional LSTM
        self.lstm = nn.LSTM(self.lstm_input_size, n_h, num_layers=lstm_layers, batch_first=True, bidirectional=True)

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



