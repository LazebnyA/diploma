import numpy as np
import torch
from torch import nn
from torchinfo import summary  # pip install torchinfo

from nn.v0.models import CNN_LSTM_CTC_V0

# Set parameters
img_height = 32
img_width = np.nan
num_channels = 1
n_classes = 80
n_h = 256

# Instantiate model
model = CNN_LSTM_CTC_V0(img_height, num_channels, n_classes, n_h)

# Use torchinfo to summarize
summary(model, input_size=(1, 1, img_height, img_width), verbose=2)
