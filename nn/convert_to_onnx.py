import torch
import torch.nn as nn

from nn.v0.models import CNN_LSTM_CTC_V2_CNN_more_filters_batch_norm

# === Параметри ===
img_height = 32
img_width = 128
num_channels = 1
n_classes = 40  # приклад: 39 символів + 1 blank
n_h = 256  # кількість прихованих нейронів у LSTM

# === Ініціалізація моделі ===
model = CNN_LSTM_CTC_V2_CNN_more_filters_batch_norm(img_height, num_channels, n_classes, n_h)
model.eval()

# === Створення вхідного тензора (dummy_input) ===
batch_size = 8
dummy_input = torch.randn(batch_size, num_channels, img_height, img_width)

# === Експорт в ONNX ===
torch.onnx.export(
    model,
    dummy_input,
    "CNN_LSTM_CTC_V2_CNN_more_filters_batch_norm.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size", 3: "width"}, "output": {0: "batch_size", 1: "sequence_length"}},
    opset_version=11,
    verbose=True
)

print("Model exported to CNN_LSTM_CTC_V2_CNN_more_filters_batch_norm.onnx")
