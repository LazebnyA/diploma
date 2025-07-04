import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from nn.archive.dataset import ProjectPaths, LabelConverter
from nn.transform import get_simple_recognize_transform
from nn.v2.models import CNN_LSTM_CTC_V2_CNN_more_filters_batch_norm_deeper_vgg16like

# Load the trained model_params
model_path = "cnn_lstm_ctc_handwritten_v1_lines_18ep_CNN-BiLSTM-CTC_CNN-VGG16_BiLSTM-1dim.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same model_params architecture
n_h = 256  # Hidden units in LSTM
img_height = 64  # Height of input images
num_channels = 1  # Grayscale images

# Load character mapping
mapping_file = "dataset/writer_independent_word_splits/v2/train_word_mappings.txt"
paths = ProjectPaths()
label_converter = LabelConverter(mapping_file, paths)
n_classes = 80  # +1 for CTC blank character

model = CNN_LSTM_CTC_V2_CNN_more_filters_batch_norm_deeper_vgg16like(img_height, num_channels, n_classes, n_h)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


# Image preprocessing
def resize_with_aspect(image, target_height=32):
    w, h = image.size
    new_w = int(w * (target_height / h))

    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

    return image.resize((new_w, target_height))


transform = get_simple_recognize_transform()


# Load and preprocess test image
def predict(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(image)  # (T, batch, n_classes)
        output = F.log_softmax(output, dim=2)
        output = torch.argmax(output, dim=2)  # Get predicted character indices
        output = output.squeeze(1).tolist()  # Convert to list

    pred_text = label_converter.decode(output)
    return pred_text


# Test the model_params on an example image
image_path = "img_1.png"  # Change this to your test image path
predicted_text = predict(image_path)

# Display the image and prediction
image = Image.open(image_path)
plt.imshow(image, cmap='gray')
plt.title(f"Predicted: {predicted_text}")
plt.axis('off')
plt.show()
