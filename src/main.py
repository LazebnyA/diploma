# main.py
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import io
import numpy as np
import uuid

# Import the model_params and other necessary components
# Assuming these modules are available in your nn structure
from project.v7.models import CNNBiLSTMResBlocksNoDenseBetweenCNN
from project.dataset import ProjectPaths, LabelConverter

app = FastAPI(title="Handwritten Text Recognition")

# Create necessary directories
UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja templates
templates = Jinja2Templates(directory="templates")


# Initialize model_params
def initialize_model():
    model_path = "project/v7/without_transition/cnn_lstm_ctc_handwritten_v5_75ep_2-Layered-BiLSTM-3-CNN-Blocks.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model_params parameters
    n_h = 256  # Hidden units in LSTM
    img_height = 32  # Height of input images
    num_channels = 1  # Grayscale images

    # Load character mapping
    mapping_file = "project/word_mappings.txt"
    paths = ProjectPaths()
    label_converter = LabelConverter(mapping_file, paths)
    n_classes = len(label_converter.chars) + 1  # +1 for CTC blank character

    # Initialize model_params
    model = CNNBiLSTMResBlocksNoDenseBetweenCNN(img_height, num_channels, n_classes, n_h)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, device, label_converter


# Simple image enhancement with lightening
def enhance_image(image):
    """
    Apply simple image enhancement techniques to lighten the background.
    """
    # Convert to grayscale if not already
    if image.mode != 'L':
        image = image.convert('L')

    # Step 1: Increase brightness significantly
    enhancer = ImageEnhance.Brightness(image)
    brightened = enhancer.enhance(2.0)  # Significantly increase brightness

    # Step 2: Increase contrast to make text stand out
    enhancer = ImageEnhance.Contrast(brightened)
    contrasted = enhancer.enhance(2.0)

    # Step 3: Apply a simple thresholding using numpy
    img_array = np.array(contrasted)

    # Calculate a threshold - a high value to ensure background becomes white
    # This is a simple method to make background white while keeping text dark
    threshold = np.percentile(img_array, 90)  # 80th percentile

    # Apply the threshold
    lightened = np.where(img_array > threshold, 255, img_array)

    # Convert back to PIL Image
    enhanced_image = Image.fromarray(lightened.astype(np.uint8))

    return enhanced_image


# Image preprocessing
def preprocess_image(image, target_height=32):
    # First enhance the image
    enhanced_image = enhance_image(image)

    # Generate a unique filename for the enhanced image
    enhanced_filename = f"enhanced_{uuid.uuid4().hex}.png"
    enhanced_path = UPLOAD_DIR / enhanced_filename

    # Save enhanced image for display
    enhanced_image.save(enhanced_path)

    # Resize while maintaining aspect ratio
    w, h = enhanced_image.size
    new_w = int(w * (target_height / h))
    resized_image = enhanced_image.resize((new_w, target_height))

    # Apply transformations
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    return transform(resized_image), enhanced_filename


# Load the model_params at startup
model, device, label_converter = initialize_model()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the home page."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": None, "image_path": None, "enhanced_image_path": None}
    )


@app.post("/recognize", response_class=HTMLResponse)
async def recognize_handwriting(request: Request, file: UploadFile = File(...)):
    """Process the uploaded image and return the recognized text."""
    # Generate a unique filename for the uploaded file
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{file_extension}"

    # Save the uploaded file
    file_path = UPLOAD_DIR / unique_filename
    contents = await file.read()

    # Save file for display
    with open(file_path, "wb") as f:
        f.write(contents)

    # Process the image
    image = Image.open(io.BytesIO(contents))

    # Preprocess and enhance the image
    image_tensor, enhanced_filename = preprocess_image(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        output = F.log_softmax(output, dim=2)
        output = torch.argmax(output, dim=2)
        output = output.squeeze(1).tolist()

    # Decode the prediction
    predicted_text = label_converter.decode(output)

    # Relative paths for template
    original_path = f"uploads/{unique_filename}"
    enhanced_path = f"uploads/{enhanced_filename}"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": predicted_text,
            "image_path": original_path,
            "enhanced_image_path": enhanced_path
        }
    )