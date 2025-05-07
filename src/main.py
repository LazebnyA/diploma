# main.py
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import io
import numpy as np
import uuid
import cv2
from typing import List, Optional

# Import the model_params and other necessary components
# Assuming these modules are available in your nn structure
from project.word_model.models import resnet18_htr_sequential
from project.dataset import ProjectPaths, LabelConverter
from project.transform_2 import (resize_without_padding, remove_noise, enhance_document, adaptive_binarization,
                              get_validation_transform)

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
    model_path = "project/word_model/parameters/cnn_lstm_ctc_handwritten_v0_word_best_CNN-BiLSTM-CTC_resnet18.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model_params parameters
    n_h = 1024  # Hidden units in LSTM
    img_height = 64  # Height of input images
    num_channels = 1  # Grayscale images
    out_channels = 64

    # Load character mapping
    mapping_file = "project/train_word_mappings.txt"
    paths = ProjectPaths()
    label_converter = LabelConverter(mapping_file, paths)
    n_classes = len(label_converter.chars) + 1  # +1 for CTC blank character

    # Initialize model_params
    model = resnet18_htr_sequential(
        img_height=img_height,
        num_channels=num_channels,
        n_classes=n_classes,
        out_channels=out_channels,
        n_h=n_h,
        lstm_layers=1)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, device, label_converter


# Image preprocessing techniques
def convert_to_grayscale(image):
    """Convert image to grayscale"""
    return image.convert('L')


def adjust_contrast(image, factor=2.0):
    """Adjust image contrast"""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def adjust_brightness(image, factor=2.0):
    """Adjust image brightness"""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def adjust_sharpness(image, factor=1.5):
    """Adjust image sharpness"""
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)


# def apply_thresholding(image, threshold=90):
#     """Apply thresholding to make the background white and text dark"""
#     if image.mode != 'L':
#         image = image.convert('L')
#
#     img_array = np.array(image)
#     threshold_value = np.percentile(img_array, threshold)
#     lightened = np.where(img_array > threshold_value, 255, img_array)
#     return Image.fromarray(lightened.astype(np.uint8))


def remove_noise(image, median_kernel_size=3, gaussian_kernel_size=3, gaussian_sigma=0.5):
    """Remove noise using median and Gaussian filters"""
    # Ensure kernel sizes are odd
    if median_kernel_size % 2 == 0:
        median_kernel_size += 1
    if gaussian_kernel_size % 2 == 0:
        gaussian_kernel_size += 1

    # Convert to numpy array
    img_np = np.array(image.convert('L'))

    # Apply median filter
    img_denoised = cv2.medianBlur(img_np, median_kernel_size)

    # Apply Gaussian filter
    img_denoised = cv2.GaussianBlur(
        img_denoised,
        (gaussian_kernel_size, gaussian_kernel_size),
        gaussian_sigma
    )

    return Image.fromarray(img_denoised)


def apply_adaptive_binarization(image, block_size=11, constant=2):
    """Apply adaptive binarization"""
    # Ensure block size is odd
    if block_size % 2 == 0:
        block_size += 1

    # Convert to numpy array in grayscale
    img_np = np.array(image.convert('L'))

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        img_np,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        constant
    )

    return Image.fromarray(binary)


# def deskew_image(image):
#     """Correct image skew"""
#     if image.mode != 'L':
#         image = image.convert('L')
#
#     # Convert to numpy array
#     img_np = np.array(image)
#
#     # Calculate skew angle
#     coords = np.column_stack(np.where(img_np < 200))
#     angle = cv2.minAreaRect(coords)[-1]
#
#     # Adjust angle
#     if angle < -45:
#         angle = -(90 + angle)
#     else:
#         angle = -angle
#
#     # Rotate image to deskew
#     h, w = img_np.shape
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated = cv2.warpAffine(img_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
#
#     return Image.fromarray(rotated)


# def invert_image(image):
#     """Invert image colors"""
#     return ImageOps.invert(image)


# Dictionary of available preprocessing techniques
preprocessing_techniques = {
    "grayscale": {"func": convert_to_grayscale, "params": {}, "display_name": "Convert to Grayscale"},
    "contrast": {"func": adjust_contrast, "params": {"factor": 2.0}, "display_name": "Adjust Contrast"},
    "brightness": {"func": adjust_brightness, "params": {"factor": 2.0}, "display_name": "Adjust Brightness"},
    "sharpness": {"func": adjust_sharpness, "params": {"factor": 1.5}, "display_name": "Enhance Sharpness"},
    "noise_removal": {"func": remove_noise,
                      "params": {"median_kernel_size": 3, "gaussian_kernel_size": 3, "gaussian_sigma": 0.5},
                      "display_name": "Remove Noise"},
    "adaptive_binarization": {"func": apply_adaptive_binarization, "params": {"block_size": 11, "constant": 2},
                              "display_name": "Adaptive Binarization"}
}


# Function to apply chosen preprocessing techniques in order
def apply_preprocessing(image, techniques):
    """Apply a series of preprocessing techniques to an image"""
    processed_image = image
    steps = []

    for technique in techniques:
        technique_info = preprocessing_techniques.get(technique)
        if technique_info:
            processed_image = technique_info["func"](processed_image, **technique_info["params"])
            steps.append(technique_info["display_name"])

    return processed_image, steps


# Image preprocessing for model input
def preprocess_image(image, preprocessing_techniques=None, target_height=64):
    image = convert_to_grayscale(image)

    # Process the image according to selected techniques
    if preprocessing_techniques:
        processed_image, applied_steps = apply_preprocessing(image, preprocessing_techniques)
    else:
        processed_image = adjust_contrast(image)
        processed_image = adjust_brightness(processed_image)
        applied_steps = ["Convert to Grayscale", "Adjust Contrast", "Adjust Brightness"]

    # Generate a unique filename for the processed image
    processed_filename = f"processed_{uuid.uuid4().hex}.png"
    processed_path = UPLOAD_DIR / processed_filename

    # Save processed image for display
    processed_image.save(processed_path)

    resized_image = resize_without_padding(image)

    # Apply transformations for model input
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    return transform(resized_image), processed_filename, applied_steps


# Load the model_params at startup
model, device, label_converter = initialize_model()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the home page."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": None,
            "image_path": None,
            "processed_image_path": None,
            "preprocessing_techniques": preprocessing_techniques,
            "applied_steps": None
        }
    )


@app.post("/recognize", response_class=HTMLResponse)
async def recognize_handwriting(
        request: Request,
        file: UploadFile = File(...),
        techniques: List[str] = Form(None)
):
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

    # Apply preprocessing techniques and prepare for model
    image_tensor, processed_filename, applied_steps = preprocess_image(image, techniques)
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
    processed_path = f"uploads/{processed_filename}"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": predicted_text,
            "image_path": original_path,
            "processed_image_path": processed_path,
            "preprocessing_techniques": preprocessing_techniques,
            "applied_steps": applied_steps
        }
    )