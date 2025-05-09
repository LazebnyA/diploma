import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import io
import numpy as np
import uuid
import cv2
from typing import List, Optional
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Import the model_params and other necessary components
# Assuming these modules are available in your nn structure
from project.word_model.models import resnet18_htr_sequential
from project.dataset import ProjectPaths, LabelConverter
from project.transform_2 import (resize_without_padding, remove_noise, enhance_document, adaptive_binarization,
                                 get_validation_transform, resize_with_padding)

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
    try:
        model_path = "project/word_model/parameters/cnn_lstm_ctc_handwritten_v0_word_best_CNN-BiLSTM-CTC_resnet18.pth"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

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
        logger.info("Model initialized successfully")

        return model, device, label_converter
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        logger.error(traceback.format_exc())
        raise


# Image preprocessing techniques
def convert_to_grayscale(image):
    """Convert image to grayscale"""
    try:
        return image.convert('L')
    except Exception as e:
        logger.error(f"Error in convert_to_grayscale: {str(e)}")
        return image  # Return original image if conversion fails


def adjust_contrast(image, factor=2.0):
    """Adjust image contrast"""
    try:
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    except Exception as e:
        logger.error(f"Error in adjust_contrast: {str(e)}")
        return image


def adjust_brightness(image, factor=2.0):
    """Adjust image brightness"""
    try:
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    except Exception as e:
        logger.error(f"Error in adjust_brightness: {str(e)}")
        return image


def adjust_sharpness(image, factor=1.5):
    """Adjust image sharpness"""
    try:
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)
    except Exception as e:
        logger.error(f"Error in adjust_sharpness: {str(e)}")
        return image


def remove_noise(image, median_kernel_size=3, gaussian_kernel_size=3, gaussian_sigma=0.5):
    """Remove noise using median and Gaussian filters"""
    try:
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
    except Exception as e:
        logger.error(f"Error in remove_noise: {str(e)}")
        logger.error(traceback.format_exc())
        # Return original image if processing fails
        return image


def apply_adaptive_binarization(image, block_size=11, constant=2):
    """Apply adaptive binarization"""
    try:
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
    except Exception as e:
        logger.error(f"Error in apply_adaptive_binarization: {str(e)}")
        logger.error(traceback.format_exc())
        # Return original image if processing fails
        return image


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
            try:
                processed_image = technique_info["func"](processed_image, **technique_info["params"])
                steps.append(technique_info["display_name"])
            except Exception as e:
                logger.error(f"Error applying {technique}: {str(e)}")
                logger.error(traceback.format_exc())
                # Continue with next technique if one fails

    return processed_image, steps


# Image preprocessing for model input
def preprocess_image(image, preprocessing_techniques=None, target_height=64):
    try:
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

        try:
            resized_image = resize_without_padding(image)
        except Exception as e:
            logger.warning(f"Error in resize_without_padding: {str(e)}, trying alternative resize")
            # Fallback to a simpler resize if the custom function fails
            resized_image = image.resize((image.width, target_height), Image.LANCZOS)

        # Apply transformations for model input
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        return transform(resized_image), processed_filename, applied_steps
    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}")
        logger.error(traceback.format_exc())
        raise


# Safe image loading function
def safe_open_image(file_bytes):
    """Safely open image data and verify it's a valid image"""
    try:
        image = Image.open(io.BytesIO(file_bytes))
        # Test accessing image data to verify it's actually an image
        image.verify()  # Verify the file is a valid image
        # Re-open after verify (which closes the file)
        image = Image.open(io.BytesIO(file_bytes))
        return image, None
    except UnidentifiedImageError:
        return None, "The uploaded file is not a recognized image format."
    except Exception as e:
        return None, f"Error opening image: {str(e)}"


# Load the model_params at startup
try:
    model, device, label_converter = initialize_model()
except Exception as e:
    logger.critical(f"Failed to initialize model: {str(e)}")
    model, device, label_converter = None, None, None


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
            "applied_steps": None,
            "error_message": None
        }
    )


@app.post("/recognize", response_class=HTMLResponse)
async def recognize_handwriting(
        request: Request,
        file: UploadFile = File(...),
        techniques: List[str] = Form(None),
        # Parameters for preprocessing
        contrast_factor: float = Form(2.0),
        brightness_factor: float = Form(2.0),
        sharpness_factor: float = Form(1.5),
        median_kernel_size: int = Form(3),
        gaussian_kernel_size: int = Form(3),
        gaussian_sigma: float = Form(0.5),
        block_size: int = Form(11),
        constant: int = Form(2)
):
    """Process the uploaded image and return the recognized text."""
    try:
        # Check if model was initialized successfully
        if model is None:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "preprocessing_techniques": preprocessing_techniques,
                    "error_message": "Model initialization failed. Please check server logs."
                }
            )

        # Generate a unique filename for the uploaded file
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"
        logger.info(f"Processing file: {file.filename}, saved as: {unique_filename}")

        # Read file content
        contents = await file.read()

        # Check if the file is empty
        if len(contents) == 0:
            logger.warning("Uploaded file is empty")
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "preprocessing_techniques": preprocessing_techniques,
                    "error_message": "Uploaded file is empty. Please upload a valid image."
                }
            )

        # Save file for display
        file_path = UPLOAD_DIR / unique_filename
        with open(file_path, "wb") as f:
            f.write(contents)

        # Safely open and validate the image
        image, error_msg = safe_open_image(contents)
        if error_msg:
            logger.error(f"Image validation error: {error_msg}")
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "preprocessing_techniques": preprocessing_techniques,
                    "error_message": error_msg
                }
            )

        # Update preprocessing parameters based on form input
        if techniques and 'contrast' in techniques:
            preprocessing_techniques['contrast']['params']['factor'] = contrast_factor
        if techniques and 'brightness' in techniques:
            preprocessing_techniques['brightness']['params']['factor'] = brightness_factor
        if techniques and 'sharpness' in techniques:
            preprocessing_techniques['sharpness']['params']['factor'] = sharpness_factor
        if techniques and 'noise_removal' in techniques:
            preprocessing_techniques['noise_removal']['params']['median_kernel_size'] = median_kernel_size
            preprocessing_techniques['noise_removal']['params']['gaussian_kernel_size'] = gaussian_kernel_size
            preprocessing_techniques['noise_removal']['params']['gaussian_sigma'] = gaussian_sigma
        if techniques and 'adaptive_binarization' in techniques:
            preprocessing_techniques['adaptive_binarization']['params']['block_size'] = block_size
            preprocessing_techniques['adaptive_binarization']['params']['constant'] = constant

        # Apply base preprocessing
        processed = image
        applied_steps = []

        try:
            # Apply user-selected techniques
            for tech in techniques or []:
                if tech == 'grayscale':
                    processed = convert_to_grayscale(processed)
                    applied_steps.append('Convert to Grayscale')
                elif tech == 'contrast':
                    processed = adjust_contrast(processed, factor=contrast_factor)
                    applied_steps.append(f'Adjust Contrast (factor={contrast_factor})')
                elif tech == 'brightness':
                    processed = adjust_brightness(processed, factor=brightness_factor)
                    applied_steps.append(f'Adjust Brightness (factor={brightness_factor})')
                elif tech == 'sharpness':
                    processed = adjust_sharpness(processed, factor=sharpness_factor)
                    applied_steps.append(f'Enhance Sharpness (factor={sharpness_factor})')
                elif tech == 'noise_removal':
                    processed = remove_noise(
                        processed,
                        median_kernel_size=median_kernel_size,
                        gaussian_kernel_size=gaussian_kernel_size,
                        gaussian_sigma=gaussian_sigma
                    )
                    applied_steps.append(
                        f'Remove Noise (median={median_kernel_size}, gaussian={gaussian_kernel_size}, sigma={gaussian_sigma})')
                elif tech == 'adaptive_binarization':
                    processed = apply_adaptive_binarization(
                        processed,
                        block_size=block_size,
                        constant=constant
                    )
                    applied_steps.append(f'Adaptive Binarization (block_size={block_size}, constant={constant})')
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            logger.error(traceback.format_exc())
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "preprocessing_techniques": preprocessing_techniques,
                    "error_message": f"Error during image preprocessing: {str(e)}",
                    "image_path": f"uploads/{unique_filename}",
                }
            )

        # Save processed image
        processed_filename = f"processed_{uuid.uuid4().hex}.png"
        processed_path = UPLOAD_DIR / processed_filename

        try:
            processed.save(processed_path)
        except Exception as e:
            logger.error(f"Error saving processed image: {str(e)}")
            logger.error(traceback.format_exc())
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "preprocessing_techniques": preprocessing_techniques,
                    "error_message": f"Error saving processed image: {str(e)}",
                    "image_path": f"uploads/{unique_filename}",
                }
            )

        # Prepare image for model
        try:
            # Get image dimensions before resizing
            orig_width, orig_height = processed.size
            logger.info(f"Original image dimensions: {orig_width}x{orig_height}")

            # Check if image is too small
            if orig_width < 20 or orig_height < 20:
                return templates.TemplateResponse(
                    "index.html",
                    {
                        "request": request,
                        "preprocessing_techniques": preprocessing_techniques,
                        "error_message": f"Image is too small for accurate recognition. Dimensions: {orig_width}x{orig_height}",
                        "image_path": f"uploads/{unique_filename}",
                        "processed_image_path": f"uploads/{processed_filename}",
                        "applied_steps": applied_steps
                    }
                )

            try:
                resized_image = resize_without_padding(processed)
                logger.info(f"Resized image dimensions: {resized_image.size[0]}x{resized_image.size[1]}")
            except Exception as resize_error:
                logger.warning(f"Custom resize failed: {str(resize_error)}, using standard resize")
                # Fall back to standard resize if custom resize fails
                target_height = 64
                ratio = target_height / float(orig_height)
                target_width = int(ratio * orig_width)
                resized_image = processed.resize((target_width, target_height), Image.LANCZOS)

            # Convert to tensor
            image_tensor = transforms.ToTensor()(resized_image).unsqueeze(0).to(device)
            logger.info(f"Tensor shape: {image_tensor.shape}")

            # Check if tensor is valid
            if torch.isnan(image_tensor).any() or torch.isinf(image_tensor).any():
                raise ValueError("Invalid tensor values (NaN or Inf) detected after preprocessing")

        except Exception as e:
            logger.error(f"Error resizing image or converting to tensor: {str(e)}")
            logger.error(traceback.format_exc())
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "preprocessing_techniques": preprocessing_techniques,
                    "error_message": f"Error preparing image for the model: {str(e)}",
                    "image_path": f"uploads/{unique_filename}",
                    "processed_image_path": f"uploads/{processed_filename}",
                    "applied_steps": applied_steps
                }
            )

        # Make prediction
        try:
            logger.info("Running prediction")
            with torch.no_grad():
                output = model(image_tensor)
                output = F.log_softmax(output, dim=2)
                output = torch.argmax(output, dim=2)
                output = output.squeeze(1).tolist()

            # Decode the prediction
            predicted_text = label_converter.decode(output)
            logger.info(f"Prediction result: '{predicted_text}'")

            # Check if prediction is empty
            if not predicted_text or predicted_text.isspace():
                logger.warning("Empty prediction result")
                return templates.TemplateResponse(
                    "index.html",
                    {
                        "request": request,
                        "preprocessing_techniques": preprocessing_techniques,
                        "error_message": "No text could be recognized in this image. Try adjusting preprocessing parameters or using a clearer image.",
                        "image_path": f"uploads/{unique_filename}",
                        "processed_image_path": f"uploads/{processed_filename}",
                        "applied_steps": applied_steps
                    }
                )

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "preprocessing_techniques": preprocessing_techniques,
                    "error_message": f"Error during text recognition: {str(e)}",
                    "image_path": f"uploads/{unique_filename}",
                    "processed_image_path": f"uploads/{processed_filename}",
                    "applied_steps": applied_steps
                }
            )

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
                "applied_steps": applied_steps,
                "error_message": None
            }
        )

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "preprocessing_techniques": preprocessing_techniques,
                "error_message": f"An unexpected error occurred: {str(e)}"
            }
        )