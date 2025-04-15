import sys
import torch
import os


def count_parameters(model_path):
    """
    Count the number of parameters in a PyTorch model.

    Args:
        model_path (str): Path to the .torch or .pt model file

    Returns:
        int: Number of parameters in the model
    """
    # Check if the file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return -1

    # Check if the file has a valid extension
    if not model_path.endswith(('.pt', '.pth', '.torch')):
        print(f"Warning: File {model_path} doesn't have a standard PyTorch model extension (.pt, .pth, .torch)")

    try:
        # Load the model
        model = torch.load(model_path, map_location=torch.device('cpu'))

        # Handle different model formats
        if isinstance(model, torch.nn.Module):
            # Model is directly a nn.Module
            return sum(p.numel() for p in model.parameters())
        elif isinstance(model, dict) and 'state_dict' in model:
            # Model is a checkpoint dictionary with a state_dict
            state_dict = model['state_dict']
            return sum(param.numel() for param in state_dict.values())
        elif isinstance(model, dict):
            # Model might be a straight state_dict
            return sum(param.numel() for param in model.values() if isinstance(param, torch.Tensor))
        else:
            print(f"Error: Unsupported model format")
            return -1

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return -1


if __name__ == "__main__":
    # You can specify the model path here
    model_path = "v0/apply_modifications/cnn_deeper/max_pool_x4_width/cnn_lstm_ctc_handwritten_v0_lines_11ep_CNN-BiLSTM-CTC_CNN-24-48-96_BiLSTM-1dim.pth"

    num_params = count_parameters(model_path)

    if num_params >= 0:
        print(f"The model has {num_params:,} parameters")
        # For models with millions of parameters, also show in millions
        if num_params >= 1_000_000:
            print(f"That's {num_params / 1_000_000:.2f}M parameters")