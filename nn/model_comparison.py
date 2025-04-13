import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Directory where all the model_params versions are stored
base_dir = Path(__file__).parent

# Find all version directories
version_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('v') and d.name[1:].isdigit()]
version_dirs.sort(key=lambda x: int(x.name[1:]))  # Sort by version number

print(f"Found {len(version_dirs)} version directories: {[d.name for d in version_dirs]}")

# Data structures to store metrics
models = []
avg_cer_values = []
avg_wer_values = []
avg_ld_values = []

def process_eval_results(eval_results_path, model_name):
    """Process evaluation results from a given path and add to data structures"""
    if eval_results_path.exists():
        try:
            with open(eval_results_path, 'r') as f:
                results = json.load(f)
            
            models.append(model_name)
            avg_cer_values.append(results.get('avg_cer', 0))
            avg_wer_values.append(results.get('avg_wer', 0))
            avg_ld_values.append(results.get('avg_ld', 0))
            
            print(f"Successfully extracted metrics from {model_name}")
            return True
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
    else:
        print(f"No evaluation results found for {model_name}")
    return False

# Process each model_params version directory and its subdirectories
for version_dir in version_dirs:
    version_name = version_dir.name
    
    # First check the main version directory
    eval_results_path = version_dir / "evaluation_results" / "evaluation_results.json"
    if process_eval_results(eval_results_path, version_name):
        # Main directory processed successfully
        pass
    
    # Now check subdirectories
    for subdir in version_dir.iterdir():
        if subdir.is_dir() and subdir.name not in ["__pycache__", "evaluation_results"]:
            sub_model_name = f"{version_name}_{subdir.name}"
            
            # Check if there's an evaluation_results directory in this subdirectory
            sub_eval_path = subdir / "evaluation_results" / "evaluation_results.json"
            if process_eval_results(sub_eval_path, sub_model_name):
                # Subdirectory processed successfully
                pass
            
            # Check for deeper nested directories
            for nested_subdir in subdir.iterdir():
                if nested_subdir.is_dir() and nested_subdir.name not in ["__pycache__", "evaluation_results"]:
                    nested_model_name = f"{version_name}_{subdir.name}_{nested_subdir.name}"
                    
                    # Check for evaluation results in nested directory
                    nested_eval_path = nested_subdir / "evaluation_results" / "evaluation_results.json"
                    process_eval_results(nested_eval_path, nested_model_name)

print(f"Total models found with evaluation results: {len(models)}")

# Create a bar chart of average CER values
if models:
    # Ensure consistent figure size for all plots
    figsize = (max(10, len(models)), 10)
    bar_width = 0.8
    
    # Creating a bar chart for average CER
    plt.figure(figsize=figsize, dpi=100)
    bars = plt.bar(range(len(models)), avg_cer_values, color='skyblue', width=bar_width)
    plt.title('Average Character Error Rate (CER) Comparison', fontsize=16)
    plt.xlabel('Model Version', fontsize=14)
    plt.ylabel('Average CER', fontsize=14)
    plt.xticks(range(len(models)), models, fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add the values on top of the bars
    for bar, value in zip(bars, avg_cer_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(base_dir / 'avg_cer_comparison.png')
    plt.close()
    print("Created Average CER comparison chart")
    
    # Creating a bar chart for average WER
    plt.figure(figsize=figsize, dpi=100)
    bars = plt.bar(range(len(models)), avg_wer_values, color='lightgreen', width=bar_width)
    plt.title('Average Word Error Rate (WER) Comparison', fontsize=16)
    plt.xlabel('Model Version', fontsize=14)
    plt.ylabel('Average WER', fontsize=14)
    plt.xticks(range(len(models)), models, fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add the values on top of the bars
    for bar, value in zip(bars, avg_wer_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(base_dir / 'avg_wer_comparison.png')
    plt.close()
    print("Created Average WER comparison chart")
    
    # Creating a bar chart for average LD
    plt.figure(figsize=figsize, dpi=100)
    bars = plt.bar(range(len(models)), avg_ld_values, color='salmon', width=bar_width)
    plt.title('Average Levenshtein Distance (LD) Comparison', fontsize=16)
    plt.xlabel('Model Version', fontsize=14)
    plt.ylabel('Average LD', fontsize=14)
    plt.xticks(range(len(models)), models, fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add the values on top of the bars
    for bar, value in zip(bars, avg_ld_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(base_dir / 'avg_ld_comparison.png')
    plt.close()
    print("Created Average LD comparison chart")
    
    print("\nAll comparison charts have been saved to:", base_dir)
else:
    print("No models with evaluation results found.") 