import os
import shutil
from pathlib import Path

def organize_form_images(input_file, output_base_dir="test_lines"):
    """
    Organize images from g06-042 form into folders by writer letter.
    Writer letter is the last character in the form ID (e.g., 'a' from 'g06-042a').
    """
    # Create base output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Dictionary to track files by writer
    writer_files = {}
    
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' not in line:
                continue
                
            path, text = line.strip().split('\t')
            
            # Extract writer letter from the form ID (e.g., 'a' from 'g06-042a')
            writer_letter = path.split('/')[-2][-1]  # get last character from folder name
            
            # Create writer directory if it doesn't exist
            writer_dir = os.path.join(output_base_dir, writer_letter)
            os.makedirs(writer_dir, exist_ok=True)
            
            # Copy the image file
            try:
                src_path = 'iam_lines/' + path
                dst_path = os.path.join(writer_dir, os.path.basename(path))
                
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    
                    # Track files for this writer
                    if writer_letter not in writer_files:
                        writer_files[writer_letter] = []
                    writer_files[writer_letter].append((dst_path, text))
                    
                    print(f"Copied {os.path.basename(path)} to writer {writer_letter}")
                else:
                    print(f"Warning: Source file not found: {src_path}")
            
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
    
    # Create annotation files for each writer
    for writer_letter, files in writer_files.items():
        annotation_file = os.path.join(output_base_dir, f"{writer_letter}_lines.txt")
        with open(annotation_file, 'w', encoding='utf-8') as f:
            for path, text in files:
                rel_path = os.path.relpath(path, output_base_dir)
                f.write(f"{rel_path}\t{text}\n")
        
        print(f"\nWriter {writer_letter}: {len(files)} images")
    
    return writer_files

if __name__ == "__main__":
    input_file = "g06_042_lines.txt"
    
    # Organize images
    writer_files = organize_form_images(input_file)
    
    print("\nOrganization complete!")
    print(f"Total writers: {len(writer_files)}")