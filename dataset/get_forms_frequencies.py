import os
from collections import defaultdict
import re

def analyze_form_frequencies(metadata_dir):
    """
    Analyze form frequencies from metadata filenames.
    Returns dictionaries with form frequencies and writer frequencies.
    """
    # Dictionary to store form frequencies
    form_frequencies = defaultdict(int)
    writer_frequencies = defaultdict(int)
    
    # Regular expression to extract form ID and writer ID
    form_pattern = re.compile(r'([a-z]\d{2}-\d{3}[a-z]?)\.xml')
    
    # Scan metadata directory
    for filename in os.listdir(metadata_dir):
        if not filename.endswith('.xml'):
            continue
            
        match = form_pattern.match(filename)
        if match:
            # Extract form ID (e.g., 'a01-000')
            form_id = match.group(1)
            print(form_id)
            if form_id[-1].isalpha():
                form_id = form_id[:-1]
            
            form_frequencies[form_id] += 1
    
    return form_frequencies

def print_statistics(form_frequencies):
    """Print sorted statistics about forms and writers."""
    print("Form Frequencies:")
    print("-" * 40)
    
    # Sort forms by frequency
    sorted_forms = sorted(form_frequencies.items(), 
                         key=lambda x: (-x[1], x[0]))
    
    for form_id, freq in sorted_forms[:20]:  # Top 20 forms
        print(f"{form_id}: {freq} occurrences")
    
    # Print total statistics
    print("\nSummary:")
    print("-" * 40)
    print(f"Total unique forms: {len(form_frequencies)}")
    print(f"Total form instances: {sum(form_frequencies.values())}")

def save_statistics(form_frequencies, output_file="form_statistics.txt"):
    """Save detailed statistics to a file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Form Frequencies:\n")
        f.write("-" * 40 + "\n")
        
        sorted_forms = sorted(form_frequencies.items(), 
                            key=lambda x: (-x[1], x[0]))
        
        for form_id, freq in sorted_forms:
            f.write(f"{form_id}: {freq} occurrences\n")
        
        f.write("\nWriter Frequencies:\n")
        f.write("-" * 40 + "\n")
        
        f.write("\nSummary:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total unique forms: {len(form_frequencies)}\n")
        f.write(f"Total form instances: {sum(form_frequencies.values())}\n")

if __name__ == "__main__":
    metadata_dir = "metadata"
    
    # Analyze frequencies
    form_freqs= analyze_form_frequencies(metadata_dir)
    
    # Print statistics to console
    print_statistics(form_freqs)
    
    # Save detailed statistics to file
    save_statistics(form_freqs)