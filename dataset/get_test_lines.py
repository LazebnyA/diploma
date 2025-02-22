import re


def find_form_lines(mappings_file, form_id):
    """Find all lines from a specific form."""
    form_lines = []
    form_pattern = rf'data/{form_id[:3]}/{form_id}[a-z]/'

    with open(mappings_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' not in line:
                continue

            path, text = line.strip().split('\t')

            # Check if path matches the form pattern
            if re.match(form_pattern, path):
                form_lines.append((path, text))

    return form_lines


def save_results(lines, output_file):
    """Save found lines to a file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for path, text in lines:
            f.write(f"{path}\t{text}\n")


if __name__ == "__main__":
    mappings_file = "line_mappings.txt"
    form_id = "g06-042"  # Form to search for
    output_file = "g06_042_lines.txt"

    # Find lines
    found_lines = find_form_lines(mappings_file, form_id)

    # Print results
    print(f"Found {len(found_lines)} lines from form {form_id}:")
    print("-" * 60)

    for path, text in found_lines:
        print(f"{path}\t{text}")

    # Save results
    save_results(found_lines, output_file)
    print(f"\nResults saved to {output_file}")
