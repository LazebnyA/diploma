import tarfile
import os

# Path to the .tgz file
# tgz_path = "./iam_lines/lines.tgz"
# tgz_path = "./iam_words_orig/words.tgz"
tgz_path = "./iam_words_orig/xml.tgz"

# Path to the directory where you want to extract the files
extract_to = "./metadata"

# Ensure the destination directory exists
os.makedirs(extract_to, exist_ok=True)

# Extract the .tgz archive
try:
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    print(f"Extraction completed successfully. Files are in: {extract_to}")
except Exception as e:
    print(f"Error during extraction: {e}")