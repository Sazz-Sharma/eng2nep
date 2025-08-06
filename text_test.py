from pathlib import Path
import sys

corpus_files = ("dataset/archive/1_Eng.txt", "dataset/archive/1_Nepali.txt")

if not Path(corpus_files[0]).exists() or not Path(corpus_files[1]).exists():
    print("Corpus files not found. Please ensure the dataset is available.")
    sys.exit(1)

# count lines
def count_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

for file in corpus_files:
    num_lines = count_lines(file)
    print(f"File: {file} has {num_lines} lines.")
    