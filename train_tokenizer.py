from pathlib import Path
import sys

from custom_tokenizer import CustomBPETokenizer

def train_and_save_tokenizer():
    # Initialize the tokenizer with a specific vocabulary size
    tokenizer = CustomBPETokenizer()

    corpus_files = ("dataset/archive/1_Eng.txt", "dataset/archive/1_Nepali.txt")

    if not Path(corpus_files[0]).exists() or not Path(corpus_files[1]).exists():
        print("Corpus files not found. Please ensure the dataset is available.")
        return
    
    tokenizer = CustomBPETokenizer()
    tokenizer.train(corpus_files)
    save_path = "trainedBPE/tokenizer.json"
    tokenizer.save(save_path)

train_and_save_tokenizer()
