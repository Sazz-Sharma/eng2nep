# English → Nepali NMT (from Scratch)

Status: Under construction

This project builds a Neural Machine Translation (NMT) system from scratch to translate English to Nepali. It includes a custom Byte-Pair Encoding (BPE) tokenizer, embedding layers (token + positional), and a Transformer architecture implemented with PyTorch.

Most of the test cases and scaffolding were authored with the help of AI QA agents.

## Dataset

Primary data source (as provided):
- Kaggle: https://www.kaggle.com/datasets/jigarpanjiyar/english-to-manipuri-dataset
- Files used in this project:
  - `dataset/archive/1_Eng.txt` (∼30M English sentences)
  - `dataset/archive/1_Nepali.txt` (∼30M corresponding Nepali sentences)
  - `dataset/english-nepali.xlsx` (for lightweight testing/spot checks)

Notes:
- The paired text files are used line-aligned (line i in English matches line i in Nepali).
- Files are very large. Use streaming (line-by-line) I/O to avoid MemoryError.
- Consider building smaller samples for quick iterations while developing.

## Repository structure (key files)

- `custom_tokenizer.py` — Custom BPE tokenizer built with Hugging Face Tokenizers.
- `train_tokenizer.py` — Script to (optionally) retrain the tokenizer.
- `tokenizer/tokenizer.json` — Pretrained BPE tokenizer saved (~40k tokens).
- `dummy_corpus.txt` — Small bilingual sample for tokenizer experimentation.
- `embeddings/token_embedding.py` — Trainable token embedding layer with optional scaling and dropout.
- `embeddings/positional_encoding.py` — Sinusoidal (and/or learned) positional encodings.
- `embeddings/test.py` — Quick tests for embeddings and positional encoding.
- `tokenizer/test/test_tokenizer.py` — Tokenizer training/inspection test from within a subpackage.
- `tokenizer/test/inspect_tokenizer.py` — Inspect and load the saved tokenizer.
- `dataset/archive/` — Large raw text files (`1_Eng.txt`, `1_Nepali.txt`).

## Setup

1) Create a Python environment (Python 3.12.3 recommended and used while developing).
2) Install dependencies (adjust to your setup):

```
pip install torch tokenizers transformers numpy requests
```

If you use conda, keep PyTorch installs consistent (do not mix pip/conda for torch in one environment).

## Tokenizer

A pretrained BPE tokenizer is already included and saved at:

- `tokenizer/tokenizer.json`
- Vocabulary size: ~40,000 tokens

Load and use it:

```
from custom_tokenizer import CustomBPETokenizer

tok = CustomBPETokenizer.load("tokenizer/tokenizer.json")
print("vocab:", tok.get_vocab_size())      # ≈ 40000
pad_id = tok.get_pad_token_id()
ids = tok.encode("Hello नमस्ते world!")
text = tok.decode(ids)
print(ids)
print(text)
```

Optional — Retrain (advanced, not required):
- Warning: Full retraining on ~30M aligned sentence pairs is expensive (time/IO/CPU). Do not load entire files into RAM; use line-by-line streaming. Retraining will overwrite `tokenizer/tokenizer.json` — back it up first.
- Command (if you must):
```
python train_tokenizer.py --mode train
```
- For quick experiments, train on a small sample (e.g., `dummy_corpus.txt`) before attempting the full dataset.

## Embeddings

- Token embeddings: `embeddings/token_embedding.py`
  - `TokenEmbedding(vocab_size, embedding_dim, padding_idx=None, dropout=0.0, scale=True)`
  - If `scale=True`, multiplies outputs by sqrt(embedding_dim) (common in Transformers).
- Positional encodings: `embeddings/positional_encoding.py`
  - Sinusoidal positional encoding (no trainable params).
  - Optionally learned positional encoding can be added.

Quick test runner:

```
python embeddings/test.py
```

## Data quality checks (large files)

For extremely large corpora (~30M lines):
- Read line-by-line, not with `f.read()`.
- Detect duplicates and empty lines using streaming passes.
- Create small samples for rapid iteration.

Example shell commands:

- Top repeated lines (approx, external sort):
```
LC_ALL=C sort -S 50% -T /tmp dataset/archive/1_Eng.txt | uniq -c | sort -nr | head -20
```
- Show only duplicated lines:
```
LC_ALL=C sort dataset/archive/1_Eng.txt | uniq -d | head -50
```
- Count empty/whitespace-only lines:
```
grep -c -E '^\s*$' dataset/archive/1_Eng.txt
```

## Roadmap (WIP)

- [x] Custom BPE tokenizer
- [x] Token + Positional embeddings
- [X] Encoder from scratch
- [ ] Decoder from scratch
- [ ] Seq2Seq Transformer 
- [ ] Dataloaders (streaming, bucketing, padding masks)
- [ ] Training loop (mixed precision, checkpoints, metrics)
- [ ] Evaluation and BLEU/SacreBLEU
- [ ] Inference script and minimal UI

This project is actively evolving; expect breaking changes.

## Testing

- Many unit/integration tests were generated and refined by AI QA agents.
- Run quick checks:
```
python embeddings/test.py
python tokenizer/test/test_tokenizer.py
python tokenizer/test/inspect_tokenizer.py
```

## Acknowledgements

- Kaggle dataset: https://www.kaggle.com/datasets/jigarpanjiyar/english-to-manipuri-dataset
- PyTorch, Hugging Face Tokenizers/Transformers

## Disclaimer

This repository is a research/learning project and is still under construction. Large-scale training requires significant compute and careful data handling.
