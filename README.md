# French-to-English Neural Machine Translation with Attention

A PyTorch implementation of a **French → English neural machine translation (NMT)** pipeline using:

- **WMT14 fr-en** data
- **SentencePiece BPE** tokenization
- **Bidirectional LSTM encoder**
- **Attention-based LSTM decoder**
- **Greedy decoding and beam search**
- **BLEU evaluation**

This project includes the full workflow from dataset creation and preprocessing to training, decoding, and evaluation.

---

## Project Overview

This repository implements a compact end-to-end NMT system for translating **French to English**.

### Main components
- **Dataset preparation** from the Hugging Face `wmt/wmt14` dataset
- **SentencePiece BPE tokenization** with separate source and target vocabularies
- **Sequence-to-sequence model** with:
  - source embeddings
  - target embeddings
  - bidirectional LSTM encoder
  - attention mechanism
  - LSTMCell decoder
- **Training loop** with:
  - label smoothing
  - gradient clipping
  - learning rate scheduling
  - early stopping
- **Inference** with greedy decoding and beam search
- **Evaluation** using corpus BLEU and qualitative decoded examples

---

## Repository Structure

```text
.
├── data/
│   ├── raw/                # raw parallel text files
│   ├── bpe/                # SentencePiece models
│   ├── tokenized/          # optional tokenized intermediates
│   └── processed/          # id-based datasets and vocab json files
│
├── outputs/
│   ├── checkpoints/        # saved model checkpoints
│   ├── logs/               # training history and loss curves
│   └── predictions/        # decoded examples and evaluation results
│
├── src/
│   ├── toy_dataset.py      # dataset sampling/filtering from WMT14
│   ├── preprocess.py       # SentencePiece training + id conversion
│   ├── model.py            # NMT model definition
│   ├── train.py            # training pipeline
│   ├── decode.py           # greedy and beam-search decoding
│   └── evaluate.py         # BLEU evaluation on test set
│
└── README.md
