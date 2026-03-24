from pathlib import Path
import json
import sentencepiece as spm
import torch

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parents[1]

# Data directories
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
TOKEN_DIR = ROOT_DIR / "data" / "tokenized"
BPE_MODEL_DIR = ROOT_DIR / "data" / "bpe"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

# Language settings
SRC_LANGUAGE = "fr"
TGT_LANGUAGE = "en"

# BPE and filtering parameters
BPE_VOCAB_SIZE = 12000
MAX_SEQUENCE_LENGTH = 50

# Special tokens used by SentencePiece
SPECIAL_TOKEN_MAP = {
    "pad": "<pad>",
    "unk": "<unk>",
    "bos": "<bos>",
    "eos": "<eos>",
}

# Ensure required directories exist
for folder in [TOKEN_DIR, BPE_MODEL_DIR, PROCESSED_DATA_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def get_device():
    # Choose computation device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_lines(file_path):
    # Read all lines from a text file
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(text_lines, file_path):
    # Write list of lines to a file
    with open(file_path, "w", encoding="utf-8") as f:
        for line in text_lines:
            f.write(line + "\n")


def train_sentencepiece(input_file, model_prefix, vocab_size):
    # Train a SentencePiece BPE model
    spm.SentencePieceTrainer.train(
        input=str(input_file),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece=SPECIAL_TOKEN_MAP["pad"],
        unk_piece=SPECIAL_TOKEN_MAP["unk"],
        bos_piece=SPECIAL_TOKEN_MAP["bos"],
        eos_piece=SPECIAL_TOKEN_MAP["eos"],
        hard_vocab_limit=False,
    )


def load_sp(model_file):
    # Load a SentencePiece model
    sp_processor = spm.SentencePieceProcessor()
    sp_processor.load(str(model_file))
    return sp_processor


def export_vocab_to_json(sp_processor, json_output):
    # Save vocabulary mapping to JSON
    vocab_dict = {sp_processor.id_to_piece(i): i for i in range(sp_processor.get_piece_size())}
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)


def process_and_filter_parallel(src_file, tgt_file, src_tokenizer, tgt_tokenizer, output_file):
    # Read parallel sentences
    src_sentences = read_lines(src_file)
    tgt_sentences = read_lines(tgt_file)

    dataset_entries = []
    total_src_len = 0
    total_tgt_len = 0

    for src_text, tgt_text in zip(src_sentences, tgt_sentences):

        # Convert text to token IDs
        src_ids = src_tokenizer.encode(src_text.strip(), out_type=int) + [src_tokenizer.eos_id()]
        tgt_ids = [tgt_tokenizer.bos_id()] + tgt_tokenizer.encode(tgt_text.strip(), out_type=int) + [tgt_tokenizer.eos_id()]

        # Filter long sequences
        if len(src_ids) <= MAX_SEQUENCE_LENGTH and len(tgt_ids) <= MAX_SEQUENCE_LENGTH:
            dataset_entries.append({"src_ids": src_ids, "tgt_ids": tgt_ids})
            total_src_len += len(src_ids)
            total_tgt_len += len(tgt_ids)

    torch.save(dataset_entries, output_file)

    print(f"File: {output_file.name}")
    print(f"  - Number of sentence pairs after filtering: {len(dataset_entries)}")
    print(f"  - Average Source Length: {total_src_len/len(dataset_entries):.2f}")
    print(f"  - Average Target Length: {total_tgt_len/len(dataset_entries):.2f}")


def pad_sequences(sequence_list, pad_token_id, device=torch.device("cpu")):
    # Pad sequences to the same length
    longest = max(len(seq) for seq in sequence_list)
    padded_sequences = [seq + [pad_token_id] * (longest - len(seq)) for seq in sequence_list]
    return torch.tensor(padded_sequences, dtype=torch.long, device=device)


def create_encoder_padding_mask(src_tensor, pad_token_id):
    # Create mask for padded positions in encoder
    mask_tensor = torch.zeros(src_tensor.shape, dtype=torch.float32, device=src_tensor.device)
    mask_tensor = mask_tensor.masked_fill(src_tensor == pad_token_id, float("-inf"))
    return mask_tensor


def collate_batch(sample_batch, src_pad_token, tgt_pad_token, device=torch.device("cpu")):
    # Prepare a batch for model training
    src_sequences = [sample["src_ids"] for sample in sample_batch]
    tgt_sequences = [sample["tgt_ids"] for sample in sample_batch]

    src_batch_tensor = pad_sequences(src_sequences, src_pad_token, device=device)
    tgt_batch_tensor = pad_sequences(tgt_sequences, tgt_pad_token, device=device)

    src_mask_tensor = create_encoder_padding_mask(src_batch_tensor, src_pad_token)

    return {
        "src_ids": src_batch_tensor,
        "tgt_ids": tgt_batch_tensor,
        "src_padding_mask": src_mask_tensor,
    }


def main():
    device = get_device()

    # Train BPE models
    train_sentencepiece(RAW_DATA_DIR / "train.fr", BPE_MODEL_DIR / "spm_fr", BPE_VOCAB_SIZE)
    train_sentencepiece(RAW_DATA_DIR / "train.en", BPE_MODEL_DIR / "spm_en", BPE_VOCAB_SIZE)

    # Load trained tokenizers
    src_tokenizer = load_sp(BPE_MODEL_DIR / "spm_fr.model")
    tgt_tokenizer = load_sp(BPE_MODEL_DIR / "spm_en.model")

    # Process dataset splits
    process_and_filter_parallel(RAW_DATA_DIR / "train.fr", RAW_DATA_DIR / "train.en", src_tokenizer, tgt_tokenizer, PROCESSED_DATA_DIR / "train_ids.pt")
    process_and_filter_parallel(RAW_DATA_DIR / "valid.fr", RAW_DATA_DIR / "valid.en", src_tokenizer, tgt_tokenizer, PROCESSED_DATA_DIR / "valid_ids.pt")
    process_and_filter_parallel(RAW_DATA_DIR / "test.fr", RAW_DATA_DIR / "test.en", src_tokenizer, tgt_tokenizer, PROCESSED_DATA_DIR / "test_ids.pt")

    # Save vocabularies
    export_vocab_to_json(src_tokenizer, PROCESSED_DATA_DIR / "src_vocab.json")
    export_vocab_to_json(tgt_tokenizer, PROCESSED_DATA_DIR / "tgt_vocab.json")

    print(f"\nVocabulary sizes: Src={src_tokenizer.get_piece_size()}, Tgt={tgt_tokenizer.get_piece_size()}")
    print(f"Preprocessing completed. Files saved in {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()