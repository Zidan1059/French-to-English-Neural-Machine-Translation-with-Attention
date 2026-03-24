from pathlib import Path
import json

import torch
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from model import NMTModel, load_vocab
from decode import (
    get_device,
    reverse_vocab,  # Changed from invert_vocab
    ids_to_token_list,  # Changed from ids_to_tokens
    sentencepiece_to_sentence,  # Changed from sentencepiece_tokens_to_text
    greedy_decode,
    beam_search_decode,
    PAD_TOKEN,  # Changed from PAD_ID
    BOS_TOKEN,  # Changed from BOS_ID
    EOS_TOKEN,  # Changed from EOS_ID
    BEAM_WIDTH,  # Changed from BEAM_SIZE
    MAX_LEN,  # Changed from MAX_DECODING_LEN
)


# -------------------------------
# Project directories
# -------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_FOLDER = ROOT_DIR / "data" / "processed"
CKPT_FOLDER = ROOT_DIR / "outputs" / "checkpoints"
OUTPUT_FOLDER = ROOT_DIR / "outputs" / "predictions"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

SRC_VOCAB_FILE = DATA_FOLDER / "src_vocab.json"
TGT_VOCAB_FILE = DATA_FOLDER / "tgt_vocab.json"
TEST_FILE = DATA_FOLDER / "test_ids.pt"
MODEL_FILE = CKPT_FOLDER / "best_model.pt"

NUM_SAMPLES_TO_SAVE = 5


def load_model_and_dataset(device):
    """Load trained model and test dataset."""

    src_vocab_dict = load_vocab(SRC_VOCAB_FILE)
    tgt_vocab_dict = load_vocab(TGT_VOCAB_FILE)

    id_to_src = reverse_vocab(src_vocab_dict)
    id_to_tgt = reverse_vocab(tgt_vocab_dict)

    checkpoint_data = torch.load(MODEL_FILE, map_location="cpu")
    config = checkpoint_data["config"]

    model = NMTModel(
        src_vocab_size=config["src_vocab_size"],
        tgt_vocab_size=config["tgt_vocab_size"],
        embed_size=config["embed_size"],
        hidden_size=config["hidden_size"],
        dropout=config["dropout"],
        src_pad_idx=PAD_TOKEN,
        tgt_pad_idx=PAD_TOKEN,
    )

    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.to(device)
    model.eval()

    test_data = torch.load(TEST_FILE, map_location="cpu")

    return model, id_to_src, id_to_tgt, test_data


def evaluate():
    """Run evaluation on the test dataset."""

    device = get_device()

    model, id_to_src, id_to_tgt, test_data = load_model_and_dataset(device)

    bleu_references = []
    greedy_predictions = []
    beam_predictions = []

    saved_examples = []

    progress = tqdm(
        enumerate(test_data),
        total=len(test_data),
        desc="Evaluating test set",
    )

    for idx, item in progress:

        src_sequence = item["src_ids"]
        tgt_sequence = item["tgt_ids"]

        greedy_output_ids = greedy_decode(model, src_sequence, device)

        # Corrected variable names to match decode.py
        beam_output_ids, beam_candidates = beam_search_decode(
            model,
            src_sequence,
            device
        )

        src_tokens = ids_to_token_list(src_sequence, id_to_src)
        ref_tokens = ids_to_token_list(tgt_sequence, id_to_tgt, stop_eos=True)

        greedy_tokens = ids_to_token_list(greedy_output_ids, id_to_tgt)
        beam_tokens = ids_to_token_list(beam_output_ids, id_to_tgt)

        bleu_references.append([ref_tokens])
        greedy_predictions.append(greedy_tokens)
        beam_predictions.append(beam_tokens)

        # Save a few examples for qualitative inspection
        if idx < NUM_SAMPLES_TO_SAVE:
            saved_examples.append({
                "example_id": idx,
                "source_text": sentencepiece_to_sentence(src_tokens),
                "reference_text": sentencepiece_to_sentence(ref_tokens),
                "greedy_text": sentencepiece_to_sentence(greedy_tokens),
                "beam_text": sentencepiece_to_sentence(beam_tokens),
                "source_tokens": src_tokens,
                "reference_tokens": ref_tokens,
                "greedy_tokens": greedy_tokens,
                "beam_tokens": beam_tokens,
            })

    smoothing = SmoothingFunction().method1

    greedy_bleu_score = corpus_bleu(
        bleu_references,
        greedy_predictions,
        smoothing_function=smoothing,
    )

    beam_bleu_score = corpus_bleu(
        bleu_references,
        beam_predictions,
        smoothing_function=smoothing,
    )

    evaluation_results = {
        "greedy_bleu": greedy_bleu_score,
        "beam_bleu": beam_bleu_score,
        "beam_size": BEAM_WIDTH,
        "max_decoding_length": MAX_LEN,
        "num_test_examples": len(test_data),
        "examples": saved_examples,
    }

    json_output = OUTPUT_FOLDER / "evaluation_results.json"
    txt_output = OUTPUT_FOLDER / "evaluation_results.txt"

    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

    with open(txt_output, "w", encoding="utf-8") as f:

        f.write("Evaluation on Test Set\n")
        f.write(f"Number of test examples: {len(test_data)}\n")
        f.write(f"Greedy BLEU: {greedy_bleu_score:.4f}\n")
        f.write(f"Beam BLEU: {beam_bleu_score:.4f}\n")
        f.write(f"Beam size: {BEAM_WIDTH}\n")
        f.write(f"Max decoding length: {MAX_LEN}\n")
        f.write("\n")

        f.write("Five Translation Examples\n")
        f.write("=" * 80 + "\n")

        for example in saved_examples:

            f.write(f"Example {example['example_id']}\n")
            f.write(f"Source   : {example['source_text']}\n")
            f.write(f"Reference: {example['reference_text']}\n")
            f.write(f"Greedy   : {example['greedy_text']}\n")
            f.write(f"Beam     : {example['beam_text']}\n")
            f.write("-" * 80 + "\n")

    print("Evaluation completed successfully.")
    print(f"Device: {device}")
    print(f"Number of test examples: {len(test_data)}")
    print(f"Greedy BLEU: {greedy_bleu_score:.4f}")
    print(f"Beam BLEU: {beam_bleu_score:.4f}")
    print(f"Saved JSON results to: {json_output}")
    print(f"Saved text results to: {txt_output}")


if __name__ == "__main__":
    evaluate()