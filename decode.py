from pathlib import Path
import json
import torch

# Ensure your model.py is in the same directory or python path
from model import NMTModel, load_vocab

# -------------------------------
# Project directory configuration
# -------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
CKPT_DIR = ROOT_DIR / "outputs" / "checkpoints"
RESULT_DIR = ROOT_DIR / "outputs" / "predictions"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

SRC_VOCAB_FILE = DATA_DIR / "src_vocab.json"
TGT_VOCAB_FILE = DATA_DIR / "tgt_vocab.json"
TEST_DATA_FILE = DATA_DIR / "test_ids.pt"
MODEL_FILE = CKPT_DIR / "best_model.pt"

# -------------------------------
# Special token IDs
# -------------------------------
PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN = 0, 1, 2, 3

# -------------------------------
# Decoding configuration
# -------------------------------
BEAM_WIDTH = 3 # As required by Problem 2(d)
MAX_LEN = 50   # As required by Problem 2(a)
NUM_SAMPLES = 10

def get_device():
    """Return available device (MPS for Mac, CUDA for Nvidia, otherwise CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def reverse_vocab(vocab_dict):
    """Create ID -> token mapping."""
    return {idx: token for token, idx in vocab_dict.items()}

def pad_batch_sequences(seq_list, pad_token):
    """Pad a list of sequences to the same length."""
    longest_len = max(len(seq) for seq in seq_list)
    padded_list = [seq + [pad_token] * (longest_len - len(seq)) for seq in seq_list]
    return torch.tensor(padded_list, dtype=torch.long)

def build_encoder_mask(src_tensor, pad_token):
    """Create padding mask for encoder."""
    mask_tensor = torch.zeros_like(src_tensor, dtype=torch.float32)
    mask_tensor = mask_tensor.masked_fill(src_tensor == pad_token, float("-inf"))
    return mask_tensor

def ids_to_token_list(id_list, id2word, stop_eos=False):
    """Convert token IDs to token strings, filtering special tokens."""
    token_list = []
    for token_id in id_list:
        token_id = int(token_id)
        if stop_eos and token_id == EOS_TOKEN:
            break
        if token_id in {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN}:
            continue
        token_list.append(id2word.get(token_id, "<unk>"))
    return token_list

def sentencepiece_to_sentence(token_list):
    """Convert sentencepiece tokens to readable text."""
    text = "".join(token_list).replace(" ", " ").replace("▁", " ").strip()
    return " ".join(text.split())

class BeamHypothesis:
    """Represents a hypothesis in beam search."""
    def __init__(self, tokens, hidden_state, cell_state, prev_output, score):
        self.tokens = tokens
        self.hidden_state = hidden_state
        self.cell_state = cell_state
        self.prev_output = prev_output
        self.score = score 

    @property
    def last_token(self):
        return self.tokens[-1]

    def is_finished(self):
        return self.last_token == EOS_TOKEN

    @property
    def normalized_score(self):
        # Length penalty to normalize scores by length
        return self.score / (len(self.tokens) ** 0.6) if len(self.tokens) > 0 else self.score

def load_model_and_dataset(device):
    """Load vocabularies, model checkpoint, and test samples."""
    src_vocab = load_vocab(SRC_VOCAB_FILE)
    tgt_vocab = load_vocab(TGT_VOCAB_FILE)
    id2src = reverse_vocab(src_vocab)
    id2tgt = reverse_vocab(tgt_vocab)

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

    test_examples = torch.load(TEST_DATA_FILE, map_location="cpu")
    return model, src_vocab, tgt_vocab, id2src, id2tgt, test_examples

def encode_input(model, src_ids, device):
    """Run encoder forward pass for a source sentence."""
    src_tensor = pad_batch_sequences([src_ids], PAD_TOKEN).to(device)
    src_mask = build_encoder_mask(src_tensor, PAD_TOKEN).to(device)

    # Returns enc_states and (init_hidden, init_cell)
    enc_outputs, decoder_init = model.encode(src_tensor)
    
    # Matching your model.py attribute: attention_proj
    enc_proj = model.attention_proj(enc_outputs)

    return src_tensor, src_mask, enc_outputs, enc_proj, decoder_init

def greedy_decode(model, src_ids, device):
    """Greedy decoding implementation."""
    _, src_mask, enc_outputs, enc_proj, decoder_init = encode_input(model, src_ids, device)

    hidden_state, cell_state = decoder_init
    prev_output = torch.zeros(1, model.hidden_size, device=device)
    current_token = torch.tensor([BOS_TOKEN], dtype=torch.long, device=device)

    generated_ids = []

    for _ in range(MAX_LEN):
        # Updated keyword argument 'prev_output' to match NMTModel.step()
        (hidden_state, cell_state), prev_output, logits, _ = model.step(
            y_t=current_token,
            dec_state=(hidden_state, cell_state),
            enc_hiddens=enc_outputs,
            enc_hiddens_proj=enc_proj,
            enc_mask=src_mask,
            prev_output=prev_output
        )

        current_token = torch.argmax(logits, dim=1)
        predicted_id = int(current_token.item())

        if predicted_id == EOS_TOKEN:
            break
        generated_ids.append(predicted_id)

    return generated_ids

def beam_search_decode(model, src_ids, device):
    """Beam search decoding implementation."""
    _, src_mask, enc_outputs, enc_proj, decoder_init = encode_input(model, src_ids, device)

    init_hidden, init_cell = decoder_init
    init_prev_output = torch.zeros(1, model.hidden_size, device=device)

    active_beams = [
        BeamHypothesis(
            tokens=[BOS_TOKEN],
            hidden_state=init_hidden,
            cell_state=init_cell,
            prev_output=init_prev_output,
            score=0.0,
        )
    ]

    finished_beams = []

    for _ in range(MAX_LEN):
        if len(active_beams) == 0:
            break

        candidate_beams = []
        for beam in active_beams:
            if beam.is_finished():
                finished_beams.append(beam)
                continue

            current_token = torch.tensor([beam.last_token], dtype=torch.long, device=device)

            # Updated keyword argument 'prev_output' to match NMTModel.step()
            (next_h, next_c), next_out, logits, _ = model.step(
                y_t=current_token,
                dec_state=(beam.hidden_state, beam.cell_state),
                enc_hiddens=enc_outputs,
                enc_hiddens_proj=enc_proj,
                enc_mask=src_mask,
                prev_output=beam.prev_output
            )

            log_probs = torch.log_softmax(logits, dim=1).squeeze(0)
            top_scores, top_tokens = torch.topk(log_probs, BEAM_WIDTH)

            for score, token in zip(top_scores.tolist(), top_tokens.tolist()):
                new_beam = BeamHypothesis(
                    tokens=beam.tokens + [int(token)],
                    hidden_state=next_h.clone(),
                    cell_state=next_c.clone(),
                    prev_output=next_out.clone(),
                    score=beam.score + float(score),
                )
                candidate_beams.append(new_beam)

        if not candidate_beams:
            break

        candidate_beams.sort(key=lambda h: h.normalized_score, reverse=True)
        active_beams = []

        for beam in candidate_beams:
            if beam.is_finished():
                finished_beams.append(beam)
            else:
                active_beams.append(beam)

            if len(active_beams) >= BEAM_WIDTH:
                break

        if len(finished_beams) >= BEAM_WIDTH:
            finished_beams.sort(key=lambda h: h.normalized_score, reverse=True)
            finished_beams = finished_beams[:BEAM_WIDTH]

        if not active_beams:
            break

    # If no beams finished naturally, take the best active ones
    final_pool = finished_beams if finished_beams else active_beams
    final_pool.sort(key=lambda h: h.normalized_score, reverse=True)
    
    best_beam = final_pool[0]
    decoded_ids = [t for t in best_beam.tokens[1:] if t != EOS_TOKEN]
    return decoded_ids, final_pool[:BEAM_WIDTH]

def decode_examples():
    device = get_device()
    model, _, _, id2src, id2tgt, test_samples = load_model_and_dataset(device)

    all_results = []

    for idx, example in enumerate(test_samples[:NUM_SAMPLES]):
        src_ids = example["src_ids"]
        tgt_ids = example["tgt_ids"]

        greedy_output = greedy_decode(model, src_ids, device)
        beam_output, beam_candidates = beam_search_decode(model, src_ids, device)

        src_tokens = ids_to_token_list(src_ids, id2src)
        ref_tokens = ids_to_token_list(tgt_ids, id2tgt, stop_eos=True)
        greedy_tokens = ids_to_token_list(greedy_output, id2tgt)
        beam_tokens = ids_to_token_list(beam_output, id2tgt)

        result = {
            "example_id": idx,
            "source_text": sentencepiece_to_sentence(src_tokens),
            "reference_text": sentencepiece_to_sentence(ref_tokens),
            "greedy_text": sentencepiece_to_sentence(greedy_tokens),
            "beam_text": sentencepiece_to_sentence(beam_tokens),
            "beam_scores": [beam.score for beam in beam_candidates],
        }
        all_results.append(result)

    json_path = RESULT_DIR / "decoded_examples.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    txt_path = RESULT_DIR / "decoded_examples.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(f"Example {item['example_id']}\n")
            f.write(f"Source   : {item['source_text']}\n")
            f.write(f"Reference: {item['reference_text']}\n")
            f.write(f"Greedy   : {item['greedy_text']}\n")
            f.write(f"Beam     : {item['beam_text']}\n")
            f.write("-" * 80 + "\n")

    print(f"Decoding finished successfully. Results saved to {RESULT_DIR}")

if __name__ == "__main__":
    decode_examples()