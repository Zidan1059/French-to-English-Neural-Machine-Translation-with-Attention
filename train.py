from pathlib import Path
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NMTModel, load_vocab


# project directories
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
CKPT_DIR = ROOT_DIR / "outputs" / "checkpoints"
LOG_OUTPUT_DIR = ROOT_DIR / "outputs" / "logs"

CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_PROCESSED_DIR / "train_ids.pt"
VALID_FILE = DATA_PROCESSED_DIR / "valid_ids.pt"
SRC_VOCAB_FILE = DATA_PROCESSED_DIR / "src_vocab.json"
TGT_VOCAB_FILE = DATA_PROCESSED_DIR / "tgt_vocab.json"


# training hyperparameters
RANDOM_SEED = 42
BATCH_SZ = 64
EMBED_DIM = 256
HIDDEN_DIM = 256
DROPOUT_RATE = 0.4
LR = 0.0009
EPOCH_COUNT = 20
GRAD_CLIP = 5.0
SMOOTHING = 0.1


# token ids
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3


def set_seed(seed_value):
    # set random seeds for reproducibility
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


def get_device():
    # select device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TranslationDataset(Dataset):
    # dataset wrapper for translation samples
    def __init__(self, file_path):
        self.dataset_samples = torch.load(file_path, map_location="cpu")

    def __len__(self):
        return len(self.dataset_samples)

    def __getitem__(self, index):
        return self.dataset_samples[index]


def pad_sequences(sequence_list, pad_token):
    # pad sequences to equal length
    longest_len = max(len(seq) for seq in sequence_list)
    padded_sequences = [seq + [pad_token] * (longest_len - len(seq)) for seq in sequence_list]
    return torch.tensor(padded_sequences, dtype=torch.long)


def create_encoder_padding_mask(src_tensor, pad_token):
    # create mask for padded tokens
    mask_tensor = torch.zeros_like(src_tensor, dtype=torch.float32)
    mask_tensor = mask_tensor.masked_fill(src_tensor == pad_token, float("-inf"))
    return mask_tensor


def collate_fn(batch_samples):
    # prepare batch tensors
    src_seq_list = [sample["src_ids"] for sample in batch_samples]
    tgt_seq_list = [sample["tgt_ids"] for sample in batch_samples]

    src_batch_tensor = pad_sequences(src_seq_list, PAD_TOKEN_ID)
    tgt_batch_tensor = pad_sequences(tgt_seq_list, PAD_TOKEN_ID)

    src_mask_tensor = create_encoder_padding_mask(src_batch_tensor, PAD_TOKEN_ID)

    return {
        "src_ids": src_batch_tensor,
        "tgt_ids": tgt_batch_tensor,
        "src_mask": src_mask_tensor,
    }


def compute_batch_loss(model, batch_data, loss_fn, device):
    # compute training loss for one batch
    src_ids = batch_data["src_ids"].to(device)
    tgt_ids = batch_data["tgt_ids"].to(device)
    src_mask = batch_data["src_mask"].to(device)

    logits = model(src_ids, tgt_ids, src_mask)

    target_tokens = tgt_ids[:, 1:]
    vocab_size = logits.size(-1)

    loss = loss_fn(
        logits.reshape(-1, vocab_size),
        target_tokens.reshape(-1)
    )

    batch_sz = src_ids.size(0)
    loss = loss / batch_sz

    return loss


def evaluate(model, data_loader, loss_fn, device):
    # validation evaluation
    model.eval()

    total_loss_value = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch_data in data_loader:
            loss = compute_batch_loss(model, batch_data, loss_fn, device)
            total_loss_value += loss.item()
            total_batches += 1

    return total_loss_value / total_batches if total_batches > 0 else float("inf")


def plot_losses(train_loss_list, valid_loss_list, output_file):
    # plot training and validation curves
    epoch_indices = list(range(1, len(train_loss_list) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_indices, train_loss_list, marker="o", label="Training Loss")
    plt.plot(epoch_indices, valid_loss_list, marker="o", label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_file)
    plt.close()


def main():
    set_seed(RANDOM_SEED)
    device = get_device()

    src_vocab = load_vocab(SRC_VOCAB_FILE)
    tgt_vocab = load_vocab(TGT_VOCAB_FILE)

    train_dataset = TranslationDataset(TRAIN_FILE)
    valid_dataset = TranslationDataset(VALID_FILE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SZ,
        shuffle=True,
        collate_fn=collate_fn,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SZ,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = NMTModel(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_size=EMBED_DIM,
        hidden_size=HIDDEN_DIM,
        dropout=DROPOUT_RATE,
        src_pad_idx=PAD_TOKEN_ID,
        tgt_pad_idx=PAD_TOKEN_ID,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=PAD_TOKEN_ID,
        reduction="sum",
        label_smoothing=SMOOTHING
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    train_loss_history = []
    valid_loss_history = []

    best_valid_loss = float("inf")
    best_model_path = CKPT_DIR / "best_model.pt"

    patience = 5
    patience_counter = 0

    print("Training configuration:")
    print(f"Device: {device}")
    print(f"Learning rate: {LR}")
    print(f"Batch size: {BATCH_SZ}")
    print(f"Epochs: {EPOCH_COUNT}")
    print(f"Dropout: {DROPOUT_RATE}")
    print(f"Gradient clipping tau: {GRAD_CLIP}")
    print(f"Label smoothing: {SMOOTHING}")
    print()

    for epoch_idx in range(1, EPOCH_COUNT + 1):

        model.train()

        epoch_train_loss = 0.0
        batch_count = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch_idx}/{EPOCH_COUNT}")

        for batch_data in progress_bar:

            optimizer.zero_grad()

            loss = compute_batch_loss(model, batch_data, loss_fn, device)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)

            optimizer.step()

            epoch_train_loss += loss.item()
            batch_count += 1

            progress_bar.set_postfix({"train_loss": f"{loss.item():.4f}"})

        avg_train_loss = epoch_train_loss / batch_count
        avg_valid_loss = evaluate(model, valid_loader, loss_fn, device)

        scheduler.step(avg_valid_loss)

        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss_history.append(avg_train_loss)
        valid_loss_history.append(avg_valid_loss)

        print(f"\nEpoch {epoch_idx} completed")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_valid_loss:.4f}")

        epoch_ckpt_path = CKPT_DIR / f"model_epoch_{epoch_idx}.pt"

        torch.save(
            {
                "epoch": epoch_idx,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "valid_loss": avg_valid_loss,
                "config": {
                    "src_vocab_size": len(src_vocab),
                    "tgt_vocab_size": len(tgt_vocab),
                    "embed_size": EMBED_DIM,
                    "hidden_size": HIDDEN_DIM,
                    "dropout": DROPOUT_RATE,
                    "batch_size": BATCH_SZ,
                    "learning_rate": LR,
                    "num_epochs": EPOCH_COUNT,
                    "clip_tau": GRAD_CLIP,
                    "label_smoothing": SMOOTHING,
                },
            },
            epoch_ckpt_path,
        )

        if avg_valid_loss < best_valid_loss:

            best_valid_loss = avg_valid_loss

            torch.save(
                {
                    "epoch": epoch_idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "valid_loss": avg_valid_loss,
                    "config": {
                        "src_vocab_size": len(src_vocab),
                        "tgt_vocab_size": len(tgt_vocab),
                        "embed_size": EMBED_DIM,
                        "hidden_size": HIDDEN_DIM,
                        "dropout": DROPOUT_RATE,
                        "batch_size": BATCH_SZ,
                        "learning_rate": LR,
                        "num_epochs": EPOCH_COUNT,
                        "clip_tau": GRAD_CLIP,
                        "label_smoothing": SMOOTHING,
                    },
                },
                best_model_path,
            )

            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch_idx}")
                break

    plot_losses(
        train_loss_history,
        valid_loss_history,
        LOG_OUTPUT_DIR / "loss_curve.png",
    )

    training_history = {
        "train_losses": train_loss_history,
        "valid_losses": valid_loss_history,
        "best_valid_loss": best_valid_loss,
        "hyperparameters": {
            "learning_rate": LR,
            "batch_size": BATCH_SZ,
            "epochs": EPOCH_COUNT,
            "dropout": DROPOUT_RATE,
            "clip_tau": GRAD_CLIP,
            "embed_size": EMBED_DIM,
            "hidden_size": HIDDEN_DIM,
            "label_smoothing": SMOOTHING,
        },
    }

    with open(LOG_OUTPUT_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)

    print("\nTraining finished successfully.")
    print(f"Best validation loss: {best_valid_loss:.4f}")
    print(f"Best checkpoint saved to: {best_model_path}")
    print(f"Loss plot saved to: {LOG_OUTPUT_DIR / 'loss_curve.png'}")
    print(f"Training history saved to: {LOG_OUTPUT_DIR / 'training_history.json'}")


if __name__ == "__main__":
    main()