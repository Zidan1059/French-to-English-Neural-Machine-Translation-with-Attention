import torch
import torch.nn as nn
import torch.nn.functional as F
import json


class NMTModel(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        embed_size=256,
        hidden_size=256,
        dropout=0.3,
        src_pad_idx=0,
        tgt_pad_idx=0,
    ):
        super().__init__()

        # store hidden dimension
        self.hidden_size = hidden_size

        # source and target embeddings
        self.src_embed_layer = nn.Embedding(src_vocab_size, embed_size, padding_idx=src_pad_idx)
        self.tgt_embed_layer = nn.Embedding(tgt_vocab_size, embed_size, padding_idx=tgt_pad_idx)

        # bidirectional encoder
        self.encoder_lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # decoder LSTM cell
        self.decoder_cell = nn.LSTMCell(
            input_size=embed_size + hidden_size,
            hidden_size=hidden_size,
        )

        # projection layers
        self.hidden_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.cell_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)

        self.attention_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)

        self.output_proj = nn.Linear(3 * hidden_size, hidden_size, bias=False)

        self.vocab_proj = nn.Linear(hidden_size, tgt_vocab_size, bias=False)

        # dropout layer
        self.dropout_layer = nn.Dropout(dropout)

    def encode(self, src_ids):
        """Encode source sentence"""

        # embed source tokens
        src_embedded = self.dropout_layer(self.src_embed_layer(src_ids))

        # run bidirectional encoder
        enc_states, (final_hidden, final_cell) = self.encoder_lstm(src_embedded)

        # concatenate forward and backward states
        hidden_concat = torch.cat([final_hidden[0], final_hidden[1]], dim=1)
        cell_concat = torch.cat([final_cell[0], final_cell[1]], dim=1)

        # project into decoder space
        decoder_init_hidden = self.hidden_proj(hidden_concat)
        decoder_init_cell = self.cell_proj(cell_concat)

        return enc_states, (decoder_init_hidden, decoder_init_cell)

    def step(self, y_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_mask, prev_output):
        """Single decoding step"""

        prev_hidden, prev_cell = dec_state

        # embed target token
        y_embed = self.dropout_layer(self.tgt_embed_layer(y_t))

        # concatenate with previous combined output
        decoder_input = torch.cat([y_embed, prev_output], dim=1)

        # decoder LSTM step
        next_hidden, next_cell = self.decoder_cell(decoder_input, (prev_hidden, prev_cell))

        # attention score computation
        attention_scores = torch.bmm(
            enc_hiddens_proj,
            next_hidden.unsqueeze(2)
        ).squeeze(2)

        if enc_mask is not None:
            attention_scores = attention_scores.masked_fill(enc_mask == float("-inf"), -1e9)

        # attention distribution
        attention_weights = F.softmax(attention_scores, dim=1)

        # compute context vector
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),
            enc_hiddens
        ).squeeze(1)

        # combine context and decoder hidden state
        combined_input = torch.cat([context_vector, next_hidden], dim=1)

        combined_output = self.dropout_layer(torch.tanh(self.output_proj(combined_input)))

        # vocabulary logits
        vocab_logits = self.vocab_proj(combined_output)

        return (next_hidden, next_cell), combined_output, vocab_logits, attention_weights

    def forward(self, src_ids, tgt_ids, enc_mask=None):
        """Forward pass for training"""

        enc_hiddens, decoder_init = self.encode(src_ids)

        enc_hiddens_projected = self.attention_proj(enc_hiddens)

        batch_size = src_ids.size(0)
        tgt_length = tgt_ids.size(1)

        dec_state = decoder_init

        prev_output = torch.zeros(batch_size, self.hidden_size, device=src_ids.device)

        logits_collection = []

        for t in range(tgt_length - 1):

            current_token = tgt_ids[:, t]

            dec_state, prev_output, step_logits, _ = self.step(
                current_token,
                dec_state,
                enc_hiddens,
                enc_hiddens_projected,
                enc_mask,
                prev_output
            )

            logits_collection.append(step_logits.unsqueeze(1))

        return torch.cat(logits_collection, dim=1)


def load_vocab(vocab_path):
    """Load vocabulary JSON file"""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    return vocab_data