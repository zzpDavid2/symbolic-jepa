"""
Symbolic Transformer decoder.

T-Net embedding is prepended as a data token to the equation tokens,
then standard causal self-attention over the whole sequence.
Includes greedy and beam-search inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymbolicTransformer(nn.Module):
    def __init__(self, encoder, vocab_size: int, d_model: int = 512,
                 n_heads: int = 8, n_layers: int = 4, d_ff: int = 2048,
                 max_seq_len: int = 128, dropout: float = 0.2, pad_id: int = 0):
        super().__init__()
        self.encoder = encoder
        self.d_model = d_model
        self.pad_id = pad_id
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len + 1, d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_embed.weight  # weight tying

    def _build_inputs(self, points, input_ids):
        batch, seq = input_ids.shape
        z_num = self.encoder(points)                     # (batch, d_model)
        data_token = z_num.unsqueeze(1)                  # (batch, 1, d_model)
        tok_emb = self.tok_embed(input_ids)
        x = torch.cat([data_token, tok_emb], dim=1)
        pos = torch.arange(seq + 1, device=x.device)
        x = x + self.pos_embed(pos).unsqueeze(0)
        return self.drop(x), z_num

    def forward(self, points, input_ids, attn_mask=None):
        x, z_num = self._build_inputs(points, input_ids)
        seq_len = x.shape[1]

        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        if attn_mask is not None:
            data_attn = torch.ones(
                attn_mask.shape[0], 1, device=attn_mask.device, dtype=attn_mask.dtype,
            )
            full_attn = torch.cat([data_attn, attn_mask], dim=1)
            key_padding_mask = (full_attn == 0)
        else:
            key_padding_mask = None

        h = self.transformer(x, mask=causal, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        logits = self.head(h)  # (batch, 1+seq_len, vocab_size)

        # logits[:, 0, :] is the data-token predicting <sos> — trivially
        # learnable and inflates metrics.  Skip it: targets start at position 1
        # of input_ids (the first real equation token after <sos>).
        pred_logits = logits[:, 1:-1, :]      # (batch, seq-1, vocab)
        targets = input_ids[:, 1:]             # (batch, seq-1)

        flat_logits = pred_logits.reshape(-1, pred_logits.size(-1))
        flat_targets = targets.reshape(-1)
        loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=self.pad_id)
        n_valid = (flat_targets != self.pad_id).sum().item()
        return {'loss': loss, 'logits': logits, 'z_num': z_num, 'n_tokens': n_valid}

    def encode_expression(self, input_ids, attn_mask=None):
        """Encode equation tokens without data token -> last-token-pooled z_sym.

        Used for JEPA alignment: produces a symbolic embedding that depends
        only on the expression (no numeric point-cloud information).

        Args:
            input_ids: (batch, seq) token IDs.
            attn_mask: (batch, seq) 1 for real tokens, 0 for padding.
        Returns:
            (batch, d_model) last-token-pooled representation.
        """
        batch, seq = input_ids.shape
        tok_emb = self.tok_embed(input_ids)
        # Positions 1..seq: match where equation tokens sit during normal
        # forward (position 0 is the data-token slot, skipped here).
        pos = torch.arange(1, seq + 1, device=input_ids.device)
        x = self.drop(tok_emb + self.pos_embed(pos).unsqueeze(0))

        causal = torch.triu(
            torch.ones(seq, seq, device=x.device, dtype=torch.bool), diagonal=1,
        )
        key_padding_mask = (attn_mask == 0) if attn_mask is not None else None

        h = self.norm(
            self.transformer(x, mask=causal, src_key_padding_mask=key_padding_mask)
        )  # (batch, seq, d_model)

        # Last non-pad token (sees full expression via causal attention)
        if attn_mask is not None:
            lengths = attn_mask.sum(dim=1).long()       # (batch,)
            last_idx = (lengths - 1).clamp(min=0)       # (batch,)
            return h[torch.arange(batch, device=h.device), last_idx]
        return h[:, -1]

    @torch.no_grad()
    def generate(self, points, tokenizer, max_new_tokens: int = 64):
        """Greedy decoding."""
        self.eval()
        batch = points.shape[0]
        ids = torch.full((batch, 1), tokenizer.sos_id,
                         dtype=torch.long, device=points.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=points.device)

        for _ in range(max_new_tokens):
            out = self.forward(points, ids)
            next_tok = out['logits'][:, -1, :].argmax(dim=-1, keepdim=True)
            finished = finished | (next_tok.squeeze(-1) == tokenizer.eos_id)
            ids = torch.cat([ids, next_tok], dim=1)
            if finished.all():
                break

        return [tokenizer.decode(seq.tolist()) for seq in ids]

    @torch.no_grad()
    def generate_beam(self, points, tokenizer, max_new_tokens: int = 64,
                      beam_width: int = 5, length_penalty: float = 0.0):
        """Beam search decoding. Batch size 1 only."""
        self.eval()
        assert points.shape[0] == 1, 'Beam search supports batch size 1 only'

        beams = [(
            torch.tensor([[tokenizer.sos_id]], device=points.device),
            0.0,
            False,
        )]

        for _ in range(max_new_tokens):
            if all(b[2] for b in beams):
                break

            candidates = []
            for ids, log_prob, finished in beams:
                if finished:
                    candidates.append((ids, log_prob, True))
                    continue

                out = self.forward(points, ids)
                logits = out['logits'][:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                top_lp, top_ids = log_probs[0].topk(beam_width)

                for tok_lp, tok_id in zip(top_lp.tolist(), top_ids.tolist()):
                    new_ids = torch.cat([
                        ids,
                        torch.tensor([[tok_id]], device=points.device),
                    ], dim=1)
                    candidates.append((
                        new_ids,
                        log_prob + tok_lp,
                        tok_id == tokenizer.eos_id,
                    ))

            def score(c):
                return c[1] / (c[0].shape[1] ** length_penalty)

            candidates.sort(key=score, reverse=True)
            beams = candidates[:beam_width]

        best = max(beams, key=lambda c: c[1] / (c[0].shape[1] ** length_penalty))
        return [tokenizer.decode(best[0].squeeze(0).tolist())]
