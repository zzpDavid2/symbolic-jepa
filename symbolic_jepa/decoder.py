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
        data_token = self.encoder(points).unsqueeze(1)  # (batch, 1, d_model)
        tok_emb = self.tok_embed(input_ids)
        x = torch.cat([data_token, tok_emb], dim=1)
        pos = torch.arange(seq + 1, device=x.device)
        x = x + self.pos_embed(pos).unsqueeze(0)
        return self.drop(x)

    def forward(self, points, input_ids, attn_mask=None):
        x = self._build_inputs(points, input_ids)
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

        pred_logits = logits[:, :-1, :]
        targets = input_ids

        loss = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.pad_id,
        )
        return {'loss': loss, 'logits': logits}

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
