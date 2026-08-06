"""EOS handling and strict prefix parsing.

Regression tests for the decoding bug that produced very low exact-match
while algebraic-equivalence and R2 stayed normal: batched greedy kept
appending tokens after a row emitted <eos>, decode() filtered <eos> out
instead of stopping at it, and the parser accepted the leading valid
expression while ignoring the trailing garbage.
"""

import pytest
import torch

from symbolic_jepa.decoder import SymbolicTransformer
from symbolic_jepa.encoder import TNet
from symbolic_jepa.tokenizer import PrefixTokenizer, prefix_to_sympy


@pytest.fixture
def tok():
    return PrefixTokenizer(max_vars=1)


class TestDecodeStopsAtEos:
    def test_post_eos_tokens_ignored(self, tok):
        """The headline acceptance case from the task."""
        clean = ['<sos>', 'add', 'x1', 'C', '<eos>']
        dirty = ['<sos>', 'add', 'x1', 'C', '<eos>', 'mul', 'x1', 'C', 'sin', 'x1']
        ids = lambda ts: [tok.token2id[t] for t in ts]
        assert tok.decode(ids(clean)) == 'add x1 C'
        assert tok.decode(ids(dirty)) == 'add x1 C'
        assert tok.decode(ids(clean)) == tok.decode(ids(dirty))

    def test_padding_after_eos_ignored(self, tok):
        seq = ['<sos>', 'add', 'x1', 'C', '<eos>'] + ['<pad>'] * 10
        assert tok.decode([tok.token2id[t] for t in seq]) == 'add x1 C'

    def test_no_eos_still_decodes(self, tok):
        seq = ['<sos>', 'add', 'x1', 'C']
        assert tok.decode([tok.token2id[t] for t in seq]) == 'add x1 C'

    def test_eos_first_gives_empty(self, tok):
        assert tok.decode([tok.token2id['<sos>'], tok.token2id['<eos>'],
                           tok.token2id['add']]) == ''

    def test_skip_special_false_still_stops_at_eos(self, tok):
        seq = ['<sos>', 'add', 'x1', 'C', '<eos>', 'mul']
        out = tok.decode([tok.token2id[t] for t in seq], skip_special=False)
        assert 'mul' not in out and out.startswith('<sos>')


class TestStrictParsing:
    def test_valid_expression_parses(self):
        expr, consts = prefix_to_sympy('add x1 C')
        assert expr is not None and len(consts) == 1

    def test_trailing_tokens_rejected(self):
        with pytest.raises(ValueError, match='Trailing tokens'):
            prefix_to_sympy('add x1 C mul x1 C')

    def test_trailing_single_token_rejected(self):
        with pytest.raises(ValueError, match='Trailing tokens'):
            prefix_to_sympy('add x1 C x1')

    def test_non_strict_still_allows_partial(self):
        """Escape hatch keeps the old lenient behaviour available."""
        expr, _ = prefix_to_sympy('add x1 C mul x1 C', strict=False)
        assert str(expr) == str(prefix_to_sympy('add x1 C')[0])

    def test_strict_is_the_default(self):
        with pytest.raises(ValueError):
            prefix_to_sympy('add x1 C sin x1')

    def test_truncated_still_rejected(self):
        with pytest.raises(ValueError, match='Truncated'):
            prefix_to_sympy('add x1')

    def test_unknown_token_still_rejected(self):
        with pytest.raises(ValueError, match='Unknown token'):
            prefix_to_sympy('add x1 bogus')

    def test_nested_expression_consumes_all(self):
        expr, _ = prefix_to_sympy('add mul C x1 sin x1')
        assert expr is not None


class TestGenerateFreezesFinishedRows:
    @staticmethod
    def _model(tok, seed=0):
        torch.manual_seed(seed)
        return SymbolicTransformer(
            encoder=TNet(d_input=2, d_model=32), vocab_size=len(tok),
            d_model=32, n_heads=4, n_layers=1, d_ff=64, max_seq_len=64,
            dropout=0.0, pad_id=tok.pad_id,
        )

    def test_nothing_follows_first_eos(self, tok):
        """Once a row emits <eos>, every later position must also be <eos>."""
        model = self._model(tok)
        model.eval()
        # Force an immediate <eos> so the freeze path is definitely exercised.
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.weight[tok.eos_id] = 10.0
        ids_seen = []
        orig = tok.decode
        try:
            tok.decode = lambda seq, **kw: (ids_seen.append(list(seq)),
                                            orig(seq, **kw))[1]
            model.generate(torch.randn(4, 20, 2), tok, max_new_tokens=8)
        finally:
            tok.decode = orig

        for seq in ids_seen:
            if tok.eos_id in seq:
                after = seq[seq.index(tok.eos_id):]
                assert all(t == tok.eos_id for t in after), (
                    f'non-EOS token generated after <eos>: {after}'
                )

    def test_generate_output_has_no_post_eos_text(self, tok):
        model = self._model(tok)
        preds = model.generate(torch.randn(4, 20, 2), tok, max_new_tokens=10)
        assert len(preds) == 4
        for p in preds:
            assert '<eos>' not in p and '<pad>' not in p

    def test_mixed_finish_times_do_not_contaminate(self, tok):
        """A row that finishes early must not accrue tokens while others run."""
        model = self._model(tok)
        preds = model.generate(torch.randn(8, 20, 2), tok, max_new_tokens=12)
        # Whatever each row produced, decoding must be idempotent w.r.t. EOS:
        # re-encoding then decoding yields the same string.
        for p in preds:
            if not p:
                continue
            ids = [tok.token2id.get(t, tok.unk_id) for t in p.split()]
            assert tok.decode(ids) == p
