"""Prefix <-> infix conversion and the upstream SymbolicGPT dataset interface.

The baseline notebook feeds our expressions to upstream SymbolicGPT as
character-level infix skeletons and scores its predictions with our prefix-based
evaluator.  Both directions therefore have to be exact: a lossy conversion would
show up as a fake accuracy gap between the baseline and our own runs.
"""

import math

import pytest
import torch

from symbolic_jepa.expressions import Expression, VarMeta
from symbolic_jepa.tokenizer import PrefixTokenizer, prefix_to_sympy
from symbolic_jepa.dataset import PointCloudDataset
from symbolic_jepa.symbolicgpt_adapter import (
    SymbolicGPTDataset, build_char_vocab, check_roundtrip, decode_prediction,
    infix_to_prefix, prefix_to_infix,
)


PREFIXES = [
    'x1',
    'C',
    'pi',
    'e',
    'two',
    'neg1',
    'half',
    'neghalf',
    'add x1 C',
    'mul C x1',
    'pow x1 two',
    'pow x1 neg2',
    'pow add C mul C x1 neg1',
    'sin add C mul C x1',
    'mul mul C cosh mul C x1 exp mul C pow x1 two',
    'add mul C log add C pow x1 two tanh add C mul C x1',
    'neg x1',
    'neg add x1 C',
    'mul neg1 pow x1 half',
]


class TestRoundTrip:
    @pytest.mark.parametrize('prefix', PREFIXES)
    def test_exact_roundtrip(self, prefix):
        assert infix_to_prefix(prefix_to_infix(prefix)) == prefix

    def test_infix_is_readable(self):
        assert prefix_to_infix('add mul C x1 sin x1') == '((C*x1)+sin(x1))'
        assert prefix_to_infix('pow x1 neg2') == '(x1**(-2))'
        assert prefix_to_infix('neg x1') == '(-x1)'

    def test_semantics_preserved(self):
        """Round-tripped prefix parses to the same SymPy expression."""
        for prefix in PREFIXES:
            before, _ = prefix_to_sympy(prefix)
            after, _ = prefix_to_sympy(infix_to_prefix(prefix_to_infix(prefix)))
            assert (before - after).simplify() == 0

    def test_check_roundtrip_reports_clean(self):
        report = check_roundtrip(PREFIXES)
        assert report['n_checked'] == len(PREFIXES)
        assert report['n_failed'] == 0


class TestRejectsBadInput:
    """A malformed prediction must raise, so the evaluator counts it unparseable."""

    @pytest.mark.parametrize('text', [
        '',
        '(((C*x1)',            # unbalanced
        '(C*x1))',             # trailing text
        '(C%x1)',              # unknown operator
        'foo(x1)',             # unknown identifier
        '(C*7)',               # numeric literal outside the vocabulary
        '(C*)',                # missing operand
        'sin x1',              # function without parentheses
    ])
    def test_raises(self, text):
        with pytest.raises(ValueError):
            infix_to_prefix(text)

    def test_prefix_side_rejects_garbage(self):
        with pytest.raises(ValueError):
            prefix_to_infix('add x1')          # truncated
        with pytest.raises(ValueError):
            prefix_to_infix('x1 C')            # trailing tokens
        with pytest.raises(ValueError):
            prefix_to_infix('frobnicate x1')   # unknown token


class TestCharVocab:
    def test_contains_upstream_specials(self):
        vocab = build_char_vocab(['(C*x1)'])
        for ch in ('_', 'T', '<', '>', ':'):
            assert ch in vocab
        assert vocab == sorted(set(vocab)), 'vocabulary must be sorted and unique'


@pytest.fixture
def tiny_dataset():
    tok = PrefixTokenizer(max_vars=1)
    var = [VarMeta(name='x', low=-math.pi, high=math.pi)]
    exprs = [
        Expression.from_infix('sin(x)', variables=var),
        Expression.from_infix('2*x + 1', variables=var),
    ]
    base = PointCloudDataset(exprs, tok, n_points=32, max_seq_len=64,
                             max_vars=1, resample=False)
    eqs = [prefix_to_infix(tok.decode(s['input_ids'].tolist()))
           for s in base.samples]
    chars = build_char_vocab(eqs)
    stoi = {c: i for i, c in enumerate(chars)}
    block = max(len(e) + 2 for e in eqs)
    return base, tok, stoi, block


class TestSymbolicGPTDataset:
    def test_upstream_tuple_and_shapes(self, tiny_dataset):
        base, tok, stoi, block = tiny_dataset
        ds = SymbolicGPTDataset(base, tok, stoi, block)

        inputs, outputs, points, num_vars = ds[0]
        assert inputs.shape == (block,) and outputs.shape == (block,)
        assert inputs.dtype == torch.long and outputs.dtype == torch.long
        # channel-first (numVars + numYs, numPoints), as upstream's tNet wants
        assert points.shape == (2, 32)
        assert int(num_vars) == 1

    def test_next_token_shift_and_padding(self, tiny_dataset):
        base, tok, stoi, block = tiny_dataset
        ds = SymbolicGPTDataset(base, tok, stoi, block)
        itos = ds.itos

        inputs, outputs, _, _ = ds[0]
        eq = ds.equation_strings[0]
        assert ''.join(itos[int(i)] for i in inputs).rstrip('_') == '<' + eq
        assert ''.join(itos[int(i)] for i in outputs).rstrip('_') == eq + '>'
        # outputs are inputs shifted by one over the non-padded region
        assert inputs[1:len(eq) + 1].tolist() == outputs[:len(eq)].tolist()

    def test_padding_id_matches_upstream_convention(self, tiny_dataset):
        base, tok, stoi, block = tiny_dataset
        ds = SymbolicGPTDataset(base, tok, stoi, block)
        assert ds.paddingToken == '_'
        assert ds.paddingID == stoi['_']
        assert ds.vocab_size == len(stoi)
        assert ds.itos[ds.paddingID] == '_'

    def test_prefix_strings_align_with_base_samples(self, tiny_dataset):
        """Evaluator alignment: prefix_strings[i] must be base.samples[i]'s GT."""
        base, tok, stoi, block = tiny_dataset
        ds = SymbolicGPTDataset(base, tok, stoi, block)
        for i, s in enumerate(base.samples):
            assert ds.prefix_strings[i] == tok.decode(s['input_ids'].tolist())
            assert prefix_to_infix(ds.prefix_strings[i]) == ds.equation_strings[i]

    def test_points_match_the_wrapped_dataset(self, tiny_dataset):
        """The adapter only transposes; it must not resample or renormalise."""
        base, tok, stoi, block = tiny_dataset
        ds = SymbolicGPTDataset(base, tok, stoi, block)
        _, _, points, _ = ds[0]
        assert torch.allclose(points, base[0]['points'].transpose(0, 1))


class TestDecodePrediction:
    def test_cuts_at_eos_and_strips_padding(self):
        chars = build_char_vocab(['(C*x1)'])
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for c, i in stoi.items()}
        text = '<(C*x1)>____'
        ids = [stoi[c] for c in text]
        assert decode_prediction(ids, itos) == '(C*x1)'

    def test_unterminated_prediction_still_returns_text(self):
        chars = build_char_vocab(['(C*x1)'])
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for c, i in stoi.items()}
        ids = [stoi[c] for c in '<(C*x1']
        assert decode_prediction(ids, itos) == '(C*x1'
