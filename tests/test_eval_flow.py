"""Test the full evaluation flow locally to catch hangs/leaks.

Exercises: evaluate_predictions → cleanup_eval_pool → next run cycle.
"""

import time
import torch
import numpy as np

from symbolic_jepa.tokenizer import PrefixTokenizer
from symbolic_jepa.expressions import Expression
from symbolic_jepa.dataset import PointCloudDataset
from symbolic_jepa.encoder import TNet
from symbolic_jepa.decoder import SymbolicTransformer
from symbolic_jepa.evaluation import evaluate_predictions, cleanup_eval_pool


def _build_small_model():
    tok = PrefixTokenizer(max_vars=1)
    enc = TNet(d_input=2, d_model=32)
    model = SymbolicTransformer(
        encoder=enc, vocab_size=len(tok), d_model=32,
        n_heads=4, n_layers=1, d_ff=64,
        max_seq_len=64, dropout=0.0, pad_id=tok.pad_id,
    )
    model.eval()
    return model, tok


def test_eval_does_not_hang():
    """evaluate_predictions + cleanup should complete in < 60s."""
    model, tok = _build_small_model()

    formulas = ["x + 1", "x**2", "sin(x)", "cos(x)", "x**3"]
    exprs = [Expression.from_infix(f) for f in formulas]
    ds = PointCloudDataset(exprs, tok, n_points=100, max_vars=1, resample=False)

    # Generate predictions (greedy)
    preds = []
    for i in range(len(ds)):
        sample = ds[i]
        points = sample['points'].unsqueeze(0)
        input_ids = sample['input_ids']
        pred_str = model.generate(points, tok, max_new_tokens=64)[0]
        gt_str = tok.decode(input_ids.tolist())
        preds.append((gt_str, pred_str))

    # This is the call that was hanging
    t0 = time.time()
    results = evaluate_predictions(preds, ds, tok)
    cleanup_eval_pool()
    elapsed = time.time() - t0

    print(f"  evaluate_predictions + cleanup: {elapsed:.1f}s")
    assert elapsed < 60, f"Evaluation took {elapsed:.0f}s — likely hanging"
    assert 'exact_match' in results
    assert results['n_total'] == len(preds)


def test_two_consecutive_evals():
    """Simulate two sweep runs back-to-back — the second must not hang."""
    model, tok = _build_small_model()

    formulas = ["x + 1", "x**2", "sin(x)"]
    exprs = [Expression.from_infix(f) for f in formulas]
    ds = PointCloudDataset(exprs, tok, n_points=100, max_vars=1, resample=False)

    preds = []
    for i in range(len(ds)):
        sample = ds[i]
        points = sample['points'].unsqueeze(0)
        gt_str = tok.decode(sample['input_ids'].tolist())
        pred_str = model.generate(points, tok, max_new_tokens=64)[0]
        preds.append((gt_str, pred_str))

    for run in range(2):
        t0 = time.time()
        results = evaluate_predictions(preds, ds, tok)
        cleanup_eval_pool()
        elapsed = time.time() - t0
        print(f"  Run {run+1}: {elapsed:.1f}s, exact={results['exact_match']*100:.0f}%")
        assert elapsed < 60, f"Run {run+1} took {elapsed:.0f}s"


def test_eval_with_mismatched_predictions():
    """Mismatched pred/gt forces sympy equivalence checking (slow path)."""
    model, tok = _build_small_model()

    formulas = ["x + 1", "x**2", "sin(x)", "cos(x)", "x**3"]
    exprs = [Expression.from_infix(f) for f in formulas]
    ds = PointCloudDataset(exprs, tok, n_points=100, max_vars=1, resample=False)

    # Deliberately wrong predictions to force equations_equivalent calls
    wrong_preds = [
        "add x1 C", "mul x1 x1", "sin add x1 C",
        "cos mul C x1", "pow x1 three",
    ]
    preds = []
    for i in range(len(ds)):
        gt_str = tok.decode(ds[i]['input_ids'].tolist())
        pred_str = wrong_preds[i % len(wrong_preds)]
        preds.append((gt_str, pred_str))

    t0 = time.time()
    results = evaluate_predictions(preds, ds, tok)
    cleanup_eval_pool()
    elapsed = time.time() - t0

    print(f"  Mismatched eval + cleanup: {elapsed:.1f}s")
    assert elapsed < 60, f"Took {elapsed:.0f}s — likely hanging"


if __name__ == "__main__":
    print("Test 1: eval does not hang")
    test_eval_does_not_hang()
    print("  PASSED\n")

    print("Test 2: two consecutive evals")
    test_two_consecutive_evals()
    print("  PASSED\n")

    print("Test 3: mismatched predictions (forces sympy)")
    test_eval_with_mismatched_predictions()
    print("  PASSED\n")

    print("All tests passed!")
