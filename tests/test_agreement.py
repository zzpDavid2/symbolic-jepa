"""Per-example agreement analysis.

The aggregate gap between two arms cannot distinguish a nested improvement
(everything the baseline solved, plus more) from a rotation (different examples
of similar count).  These tests pin that the analysis actually separates them,
and that it refuses to pair runs that were not scored on the same test set.
"""

import json

import pytest

from symbolic_jepa.agreement import (
    agreement,
    compare_runs,
    load_details,
    load_run_matrix,
    mcnemar_exact,
    stratified_agreement,
)


def _details(flags, gts=None, r2=None):
    """Minimal `details` records: one per test example."""
    n = len(flags)
    gts = gts or [f'add x1 C {i}' for i in range(n)]
    return [
        {'gt': gts[i], 'pred': 'x', 'exact': False, 'equiv': bool(flags[i]),
         'r2': (r2[i] if r2 else (1.0 if flags[i] else 0.0)),
         'parseable': True, 'r2_status': 'ok'}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Nested vs rotated — the distinction the whole module exists to draw
# ---------------------------------------------------------------------------

def test_detects_a_nested_improvement():
    a = _details([1, 1, 0, 0, 0])
    b = _details([1, 1, 1, 1, 0])       # everything A got, plus two

    t = agreement(a, b, name_a='base', name_b='jepa')
    assert t.both == 2 and t.only_a == 0 and t.only_b == 2 and t.neither == 1
    assert t.is_nested
    assert t.churn == 1.0, 'a pure gain has no churn'
    assert 'NESTED' in t.report()


def test_detects_a_rotation_with_the_same_headline_number():
    """Same net gain as the nested case, but a completely different story."""
    a = _details([1, 1, 1, 1, 0, 0, 0, 0])
    b = _details([0, 0, 1, 1, 1, 1, 1, 0])

    t = agreement(a, b, name_a='base', name_b='jepa')
    assert t.a_correct == 4 and t.b_correct == 5      # +1 net, as headline
    assert t.only_a == 2 and t.only_b == 3
    assert not t.is_nested
    assert t.churn == 5.0, 'five examples moved to gain one'
    assert 'NOT nested' in t.report()


def test_churn_flags_a_pure_reshuffle():
    a = _details([1, 1, 0, 0])
    b = _details([0, 0, 1, 1])
    t = agreement(a, b)
    assert t.a_correct == t.b_correct
    assert t.discordant == 4
    assert t.churn == float('inf'), 'no net movement, maximal churn'


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------

def test_mcnemar_uses_only_discordant_pairs():
    """Concordant examples carry no information about the difference."""
    small = agreement(_details([1, 0, 0]), _details([1, 0, 1]))
    padded = agreement(_details([1, 0, 0] + [1] * 500),
                       _details([1, 0, 1] + [1] * 500))
    assert small.pvalue() == padded.pvalue()


def test_mcnemar_exact_matches_the_binomial():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    # 0 vs 8 discordant: two-sided p = 2 * 0.5**8
    assert mcnemar_exact(0, 8) == pytest.approx(2 * 0.5 ** 8)
    assert mcnemar_exact(8, 0) == pytest.approx(2 * 0.5 ** 8)
    assert mcnemar_exact(1, 9) < 0.05
    assert mcnemar_exact(4, 6) > 0.5


def test_metric_selection():
    a = _details([1, 1, 0], r2=[0.95, 0.5, 0.99])
    b = _details([1, 0, 1], r2=[0.99, 0.95, 0.2])

    eq = agreement(a, b, metric='equiv')
    assert (eq.both, eq.only_a, eq.only_b) == (1, 1, 1)

    r2 = agreement(a, b, metric='r2>0.9')
    assert (r2.both, r2.only_a, r2.only_b) == (1, 1, 1)
    assert r2.metric == 'r2>0.9'


def test_non_finite_r2_counts_as_a_miss():
    a = _details([1, 1], r2=[float('nan'), 1.0])
    b = _details([1, 1], r2=[1.0, 1.0])
    t = agreement(a, b, metric='r2>0.9')
    assert t.only_b == 1 and t.both == 1


# ---------------------------------------------------------------------------
# Pairing safety
# ---------------------------------------------------------------------------

def test_refuses_runs_of_different_length():
    with pytest.raises(ValueError, match='not scored on the same test set'):
        agreement(_details([1, 0]), _details([1, 0, 1]))


def test_refuses_runs_with_a_different_test_set():
    """Element-wise pairing is only valid if index i means the same example."""
    a = _details([1, 0, 1], gts=['add x1 C', 'mul x1 C', 'sin x1'])
    b = _details([1, 0, 1], gts=['add x1 C', 'sin x1', 'mul x1 C'])
    with pytest.raises(ValueError, match='ground truth differs'):
        agreement(a, b)


# ---------------------------------------------------------------------------
# Multi-seed pooling
# ---------------------------------------------------------------------------

def test_compare_runs_pools_and_reports_per_seed(capsys):
    runs_a = {42: _details([1, 1, 0, 0]), 7: _details([1, 0, 0, 0])}
    runs_b = {42: _details([1, 1, 1, 0]), 7: _details([1, 1, 0, 0])}

    out = compare_runs(runs_a, runs_b, metric='equiv')
    pooled = out['pooled']
    assert pooled.n == 8
    assert pooled.only_b == 2 and pooled.only_a == 0
    assert pooled.is_nested
    assert out['seeds'] == [7, 42]

    text = capsys.readouterr().out
    assert 'POOLED' in text and 'nested in 2/2' in text


def test_compare_runs_needs_shared_seeds():
    with pytest.raises(ValueError, match='no seeds in common'):
        compare_runs({1: _details([1])}, {2: _details([1])})


def test_stratified_agreement_localises_the_gain():
    """Short expressions gained, long ones unchanged."""
    gts = ['a b', 'a b', 'a b c d e f g h i j k', 'a b c d e f g h i j k']
    a = _details([0, 0, 1, 0], gts=gts)
    b = _details([1, 1, 1, 0], gts=gts)

    strat = stratified_agreement(a, b, bins=(0, 5, 100))
    assert strat['0-5']['only_b'] == 2
    assert strat['5-100']['only_b'] == 0
    assert strat['5-100']['n'] == 2


# ---------------------------------------------------------------------------
# Reading a real checkpoint tree
# ---------------------------------------------------------------------------

def test_load_run_matrix_skips_missing_runs(tmp_path):
    for pe, seed in ((0, 42), (10, 42)):
        d = tmp_path / f'pretrain_{pe}' / f'seed_{seed}' / 'supervised'
        d.mkdir(parents=True)
        (d / 'metrics.json').write_text(json.dumps(
            {'details': _details([1, 0, 1])}))
    # pretrain_10/seed_7 is deliberately absent.
    (tmp_path / 'pretrain_0' / 'seed_7' / 'supervised').mkdir(parents=True)
    (tmp_path / 'pretrain_0' / 'seed_7' / 'supervised' / 'metrics.json'
     ).write_text(json.dumps({'details': _details([1, 1, 1])}))

    matrix = load_run_matrix(tmp_path, [0, 10], [42, 7])
    assert sorted(matrix[0]) == [7, 42]
    assert sorted(matrix[10]) == [42]

    out = compare_runs(matrix[0], matrix[10], verbose=False)
    assert out['seeds'] == [42]


def test_load_details_rejects_metrics_without_details(tmp_path):
    p = tmp_path / 'metrics.json'
    p.write_text(json.dumps({'exact_match': 0.0}))
    with pytest.raises(ValueError, match='no per-example'):
        load_details(p)
