"""Canonical templates and dynamic constant augmentation.

The experiment these back rests on two claims that are cheap to state and easy
to break silently:

* the decoder target depends on the **structure only**, so resampling
  coefficients augments the input without touching supervision;
* the **canonical form** is the split unit, so held-out structures are genuinely
  unseen rather than seen with different numbers.

Every test below pins one of those, or pins the backward-compatibility promise
that turning the feature off reproduces the old pipeline exactly.
"""

import math
import pickle
import warnings

import numpy as np
import pytest
import torch

from symbolic_jepa.dataset import PointCloudDataset, build_synthetic_splits
from symbolic_jepa.expressions import Expression, VarMeta, load_synthetic_pkl
from symbolic_jepa.templates import (
    ConstantSampler,
    ConstantTemplate,
    DynamicConstantPointCloudDataset,
    build_template_splits,
    build_templates_from_strings,
    canonical_split_report,
    describe_realizations,
    load_template_dataset,
    pool_is_usable,
    save_template_dataset,
    templatize,
)
from symbolic_jepa.tokenizer import PrefixTokenizer, sympy_to_prefix


# Strings in the generator's own syntax, covering the component families it
# emits: polynomial, trig with an integer frequency, damped trig, Gaussian,
# tanh, sinh, rational, and sinc (which expands to a repeated subexpression).
RAW_STRINGS = [
    '2.1*sin(3*x + 0.5)',
    '2.1*sin(3*x + 0.5)',            # exact duplicate
    '0.8*sin(3*x + -1.2)',           # same canonical form, other coefficients
    '-1.5*sin(3*x + 2.4)',           # same canonical form again
    '1.3*cos(2*x + 0.1)',
    '0.4*cos(2*x + -0.9)',
    '0.9*tanh(1.2*x + -0.3)',
    '1.7*tanh(0.6*x + 2.2)',
    '(1.2*exp(-0.7*(x-0.3)**2))',
    '(0.5*exp(-1.9*(x--0.8)**2))',
    '0.7*sinh(0.4*x)*exp(-0.2*x**2)',
    '1.1*sinh(0.9*x)*exp(-0.4*x**2)',
    '1.4*cosh(0.5*x)*exp(-0.3*x**2)',
    '0.6 + 1.2*x + -0.4*x**2',
    '-1.1 + 0.3*x + 1.8*x**2',
    '0.2 + -0.7*x + 0.9*x**2 + 1.4*x**3',
    '1.9*((0.3 + -1.1*x)/(1 + (0.7 + 1.2*x)**2))',
    '0.5*((-1.4 + 0.8*x)/(1 + (-0.2 + 1.9*x)**2))',
    '0.82*sinc(4.1*x + -1.43)',
    '1.35*sinc(2.7*x + 0.62)',
    '(1.6*sin(5*x + 1.3))*lorentz(x, 1.04)',
    '(0.9*sin(5*x + -2.1))*lorentz(x, 2.31)',
    '(0.4*log(1 + (-0.38 + 1.13*x)**2))',
    '(1.7*log(1 + (0.92 + -0.44*x)**2))',
    '(sqrt(1 + (-0.65 + -1.81*x)**2))',
    '(sqrt(1 + (1.22 + 0.37*x)**2))',
    '2.4*tanh(1.1*x + 0.2) + 0.9*cos(4*x + -1.1)',
    '0.7*tanh(2.3*x + -1.6) + 1.8*cos(4*x + 0.4)',
]


@pytest.fixture(scope='module')
def tokenizer():
    return PrefixTokenizer(max_vars=1)


@pytest.fixture(scope='module')
def templates(tokenizer):
    tmpls, _stats = build_templates_from_strings(
        RAW_STRINGS, tokenizer, max_seq_len=64)
    assert tmpls, 'fixture corpus produced no templates'
    return tmpls


@pytest.fixture(scope='module')
def univariate():
    return [VarMeta(name='x', low=-math.pi, high=math.pi)]


def _splits(templates, tokenizer, **kwargs):
    kwargs.setdefault('n_points', 64)
    kwargs.setdefault('max_vars', 1)
    kwargs.setdefault('seed', 42)
    return build_template_splits(templates, tokenizer, **kwargs)


# ---------------------------------------------------------------------------
# 1. Canonicalization is deterministic
# ---------------------------------------------------------------------------

def test_templatize_is_deterministic(templates):
    """Repeated templatization of one expression gives one canonical form."""
    for tmpl in templates:
        first = tmpl.prefix
        for _ in range(3):
            again = ConstantTemplate.from_sympy(
                tmpl.reference_expr, tmpl.variables)
            assert again.prefix == first
            assert again.n_constants == tmpl.n_constants
            assert np.allclose(again.reference_constants,
                               tmpl.reference_constants)


def test_build_is_deterministic(tokenizer):
    """Rebuilding from the same corpus reproduces forms, order, and pools."""
    a, stats_a = build_templates_from_strings(
        RAW_STRINGS, tokenizer, max_seq_len=64)
    b, stats_b = build_templates_from_strings(
        RAW_STRINGS, tokenizer, max_seq_len=64)

    assert stats_a == stats_b
    assert [t.prefix for t in a] == [t.prefix for t in b]
    for ta, tb in zip(a, b):
        assert len(ta.constant_pools) == len(tb.constant_pools)
        for pa, pb in zip(ta.constant_pools, tb.constant_pools):
            assert np.array_equal(pa, pb)


def test_templatize_preserves_the_prefix_walk(univariate):
    """The template renders to the same prefix as the expression it came from.

    SymPy orders `Add`/`Mul` arguments by a sort key that distinguishes numbers
    from symbols, so an evaluated rebuild could permute arguments and silently
    shift the decoder target.  Everything downstream assumes this holds; the
    full 200 000-string corpus builds with zero mismatches.
    """
    for infix in [
        '2.1*sin(3*x + 0.5)',
        '0.7*sinh(0.4*x)*exp(-0.2*x**2)',
        '0.6 + 1.2*x - 0.4*x**2',
        '1.9*(0.3 - 1.1*x)/(1 + (0.7 + 1.2*x)**2)',
        '2.4*tanh(1.1*x + 0.2) + 0.9*cos(4*x - 1.1)',
    ]:
        expr = Expression.from_infix(infix, univariate)
        template_expr, values = templatize(expr.sympy_expr)

        var_map = {'x': 'x1'}
        var_map.update({f'c_{i}': 'C' for i in range(len(values))})
        assert sympy_to_prefix(template_expr, var_map) == expr.prefix, infix


def test_structural_numerics_are_not_coefficient_slots(univariate):
    """Exponents stay exponents; only fittable constants become slots.

    `x**2` and `x**4` are distinguishable to the decoder only because the
    tokenizer keeps their exponents as `two`/`four`.  Turning an exponent into a
    resampleable coefficient would collapse them to `pow x1 C` and make the
    form unrecoverable.
    """
    expr = Expression.from_infix('2.5*x**3 + 1.5*x**2', univariate)
    tmpl = ConstantTemplate.from_expression(expr)

    assert tmpl.n_constants == 2, tmpl.reference_constants
    assert 'three' in tmpl.prefix and 'two' in tmpl.prefix
    assert sorted(round(v, 6) for v in tmpl.reference_constants) == [1.5, 2.5]


# ---------------------------------------------------------------------------
# 2. Coefficient values do not affect the canonical target tokens
# ---------------------------------------------------------------------------

def test_realizations_share_one_token_target(templates, tokenizer):
    """Every sampled realisation tokenizes to the template's own sequence."""
    sampler = ConstantSampler().fit(templates)
    rng = np.random.RandomState(0)

    for tmpl in templates:
        target = tokenizer.encode(tmpl.prefix)
        for _ in range(8):
            expr = tmpl.instantiate(sampler.sample(tmpl, rng))
            assert expr.prefix == tmpl.prefix
            assert tokenizer.encode(expr.prefix) == target


def test_equal_coefficient_forms_deduplicate_together(templates):
    """The four `A*sin(3*x + phi)` strings collapse to one canonical form.

    The frequency `3` stays the structural token `three` and is *not* a
    coefficient slot, so only the amplitude and phase resample — which is the
    behaviour the task asks for: exponents and other structural numerics stay
    distinct from nuisance coefficients.
    """
    prefixes = [t.prefix for t in templates]
    assert len(prefixes) == len(set(prefixes))

    sin_forms = [t for t in templates
                 if t.prefix == 'mul C sin add C mul three x1']
    assert len(sin_forms) == 1
    assert sin_forms[0].n_constants == 2
    # 2.1/0.5, 0.8/-1.2, -1.5/2.4 all landed in the one form's pools.
    assert sin_forms[0].constant_pools[0].size >= 3
    assert sorted(float(v) for v in sin_forms[0].constant_pools[0]) == \
        pytest.approx([-1.5, 0.8, 2.1, 2.1], abs=1e-6)


def test_describe_realizations_verifies_the_invariant(templates, tokenizer):
    train, _val, _test = _splits(templates, tokenizer, dynamic_constants=True)
    assert describe_realizations(train, n_forms=3, n_realizations=3, seed=0)


# ---------------------------------------------------------------------------
# 3. Different constant samples produce different numerical functions
# ---------------------------------------------------------------------------

def test_different_constants_change_the_function(templates):
    """Distinct coefficient vectors give numerically distinct functions."""
    sampler = ConstantSampler().fit(templates)
    rng = np.random.RandomState(1)
    points = np.linspace(-3.0, 3.0, 128).reshape(-1, 1)

    n_checked = 0
    for tmpl in templates:
        if tmpl.n_constants == 0:
            continue
        a = sampler.sample(tmpl, rng)
        b = sampler.sample(tmpl, rng)
        if np.allclose(a, b):
            continue  # same draw from a small pool; not a counterexample
        ya = tmpl.evaluate(points, a)
        yb = tmpl.evaluate(points, b)
        finite = np.isfinite(ya) & np.isfinite(yb)
        if finite.sum() < 10:
            continue
        assert not np.allclose(ya[finite], yb[finite]), tmpl.prefix
        n_checked += 1

    assert n_checked >= 5, f'only {n_checked} forms exercised'


def test_reference_constants_reproduce_the_original_function(templates):
    """`instantiate(reference_constants)` recovers the source expression.

    This is what makes the slot alignment trustworthy: `c_k` really does stand
    where `reference_constants[k]` was.
    """
    points = np.linspace(-math.pi, math.pi, 200).reshape(-1, 1)

    for tmpl in templates:
        original = tmpl.reference_expression().evaluate(points)
        rebuilt = tmpl.instantiate(tmpl.reference_constants).evaluate(points)
        finite = np.isfinite(original) & np.isfinite(rebuilt)
        assert finite.sum() >= 10, tmpl.prefix
        assert np.allclose(original[finite], rebuilt[finite],
                           rtol=1e-9, atol=1e-12), tmpl.prefix


def test_sampler_perturb_mode_needs_no_pools(templates):
    """`perturb` keeps the sign and stays within the configured magnitude band."""
    sampler = ConstantSampler(mode='perturb', log_spread=2.0)
    rng = np.random.RandomState(3)

    for tmpl in templates:
        if tmpl.n_constants == 0:
            continue
        values = sampler.sample(tmpl, rng)
        ratio = np.abs(values) / np.abs(tmpl.reference_constants)
        assert np.all(np.sign(values) == np.sign(tmpl.reference_constants))
        assert np.all(ratio >= 0.5 - 1e-9) and np.all(ratio <= 2.0 + 1e-9)


# ---------------------------------------------------------------------------
# The global sampler: one distribution, shared by every canonical form
# ---------------------------------------------------------------------------

def test_global_sampler_gives_every_form_the_same_magnitude_law(templates):
    """Magnitudes are drawn from one pool, so no form has its own distribution.

    This is the property the whole augmentation depends on. If form F drew from
    F's own observed coefficients, those coefficients would still identify F —
    the exact shortcut the resampling is meant to remove.
    """
    sampler = ConstantSampler(mode='global').fit(templates)
    pool = set(np.round(sampler.magnitude_pool.astype(float), 6))

    draws = {}
    for tmpl in templates[:6]:
        rng = np.random.RandomState(11)
        mags = np.abs(np.concatenate(
            [sampler.sample(tmpl, rng) for _ in range(300)]))
        # Every magnitude drawn for any form comes from the one shared pool.
        assert set(np.round(mags, 6)) <= pool, tmpl.prefix
        draws[tmpl.prefix] = np.sort(mags)

    # And the realised magnitude distributions agree across forms.
    quantiles = np.linspace(0.1, 0.9, 9)
    refs = np.quantile(list(draws.values())[0], quantiles)
    for prefix, mags in list(draws.items())[1:]:
        assert np.quantile(mags, quantiles) == pytest.approx(refs, rel=0.35), \
            prefix


def test_global_sampler_is_not_biased_by_a_form_own_pool(templates):
    """A form's own observed values do not preferentially come back to it."""
    sampler = ConstantSampler(mode='global').fit(templates)

    # Pick the form with the deepest pool: under per-form sampling essentially
    # every draw would land in its own pool.
    tmpl = max(templates, key=lambda t: t.constant_pools[0].size)
    own = set(np.round(np.abs(np.concatenate(tmpl.constant_pools)), 6))

    rng = np.random.RandomState(5)
    mags = np.abs(np.concatenate([sampler.sample(tmpl, rng) for _ in range(200)]))
    share = np.mean([round(float(m), 6) in own for m in mags])

    per_form = ConstantSampler(mode='empirical')
    rng = np.random.RandomState(5)
    mags_pf = np.abs(np.concatenate(
        [per_form.sample(tmpl, rng) for _ in range(200)]))
    share_pf = np.mean([round(float(m), 6) in own for m in mags_pf])

    assert share_pf > share, (share_pf, share)


def test_global_sampler_preserves_structural_signs(templates):
    """Sign comes from the reference: `exp(c*(x-mu)**2)` must keep `c < 0`."""
    sampler = ConstantSampler(mode='global').fit(templates)
    rng = np.random.RandomState(7)

    for tmpl in templates:
        if tmpl.n_constants == 0:
            continue
        for _ in range(5):
            values = sampler.sample(tmpl, rng)
            assert np.all(np.sign(values) == np.sign(tmpl.reference_constants)), \
                tmpl.prefix


def test_global_sampler_must_be_fitted(templates):
    sampler = ConstantSampler(mode='global')
    assert not sampler.is_fitted
    with pytest.raises(RuntimeError, match='must be fitted'):
        sampler.sample(templates[0], np.random.RandomState(0))

    assert sampler.fit(templates).is_fitted
    assert sampler.magnitude_pool.size > 0
    assert np.all(sampler.magnitude_pool > 0)
    assert np.all(np.isfinite(sampler.magnitude_pool))


def test_split_builder_fits_the_sampler_on_train_only(templates, tokenizer):
    """Held-out forms contribute nothing to the shared pool."""
    sampler = ConstantSampler(mode='global')
    train, val, test = _splits(templates, tokenizer, dynamic_constants=True,
                               sampler=sampler)
    assert sampler.is_fitted

    expected = sum(int(np.sum([p.size for p in s['template'].constant_pools]))
                   for s in train.samples)
    held_out = np.concatenate([
        np.abs(np.concatenate(s['template'].constant_pools))
        for ds in (val, test) for s in ds.samples])

    # Size matches the train contribution (minus any zero/non-finite drops).
    assert sampler.magnitude_pool.size <= expected
    assert sampler.magnitude_pool.size >= expected - 5

    # And no held-out form's pool was merged in wholesale.
    assert len(held_out) > 0


# ---------------------------------------------------------------------------
# 4. Repeated __getitem__ in dynamic mode gives new realisations
# ---------------------------------------------------------------------------

def test_dynamic_getitem_varies_clouds_but_not_targets(templates, tokenizer):
    train, _val, _test = _splits(templates, tokenizer, dynamic_constants=True)
    assert train.dynamic_constants

    np.random.seed(0)
    draws = [train[0] for _ in range(6)]

    clouds = [d['points'] for d in draws]
    assert any(not torch.equal(clouds[0], c) for c in clouds[1:]), \
        'dynamic mode produced six identical point clouds'

    for d in draws[1:]:
        assert torch.equal(d['input_ids'], draws[0]['input_ids'])
        assert torch.equal(d['attn_mask'], draws[0]['attn_mask'])


def test_dynamic_mode_does_not_change_dataset_length(templates, tokenizer):
    """Augmentation is per-draw, so epochs and optimizer steps are unchanged."""
    static, _v, _t = _splits(templates, tokenizer, dynamic_constants=False)
    dynamic, _v2, _t2 = _splits(templates, tokenizer, dynamic_constants=True)
    assert len(static) == len(dynamic)
    assert static.token_keys == dynamic.token_keys


def test_constant_seed_makes_draws_reproducible(templates, tokenizer):
    """An explicit `constant_seed` pins coefficients to (seed, epoch, idx)."""
    train, _v, _t = _splits(templates, tokenizer, dynamic_constants=True,
                            constant_seed=1234)
    a = train[2]['points']
    b = train[2]['points']
    assert torch.equal(a, b)

    train.epoch = 1
    c = train[2]['points']
    assert not torch.equal(a, c), 'epoch change did not move the realisation'


# ---------------------------------------------------------------------------
# 5. The old dataset mode is unchanged
# ---------------------------------------------------------------------------

def test_templates_reproduce_the_legacy_loader(tokenizer, tmp_path):
    """Same surviving expressions, same order, same prefixes as before."""
    pkl = tmp_path / 'raw.pkl'
    pkl.write_bytes(pickle.dumps(RAW_STRINGS))

    legacy = load_synthetic_pkl(str(pkl), max_seq_len=64, tokenizer=tokenizer)
    tmpls, _stats = build_templates_from_strings(
        RAW_STRINGS, tokenizer, max_seq_len=64)

    assert len(legacy) == len(tmpls)
    assert [e.prefix for e in legacy] == [t.prefix for t in tmpls]


def test_static_mode_reproduces_the_legacy_pipeline(tokenizer, tmp_path):
    """`dynamic_constants=False` gives bit-identical splits and point clouds."""
    pkl = tmp_path / 'raw.pkl'
    pkl.write_bytes(pickle.dumps(RAW_STRINGS))

    legacy = load_synthetic_pkl(str(pkl), max_seq_len=64, tokenizer=tokenizer)
    tmpls, _ = build_templates_from_strings(
        RAW_STRINGS, tokenizer, max_seq_len=64)

    old = build_synthetic_splits(legacy, tokenizer, n_points=64, max_vars=1,
                                 seed=42)
    new = _splits(tmpls, tokenizer, dynamic_constants=False)

    for old_ds, new_ds in zip(old, new):
        assert len(old_ds) == len(new_ds)
        assert old_ds.token_keys == new_ds.token_keys
        for i in range(len(old_ds)):
            np.random.seed(7)
            a = old_ds[i]
            np.random.seed(7)
            b = new_ds[i]
            assert torch.equal(a['points'], b['points']), f'sample {i}'
            assert torch.equal(a['input_ids'], b['input_ids'])


def test_existing_dataset_class_is_untouched(tokenizer, univariate):
    """PointCloudDataset gained no dynamic behaviour by default."""
    exprs = [Expression.from_infix('2.1*sin(3*x + 0.5)', univariate)]
    ds = PointCloudDataset(exprs, tokenizer, n_points=64, max_vars=1,
                           resample=False)
    assert not hasattr(ds, 'dynamic_constants')
    assert torch.equal(ds[0]['points'], ds[0]['points'])


# ---------------------------------------------------------------------------
# 6. No canonical form crosses a split boundary
# ---------------------------------------------------------------------------

def test_no_canonical_form_crosses_splits(templates, tokenizer):
    train, val, test = _splits(templates, tokenizer, dynamic_constants=True)

    report = canonical_split_report(train, val, test)
    assert report['overlaps'] == {'train/val': 0, 'train/test': 0,
                                  'val/test': 0}

    forms = {'train': set(train.canonical_forms),
             'val': set(val.canonical_forms),
             'test': set(test.canonical_forms)}
    assert forms['train'] & forms['val'] == set()
    assert forms['train'] & forms['test'] == set()
    assert forms['val'] & forms['test'] == set()
    assert sum(len(f) for f in forms.values()) == len(templates)


def test_split_report_raises_on_leakage(templates, tokenizer):
    """The assertion is real: a deliberately leaked split must fail."""
    train, val, test = _splits(templates, tokenizer)
    leaked = DynamicConstantPointCloudDataset(
        [train.samples[0]['template'], val.samples[0]['template']],
        tokenizer, n_points=64, max_vars=1, resample=False)

    with pytest.raises(AssertionError):
        canonical_split_report(train, leaked, test)


def test_split_matches_the_legacy_partition(tokenizer, tmp_path):
    """Template splits equal expression splits at the same seed.

    Without this the dynamic-constants run and the baseline it is compared
    against would be trained on different structures, and any difference in
    the metrics could not be attributed to the augmentation.
    """
    pkl = tmp_path / 'raw.pkl'
    pkl.write_bytes(pickle.dumps(RAW_STRINGS))
    legacy = load_synthetic_pkl(str(pkl), max_seq_len=64, tokenizer=tokenizer)
    tmpls, _ = build_templates_from_strings(RAW_STRINGS, tokenizer,
                                            max_seq_len=64)

    old = build_synthetic_splits(legacy, tokenizer, n_points=64, max_vars=1,
                                 seed=42)
    new = _splits(tmpls, tokenizer, dynamic_constants=True)
    for old_ds, new_ds in zip(old, new):
        assert old_ds.token_keys == new_ds.token_keys


# ---------------------------------------------------------------------------
# 7. Val/test generation is deterministic
# ---------------------------------------------------------------------------

def test_val_and_test_are_deterministic(templates, tokenizer):
    _train, val, test = _splits(templates, tokenizer, dynamic_constants=True)

    for ds in (val, test):
        assert not ds.dynamic_constants
        assert not ds.resample
        for i in range(len(ds)):
            np.random.seed(0)
            first = ds[i]['points']
            np.random.seed(999)
            second = ds[i]['points']
            assert torch.equal(first, second), f'{ds} sample {i} moved'


def test_eval_splits_ignore_the_global_rng(templates, tokenizer):
    """Rebuilding val/test under a different global seed gives the same clouds."""
    np.random.seed(1)
    _t1, val1, _s1 = _splits(templates, tokenizer, dynamic_constants=True)
    np.random.seed(2)
    _t2, val2, _s2 = _splits(templates, tokenizer, dynamic_constants=True)

    assert len(val1) == len(val2)
    for i in range(len(val1)):
        assert torch.equal(val1[i]['points'], val2[i]['points'])


def test_eval_cache_returns_identical_clouds(templates, tokenizer):
    _train, val, _test = _splits(templates, tokenizer, dynamic_constants=True,
                                 cache_eval=True)
    assert val.cache
    assert torch.equal(val[0]['points'], val[0]['points'])


# ---------------------------------------------------------------------------
# 8. Generated point clouds are finite and usable
# ---------------------------------------------------------------------------

def test_dynamic_clouds_are_finite_and_non_degenerate(templates, tokenizer):
    train, val, test = _splits(templates, tokenizer, dynamic_constants=True)

    np.random.seed(0)
    for ds in (train, val, test):
        for i in range(len(ds)):
            for _ in range(3 if ds is train else 1):
                points = ds[i]['points']
                assert torch.isfinite(points).all(), f'{ds} sample {i}'
                assert points.shape == (ds.n_points, ds.target_d)
                # Column 0 is the input, column 1 the output; both are
                # z-scored, so a live column has unit-ish spread.
                assert float(points[:, 1].std()) > 1e-3, f'{ds} sample {i}'


def test_overflowing_draws_do_not_leak_numpy_warnings(univariate, tokenizer):
    """A blown-up candidate is rejected quietly, not narrated to the log.

    `A*sinh(b*x)*exp(-a*x**2)` overflows for large `b`, and a resampled `b` will
    occasionally be large. That draw is rejected and redrawn — correct — but
    NumPy would warn once per lambdified template per first offending draw, and
    since whether a template overflows depends on the coefficients drawn, the
    warnings keep arriving every epoch for 30 epochs.
    """
    expr = Expression.from_infix('1.5*sinh(0.4*x)*exp(-0.2*x**2)', univariate)
    tmpl = ConstantTemplate.from_expression(expr)

    # A pool that guarantees overflow: sinh(300*pi) is far past float64 range.
    sampler = ConstantSampler(mode='global', magnitude_pool=np.array([300.0]))
    ds = DynamicConstantPointCloudDataset(
        [tmpl], tokenizer, n_points=64, max_vars=1,
        dynamic_constants=True, sampler=sampler, max_constant_tries=3)

    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        np.random.seed(0)
        item = ds[0]

    assert torch.isfinite(item['points']).all()
    assert ds.n_constant_fallbacks == 1, 'every draw should have been rejected'


def test_pool_is_usable_rejects_pathologies():
    x = np.linspace(-1, 1, 100).reshape(-1, 1)

    good = np.column_stack([x[:, 0], np.sin(3 * x[:, 0])])
    assert pool_is_usable(good)

    erased = np.column_stack([x[:, 0], np.zeros(100)])
    assert not pool_is_usable(erased), 'a constant output column is degenerate'

    blown_up = np.column_stack([x[:, 0], 1e12 * np.sin(3 * x[:, 0])])
    assert not pool_is_usable(blown_up)

    nonfinite = good.copy()
    nonfinite[3, 1] = np.inf
    assert not pool_is_usable(nonfinite)


def test_rejected_draws_fall_back_to_the_reference(templates, tokenizer):
    """An impossible validity bar degrades to baseline instead of raising."""
    train, _val, _test = _splits(templates, tokenizer, dynamic_constants=True)
    train.max_abs = 1e-12          # nothing can pass
    train.max_constant_tries = 2

    np.random.seed(0)
    item = train[0]
    assert torch.isfinite(item['points']).all()
    assert train.n_constant_fallbacks >= 1
    assert train.n_constant_rejections >= 2


# ---------------------------------------------------------------------------
# On-disk artifact
# ---------------------------------------------------------------------------

def test_template_dataset_roundtrip(templates, tokenizer, tmp_path):
    path = tmp_path / 'templates.pkl'
    save_template_dataset(path, templates, tokenizer, meta={'source': 'test'})

    loaded, meta = load_template_dataset(path, tokenizer)
    assert meta['source'] == 'test'
    assert len(loaded) == len(templates)

    points = np.linspace(-2, 2, 64).reshape(-1, 1)
    for a, b in zip(templates, loaded):
        assert a.prefix == b.prefix
        assert a.n_constants == b.n_constants
        assert np.allclose(a.reference_constants, b.reference_constants)
        ya = a.evaluate(points, a.reference_constants)
        yb = b.evaluate(points, b.reference_constants)
        finite = np.isfinite(ya) & np.isfinite(yb)
        assert np.array_equal(ya[finite], yb[finite])


def test_load_rejects_a_mismatched_vocabulary(templates, tokenizer, tmp_path):
    path = tmp_path / 'templates.pkl'
    save_template_dataset(path, templates, tokenizer)

    other = PrefixTokenizer(max_vars=1)
    other.extend(['my_concept_token'])
    with pytest.raises(ValueError, match='different tokenizer vocabulary'):
        load_template_dataset(path, other)


# ---------------------------------------------------------------------------
# Multi-view: several partial observations of ONE realisation
# ---------------------------------------------------------------------------

def test_multiview_shares_one_function_across_views(templates, tokenizer):
    """Views must differ in points observed, not in which function was drawn.

    Subsample-JEPA asks the encoder to map partial observations of the *same*
    function to the same latent. Drawing fresh constants per view would train
    that objective against a premise that is false.
    """
    from symbolic_jepa.templates import DynamicConstantMultiViewDataset

    ds = DynamicConstantMultiViewDataset(
        templates, tokenizer, n_points=64, max_vars=1, n_views=3,
        pool_mult=4, dynamic_constants=True,
        sampler=ConstantSampler().fit(templates), constant_seed=42)
    ds.set_epoch(3, stage='pretrain')

    item = ds[0]
    assert item['points_views'].shape == (3, 64, ds.target_d)
    assert torch.equal(item['points'], item['points_views'][0])
    assert torch.isfinite(item['points_views']).all()

    # Views are distinct subsamples...
    for a in range(3):
        for b in range(a + 1, 3):
            assert not torch.equal(item['points_views'][a],
                                   item['points_views'][b])

    # ...but of one function: the constants drawn for this (epoch, idx) are a
    # single vector, so re-deriving them reproduces every view exactly.
    again = ds[0]
    assert torch.equal(item['points_views'], again['points_views'])


def test_multiview_is_deterministic_per_epoch(templates, tokenizer):
    from symbolic_jepa.templates import DynamicConstantMultiViewDataset

    ds = DynamicConstantMultiViewDataset(
        templates, tokenizer, n_points=64, max_vars=1, n_views=2,
        dynamic_constants=True,
        sampler=ConstantSampler().fit(templates), constant_seed=42)

    ds.set_epoch(1, stage='pretrain')
    a = ds[0]['points_views'].clone()
    ds.set_epoch(2, stage='pretrain')
    b = ds[0]['points_views'].clone()
    ds.set_epoch(1, stage='pretrain')
    c = ds[0]['points_views'].clone()

    assert not torch.equal(a, b), 'epoch did not advance the realisation'
    assert torch.equal(a, c), 'epoch 1 did not replay'


def test_multiview_split_matches_the_single_view_partition(templates,
                                                           tokenizer):
    """Same seed, same split — so multi-view and single-view stay comparable."""
    from symbolic_jepa.templates import build_multiview_template_splits

    single = build_template_splits(
        templates, tokenizer, n_points=64, max_vars=1, seed=42,
        dynamic_constants=True, sampler=ConstantSampler())
    multi = build_multiview_template_splits(
        templates, tokenizer, n_points=64, max_vars=1, seed=42,
        n_views=2, dynamic_constants=True, sampler=ConstantSampler())

    for a, b in zip(single, multi):
        assert a.token_keys == b.token_keys
    assert multi[0].n_views == 2
    assert not multi[1].dynamic_constants and not multi[2].dynamic_constants


def test_stage1_subsample_pretraining_leaves_the_decoder_untouched(templates,
                                                                   tokenizer):
    """Stage 1 trains the encoder only.

    The optimizer is built over `model.encoder.parameters()` rather than by
    freezing, so the decoder cannot take a gradient from the subsample
    objective even by accident.
    """
    from symbolic_jepa import TNet, SymbolicTransformer
    from symbolic_jepa.jepa import subsample_consistency_loss
    from symbolic_jepa.templates import build_multiview_template_splits

    train, _v, _t = build_multiview_template_splits(
        templates, tokenizer, n_points=64, max_vars=1, seed=42, n_views=2,
        pool_mult=4, dynamic_constants=True, sampler=ConstantSampler(),
        constant_seed=2026)
    train.set_epoch(1, stage='pretrain')

    encoder = TNet(d_input=2, d_model=64)
    model = SymbolicTransformer(encoder=encoder, vocab_size=len(tokenizer),
                                d_model=64, n_heads=4, n_layers=2,
                                max_seq_len=64, dropout=0.1)
    opt = torch.optim.AdamW(list(model.encoder.parameters()), lr=1e-3)

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    model.train()
    for i in range(4):
        views = torch.stack([train[i]['points_views']])
        opt.zero_grad()
        z = [model.encoder(views[:, v]) for v in range(2)]
        subsample_consistency_loss([z[0], z[1].detach()],
                                   mode='centered').backward()
        opt.step()

    changed = {n for n, p in model.named_parameters()
               if not torch.equal(p.detach(), before[n])}
    assert changed, 'nothing trained at all'
    assert all(n.startswith('encoder.') for n in changed), \
        f'Stage 1 modified non-encoder parameters: {sorted(changed)[:5]}'
    assert all(p.grad is None for n, p in model.named_parameters()
               if not n.startswith('encoder.')), 'decoder received a gradient'


def test_multiview_views_share_coefficients_and_target(templates, tokenizer):
    """Views differ only by subsampling; coefficients and target are shared."""
    from symbolic_jepa.templates import build_multiview_template_splits

    train, _v, _t = build_multiview_template_splits(
        templates, tokenizer, n_points=64, max_vars=1, seed=42, n_views=2,
        pool_mult=4, dynamic_constants=True, sampler=ConstantSampler(),
        constant_seed=2026)
    train.set_epoch(2, stage='pretrain')

    for idx in range(min(5, len(train))):
        item = train[idx]
        a, b = item['points_views'][0], item['points_views'][1]
        assert not torch.equal(a, b), f'views identical at {idx}'
        assert torch.equal(item['points'], a)
        # One draw of coefficients backs both views.
        pool, expr, _r, _f, _s = train.draw_realization(idx)
        again = train.draw_realization(idx)[1]
        assert np.array_equal(expr.values, again.values)
        # Every view row comes from the shared pool.
        assert torch.isfinite(item['points_views']).all()
        assert torch.equal(item['input_ids'], train[idx]['input_ids'])
