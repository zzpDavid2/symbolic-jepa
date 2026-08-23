"""Resuming a run must be scientifically equivalent to never interrupting it.

The JEPA effect being measured is ~0.7 pp.  Augmentation that depends on
DataLoader worker RNG state would make a resumed run differ from an
uninterrupted one by an uncontrolled amount, and there is no way to tell that
apart from the effect afterwards.

The fix is stateless: every realisation is a pure function of
``(base_seed, stage, epoch, sample_idx, stream, attempt)``.  Nothing reads
mutable global RNG state, so worker count, worker scheduling, process restart
and checkpoint resume are all irrelevant by construction.  These tests pin that.
"""

import hashlib

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from symbolic_jepa.templates import (
    AUGMENTATION_VERSION,
    ConstantSampler,
    augmentation_seed,
    build_template_splits,
    build_templates_from_strings,
)
from symbolic_jepa.tokenizer import PrefixTokenizer
from tests.test_constant_templates import RAW_STRINGS

BASE_SEED = 42


@pytest.fixture(scope='module')
def tokenizer():
    return PrefixTokenizer(max_vars=1)


@pytest.fixture(scope='module')
def templates(tokenizer):
    tmpls, _ = build_templates_from_strings(
        RAW_STRINGS, tokenizer, max_seq_len=64)
    return tmpls


def _train_split(templates, tokenizer, constant_seed=BASE_SEED):
    train, val, test = build_template_splits(
        templates, tokenizer, n_points=64, max_vars=1, seed=42,
        dynamic_constants=True, sampler=ConstantSampler(),
        constant_seed=constant_seed)
    return train, val, test


# ---------------------------------------------------------------------------
# 1. The same (epoch, sample) is always the same realisation
# ---------------------------------------------------------------------------

def test_same_epoch_and_index_is_bit_identical(templates, tokenizer):
    train, _v, _t = _train_split(templates, tokenizer)
    train.set_epoch(5, stage='supervised')

    idx = min(3, len(train) - 1)
    first = train[idx]

    for _ in range(4):
        # Move the global RNG between calls: a stateless implementation cannot
        # notice, a worker-RNG one would.
        np.random.seed(np.random.randint(1 << 30))
        again = train[idx]
        assert torch.equal(first['points'], again['points'])
        assert torch.equal(first['input_ids'], again['input_ids'])
        assert torch.equal(first['attn_mask'], again['attn_mask'])


def test_constants_and_points_are_reproducible_and_separable(templates,
                                                             tokenizer):
    """Constants, x and y all replay; the two streams are independent."""
    train, _v, _t = _train_split(templates, tokenizer)
    train.set_epoch(5, stage='supervised')
    idx = min(3, len(train) - 1)

    pool_a, expr_a, _, _, _ = train.draw_realization(idx)
    pool_b, expr_b, _, _, _ = train.draw_realization(idx)

    assert np.array_equal(expr_a.values, expr_b.values)      # constants
    assert np.array_equal(pool_a[:, :-1], pool_b[:, :-1])    # x
    assert np.array_equal(pool_a[:, -1], pool_b[:, -1])      # y

    # Independent streams: the constant and point seeds must not coincide, or a
    # change in how many draws the coefficients consume would shift the cloud.
    c = augmentation_seed(BASE_SEED, 'supervised', 5, idx, 'constants')
    p = augmentation_seed(BASE_SEED, 'supervised', 5, idx, 'points')
    s = augmentation_seed(BASE_SEED, 'supervised', 5, idx, 'subsample')
    assert len({c, p, s}) == 3


def test_seed_is_process_stable_not_python_hash(templates, tokenizer):
    """The mixing function is SHA-256, so it survives a new interpreter.

    `hash()` on a str is salted per process; using it would silently make every
    fresh Colab runtime a different experiment.
    """
    expected = int.from_bytes(
        hashlib.sha256(b'42|supervised|5|123|constants|0').digest()[:4], 'big')
    assert augmentation_seed(42, 'supervised', 5, 123, 'constants', 0) == expected


# ---------------------------------------------------------------------------
# 2. Different epochs / stages do augment
# ---------------------------------------------------------------------------

def test_different_epochs_augment_but_keep_the_target(templates, tokenizer):
    train, _v, _t = _train_split(templates, tokenizer)

    per_epoch = []
    for epoch in range(6):
        train.set_epoch(epoch, stage='supervised')
        per_epoch.append([train[i] for i in range(len(train))])

    for i in range(len(train)):
        target = per_epoch[0][i]['input_ids']
        for e in range(1, 6):
            assert torch.equal(per_epoch[e][i]['input_ids'], target)

    changed = sum(
        1 for i in range(len(train))
        if not torch.equal(per_epoch[0][i]['points'], per_epoch[1][i]['points'])
    )
    assert changed == len(train), f'{changed}/{len(train)} examples changed'


def test_stage1_and_stage2_do_not_collide(templates, tokenizer):
    """Same epoch number in the two stages must not replay one realisation."""
    train, _v, _t = _train_split(templates, tokenizer)
    idx = min(3, len(train) - 1)

    train.set_epoch(3, stage='pretrain')
    a = train[idx]['points']
    train.set_epoch(3, stage='supervised')
    b = train[idx]['points']
    assert not torch.equal(a, b)


# ---------------------------------------------------------------------------
# 3. Worker-count independence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('num_workers', [0, 2])
def test_batches_are_identical_across_worker_counts(templates, tokenizer,
                                                    num_workers):
    """A given epoch yields the same per-example content at any worker count."""
    train, _v, _t = _train_split(templates, tokenizer)
    train.set_epoch(7, stage='supervised')

    loader = DataLoader(train, batch_size=4, shuffle=False,
                        num_workers=num_workers)
    got = torch.cat([b['points'] for b in loader], dim=0)

    reference = torch.stack([train[i]['points'] for i in range(len(train))])
    assert torch.equal(got, reference)


# ---------------------------------------------------------------------------
# 4. Resume equivalence — the integration test
# ---------------------------------------------------------------------------

def _tiny_model(vocab, d_model=32):
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(64 * 2, d_model),
        torch.nn.ReLU(),
        torch.nn.Linear(d_model, vocab),
    )


def _run_epochs(train, model, opt, epochs, tokenizer):
    """Train over `epochs` (1-indexed), returning the per-epoch loss."""
    losses = []
    for epoch in epochs:
        train.set_epoch(epoch, stage='supervised')
        loader = DataLoader(train, batch_size=4, shuffle=False, num_workers=0)
        total = 0.0
        for batch in loader:
            opt.zero_grad()
            logits = model(batch['points'])
            target = batch['input_ids'][:, 1]      # one token: enough to bind
            loss = torch.nn.functional.cross_entropy(logits, target)
            loss.backward()
            opt.step()
            total += float(loss.detach())
        losses.append(total)
    return losses


def test_resume_matches_an_uninterrupted_run(templates, tokenizer):
    """A: 6 epochs straight.  B: 3 epochs, checkpoint, rebuild, 3 more.

    Final weights and the loss history must match to the bit.  This is the
    property the whole task exists for.
    """
    vocab = len(tokenizer)

    train_a, _v, _t = _train_split(templates, tokenizer)
    model_a = _tiny_model(vocab)
    opt_a = torch.optim.SGD(model_a.parameters(), lr=0.05)
    losses_a = _run_epochs(train_a, model_a, opt_a, range(1, 7), tokenizer)

    # --- B: interrupt after epoch 3 ---
    train_b, _v2, _t2 = _train_split(templates, tokenizer)
    model_b = _tiny_model(vocab)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=0.05)
    losses_b1 = _run_epochs(train_b, model_b, opt_b, range(1, 4), tokenizer)

    ckpt = {
        'epoch': 3,
        'model': {k: v.clone() for k, v in model_b.state_dict().items()},
        'optimizer': opt_b.state_dict(),
        'augmentation_version': AUGMENTATION_VERSION,
        'constant_seed': train_b.constant_seed,
    }

    # Everything is rebuilt from scratch, as a restarted runtime would.
    train_c, _v3, _t3 = _train_split(templates, tokenizer)
    model_c = _tiny_model(vocab)
    model_c.load_state_dict(ckpt['model'])
    opt_c = torch.optim.SGD(model_c.parameters(), lr=0.05)
    opt_c.load_state_dict(ckpt['optimizer'])

    # start_epoch = completed + 1; epochs 4,5,6 must match A's 4,5,6.
    start_epoch = ckpt['epoch'] + 1
    assert start_epoch == 4, 'off-by-one in the resume epoch'
    losses_b2 = _run_epochs(train_c, model_c, opt_c, range(start_epoch, 7),
                            tokenizer)

    assert losses_b1 + losses_b2 == pytest.approx(losses_a, rel=0, abs=0), (
        losses_a, losses_b1 + losses_b2)

    for (k, va), (_, vc) in zip(model_a.state_dict().items(),
                                model_c.state_dict().items()):
        assert torch.equal(va, vc), f'parameter {k} diverged after resume'


def test_resume_at_the_wrong_epoch_is_detectable(templates, tokenizer):
    """Sanity: the test above would actually fail on an off-by-one."""
    train, _v, _t = _train_split(templates, tokenizer)
    vocab = len(tokenizer)

    model = _tiny_model(vocab)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    correct = _run_epochs(train, model, opt, range(4, 7), tokenizer)

    model2 = _tiny_model(vocab)
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.05)
    off_by_one = _run_epochs(train, model2, opt2, range(3, 6), tokenizer)

    assert correct != off_by_one


# ---------------------------------------------------------------------------
# 5. Val / test are untouched by any of this
# ---------------------------------------------------------------------------

def test_eval_splits_stay_fixed_across_epochs_and_seeds(templates, tokenizer):
    _train, val, test = _train_split(templates, tokenizer)

    for ds in (val, test):
        assert not ds.dynamic_constants
        assert ds.constant_seed is None, 'eval must not carry a constant seed'
        baseline = [ds[i]['points'].clone() for i in range(len(ds))]

        for epoch in (0, 5, 29):
            ds.set_epoch(epoch, stage='supervised')
            np.random.seed(epoch + 1)
            for i in range(len(ds)):
                assert torch.equal(ds[i]['points'], baseline[i])

    other, val2, _ = _train_split(templates, tokenizer, constant_seed=999)
    assert len(val2) == len(val)
    for i in range(len(val)):
        assert torch.equal(val2[i]['points'], val[i]['points'])
    assert other.constant_seed == 999


def test_deterministic_flag_reports_honestly(templates, tokenizer):
    det, _v, _t = _train_split(templates, tokenizer, constant_seed=BASE_SEED)
    assert det.deterministic_augmentation

    stochastic, _v2, _t2 = build_template_splits(
        templates, tokenizer, n_points=64, max_vars=1, seed=42,
        dynamic_constants=True, sampler=ConstantSampler(), constant_seed=None)
    assert not stochastic.deterministic_augmentation

    # The stochastic mode is the legacy path and is genuinely not reproducible.
    np.random.seed(0)
    a = stochastic[0]['points']
    np.random.seed(1)
    b = stochastic[0]['points']
    assert not torch.equal(a, b)


# ---------------------------------------------------------------------------
# 6. Decoupled model seed / data seed
# ---------------------------------------------------------------------------

def test_fingerprint_is_equal_for_arms_sharing_a_data_seed(templates,
                                                           tokenizer):
    """The paired-control requirement, as an assertion rather than an argument.

    Two runs differing only in JEPA pretraining must see the SAME Stage-2
    supervised data. Since the augmentation keys on the data seed and the stage
    — never on the model seed — that holds by construction, and this pins it.
    """
    from symbolic_jepa.templates import stage2_fingerprint

    arm_a, _v, _t = _train_split(templates, tokenizer, constant_seed=2026)
    arm_b, _v2, _t2 = _train_split(templates, tokenizer, constant_seed=2026)

    epochs = (0, 5, 15, 29)
    fa = stage2_fingerprint(arm_a, epochs=epochs, n_indices=6)
    fb = stage2_fingerprint(arm_b, epochs=epochs, n_indices=6)
    assert fa == fb, 'same data seed produced different Stage-2 data'

    # And a different data seed must move it, or the knob does nothing.
    other, _v3, _t3 = _train_split(templates, tokenizer, constant_seed=31415)
    assert stage2_fingerprint(other, epochs=epochs, n_indices=6) != fa


def test_fingerprint_ignores_global_rng_and_restores_state(templates,
                                                           tokenizer):
    """Model-side randomness must not leak into the data fingerprint."""
    from symbolic_jepa.templates import stage2_fingerprint

    train, _v, _t = _train_split(templates, tokenizer, constant_seed=2026)
    train.set_epoch(3, stage='pretrain')

    np.random.seed(1)
    torch.manual_seed(1)
    first = stage2_fingerprint(train, epochs=(0, 5), n_indices=4)
    assert (train.epoch, train.stage) == (3, 'pretrain'), 'state not restored'

    np.random.seed(999)
    torch.manual_seed(999)
    assert stage2_fingerprint(train, epochs=(0, 5), n_indices=4) == first


def test_fingerprint_covers_targets_not_just_clouds(templates, tokenizer):
    """A changed token target must change the fingerprint."""
    from symbolic_jepa.templates import stage2_fingerprint

    train, _v, _t = _train_split(templates, tokenizer, constant_seed=2026)
    before = stage2_fingerprint(train, epochs=(0, 1), n_indices=4)

    saved = train.samples[0]['input_ids'].clone()
    train.samples[0]['input_ids'] = saved.roll(1)
    try:
        assert stage2_fingerprint(train, epochs=(0, 1), n_indices=4) != before
    finally:
        train.samples[0]['input_ids'] = saved


def test_data_seed_is_independent_of_model_seed(templates, tokenizer):
    """Changing the model seed must not change what the data pipeline serves.

    This is the whole point of the decoupling: with the data seed fixed, every
    model initialisation trains on one identical augmented trajectory.
    """
    from symbolic_jepa.templates import stage2_fingerprint

    train, _v, _t = _train_split(templates, tokenizer, constant_seed=2026)
    fingerprints = set()
    for model_seed in (42, 123, 7):
        torch.manual_seed(model_seed)
        np.random.seed(model_seed)
        fingerprints.add(stage2_fingerprint(train, epochs=(0, 5), n_indices=5))
    assert len(fingerprints) == 1, fingerprints
