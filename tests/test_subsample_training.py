"""Training-step level tests for subsample JEPA.

These mirror the training step in `subsample_jepa.ipynb` and cover the checks
that used to live in the notebook's smoke-test cells: that the subsample term
actually reaches the encoder, that the lambda=0 baseline skips the extra
encoder forwards, and that the objective is genuinely optimisable.

Component-level behaviour (view seeding, loss modes, diagnostics) lives in
test_subsample_jepa.py.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from symbolic_jepa.dataset import MultiViewPointCloudDataset
from symbolic_jepa.decoder import SymbolicTransformer
from symbolic_jepa.encoder import TNet
from symbolic_jepa.expressions import Expression
from symbolic_jepa.jepa import subsample_consistency_loss
from symbolic_jepa.tokenizer import PrefixTokenizer

FORMULAS = ["x + 1", "x**2", "sin(x)", "cos(x)", "x**3", "exp(x)",
            "x**2 + 1", "sin(x) + x"]
N_POINTS, N_VIEWS, D_MODEL = 64, 2, 32


@pytest.fixture
def tok():
    return PrefixTokenizer(max_vars=1)


@pytest.fixture
def dataset(tok):
    exprs = [Expression.from_infix(f) for f in FORMULAS]
    return MultiViewPointCloudDataset(
        exprs, tok, n_points=N_POINTS, max_vars=1, n_views=N_VIEWS,
    )


def _model(tok, seed=0):
    torch.manual_seed(seed)
    enc = TNet(d_input=2, d_model=D_MODEL)
    model = SymbolicTransformer(
        encoder=enc, vocab_size=len(tok), d_model=D_MODEL, n_heads=4,
        n_layers=1, d_ff=2 * D_MODEL, max_seq_len=64, dropout=0.0,
        pad_id=tok.pad_id,
    )
    return model, enc


def _batch(dataset, size=8, epoch=1):
    dataset.epoch = epoch
    return next(iter(DataLoader(dataset, batch_size=size, shuffle=False)))


def _train_step(model, batch, lam, mode='centered'):
    """The notebook's training step, verbatim in structure.

    View 0 drives the CE forward pass; the remaining views go through the
    encoder only. Returns (total, ce, sub) with sub=None when lam == 0.
    """
    pv = batch['points_views']
    out = model(pv[:, 0], batch['input_ids'], attn_mask=batch['attn_mask'])
    if lam > 0:
        z_views = [out['z_num']] + [
            model.encoder(pv[:, v]) for v in range(1, pv.shape[1])
        ]
        sub = subsample_consistency_loss(z_views, mode=mode)
        return out['loss'] + lam * sub, out['loss'], sub
    return out['loss'], out['loss'], None


class TestTrainingStep:
    def test_losses_are_finite(self, tok, dataset):
        model, _ = _model(tok)
        total, ce, sub = _train_step(model, _batch(dataset), lam=0.3)
        for name, t in [('total', total), ('ce', ce), ('sub', sub)]:
            assert torch.isfinite(t), f'{name} loss is not finite'
        assert sub.item() >= 0

    def test_total_is_ce_plus_weighted_sub(self, tok, dataset):
        model, _ = _model(tok)
        lam = 0.3
        total, ce, sub = _train_step(model, _batch(dataset), lam=lam)
        assert total.item() == pytest.approx(ce.item() + lam * sub.item(),
                                             abs=1e-5)

    def test_subsample_term_alone_reaches_encoder(self, tok, dataset):
        """Guards against the subsample loss being detached from the encoder.

        A combined CE+subsample backward would show encoder gradients even if
        the subsample term contributed nothing, so back-prop it in isolation.
        """
        model, enc = _model(tok)
        batch = _batch(dataset)
        pv = batch['points_views']
        z_views = [model.encoder(pv[:, v]) for v in range(N_VIEWS)]
        subsample_consistency_loss(z_views, mode='centered').backward()

        grads = [p.grad for p in enc.parameters() if p.grad is not None]
        assert grads, 'subsample loss produced no encoder gradients'
        assert all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0

    def test_lambda_zero_skips_extra_encoder_forwards(self, tok, dataset):
        """The lambda=0 baseline must not pay for views it never uses."""
        model, enc = _model(tok)
        batch = _batch(dataset)
        calls = []
        orig = enc.forward
        enc.forward = lambda x, _o=orig, _c=calls: (_c.append(1), _o(x))[1]

        _train_step(model, batch, lam=0.0)
        n_baseline = len(calls)
        calls.clear()
        _train_step(model, batch, lam=0.3)
        n_jepa = len(calls)

        enc.forward = orig
        assert n_baseline == 1, f'baseline encoded {n_baseline}x, expected 1'
        assert n_jepa == N_VIEWS, f'jepa encoded {n_jepa}x, expected {N_VIEWS}'

    def test_lambda_zero_total_equals_ce(self, tok, dataset):
        model, _ = _model(tok)
        total, ce, sub = _train_step(model, _batch(dataset), lam=0.0)
        assert sub is None
        assert total is ce


class TestOptimisability:
    """The objective must actually be reducible by gradient descent."""

    @staticmethod
    def _optimise(tok, dataset, mode, lam=1.0, steps=25):
        model, _ = _model(tok, seed=0)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch = _batch(dataset)
        first = last = None
        for i in range(steps):
            opt.zero_grad()
            total, _, sub = _train_step(model, batch, lam=lam, mode=mode)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if i == 0:
                first = sub.item()
            last = sub.item()
        return first, last

    def test_centered_objective_decreases(self, tok, dataset):
        first, last = self._optimise(tok, dataset, mode='centered')
        assert last < first, (
            f'centered subsample loss did not decrease: {first} -> {last}'
        )

    def test_raw_objective_starts_near_zero(self, tok, dataset):
        """Documents why 'centered' is the default in the notebook.

        The T-Net max-pools ReLU features into the positive orthant, so raw
        cosine between any two embeddings starts near 1 and the raw loss has
        almost no gradient to give.
        """
        model, _ = _model(tok)
        batch = _batch(dataset)
        pv = batch['points_views']
        z_views = [model.encoder(pv[:, v]) for v in range(N_VIEWS)]
        raw = subsample_consistency_loss(z_views, mode='cosine').item()
        cen = subsample_consistency_loss(z_views, mode='centered').item()
        assert raw < 0.05, f'raw loss {raw} unexpectedly large'
        assert cen > raw, f'centered ({cen}) should exceed raw ({raw})'


class TestEpochLoop:
    """Multi-epoch behaviour of the notebook's loop shape."""

    def test_views_refresh_between_epochs_during_training(self, tok, dataset):
        seen = []
        model, _ = _model(tok)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for epoch in (1, 2, 3):
            dataset.epoch = epoch
            loader = DataLoader(dataset, batch_size=4, shuffle=False)
            batch = next(iter(loader))
            seen.append(batch['points_views'][0, 0].clone())
            opt.zero_grad()
            total, _, _ = _train_step(model, batch, lam=0.3)
            total.backward()
            opt.step()
            assert torch.isfinite(total)

        for a in range(len(seen)):
            for b in range(a + 1, len(seen)):
                assert not torch.allclose(seen[a], seen[b]), (
                    f'epochs {a+1} and {b+1} produced identical views'
                )

    def test_run_is_reproducible_across_identical_runs(self, tok, dataset):
        """Same seeds -> same trajectory, so lambda comparisons are paired."""
        def trajectory():
            model, _ = _model(tok, seed=7)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            losses = []
            for epoch in (1, 2):
                dataset.epoch = epoch
                batch = next(iter(DataLoader(dataset, batch_size=4,
                                             shuffle=False)))
                opt.zero_grad()
                total, _, _ = _train_step(model, batch, lam=0.3)
                total.backward()
                opt.step()
                losses.append(total.item())
            return losses

        a, b = trajectory(), trajectory()
        assert a == pytest.approx(b, abs=1e-6), f'not reproducible: {a} vs {b}'
