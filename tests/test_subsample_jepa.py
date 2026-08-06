"""Tests for the subsample-JEPA components.

Covers deterministic multi-view seeding, the consistency loss, and the
view-consistency diagnostics.
"""

import numpy as np
import pytest
import torch

from symbolic_jepa.dataset import (
    MultiViewPointCloudDataset, PointCloudDataset,
    build_multiview_synthetic_splits, build_synthetic_splits,
)
from symbolic_jepa.diagnostics import (
    generate_diagnostic_embeddings, view_consistency_diagnostics,
)
from symbolic_jepa.encoder import TNet
from symbolic_jepa.expressions import Expression
from symbolic_jepa.jepa import subsample_consistency_loss
from symbolic_jepa.tokenizer import PrefixTokenizer

FORMULAS = ["x + 1", "x**2", "sin(x)", "cos(x)", "x**3", "exp(x)"]


@pytest.fixture
def exprs():
    return [Expression.from_infix(f) for f in FORMULAS]


@pytest.fixture
def tok():
    return PrefixTokenizer(max_vars=1)


def _mv(exprs, tok, **kw):
    kw.setdefault('n_points', 100)
    kw.setdefault('max_vars', 1)
    return MultiViewPointCloudDataset(exprs, tok, **kw)


class TestMultiViewDataset:
    def test_shapes(self, exprs, tok):
        ds = _mv(exprs, tok, n_views=3)
        item = ds[0]
        assert item['points_views'].shape == (3, 100, 2)
        # view 0 is also exposed under the standard single-view key
        assert torch.equal(item['points'], item['points_views'][0])

    def test_views_differ(self, exprs, tok):
        """Independent samples of one equation must not be identical."""
        ds = _mv(exprs, tok, n_views=2)
        v = ds[0]['points_views']
        assert not torch.allclose(v[0], v[1]), "views are identical — not resampled"

    def test_deterministic_within_epoch(self, exprs, tok):
        """Repeated access at a fixed epoch returns identical views."""
        ds = _mv(exprs, tok, n_views=2)
        ds.epoch = 3
        a = ds[2]['points_views']
        # Perturb global RNG: seeded views must be unaffected by it
        np.random.seed(12345)
        _ = np.random.rand(1000)
        b = ds[2]['points_views']
        assert torch.equal(a, b)

    def test_views_change_across_epochs(self, exprs, tok):
        ds = _mv(exprs, tok, n_views=2)
        ds.epoch = 0
        a = ds[1]['points_views']
        ds.epoch = 1
        b = ds[1]['points_views']
        assert not torch.allclose(a, b), "views frozen across epochs"

    def test_views_independent_of_model_seed(self, exprs, tok):
        """Two datasets with equal view seeds agree regardless of torch seed."""
        torch.manual_seed(42)
        ds_a = _mv(exprs, tok, n_views=2, train_view_seed=1729)
        torch.manual_seed(999)
        ds_b = _mv(exprs, tok, n_views=2, train_view_seed=1729)
        ds_a.epoch = ds_b.epoch = 5
        assert torch.equal(ds_a[0]['points_views'], ds_b[0]['points_views'])

    def test_different_view_seed_gives_different_views(self, exprs, tok):
        ds_a = _mv(exprs, tok, n_views=2, train_view_seed=1729)
        ds_b = _mv(exprs, tok, n_views=2, train_view_seed=2024)
        assert not torch.allclose(
            ds_a[0]['points_views'], ds_b[0]['points_views']
        )

    def test_view_seed_is_stable_across_processes(self):
        """SHA-256 based, so not affected by PYTHONHASHSEED."""
        s = MultiViewPointCloudDataset.view_seed(1729, 2, 7, 1)
        assert s == MultiViewPointCloudDataset.view_seed(1729, 2, 7, 1)
        assert s != MultiViewPointCloudDataset.view_seed(1729, 2, 7, 0)
        assert s != MultiViewPointCloudDataset.view_seed(1729, 3, 7, 1)
        assert 0 <= s < 2 ** 32  # valid RandomState seed

    def test_collates_into_batch(self, exprs, tok):
        from torch.utils.data import DataLoader
        ds = _mv(exprs, tok, n_views=2)
        batch = next(iter(DataLoader(ds, batch_size=4)))
        assert batch['points_views'].shape == (4, 2, 100, 2)


class _EpochProbe(torch.utils.data.Dataset):
    """Picklable stand-in that reports the epoch each worker observed.

    Mirrors how MultiViewPointCloudDataset reads ``self.epoch`` inside
    __getitem__.  A real dataset can't be used here: Expression holds a
    sympy-lambdified function that cannot be pickled, so it only works
    under fork (Linux/Colab), not spawn (macOS).
    """

    def __init__(self):
        self.epoch = 0

    def __len__(self):
        return 8

    def __getitem__(self, idx):
        return torch.tensor([self.epoch, idx])


class TestEpochPropagation:
    """The training loop mutates dataset.epoch between epochs; workers must see it."""

    @staticmethod
    def _epochs_seen(persistent):
        from torch.utils.data import DataLoader
        ds = _EpochProbe()
        dl = DataLoader(ds, batch_size=4, num_workers=2,
                        persistent_workers=persistent)
        seen = []
        for epoch in (1, 2, 3):
            ds.epoch = epoch
            seen.append(int(torch.cat([b[:, 0] for b in dl]).max()))
        return seen

    def test_non_persistent_workers_see_epoch_updates(self):
        assert self._epochs_seen(persistent=False) == [1, 2, 3]

    def test_persistent_workers_freeze_epoch(self):
        """Documents why train_one must use persistent_workers=False."""
        seen = self._epochs_seen(persistent=True)
        assert seen != [1, 2, 3], (
            "persistent workers unexpectedly saw epoch updates; the "
            "persistent_workers=False requirement may no longer hold"
        )


class TestEvalCloudCache:
    """Caching deterministic eval clouds must be value-neutral."""

    def test_cached_matches_uncached(self, exprs, tok):
        kw = dict(n_points=100, max_vars=1, resample=False)
        plain = PointCloudDataset(exprs, tok, **kw)
        cached = PointCloudDataset(exprs, tok, cache=True, **kw)
        for i in range(len(plain)):
            assert torch.equal(plain[i]['points'], cached[i]['points']), (
                f'cache changed the cloud for sample {i}'
            )

    def test_cache_returns_stable_values(self, exprs, tok):
        ds = PointCloudDataset(exprs, tok, n_points=100, max_vars=1,
                               resample=False, cache=True)
        first = ds[0]['points'].clone()
        np.random.seed(999)          # perturb global RNG
        assert torch.equal(ds[0]['points'], first)

    def test_cache_populates(self, exprs, tok):
        ds = PointCloudDataset(exprs, tok, n_points=100, max_vars=1,
                               resample=False, cache=True)
        assert len(ds._cloud_cache) == 0
        ds[0]; ds[1]
        assert len(ds._cloud_cache) == 2

    def test_cache_ignored_when_resampling(self, exprs, tok):
        """Train clouds must stay fresh; caching them would be a bug."""
        ds = PointCloudDataset(exprs, tok, n_points=100, max_vars=1,
                               resample=True, cache=True)
        assert ds.cache is False
        ds[0]
        assert len(ds._cloud_cache) == 0

    def test_default_is_uncached(self, exprs, tok):
        """jepa_sweep builds datasets without `cache`; behaviour must not shift."""
        ds = PointCloudDataset(exprs, tok, n_points=100, max_vars=1,
                               resample=False)
        assert ds.cache is False

    def test_multiview_splits_cache_eval_only(self, exprs, tok):
        train, val, test = build_multiview_synthetic_splits(
            exprs * 4, tok, n_points=100, max_vars=1, seed=42, n_views=2)
        assert train.cache is False, 'training views must not be cached'
        assert val.cache is True and test.cache is True


class TestSplitParity:
    def test_same_partition_as_single_view(self, exprs, tok):
        """Multi-view splits must match the single-view sweep's partition."""
        exprs_x = exprs * 4  # need enough for a 80/10/10 split
        tr_a, va_a, te_a = build_synthetic_splits(
            exprs_x, tok, n_points=100, max_vars=1, seed=42,
        )
        tr_b, va_b, te_b = build_multiview_synthetic_splits(
            exprs_x, tok, n_points=100, max_vars=1, seed=42, n_views=2,
        )
        assert (len(tr_a), len(va_a), len(te_a)) == (len(tr_b), len(va_b), len(te_b))
        assert isinstance(tr_b, MultiViewPointCloudDataset)
        # val/test stay deterministic single-view
        assert isinstance(va_b, PointCloudDataset)
        assert not isinstance(va_b, MultiViewPointCloudDataset)
        assert va_b.resample is False


class TestSubsampleLoss:
    def test_identical_views_give_zero(self):
        z = torch.randn(8, 16)
        assert subsample_consistency_loss([z, z]).item() == pytest.approx(0.0, abs=1e-6)

    def test_opposite_views_give_two(self):
        z = torch.randn(8, 16)
        assert subsample_consistency_loss([z, -z]).item() == pytest.approx(2.0, abs=1e-5)

    def test_all_pairs_averaged(self):
        """3 views -> mean over the 3 distinct pairs."""
        torch.manual_seed(0)
        zs = [torch.randn(8, 16) for _ in range(3)]
        expected = np.mean([
            (1 - torch.nn.functional.cosine_similarity(zs[i], zs[j], dim=-1)).mean().item()
            for i, j in [(0, 1), (0, 2), (1, 2)]
        ])
        assert subsample_consistency_loss(zs).item() == pytest.approx(expected, abs=1e-6)

    def test_gradients_flow_to_all_views(self):
        zs = [torch.randn(4, 8, requires_grad=True) for _ in range(2)]
        subsample_consistency_loss(zs).backward()
        for z in zs:
            assert z.grad is not None
            assert torch.isfinite(z.grad).all()
            assert z.grad.abs().sum() > 0

    def test_single_view_is_zero_and_differentiable(self):
        z = torch.randn(4, 8, requires_grad=True)
        loss = subsample_consistency_loss([z])
        assert loss.item() == 0.0
        loss.backward()  # must not raise
        assert z.grad is not None

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match='unknown mode'):
            subsample_consistency_loss([torch.randn(4, 8)] * 2, mode='bogus')


class TestLossModes:
    """Raw vs centered subsample loss — the A/B being compared in the sweep."""

    @staticmethod
    def _positive_orthant_views(n=32, d=16, jitter=0.05):
        """Mimic T-Net output: ReLU/max-pooled features with a big common mode."""
        torch.manual_seed(0)
        base = torch.rand(n, d) + 3.0          # strictly positive, shared offset
        return [base + jitter * torch.rand(n, d) for _ in range(2)]

    def test_raw_loss_is_near_zero_on_positive_orthant(self):
        """The reason raw cosine barely trains: it starts already satisfied."""
        zs = self._positive_orthant_views()
        raw = subsample_consistency_loss(zs, mode='cosine').item()
        cen = subsample_consistency_loss(zs, mode='centered').item()
        assert raw < 0.01, f'raw loss {raw} unexpectedly large'
        assert cen > raw * 10, (
            f'centered ({cen}) should be far larger than raw ({raw})'
        )

    def test_centered_is_invariant_to_common_offset(self):
        zs = [torch.randn(16, 8) for _ in range(2)]
        shifted = [z + 100.0 for z in zs]
        assert subsample_consistency_loss(shifted, mode='centered').item() == \
            pytest.approx(subsample_consistency_loss(zs, mode='centered').item(),
                          abs=1e-4)

    def test_raw_is_not_invariant_to_common_offset(self):
        zs = [torch.randn(16, 8) for _ in range(2)]
        shifted = [z + 100.0 for z in zs]
        assert subsample_consistency_loss(shifted, mode='cosine').item() != \
            pytest.approx(subsample_consistency_loss(zs, mode='cosine').item(),
                          abs=1e-3)

    def test_collapse_minimises_raw_but_penalises_centered(self):
        """The key safety difference between the two objectives."""
        collapsed = [torch.ones(16, 8), torch.ones(16, 8)]
        assert subsample_consistency_loss(collapsed, mode='cosine').item() == \
            pytest.approx(0.0, abs=1e-6)
        assert subsample_consistency_loss(collapsed, mode='centered').item() == \
            pytest.approx(1.0, abs=1e-5)

    def test_centered_rewards_structure_preserving_agreement(self):
        """Distinct-per-equation but view-consistent -> low centered loss."""
        torch.manual_seed(1)
        base = torch.randn(16, 8)
        zs = [base + 0.01 * torch.randn(16, 8) for _ in range(2)]
        assert subsample_consistency_loss(zs, mode='centered').item() < 0.05

    def test_gradients_flow_in_both_modes(self):
        for mode in ('cosine', 'centered'):
            zs = [torch.randn(8, 16, requires_grad=True) for _ in range(2)]
            subsample_consistency_loss(zs, mode=mode).backward()
            for z in zs:
                assert z.grad is not None and torch.isfinite(z.grad).all()
                assert z.grad.abs().sum() > 0, f'no gradient in mode={mode}'


class TestViewConsistencyDiagnostics:
    def test_perfect_invariance(self):
        """Identical views per equation -> same_fn_cos == 1."""
        n_exprs, n_views, d = 5, 4, 16
        base = torch.randn(n_exprs, d)
        z = base.repeat_interleave(n_views, dim=0)
        out = view_consistency_diagnostics(z, n_views)
        assert out['same_fn_cos'] == pytest.approx(1.0, abs=1e-5)
        assert out['gap'] > 0

    def test_collapse_gives_zero_gap(self):
        """All embeddings equal -> same and diff cosine both 1, gap 0."""
        z = torch.ones(12, 16)
        out = view_consistency_diagnostics(z, n_views=3)
        assert out['gap'] == pytest.approx(0.0, abs=1e-5)

    def test_gap_is_difference(self):
        torch.manual_seed(1)
        out = view_consistency_diagnostics(torch.randn(20, 16), n_views=4)
        assert out['gap'] == pytest.approx(
            out['same_fn_cos'] - out['diff_fn_cos'], abs=1e-6
        )

    def test_degenerate_inputs_return_nan(self):
        out = view_consistency_diagnostics(torch.randn(4, 8), n_views=1)
        assert np.isnan(out['gap'])
        assert np.isnan(out['gap_centered'])

    def test_centered_variant_removes_common_mode(self):
        """A large shared offset inflates raw cosine but not the centered one."""
        torch.manual_seed(3)
        n_exprs, n_views, d = 6, 3, 16
        base = torch.randn(n_exprs, d)
        z = base.repeat_interleave(n_views, dim=0)
        z = z + 0.05 * torch.randn_like(z)          # per-view jitter
        z_off = z + 50.0                            # strong common mode

        raw = view_consistency_diagnostics(z_off, n_views)
        # Offset drives every raw pair to ~1, crushing the raw gap...
        assert raw['diff_fn_cos'] > 0.99
        assert raw['gap'] < 0.01
        # ...while the centered gap still separates same from different.
        assert raw['gap_centered'] > 0.5

        # Centering makes the metric invariant to the offset.
        no_off = view_consistency_diagnostics(z, n_views)
        assert raw['gap_centered'] == pytest.approx(
            no_off['gap_centered'], abs=1e-3
        )

    def test_generate_embeddings_shape_and_determinism(self, exprs, tok):
        ds = PointCloudDataset(exprs, tok, n_points=100, max_vars=1, resample=False)
        enc = TNet(d_input=2, d_model=16)
        enc.eval()
        kw = dict(n_exprs=3, n_views=4, eval_seed=2718, device='cpu')
        z1 = generate_diagnostic_embeddings(ds, enc, **kw)
        z2 = generate_diagnostic_embeddings(ds, enc, **kw)
        assert z1.shape == (12, 16)
        assert torch.allclose(z1, z2), "diagnostic views are not fixed"

    def test_generate_embeddings_restores_training_mode(self, exprs, tok):
        ds = PointCloudDataset(exprs, tok, n_points=100, max_vars=1, resample=False)
        enc = TNet(d_input=2, d_model=16)
        enc.train()
        generate_diagnostic_embeddings(
            ds, enc, n_exprs=2, n_views=2, eval_seed=1, device='cpu',
        )
        assert enc.training, "encoder left in eval mode"


class TestTrainingStep:
    def test_end_to_end_backward(self, exprs, tok):
        """CE + subsample loss backprops and reaches the encoder."""
        from symbolic_jepa.decoder import SymbolicTransformer
        from torch.utils.data import DataLoader

        ds = _mv(exprs, tok, n_views=2)
        enc = TNet(d_input=2, d_model=32)
        model = SymbolicTransformer(
            encoder=enc, vocab_size=len(tok), d_model=32,
            n_heads=4, n_layers=1, d_ff=64, max_seq_len=64,
            dropout=0.0, pad_id=tok.pad_id,
        )
        batch = next(iter(DataLoader(ds, batch_size=4)))
        pv = batch['points_views']

        out = model(pv[:, 0], batch['input_ids'], attn_mask=batch['attn_mask'])
        z_views = [out['z_num']] + [model.encoder(pv[:, v]) for v in range(1, 2)]
        loss_sub = subsample_consistency_loss(z_views)
        loss = out['loss'] + 0.03 * loss_sub

        assert torch.isfinite(out['loss'])
        assert torch.isfinite(loss_sub)
        loss.backward()

        enc_grads = [p.grad for p in enc.parameters() if p.grad is not None]
        assert enc_grads, "encoder received no gradient"
        assert all(torch.isfinite(g).all() for g in enc_grads)
