"""
PyTorch Dataset for symbolic regression.

Wraps Expression objects into point-cloud + token-sequence pairs.
Each __getitem__ resamples the point cloud for data augmentation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from symbolic_jepa.expressions import Expression, load_feynman_csv
from symbolic_jepa.tokenizer import PrefixTokenizer


class PointCloudDataset(Dataset):
    """Dataset of (point_cloud, token_ids) pairs.

    Each call to __getitem__ freshly samples the point cloud from the
    Expression, providing infinite augmentation.
    """

    def __init__(
        self,
        expressions: list[Expression],
        tokenizer: PrefixTokenizer,
        n_points: int = 1000,
        max_seq_len: int = 64,
        max_vars: int = 9,
        resample: bool = True,
    ):
        self.tokenizer = tokenizer
        self.n_points = n_points
        self.max_seq_len = max_seq_len
        self.max_vars = max_vars
        self.target_d = max_vars + 1  # input vars + output
        self.resample = resample

        # Pre-tokenize and filter
        self.samples: list[dict] = []
        for expr in expressions:
            try:
                ids = expr.tokenize(tokenizer)
            except (ValueError, Exception):
                continue

            if len(ids) > max_seq_len or tokenizer.unk_id in ids:
                continue

            pad = max_seq_len - len(ids)
            self.samples.append({
                'expr': expr,
                'input_ids': torch.tensor(ids + [tokenizer.pad_id] * pad, dtype=torch.long),
                'attn_mask': torch.tensor([1] * len(ids) + [0] * pad, dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        points = self._sample_points(s['expr'])
        return {
            'points': points,
            'input_ids': s['input_ids'],
            'attn_mask': s['attn_mask'],
        }

    def _sample_points(self, expr: Expression) -> torch.Tensor:
        """Sample, normalize, and pad a point cloud from the expression."""
        cloud = expr.sample(self.n_points, method='uniform')  # (n_points, n_vars+1)

        # Filter non-finite rows
        finite_mask = np.isfinite(cloud).all(axis=1)
        cloud = cloud[finite_mask]

        if len(cloud) < 10:
            # Fallback: return zeros if expression is badly behaved
            return torch.zeros(self.n_points, self.target_d, dtype=torch.float32)

        # Pad or truncate to n_points
        if len(cloud) >= self.n_points:
            if self.resample:
                idx = np.random.choice(len(cloud), self.n_points, replace=False)
                cloud = cloud[idx]
            else:
                cloud = cloud[:self.n_points]
        else:
            # Repeat with noise if too few valid points
            n_need = self.n_points - len(cloud)
            extra_idx = np.random.choice(len(cloud), n_need, replace=True)
            cloud = np.vstack([cloud, cloud[extra_idx]])

        # Normalize per-column
        cloud = (cloud - cloud.mean(axis=0)) / (cloud.std(axis=0) + 1e-8)

        # Pad dimensions to target_d
        n, d = cloud.shape
        if d < self.target_d:
            cloud = np.pad(cloud, ((0, 0), (0, self.target_d - d)))
        else:
            cloud = cloud[:, :self.target_d]

        return torch.tensor(cloud, dtype=torch.float32)


def build_feynman_splits(
    csv_path: str,
    tokenizer: PrefixTokenizer,
    n_points: int = 1000,
    max_seq_len: int = 64,
    max_vars: int = 9,
    seed: int = 42,
) -> tuple[PointCloudDataset, PointCloudDataset, PointCloudDataset]:
    """Load Feynman equations and split into train/val/test datasets.

    Returns:
        (train_ds, val_ds, test_ds)
    """
    all_exprs = load_feynman_csv(csv_path)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(all_exprs))
    rng.shuffle(idx)
    n = len(idx)

    splits = {
        'train': idx[:int(0.8 * n)],
        'val': idx[int(0.8 * n):int(0.9 * n)],
        'test': idx[int(0.9 * n):],
    }

    datasets = {}
    for name, indices in splits.items():
        exprs = [all_exprs[i] for i in indices]
        datasets[name] = PointCloudDataset(
            exprs, tokenizer,
            n_points=n_points,
            max_seq_len=max_seq_len,
            max_vars=max_vars,
            resample=(name == 'train'),
        )
        print(f'Feynman {name}: {len(datasets[name])} equations')

    return datasets['train'], datasets['val'], datasets['test']


def build_synthetic_splits(
    expressions: list[Expression],
    tokenizer: PrefixTokenizer,
    n_points: int = 1000,
    max_seq_len: int = 64,
    max_vars: int = 9,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> tuple[PointCloudDataset, PointCloudDataset, PointCloudDataset]:
    """Split synthetic expressions into train/val/test datasets.

    Returns:
        (train_ds, val_ds, test_ds)
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(expressions))
    rng.shuffle(idx)
    n = len(idx)

    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    splits = {
        'train': idx[:n_train],
        'val': idx[n_train:n_train + n_val],
        'test': idx[n_train + n_val:],
    }

    datasets = {}
    for name, indices in splits.items():
        exprs = [expressions[i] for i in indices]
        datasets[name] = PointCloudDataset(
            exprs, tokenizer,
            n_points=n_points,
            max_seq_len=max_seq_len,
            max_vars=max_vars,
            resample=(name == 'train'),
        )
        print(f'Synthetic {name}: {len(datasets[name])} equations')

    return datasets['train'], datasets['val'], datasets['test']
