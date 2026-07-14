"""JEPA diagnostics: spread, retrieval, common-mode analysis."""

import torch
import torch.nn.functional as F


def _pairwise_cos(z: torch.Tensor) -> torch.Tensor:
    """Return (B, B) pairwise cosine similarity matrix."""
    z_norm = F.normalize(z, dim=-1)
    return z_norm @ z_norm.T


def sym_spread(z_sym: torch.Tensor) -> dict:
    """Off-diagonal mean pairwise cosine, raw and mean-centered.

    High off-diag cos = collapsed / anisotropic embeddings.
    """
    B = z_sym.shape[0]
    # Raw
    cos_raw = _pairwise_cos(z_sym)
    mask = ~torch.eye(B, device=z_sym.device, dtype=torch.bool)
    raw_offdiag = cos_raw[mask].mean().item()

    # Centered
    z_cent = z_sym - z_sym.mean(0)
    cos_cent = _pairwise_cos(z_cent)
    cent_offdiag = cos_cent[mask].mean().item()

    return {"raw": raw_offdiag, "centered": cent_offdiag}


def pred_spread(z_pred: torch.Tensor) -> dict:
    """Off-diagonal mean pairwise cosine of predictor outputs.

    High value = predictor outputting near-constant vector.
    """
    B = z_pred.shape[0]
    cos = _pairwise_cos(z_pred)
    mask = ~torch.eye(B, device=z_pred.device, dtype=torch.bool)
    return {"raw": cos[mask].mean().item()}


def retrieval_top1(z_pred: torch.Tensor, z_sym: torch.Tensor) -> dict:
    """Row-argmax of cosine matrix == diagonal → retrieval accuracy.

    Reports raw and centered versions. Chance = 1/B.
    """
    B = z_pred.shape[0]

    # Raw
    cos_raw = F.normalize(z_pred, dim=-1) @ F.normalize(z_sym, dim=-1).T
    acc_raw = (cos_raw.argmax(dim=1) == torch.arange(B, device=z_pred.device)).float().mean().item()

    # Centered
    zp = z_pred - z_pred.mean(0)
    zs = z_sym - z_sym.mean(0)
    cos_cent = F.normalize(zp, dim=-1) @ F.normalize(zs, dim=-1).T
    acc_cent = (cos_cent.argmax(dim=1) == torch.arange(B, device=z_pred.device)).float().mean().item()

    return {"raw": acc_raw, "centered": acc_cent, "chance": 1.0 / B}


def common_mode(z: torch.Tensor) -> dict:
    """Common-mode analysis: mean-norm ratio and top-3 PC variance share.

    mean_norm_ratio = ||mean(z)|| / mean(||z||).  Close to 1 = strong common mode.
    pc_var_share = fraction of variance in top-3 PCs (via pca_lowrank).
    """
    mean_vec = z.mean(0)
    mean_norm = mean_vec.norm().item()
    avg_norm = z.norm(dim=-1).mean().item()
    ratio = mean_norm / (avg_norm + 1e-10)

    # Top-3 PC variance share
    z_centered = z - mean_vec
    try:
        _, S, _ = torch.pca_lowrank(z_centered, q=min(3, z.shape[0], z.shape[1]))
        var_explained = S ** 2
        total_var = z_centered.var(dim=0).sum() * z.shape[0]
        pc_share = (var_explained.sum() / (total_var + 1e-10)).item()
    except Exception:
        pc_share = float("nan")

    return {"mean_norm_ratio": ratio, "pc_top3_var_share": pc_share}
