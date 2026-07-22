"""JEPA alignment components: predictor MLP and cosine alignment loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPAPredictor(nn.Module):
    """2-layer MLP projecting z_num into z_sym's space."""

    def __init__(self, d_model: int, d_hidden: int | None = None):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, z_num: torch.Tensor) -> torch.Tensor:
        """(batch, d_model) -> (batch, d_model)"""
        return self.net(z_num)


class IdentityPredictor(nn.Module):
    """No-op predictor: z_pred = z_num."""

    def forward(self, z_num: torch.Tensor) -> torch.Tensor:
        return z_num


def jepa_loss(
    z_pred: torch.Tensor,
    z_sym: torch.Tensor,
    mode: str = "cosine",
) -> torch.Tensor:
    """Cosine alignment loss: (1 - cosine_similarity).mean().

    Args:
        z_pred: (batch, d_model) from predictor(z_num).
        z_sym: (batch, d_model) — detach at call site if needed.
        mode: "cosine" (default) or "raw" — ordinary pairwise cosine;
              "centered" — subtracts batch mean before cosine.
    Returns:
        Scalar loss in [0, 2].
    """
    if mode == "centered":
        z_pred = z_pred - z_pred.mean(0)
        z_sym = z_sym - z_sym.mean(0)
    cos = F.cosine_similarity(z_pred, z_sym, dim=-1)
    return (1 - cos).mean()
