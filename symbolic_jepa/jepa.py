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


def jepa_loss(
    z_pred: torch.Tensor,
    z_sym_detached: torch.Tensor,
    mode: str = "centered",
) -> torch.Tensor:
    """Cosine alignment loss: (1 - cosine_similarity).mean().

    Args:
        z_pred: (batch, d_model) from predictor(z_num).
        z_sym_detached: (batch, d_model) — must already be detached.
        mode: "centered" subtracts batch mean before cosine (default);
              "raw" uses vectors as-is.
    Returns:
        Scalar loss in [0, 2].
    """
    if mode == "centered":
        z_pred = z_pred - z_pred.mean(0)
        z_sym_detached = z_sym_detached - z_sym_detached.mean(0)
    cos = F.cosine_similarity(z_pred, z_sym_detached, dim=-1)
    return (1 - cos).mean()
