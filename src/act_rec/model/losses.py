from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def masked_recon_loss(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error applied only where mask == 1.

    Parameters
    ----------
    x_hat: torch.Tensor
        Predicted sequence, shape `(N, C, T, V, M)`.
    x: torch.Tensor
        Ground-truth sequence with the same shape.
    mask: torch.Tensor
        Mask broadcastable to the same shape as `x_hat`.
    """
    mask = mask.to(x_hat.dtype)
    recon = F.mse_loss(x_hat, x, reduction="none") * mask
    mask_sum = mask.sum()
    if mask_sum.item() == 0:
        raise ValueError("Reconstruction mask is empty; cannot compute masked loss.")
    return recon.sum() / mask_sum


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
