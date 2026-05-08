"""
Focal loss for multi-class classification (Lin et al. 2017).

Modulates standard cross-entropy by (1 - p_t)^gamma so that easy
correctly-classified examples contribute less and the model focuses on
the minority class. With gamma=0 reduces to plain cross-entropy.

Used by Phase A LSTM softmax heads to address the EUR/JPY collapse
diagnosed in the 2026-04-25 retrain (memory/project_phase2a_pivot.md).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        class_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.class_weight = class_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard CE with no reduction so we can apply the (1-p_t)^gamma
        # modulator per-sample.
        ce = F.cross_entropy(
            logits, targets, weight=self.class_weight, reduction="none"
        )
        # p_t = exp(-ce) gives the predicted probability of the true class.
        p_t = torch.exp(-ce)
        loss = ((1.0 - p_t) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
