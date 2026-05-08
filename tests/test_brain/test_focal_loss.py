"""
Focal loss for 3-class softmax (spec §1 Phase A — fix EUR/JPY collapse).
Down-weights easy examples by (1 - p_t)^gamma. With gamma=0 it reduces
to plain cross-entropy.
"""
import torch


def test_focal_loss_gamma_zero_equals_ce():
    """gamma=0 should reproduce plain cross-entropy."""
    from src.brain.deep_learning.focal_loss import FocalLoss

    logits = torch.tensor([[2.0, 1.0, 0.5], [0.1, 0.5, 0.3]])
    targets = torch.tensor([0, 1])

    fl = FocalLoss(gamma=0.0)
    ce = torch.nn.CrossEntropyLoss(reduction="mean")

    assert torch.allclose(fl(logits, targets), ce(logits, targets), atol=1e-6)


def test_focal_loss_downweights_easy_examples():
    """gamma>0 should produce smaller loss on confident-correct examples."""
    from src.brain.deep_learning.focal_loss import FocalLoss

    confident_logits = torch.tensor([[10.0, 0.0, 0.0]])
    confident_target = torch.tensor([0])

    fl0 = FocalLoss(gamma=0.0)
    fl2 = FocalLoss(gamma=2.0)

    loss0 = fl0(confident_logits, confident_target)
    loss2 = fl2(confident_logits, confident_target)

    assert loss2 < loss0


def test_focal_loss_with_class_weights():
    """class_weight tensor should multiply the per-class contribution."""
    from src.brain.deep_learning.focal_loss import FocalLoss

    logits = torch.tensor([[1.0, 2.0, 0.0]])
    targets = torch.tensor([1])
    weights = torch.tensor([1.0, 5.0, 1.0])

    fl_unweighted = FocalLoss(gamma=2.0)
    fl_weighted = FocalLoss(gamma=2.0, class_weight=weights)

    assert fl_weighted(logits, targets) > fl_unweighted(logits, targets) * 4.0
