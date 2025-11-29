"""Group sparsity loss for hierarchical SPD."""

import torch
from jaxtyping import Float
from torch import Tensor


def group_sparsity_loss(
    group_importances: Float[Tensor, "batch num_groups"],
    pnorm: float = 1.0,
) -> Float[Tensor, ""]:
    """Compute group sparsity loss to encourage few groups to be active.

    Args:
        group_importances: Group importance values of shape (batch, num_groups)
        pnorm: p-norm for sparsity (typically 1.0 for L1 sparsity)

    Returns:
        Scalar loss tensor
    """
    # Apply p-norm across groups dimension, then average over batch
    loss = (group_importances.abs() ** pnorm).sum(dim=1).mean()
    return loss
