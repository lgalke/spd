from typing import Any, ClassVar, override

import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from spd.metrics.base import Metric
from spd.models.component_model import ComponentModel
from spd.models.components import Components
from spd.utils.distributed_utils import all_reduce


def _orthogonality_loss_update(
    components: dict[str, Components],
) -> tuple[Float[Tensor, ""], int]:
    """Compute sum of squared off-diagonal Gram matrix entries and count.

    For each component layer, computes the Gram matrix of reconstructed weights
    G[i,j] = <W_i, W_j>_F where W_c = outer(V[:, c], U[c, :]).

    This factors as G = (V^T @ V) * (U @ U^T) (element-wise product).
    """
    assert components, "Empty components dict"
    # Get device from first component's V tensor
    first_component = next(iter(components.values()))
    device = first_component.V.device
    sum_loss = torch.tensor(0.0, device=device)
    total_pairs = 0

    for component in components.values():
        C = component.C
        # G_V[i,j] = V[:, i]^T @ V[:, j]
        G_V = component.V.T @ component.V  # (C, C)
        # G_U[i,j] = U[i, :] @ U[j, :]^T
        G_U = component.U @ component.U.T  # (C, C)
        # G[i,j] = <W_i, W_j>_F where W_c is reconstructed weight of component c
        G = G_V * G_U
        # Penalize off-diagonal entries (squared)
        mask = ~torch.eye(C, dtype=torch.bool, device=device)
        sum_loss += G[mask].pow(2).sum()
        total_pairs += C * (C - 1)  # number of off-diagonal entries

    return sum_loss, total_pairs


def _orthogonality_loss_compute(sum_loss: Float[Tensor, ""], total_pairs: int) -> Float[Tensor, ""]:
    if total_pairs == 0:
        return sum_loss  # Return 0 if C=1 (no off-diagonal entries)
    return sum_loss / total_pairs


def orthogonality_loss(components: dict[str, Components]) -> Float[Tensor, ""]:
    """Calculate orthogonality loss encouraging component weight matrices to be orthogonal.

    For each component layer, this computes the Gram matrix of reconstructed weights
    and penalizes non-diagonal entries (i.e., inner products between different components).

    The reconstructed weight for component c is W_c = outer(V[:, c], U[c, :]).
    The Frobenius inner product <W_i, W_j>_F = (V^T @ V)[i,j] * (U @ U^T)[i,j].
    This loss penalizes the mean of squared off-diagonal entries of this Gram matrix.

    Args:
        components: Dictionary of components for each layer.

    Returns:
        The orthogonality loss as a scalar tensor.
    """
    sum_loss, total_pairs = _orthogonality_loss_update(components)
    return _orthogonality_loss_compute(sum_loss, total_pairs)


class OrthogonalityLoss(Metric):
    """Penalizes non-orthogonal component weight matrices.

    Encourages the reconstructed weight matrices of different components to be
    orthogonal to each other by penalizing off-diagonal entries of the Gram matrix
    formed by inner products between component weights.
    """

    metric_section: ClassVar[str] = "loss"

    def __init__(self, model: ComponentModel, device: str) -> None:
        self.model = model
        self.sum_loss = torch.tensor(0.0, device=device)
        self.total_pairs = 0

    @override
    def update(self, **_: Any) -> None:
        sum_loss, total_pairs = _orthogonality_loss_update(self.model.components)
        self.sum_loss += sum_loss
        self.total_pairs += total_pairs

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        return _orthogonality_loss_compute(sum_loss, self.total_pairs)
