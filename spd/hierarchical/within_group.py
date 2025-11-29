"""Within-group importance networks for hierarchical SPD."""

from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn


class WithinGroupImportance(nn.Module):
    """Computes importance within a single group.

    Each group has its own small MLP that takes the inner activations of
    subcomponents in that group and outputs per-subcomponent importance values.
    """

    def __init__(self, group_size: int, hidden_dim: int = 16, leak: float = 0.01):
        """Initialize within-group importance network.

        Args:
            group_size: Number of subcomponents in this group |K_g|
            hidden_dim: MLP hidden dimension
            leak: Leak parameter for leaky hard sigmoid (for gradient flow)
        """
        super().__init__()
        self.group_size = group_size
        self.leak = leak

        if group_size == 0:
            # Empty group - no network needed
            self.mlp = None
        else:
            self.mlp = nn.Sequential(
                nn.Linear(group_size, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, group_size),
            )

    def _leaky_hard_sigmoid(
        self, x: Float[Tensor, "..."], upper_leak: bool = False
    ) -> Float[Tensor, "..."]:
        """Hard sigmoid with leaky regions for gradient flow.

        Args:
            x: Input tensor
            upper_leak: If True, leak above 1 (for importance loss). If False, leak below 0 (for forward).

        Returns:
            Leaky hard sigmoid of x
        """
        if upper_leak:
            # For importance loss: leak above 1, clamp below 0
            return torch.where(
                x <= 0,
                torch.zeros_like(x),
                torch.where(x >= 1, 1 + self.leak * (x - 1), x),
            )
        else:
            # For forward pass: leak below 0, clamp above 1
            return torch.where(
                x <= 0,
                self.leak * x,
                torch.where(x >= 1, torch.ones_like(x), x),
            )

    @override
    def forward(
        self,
        group_inner_activations: Float[Tensor, "batch group_size"],
        return_raw: bool = False,
        upper_leak: bool = False,
    ) -> Float[Tensor, "batch group_size"]:
        """Compute within-group importance.

        Args:
            group_inner_activations: Shape (batch, group_size)
            return_raw: If True, return pre-sigmoid values
            upper_leak: If True, use upper-leaky sigmoid (for loss computation)

        Returns:
            Tensor of shape (batch, group_size) with within-group importance values
        """
        if self.mlp is None:
            # Empty group - return empty tensor
            return torch.zeros(
                group_inner_activations.shape[0], 0, device=group_inner_activations.device
            )

        raw_logits = self.mlp(group_inner_activations)

        if return_raw:
            return raw_logits

        return self._leaky_hard_sigmoid(raw_logits, upper_leak=upper_leak)
