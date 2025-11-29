"""Group router for computing group-level importance in hierarchical SPD."""

from typing import Literal, override

import torch
from jaxtyping import Float
from torch import Tensor, nn


class GroupRouter(nn.Module):
    """Computes group-level importance values from aggregated inner activations.

    The router is a lightweight MLP that takes aggregated representations of
    inner activations and outputs importance values for each group.
    """

    def __init__(
        self,
        num_groups: int,
        num_layers: int,
        aggregation: Literal["per_layer_mean", "global_mean", "per_layer_max"] = "per_layer_mean",
        hidden_dim: int = 64,
        leak: float = 0.01,
    ):
        """Initialize group router.

        Args:
            num_groups: Number of groups G
            num_layers: Number of layers L (for per-layer aggregation)
            aggregation: How to aggregate inner activations:
                - 'per_layer_mean': Mean activation per layer (input_dim = num_layers)
                - 'global_mean': Mean across all layers (input_dim = 1)
                - 'per_layer_max': Max absolute activation per layer (input_dim = num_layers)
            hidden_dim: Router MLP hidden dimension
            leak: Leak parameter for leaky hard sigmoid (for gradient flow)
        """
        super().__init__()
        self.num_groups = num_groups
        self.num_layers = num_layers
        self.aggregation = aggregation
        self.leak = leak

        # Input dimension depends on aggregation strategy
        if aggregation in ["per_layer_mean", "per_layer_max"]:
            input_dim = num_layers
        elif aggregation == "global_mean":
            input_dim = 1
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_groups),
        )

    def _aggregate_inner_activations(
        self, inner_activations: dict[int, Float[Tensor, "batch C_l"]]
    ) -> Float[Tensor, "batch input_dim"]:
        """Aggregate inner activations into router input.

        Args:
            inner_activations: Dict mapping layer_idx -> tensor of shape (batch, C^l)

        Returns:
            Tensor of shape (batch, input_dim)
        """
        batch_size = next(iter(inner_activations.values())).shape[0]
        device = next(iter(inner_activations.values())).device

        if self.aggregation == "per_layer_mean":
            # Mean activation per layer
            layer_means = []
            for layer_idx in range(self.num_layers):
                if layer_idx in inner_activations:
                    layer_means.append(inner_activations[layer_idx].mean(dim=1, keepdim=True))
                else:
                    layer_means.append(torch.zeros(batch_size, 1, device=device))
            return torch.cat(layer_means, dim=1)  # (batch, L)

        elif self.aggregation == "per_layer_max":
            layer_maxs = []
            for layer_idx in range(self.num_layers):
                if layer_idx in inner_activations:
                    layer_maxs.append(inner_activations[layer_idx].abs().max(dim=1, keepdim=True)[0])
                else:
                    layer_maxs.append(torch.zeros(batch_size, 1, device=device))
            return torch.cat(layer_maxs, dim=1)  # (batch, L)

        elif self.aggregation == "global_mean":
            all_acts = torch.cat([v for v in inner_activations.values()], dim=1)
            return all_acts.mean(dim=1, keepdim=True)  # (batch, 1)

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

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
        inner_activations: dict[int, Float[Tensor, "batch C_l"]],
        return_raw: bool = False,
        upper_leak: bool = False,
    ) -> Float[Tensor, "batch num_groups"]:
        """Compute group importance values.

        Args:
            inner_activations: Dict mapping layer_idx -> (batch, C^l)
            return_raw: If True, return pre-sigmoid values
            upper_leak: If True, use upper-leaky sigmoid (for loss computation)

        Returns:
            Tensor of shape (batch, num_groups) with group importance values
        """
        aggregated = self._aggregate_inner_activations(inner_activations)
        raw_logits = self.mlp(aggregated)

        if return_raw:
            return raw_logits

        return self._leaky_hard_sigmoid(raw_logits, upper_leak=upper_leak)
