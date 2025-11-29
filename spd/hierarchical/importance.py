"""Combined hierarchical importance function for H-SPD."""

from typing import Literal, override

import torch
from jaxtyping import Float
from torch import Tensor, nn

from spd.hierarchical.group_assignment import GroupAssignment
from spd.hierarchical.router import GroupRouter
from spd.hierarchical.within_group import WithinGroupImportance


class HierarchicalImportanceFunction(nn.Module):
    """Combines router and within-group networks for hierarchical importance computation.

    The hierarchical importance factorizes as:
        g^l_c(x) = g_group(k(l,c), x) × g_within(l, c, x | group k(l,c))

    where k(l,c) is the group assignment for subcomponent (l, c).
    """

    def __init__(
        self,
        group_assignment: GroupAssignment,
        router_hidden_dim: int = 64,
        within_group_hidden_dim: int = 16,
        router_aggregation: Literal[
            "per_layer_mean", "global_mean", "per_layer_max"
        ] = "per_layer_mean",
    ):
        """Initialize hierarchical importance function.

        Args:
            group_assignment: GroupAssignment object managing subcomponent-to-group mapping
            router_hidden_dim: Hidden dimension for router MLP
            within_group_hidden_dim: Hidden dimension for within-group MLPs
            router_aggregation: How router aggregates inner activations
        """
        super().__init__()
        self.group_assignment = group_assignment

        # Router network (shared across all groups)
        self.router = GroupRouter(
            num_groups=group_assignment.num_groups,
            num_layers=group_assignment.num_layers,
            aggregation=router_aggregation,
            hidden_dim=router_hidden_dim,
        )

        # Within-group networks (one per group)
        self.within_group_nets = nn.ModuleDict()
        for g in range(group_assignment.num_groups):
            group_size = group_assignment.get_group_size(g)
            self.within_group_nets[str(g)] = WithinGroupImportance(
                group_size=group_size,
                hidden_dim=within_group_hidden_dim,
            )

    def _gather_group_activations(
        self,
        inner_activations: dict[int, Float[Tensor, "batch C_l"]],
        group_idx: int,
    ) -> Float[Tensor, "batch group_size"]:
        """Gather inner activations for all subcomponents in a group.

        Args:
            inner_activations: Dict mapping layer_idx -> (batch, C^l)
            group_idx: Group index

        Returns:
            Tensor of shape (batch, group_size)
        """
        group_flat_indices = self.group_assignment.group_members[group_idx]
        if len(group_flat_indices) == 0:
            # Empty group
            batch_size = next(iter(inner_activations.values())).shape[0]
            device = next(iter(inner_activations.values())).device
            return torch.zeros(batch_size, 0, device=device)

        activations = []
        for flat_idx in group_flat_indices:
            layer_idx, local_idx = self.group_assignment.flat_to_layer_local[flat_idx]
            activations.append(inner_activations[layer_idx][:, local_idx : local_idx + 1])

        return torch.cat(activations, dim=1)  # (batch, group_size)

    @override
    def forward(
        self, inner_activations: dict[int, Float[Tensor, "batch C_l"]]
    ) -> tuple[dict[int, Float[Tensor, "batch C_l"]], Float[Tensor, "batch num_groups"]]:
        """Compute factorized importance values for all subcomponents.

        Args:
            inner_activations: Dict mapping layer_idx -> (batch, C^l)

        Returns:
            importance_by_layer: Dict mapping layer_idx -> (batch, C^l)
            group_importances: (batch, num_groups) for group sparsity loss
        """
        batch_size = next(iter(inner_activations.values())).shape[0]
        device = next(iter(inner_activations.values())).device

        # Step 1: Compute group importances
        group_importances = self.router(inner_activations)  # (batch, G)

        # Step 2: Compute within-group importances and combine
        # Initialize output tensors
        importance_by_layer: dict[int, Float[Tensor, "batch C_l"]] = {
            layer_idx: torch.zeros(
                batch_size, self.group_assignment.layer_num_subcomponents[layer_idx], device=device
            )
            for layer_idx in range(self.group_assignment.num_layers)
        }

        for g in range(self.group_assignment.num_groups):
            group_flat_indices = self.group_assignment.group_members[g]
            if len(group_flat_indices) == 0:
                continue

            # Gather activations for this group
            group_acts = self._gather_group_activations(inner_activations, g)

            # Compute within-group importance
            within_importance = self.within_group_nets[str(g)](group_acts)  # (batch, |K_g|)

            # Combine: g_combined = g_group * g_within
            g_group = group_importances[:, g : g + 1]  # (batch, 1)
            combined_importance = g_group * within_importance  # (batch, |K_g|)

            # Scatter back to layer-organized structure
            for local_g_idx, flat_idx in enumerate(group_flat_indices):
                layer_idx, layer_local_idx = self.group_assignment.flat_to_layer_local[flat_idx]
                importance_by_layer[layer_idx][:, layer_local_idx] = combined_importance[
                    :, local_g_idx
                ]

        return importance_by_layer, group_importances

    def get_importance_for_loss(
        self, inner_activations: dict[int, Float[Tensor, "batch C_l"]]
    ) -> tuple[Float[Tensor, "batch total_subcomponents"], Float[Tensor, "batch num_groups"]]:
        """Get flattened importance values with upper-leaky sigmoid for loss computation.

        Args:
            inner_activations: Dict mapping layer_idx -> (batch, C^l)

        Returns:
            all_importances: (batch, total_subcomponents) - flattened importance values
            group_importances: (batch, num_groups) - group importance values
        """
        # For the loss, we need upper-leaky versions
        group_importances_raw = self.router(inner_activations, return_raw=True)
        group_importances = self.router._leaky_hard_sigmoid(group_importances_raw, upper_leak=True)

        batch_size = group_importances.shape[0]
        device = group_importances.device
        all_importances = torch.zeros(
            batch_size, self.group_assignment.total_subcomponents, device=device
        )

        for g in range(self.group_assignment.num_groups):
            group_flat_indices = self.group_assignment.group_members[g]
            if len(group_flat_indices) == 0:
                continue

            group_acts = self._gather_group_activations(inner_activations, g)
            within_raw = self.within_group_nets[str(g)](group_acts, return_raw=True)
            within_importance = self.within_group_nets[str(g)]._leaky_hard_sigmoid(
                within_raw, upper_leak=True
            )

            g_group = group_importances[:, g : g + 1]
            combined = g_group * within_importance

            indices = self.group_assignment.group_member_tensors[g]
            all_importances[:, indices] = combined

        return all_importances, group_importances
