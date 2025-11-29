"""Group assignment management for hierarchical SPD."""

from typing import Literal

import numpy as np
import torch
from jaxtyping import Int
from torch import Tensor


class GroupAssignment:
    """Manages mapping from subcomponents to groups in hierarchical SPD.

    Each subcomponent (layer_idx, local_idx) is assigned to exactly one group.
    Groups can span multiple layers, enabling cross-layer mechanism discovery.
    """

    def __init__(
        self,
        num_groups: int,
        layer_num_subcomponents: list[int],
        init_strategy: Literal["random", "layer_based", "kmeans"] = "random",
        device: str = "cpu",
    ):
        """Initialize group assignment.

        Args:
            num_groups: Number of groups G
            layer_num_subcomponents: List where element l is C^l (num subcomponents in layer l)
            init_strategy: How to initialize assignments:
                - 'random': Random assignment to groups
                - 'layer_based': Subcomponents from same layer go to same group (round-robin)
                - 'kmeans': Placeholder for k-means (initialized randomly, can be updated later)
            device: Device to place tensors on
        """
        self.num_groups = num_groups
        self.layer_num_subcomponents = layer_num_subcomponents
        self.num_layers = len(layer_num_subcomponents)
        self.total_subcomponents = sum(layer_num_subcomponents)
        self.device = device

        # Create flat index mapping
        self._build_index_mappings()

        # Initialize assignments
        self.assignments = self._initialize_assignments(init_strategy)

        # Build reverse mapping: group -> list of flat indices
        self._build_group_members()

    def _build_index_mappings(self):
        """Build bidirectional mappings between flat and (layer, local) indices."""
        self.flat_to_layer_local: list[tuple[int, int]] = []
        self.layer_local_to_flat: dict[tuple[int, int], int] = {}

        flat_idx = 0
        for layer_idx, num_sub in enumerate(self.layer_num_subcomponents):
            for local_idx in range(num_sub):
                self.flat_to_layer_local.append((layer_idx, local_idx))
                self.layer_local_to_flat[(layer_idx, local_idx)] = flat_idx
                flat_idx += 1

    def _initialize_assignments(self, strategy: str) -> Int[Tensor, "total_subcomponents"]:
        """Initialize group assignments.

        Returns:
            Tensor of shape (total_subcomponents,) with values in [0, num_groups)
        """
        if strategy == "random":
            return torch.randint(0, self.num_groups, (self.total_subcomponents,), device=self.device)

        elif strategy == "layer_based":
            # Assign subcomponents from same layer to same group initially
            assignments = []
            for layer_idx, num_sub in enumerate(self.layer_num_subcomponents):
                group_for_layer = layer_idx % self.num_groups
                assignments.extend([group_for_layer] * num_sub)
            return torch.tensor(assignments, dtype=torch.long, device=self.device)

        elif strategy == "kmeans":
            # For now, initialize randomly - can be updated later with actual clustering
            return torch.randint(0, self.num_groups, (self.total_subcomponents,), device=self.device)

        else:
            raise ValueError(f"Unknown init strategy: {strategy}")

    def _build_group_members(self):
        """Build dict mapping group_idx -> list of flat indices."""
        self.group_members: dict[int, list[int]] = {g: [] for g in range(self.num_groups)}
        for flat_idx, group_idx in enumerate(self.assignments.tolist()):
            self.group_members[group_idx].append(flat_idx)

        # Also store as tensors for efficient indexing
        self.group_member_tensors: dict[int, Int[Tensor, "group_size"]] = {
            g: torch.tensor(members, dtype=torch.long, device=self.device)
            for g, members in self.group_members.items()
        }

    def get_group(self, layer_idx: int, local_idx: int) -> int:
        """Get group index for a subcomponent.

        Args:
            layer_idx: Layer index
            local_idx: Subcomponent index within layer

        Returns:
            Group index in [0, num_groups)
        """
        flat_idx = self.layer_local_to_flat[(layer_idx, local_idx)]
        return self.assignments[flat_idx].item()

    def get_group_size(self, group_idx: int) -> int:
        """Get number of subcomponents in a group.

        Args:
            group_idx: Group index

        Returns:
            Number of subcomponents assigned to this group
        """
        return len(self.group_members[group_idx])

    def reassign_from_coactivation(
        self, coactivation_matrix: Tensor, method: str = "kmeans"
    ) -> None:
        """Reassign subcomponents to groups based on co-activation patterns.

        Note: This is a future feature for dynamic reassignment during training.
        Currently only supports kmeans clustering.

        Args:
            coactivation_matrix: Shape (total_subcomponents, total_subcomponents)
                Entry [i,j] = how often subcomponents i and j are both important
            method: Clustering method ('kmeans' supported)
        """
        if method == "kmeans":
            # Use co-activation matrix rows as features
            try:
                from sklearn.cluster import KMeans
            except ImportError:
                raise ImportError(
                    "scikit-learn is required for kmeans reassignment. "
                    "Install with: pip install scikit-learn"
                )

            features = coactivation_matrix.cpu().numpy()
            kmeans = KMeans(n_clusters=self.num_groups, random_state=42, n_init=10)
            new_assignments = kmeans.fit_predict(features)
            self.assignments = torch.tensor(new_assignments, dtype=torch.long, device=self.device)
        else:
            raise NotImplementedError(f"Method {method} not implemented")

        self._build_group_members()

    def to(self, device: str) -> "GroupAssignment":
        """Move tensors to device.

        Args:
            device: Device to move to

        Returns:
            Self for chaining
        """
        self.device = device
        self.assignments = self.assignments.to(device)
        self.group_member_tensors = {
            g: t.to(device) for g, t in self.group_member_tensors.items()
        }
        return self

    def state_dict(self) -> dict:
        """Return state dict for serialization.

        Returns:
            Dictionary containing all state needed to reconstruct this object
        """
        return {
            "num_groups": self.num_groups,
            "layer_num_subcomponents": self.layer_num_subcomponents,
            "assignments": self.assignments.cpu(),
            "device": self.device,
        }

    @classmethod
    def from_state_dict(cls, state_dict: dict) -> "GroupAssignment":
        """Reconstruct GroupAssignment from state dict.

        Args:
            state_dict: Dictionary from state_dict() method

        Returns:
            Reconstructed GroupAssignment instance
        """
        # Create instance with dummy init strategy (we'll override assignments)
        obj = cls(
            num_groups=state_dict["num_groups"],
            layer_num_subcomponents=state_dict["layer_num_subcomponents"],
            init_strategy="random",
            device=state_dict["device"],
        )

        # Override assignments with saved values
        obj.assignments = state_dict["assignments"].to(state_dict["device"])
        obj._build_group_members()

        return obj
