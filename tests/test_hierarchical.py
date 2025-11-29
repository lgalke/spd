"""Tests for hierarchical SPD components."""

import pytest
import torch

from spd.hierarchical import (
    GroupAssignment,
    GroupRouter,
    HierarchicalImportanceFunction,
    WithinGroupImportance,
)


class TestGroupAssignment:
    """Tests for GroupAssignment class."""

    def test_initialization_random(self):
        ga = GroupAssignment(
            num_groups=4, layer_num_subcomponents=[10, 20, 10], init_strategy="random"
        )
        assert ga.total_subcomponents == 40
        assert ga.num_groups == 4
        assert len(ga.assignments) == 40

    def test_layer_based_init(self):
        ga = GroupAssignment(
            num_groups=3, layer_num_subcomponents=[5, 5, 5], init_strategy="layer_based"
        )
        # Layer 0 -> group 0, layer 1 -> group 1, layer 2 -> group 2
        assert ga.get_group(0, 0) == 0
        assert ga.get_group(1, 0) == 1
        assert ga.get_group(2, 0) == 2

    def test_bidirectional_indexing(self):
        ga = GroupAssignment(
            num_groups=2, layer_num_subcomponents=[3, 4], init_strategy="random"
        )
        for flat_idx in range(7):
            layer, local = ga.flat_to_layer_local[flat_idx]
            assert ga.layer_local_to_flat[(layer, local)] == flat_idx

    def test_state_dict_serialization(self):
        ga = GroupAssignment(
            num_groups=4, layer_num_subcomponents=[10, 20], init_strategy="random"
        )
        state = ga.state_dict()
        ga_restored = GroupAssignment.from_state_dict(state)

        assert ga_restored.num_groups == ga.num_groups
        assert ga_restored.total_subcomponents == ga.total_subcomponents
        assert torch.equal(ga_restored.assignments, ga.assignments)


class TestGroupRouter:
    """Tests for GroupRouter class."""

    def test_forward_shape(self):
        router = GroupRouter(
            num_groups=8, num_layers=4, aggregation="per_layer_mean", hidden_dim=32
        )
        inner_acts = {
            0: torch.randn(16, 100),
            1: torch.randn(16, 100),
            2: torch.randn(16, 100),
            3: torch.randn(16, 100),
        }
        output = router(inner_acts)
        assert output.shape == (16, 8)
        # Check leaky hard sigmoid bounds
        assert (output >= -0.1).all() and (output <= 1.1).all()

    def test_global_mean_aggregation(self):
        router = GroupRouter(
            num_groups=5, num_layers=3, aggregation="global_mean", hidden_dim=16
        )
        inner_acts = {0: torch.randn(8, 50), 1: torch.randn(8, 50), 2: torch.randn(8, 50)}
        output = router(inner_acts)
        assert output.shape == (8, 5)


class TestWithinGroupImportance:
    """Tests for WithinGroupImportance class."""

    def test_forward_shape(self):
        within_group = WithinGroupImportance(group_size=10, hidden_dim=8)
        group_acts = torch.randn(16, 10)
        output = within_group(group_acts)
        assert output.shape == (16, 10)

    def test_empty_group(self):
        within_group = WithinGroupImportance(group_size=0, hidden_dim=8)
        group_acts = torch.randn(16, 0)
        output = within_group(group_acts)
        assert output.shape == (16, 0)


class TestHierarchicalImportanceFunction:
    """Tests for combined hierarchical importance function."""

    def test_end_to_end(self):
        ga = GroupAssignment(
            num_groups=4, layer_num_subcomponents=[20, 20], init_strategy="random"
        )
        hif = HierarchicalImportanceFunction(
            group_assignment=ga, router_hidden_dim=32, within_group_hidden_dim=16
        )

        inner_acts = {0: torch.randn(8, 20), 1: torch.randn(8, 20)}

        importance_by_layer, group_importance = hif(inner_acts)

        assert importance_by_layer[0].shape == (8, 20)
        assert importance_by_layer[1].shape == (8, 20)
        assert group_importance.shape == (8, 4)

    def test_factorization_constraint(self):
        """Test that combined importance values are in valid range."""
        ga = GroupAssignment(
            num_groups=2, layer_num_subcomponents=[10], init_strategy="layer_based"
        )
        hif = HierarchicalImportanceFunction(ga)

        inner_acts = {0: torch.randn(4, 10)}
        importance_by_layer, group_importance = hif(inner_acts)

        # Combined importance should be product of group and within (both in [0,1])
        # So combined should be in [0,1] as well (with small leaky tolerance)
        assert (importance_by_layer[0] >= -0.05).all()
        assert (importance_by_layer[0] <= 1.05).all()

    def test_get_importance_for_loss(self):
        """Test that get_importance_for_loss returns correct shapes."""
        ga = GroupAssignment(
            num_groups=3, layer_num_subcomponents=[15, 15], init_strategy="random"
        )
        hif = HierarchicalImportanceFunction(ga)

        inner_acts = {0: torch.randn(6, 15), 1: torch.randn(6, 15)}

        all_importances, group_importances = hif.get_importance_for_loss(inner_acts)

        assert all_importances.shape == (6, 30)  # batch_size, total_subcomponents
        assert group_importances.shape == (6, 3)  # batch_size, num_groups
