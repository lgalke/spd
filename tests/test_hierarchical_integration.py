"""Integration tests for hierarchical SPD on toy models.

Tests that hierarchical SPD works end-to-end with actual training loops.
"""

import pytest
import torch

from spd.experiments.tms.configs import TMSModelConfig
from spd.experiments.tms.models import TMSModel
from spd.losses import compute_total_loss
from spd.models.component_model import ComponentModel
from spd.utils.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset


@pytest.mark.slow
class TestHierarchicalIntegration:
    """Integration tests for hierarchical SPD."""

    @pytest.fixture
    def tiny_tms_model(self):
        """Create a tiny TMS model for testing."""
        config = TMSModelConfig(
            n_features=5,
            n_hidden=2,
            n_hidden_layers=0,
            tied_weights=False,
            init_bias_to_zero=True,
            device="cpu",
        )
        model = TMSModel(config)
        model.eval()
        model.requires_grad_(False)
        return model

    @pytest.fixture
    def tiny_dataloader(self):
        """Create a tiny dataloader for testing."""
        dataset = SparseFeatureDataset(
            n_features=5,
            feature_probability=0.05,
            data_generation_type="at_least_zero_active",
            device="cpu",
        )
        return DatasetGeneratedDataLoader(dataset, batch_size=32)

    def test_hierarchical_spd_basic_forward(self, tiny_tms_model):
        """Test that hierarchical ComponentModel can be created and run forward pass."""
        # Create hierarchical component model
        comp_model = ComponentModel(
            target_model=tiny_tms_model,
            target_module_patterns=["linear1", "linear2"],
            C=10,
            ci_fn_type="hierarchical",
            ci_fn_hidden_dims=[8],
            sigmoid_type="leaky_hard",
            pretrained_model_output_attr=None,
            # Hierarchical parameters
            num_groups=4,
            router_hidden_dim=16,
            within_group_hidden_dim=8,
            router_aggregation="per_layer_mean",
            group_init_strategy="random",
        )

        # Test forward pass
        test_input = torch.randn(8, 5)
        with torch.no_grad():
            output = comp_model.target_model(test_input)

        assert output.shape == (8, 5)
        assert not torch.isnan(output).any()

    def test_hierarchical_ci_computation(self, tiny_tms_model):
        """Test that hierarchical causal importance computation works."""
        comp_model = ComponentModel(
            target_model=tiny_tms_model,
            target_module_patterns=["linear1", "linear2"],
            C=10,
            ci_fn_type="hierarchical",
            ci_fn_hidden_dims=[8],
            sigmoid_type="leaky_hard",
            pretrained_model_output_attr=None,
            num_groups=4,
            router_hidden_dim=16,
            within_group_hidden_dim=8,
        )

        # Get activations
        test_input = torch.randn(8, 5)
        output_with_cache = comp_model(test_input, cache_type="input")

        # Compute causal importances
        ci_outputs = comp_model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache, sampling="continuous", detach_inputs=False
        )

        # Check outputs
        assert "linear1" in ci_outputs.lower_leaky
        assert "linear2" in ci_outputs.lower_leaky
        assert ci_outputs.lower_leaky["linear1"].shape == (8, 10)
        assert ci_outputs.lower_leaky["linear2"].shape == (8, 10)

        # Check group importances are present in hierarchical mode
        assert ci_outputs.group_importances is not None
        assert ci_outputs.group_importances.shape == (8, 4)
        assert not torch.isnan(ci_outputs.group_importances).any()

    def test_hierarchical_training_step(self, tiny_tms_model, tiny_dataloader):
        """Test that hierarchical SPD can run a few training steps without errors."""
        comp_model = ComponentModel(
            target_model=tiny_tms_model,
            target_module_patterns=["linear1", "linear2"],
            C=10,
            ci_fn_type="hierarchical",
            ci_fn_hidden_dims=[8],
            sigmoid_type="leaky_hard",
            pretrained_model_output_attr=None,
            num_groups=4,
            router_hidden_dim=16,
            within_group_hidden_dim=8,
        )

        # Simple optimizer
        optimizer = torch.optim.Adam(comp_model.parameters(), lr=1e-3)

        # Run a few training steps
        comp_model.train()
        batch, _ = next(iter(tiny_dataloader))  # DatasetGeneratedDataLoader returns (batch, labels)

        for _ in range(3):
            # Forward pass to get activations
            output_with_cache = comp_model(batch, cache_type="input")

            # Compute causal importances
            ci_outputs = comp_model.calc_causal_importances(
                pre_weight_acts=output_with_cache.cache, sampling="continuous"
            )

            # Compute weight deltas
            weight_deltas = comp_model.calc_weight_deltas()

            # Compute loss (simplified - just use a few loss terms)
            from spd.configs import (
                ImportanceMinimalityLossConfig,
                StochasticReconLossConfig,
            )

            loss_configs = [
                ImportanceMinimalityLossConfig(coeff=1e-3, pnorm=1.0),
                StochasticReconLossConfig(coeff=1.0),
            ]

            total_loss, _ = compute_total_loss(
                loss_metric_configs=loss_configs,
                model=comp_model,
                batch=batch,
                ci=ci_outputs,
                target_out=output_with_cache.output,
                weight_deltas=weight_deltas,
                pre_weight_acts=output_with_cache.cache,
                current_frac_of_training=0.0,
                sampling="continuous",
                use_delta_component=True,
                n_mask_samples=1,
                output_loss_type="mse",
            )

            # Check loss is valid
            assert not torch.isnan(total_loss)
            assert not torch.isinf(total_loss)
            assert total_loss.item() > 0

            # Backward and step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    def test_group_sparsity_loss_integration(self, tiny_tms_model, tiny_dataloader):
        """Test that GroupSparsityLoss works correctly in hierarchical mode."""
        comp_model = ComponentModel(
            target_model=tiny_tms_model,
            target_module_patterns=["linear1", "linear2"],
            C=10,
            ci_fn_type="hierarchical",
            ci_fn_hidden_dims=[8],
            sigmoid_type="leaky_hard",
            pretrained_model_output_attr=None,
            num_groups=4,
            router_hidden_dim=16,
            within_group_hidden_dim=8,
        )

        batch, _ = next(iter(tiny_dataloader))  # DatasetGeneratedDataLoader returns (batch, labels)

        # Forward pass
        output_with_cache = comp_model(batch, cache_type="input")

        # Compute causal importances
        ci_outputs = comp_model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache, sampling="continuous"
        )

        # Compute weight deltas
        weight_deltas = comp_model.calc_weight_deltas()

        # Include GroupSparsityLoss
        from spd.configs import GroupSparsityLossConfig, StochasticReconLossConfig

        loss_configs = [
            StochasticReconLossConfig(coeff=1.0),
            GroupSparsityLossConfig(coeff=1e-4, pnorm=1.0),
        ]

        total_loss, loss_terms = compute_total_loss(
            loss_metric_configs=loss_configs,
            model=comp_model,
            batch=batch,
            ci=ci_outputs,
            target_out=output_with_cache.output,
            weight_deltas=weight_deltas,
            pre_weight_acts=output_with_cache.cache,
            current_frac_of_training=0.0,
            sampling="continuous",
            use_delta_component=True,
            n_mask_samples=1,
            output_loss_type="mse",
        )

        # Check that GroupSparsityLoss was computed
        assert "loss/GroupSparsityLoss" in loss_terms
        assert loss_terms["loss/GroupSparsityLoss"] >= 0
        assert not torch.isnan(total_loss)

    def test_hierarchical_vs_standard_output_compatibility(self, tiny_tms_model):
        """Test that hierarchical and standard SPD have compatible output formats."""
        # Standard SPD
        standard_model = ComponentModel(
            target_model=tiny_tms_model,
            target_module_patterns=["linear1", "linear2"],
            C=10,
            ci_fn_type="mlp",
            ci_fn_hidden_dims=[8],
            sigmoid_type="leaky_hard",
            pretrained_model_output_attr=None,
        )

        # Hierarchical SPD
        hierarchical_model = ComponentModel(
            target_model=tiny_tms_model,
            target_module_patterns=["linear1", "linear2"],
            C=10,
            ci_fn_type="hierarchical",
            ci_fn_hidden_dims=[8],
            sigmoid_type="leaky_hard",
            pretrained_model_output_attr=None,
            num_groups=4,
            router_hidden_dim=16,
            within_group_hidden_dim=8,
        )

        test_input = torch.randn(8, 5)

        # Get CI outputs from both
        standard_output = standard_model(test_input, cache_type="input")
        hierarchical_output = hierarchical_model(test_input, cache_type="input")

        standard_ci = standard_model.calc_causal_importances(
            pre_weight_acts=standard_output.cache, sampling="continuous"
        )
        hierarchical_ci = hierarchical_model.calc_causal_importances(
            pre_weight_acts=hierarchical_output.cache, sampling="continuous"
        )

        # Both should have same module keys
        assert set(standard_ci.lower_leaky.keys()) == set(hierarchical_ci.lower_leaky.keys())

        # Both should have same shapes for per-module importance
        for key in standard_ci.lower_leaky.keys():
            assert standard_ci.lower_leaky[key].shape == hierarchical_ci.lower_leaky[key].shape

        # Only hierarchical should have group importances
        assert standard_ci.group_importances is None
        assert hierarchical_ci.group_importances is not None
