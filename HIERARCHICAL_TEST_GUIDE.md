# Hierarchical SPD Validation Tests

This guide explains how to validate that the hierarchical SPD implementation works correctly with the TMS toy examples.

## Overview

We've created two types of tests to validate hierarchical SPD:

1. **Automated Integration Tests** - Fast unit/integration tests that verify basic functionality
2. **Full Experiment Config** - A complete 1000-step training run to validate end-to-end behavior

## 1. Automated Integration Tests

These tests verify that hierarchical SPD works correctly in isolation with programmatically created TMS models.

### Running the Tests

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run all hierarchical integration tests
python -m pytest tests/test_hierarchical_integration.py -v --runslow

# Run a specific test
python -m pytest tests/test_hierarchical_integration.py::TestHierarchicalIntegration::test_hierarchical_training_step -v --runslow
```

### What's Tested

The integration tests ([test_hierarchical_integration.py](tests/test_hierarchical_integration.py)) verify:

1. **Basic Forward Pass** - Hierarchical ComponentModel can be created and run forward passes
2. **CI Computation** - Hierarchical causal importance computation produces correct outputs with group importances
3. **Training Steps** - Full training loop with hierarchical SPD runs without errors
4. **GroupSparsityLoss** - The new group sparsity loss integrates correctly with the loss system
5. **Output Compatibility** - Hierarchical and standard SPD produce compatible output formats

## 2. Full Experiment Config

A lightweight config for end-to-end validation with actual WandB logging and full training infrastructure.

### Config Details

**File:** [spd/experiments/tms/tms_5-2_hierarchical_test_config.yaml](spd/experiments/tms/tms_5-2_hierarchical_test_config.yaml)

**Key Parameters:**
- `ci_fn_type: "hierarchical"` - Enable hierarchical mode
- `num_groups: 8` - Number of component groups
- `router_hidden_dim: 32` - Router MLP hidden dimension
- `within_group_hidden_dim: 8` - Within-group MLP hidden dimension
- `steps: 1000` - Reduced from 10,000 for faster testing
- Includes `GroupSparsityLoss` in loss configuration

### Running via Registry

The config is registered in [spd/registry.py](spd/registry.py#L49-L55) for convenient execution.

**First, install the package if you haven't already:**

```bash
source .venv/bin/activate
pip install -e .
```

**Then run the experiment:**

```bash
# Run locally with spd-local (GPU by default)
spd-local tms_5-2_hierarchical_test

# Run locally on CPU
spd-local tms_5-2_hierarchical_test --cpu

# Or via spd-run (submits to SLURM)
spd-run --experiments tms_5-2_hierarchical_test
```

### Running Directly

You can also run the experiment script directly using Python fire CLI:

```bash
# Activate environment
source .venv/bin/activate

# Run the decomposition
python spd/experiments/tms/tms_decomposition.py \
  --config_path spd/experiments/tms/tms_5-2_hierarchical_test_config.yaml
```

### What to Check

After running the experiment, verify:

1. **Training completes successfully** - No errors during 1000 steps
2. **Group importances are logged** - Check WandB for group sparsity metrics
3. **Loss values are reasonable** - Compare to standard TMS runs
4. **Components are learned** - Verify that CI values show sparse activation patterns

## Expected Behavior

### Integration Tests

All 5 tests should **PASS**:
- ✅ test_hierarchical_spd_basic_forward
- ✅ test_hierarchical_ci_computation
- ✅ test_hierarchical_training_step
- ✅ test_group_sparsity_loss_integration
- ✅ test_hierarchical_vs_standard_output_compatibility

### Full Experiment

The experiment should:
- Complete 1000 training steps without errors
- Log standard SPD metrics (CI-L0, reconstruction loss, etc.)
- Additionally log group sparsity metrics
- Produce interpretable component decompositions

## Comparison with Standard SPD

To compare hierarchical vs standard SPD on the same toy model:

```bash
# Run standard SPD
spd-local tms_5-2

# Run hierarchical SPD
spd-local tms_5-2_hierarchical_test

# Compare in WandB
```

## Troubleshooting

### Tests are Skipped

If tests show as "skipped", you need to add the `--runslow` flag:

```bash
python -m pytest tests/test_hierarchical_integration.py -v --runslow
```

### Import Errors

Make sure you're in the activated virtual environment:

```bash
source .venv/bin/activate
```

And that the package is installed in editable mode:

```bash
pip install -e .
```

### WandB Authentication

For full experiments, ensure you have WandB configured:

```bash
wandb login
```

## Next Steps

After validation:

1. Compare hierarchical vs standard SPD performance on TMS
2. Test on larger models (tms_40-10)
3. Experiment with different numbers of groups
4. Try different group initialization strategies

## Files Reference

- **Integration Tests:** [tests/test_hierarchical_integration.py](tests/test_hierarchical_integration.py)
- **Test Config:** [spd/experiments/tms/tms_5-2_hierarchical_test_config.yaml](spd/experiments/tms/tms_5-2_hierarchical_test_config.yaml)
- **Registry Entry:** [spd/registry.py](spd/registry.py#L49-L55)
- **Hierarchical Implementation:** [spd/hierarchical/](spd/hierarchical/)
- **Implementation Status:** [hierarchical-spd-implementation-status.md](hierarchical-spd-implementation-status.md)
