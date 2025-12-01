# Quick Validation Guide for Hierarchical SPD

This is a condensed guide showing the exact commands to validate hierarchical SPD.

## Prerequisites

```bash
# Activate the virtual environment
source .venv/bin/activate
```

**Note:** The package should already be installed in your environment. You can verify by running:

```bash
spd-local --help
```

If the command is not found, the package may need to be installed. See the Troubleshooting section below.

## Option 1: Automated Tests (< 1 minute)

Run the integration tests that verify hierarchical SPD works correctly:

```bash
python -m pytest tests/test_hierarchical_integration.py -v --runslow
```

**Expected:** All 5 tests should PASS ✅

## Option 2: Full Experiment (~ 2-5 minutes)

Run a 1000-step training experiment with hierarchical SPD:

### Using spd-local (recommended)

```bash
# GPU (default)
spd-local tms_5-2_hierarchical_test

# CPU
spd-local tms_5-2_hierarchical_test --cpu
```

### Using Python directly

```bash
python spd/experiments/tms/tms_decomposition.py \
  --config_path spd/experiments/tms/tms_5-2_hierarchical_test_config.yaml
```

## What to Expect

### Integration Tests
- 5 tests covering basic functionality, CI computation, training steps, loss integration, and compatibility
- All should pass without errors
- Takes < 1 minute

### Full Experiment
- 1000 training steps (vs 10,000 in standard config)
- Hierarchical mode enabled with 8 groups
- Logs to WandB
- Should complete without errors
- Takes 2-5 minutes depending on hardware

## Verify Results

Check that:
1. Training completes successfully
2. Group sparsity metrics appear in WandB logs
3. Loss values are reasonable (similar to standard SPD)
4. Components show sparse activation patterns

## Troubleshooting

**Tests are skipped:** Add `--runslow` flag to pytest command

**spd-local command not found:**

- First verify your venv is activated: `source .venv/bin/activate`
- Check if the command exists: `ls .venv/bin/spd-*`
- If missing, you may need to reinstall. However, if you have conda interfering with pip, use: `make install-dev` instead

**Python version error when installing:**

If you get "Package 'spd' requires a different Python: 3.12.9 not in '==3.13.*'", this means conda's pip is interfering. The package is likely already installed. Just skip the install step and run the commands directly.

**Import errors:** Make sure you're in the activated venv (`source .venv/bin/activate`)

## Next Steps

After successful validation, see [HIERARCHICAL_TEST_GUIDE.md](HIERARCHICAL_TEST_GUIDE.md) for:
- Detailed explanation of what's tested
- How to compare with standard SPD
- Configuration options
- Full troubleshooting guide
