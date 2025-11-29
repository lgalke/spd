# Hierarchical SPD Implementation Status

**Version**: 1.0
**Date**: 2025-01-29
**Status**: ✅ **COMPLETE - READY FOR USE**

## Overview

Hierarchical Stochastic Parameter Decomposition (H-SPD) has been successfully implemented as an optional enhancement to the SPD framework. The implementation is **fully backward compatible** with the original SPD behavior and all existing tests pass.

## Implementation Status

### ✅ Completed Components

#### 1. Core Hierarchical Module (`spd/hierarchical/`)

All core components from the plan (Section 3.1) have been implemented:

- **`GroupAssignment`** (`group_assignment.py`)
  - Manages subcomponent-to-group mappings
  - Supports 3 initialization strategies: `random`, `layer_based`, `kmeans`
  - Includes serialization (`state_dict()` / `from_state_dict()`)
  - **Note**: Dynamic reassignment method exists but is not called during training (deferred to future work)

- **`GroupRouter`** (`router.py`)
  - Computes group-level importance: `g_group(x)`
  - Supports 3 aggregation strategies: `per_layer_mean`, `global_mean`, `per_layer_max`
  - Uses leaky hard sigmoid for gradient flow
  - Hidden dimension: configurable (default 64)

- **`WithinGroupImportance`** (`within_group.py`)
  - Per-group MLPs for fine-grained importance: `g_within(l, c, x | g)`
  - Handles empty groups gracefully
  - Hidden dimension: configurable (default 16)

- **`HierarchicalImportanceFunction`** (`importance.py`)
  - Combines router and within-group networks
  - Computes factorized importance: `g^l_c(x) = g_group(g, x) × g_within(l, c, x | g)`
  - Returns both per-subcomponent and per-group importances
  - Includes `get_importance_for_loss()` for upper-leaky variants

#### 2. Configuration System (`spd/configs.py`)

Extended `Config` class with hierarchical parameters (lines 265-293):

```python
# Hierarchical SPD parameters (only used if ci_fn_type == "hierarchical")
num_groups: PositiveInt | None = None  # Required for hierarchical mode
router_hidden_dim: PositiveInt = 64
within_group_hidden_dim: PositiveInt = 16
router_aggregation: Literal["per_layer_mean", "global_mean", "per_layer_max"] = "per_layer_mean"
group_init_strategy: Literal["random", "layer_based", "kmeans"] = "random"
group_sparsity_coeff: NonNegativeFloat = 0.0
group_sparsity_pnorm: float = 1.0
```

Added `GroupSparsityLossConfig` (lines 46-48):
```python
class GroupSparsityLossConfig(LossMetricConfig):
    classname: Literal["GroupSparsityLoss"] = "GroupSparsityLoss"
    pnorm: float = 1.0
```

Validator ensures `num_groups` is set when `ci_fn_type == "hierarchical"` (lines 547-551).

#### 3. ComponentModel Integration (`spd/models/component_model.py`)

- **Extended `__init__`** (lines 104-191)
  - Accepts hierarchical parameters
  - Detects hierarchical mode via `ci_fn_type == "hierarchical"`
  - Creates `HierarchicalImportanceFunction` instead of per-module CI functions
  - Maintains backward compatibility: empty `ci_fns` dict in hierarchical mode

- **Extended `CIOutputs`** (lines 79-84)
  ```python
  @dataclass
  class CIOutputs:
      lower_leaky: dict[str, Float[Tensor, "... C"]]
      upper_leaky: dict[str, Float[Tensor, "... C"]]
      pre_sigmoid: dict[str, Tensor]
      group_importances: Float[Tensor, "batch num_groups"] | None = None  # For hierarchical
  ```

- **New method: `_calc_hierarchical_causal_importances()`** (lines 639-706)
  - Computes inner activations for all modules
  - Calls `HierarchicalImportanceFunction.forward()`
  - Converts layer-indexed dict to module-name-indexed dict
  - Returns `CIOutputs` with `group_importances` populated

- **Modified `calc_causal_importances()`** (lines 573-637)
  - Dispatches to `_calc_hierarchical_causal_importances()` when in hierarchical mode
  - Original behavior unchanged for non-hierarchical modes

- **Updated `from_run_info()`** (lines 520-561)
  - Passes hierarchical parameters to `ComponentModel.__init__()`

#### 4. Training Integration (`spd/run_spd.py`)

- **Updated ComponentModel instantiation** (lines 148-162)
  - Passes all hierarchical parameters from config

#### 5. Loss System

- **New metric function** (`spd/metrics/group_sparsity_loss.py`)
  ```python
  def group_sparsity_loss(
      group_importances: Float[Tensor, "batch num_groups"],
      pnorm: float = 1.0,
  ) -> Float[Tensor, ""]:
      """Encourage sparse group activation via p-norm."""
      loss = (group_importances.abs() ** pnorm).sum(dim=1).mean()
      return loss
  ```

- **Updated `compute_total_loss()`** (`spd/losses.py`, lines 78-86)
  - Added case for `GroupSparsityLossConfig`
  - Validates that `ci.group_importances` is not None
  - Computes group sparsity loss using configurable p-norm

- **Exported from metrics** (`spd/metrics/__init__.py`, line 17)

#### 6. Type System

- **Extended `CiFnType`** (`spd/models/components.py`, line 12)
  ```python
  CiFnType = Literal["mlp", "vector_mlp", "shared_mlp", "hierarchical"]
  ```

#### 7. Tests (`tests/test_hierarchical.py`)

Comprehensive test suite with **11 tests, all passing**:

- `TestGroupAssignment` (4 tests)
  - Initialization strategies (random, layer_based)
  - Bidirectional indexing
  - State dict serialization

- `TestGroupRouter` (2 tests)
  - Forward pass shapes and bounds
  - Different aggregation strategies

- `TestWithinGroupImportance` (2 tests)
  - Forward pass shapes
  - Empty group handling

- `TestHierarchicalImportanceFunction` (3 tests)
  - End-to-end functionality
  - Factorization constraints
  - Loss variant (`get_importance_for_loss()`)

### 🔄 Not Implemented (Deferred to Future Work)

These items from the original plan are **intentionally deferred**:

1. **Dynamic Group Reassignment During Training** (Plan Section 4.5-4.6)
   - `ImportanceHistoryTracker` not created
   - `reassign_from_coactivation()` exists but is not called
   - `_reassign_groups()` training loop logic not implemented
   - **Rationale**: Reduces implementation complexity, can be added later

2. **Advanced Loss Terms**
   - `L_group-balance` loss (Plan Section 2.7c)
   - **Rationale**: Not critical for initial functionality

3. **Validation Experiments** (Plan Section 5.4)
   - TMS5-2, TMS40-10 experiments with hierarchical mode
   - **Rationale**: Requires access to experiment infrastructure

4. **Future Extensions** (Plan Section 8)
   - Soft group assignments
   - Multi-level hierarchies
   - Learned reassignment

## Usage Guide

### Basic Example

```yaml
# In your experiment config YAML file

# Enable hierarchical mode
ci_fn_type: "hierarchical"
num_groups: 32  # Required when ci_fn_type is "hierarchical"

# Optional: Configure hierarchical parameters (defaults shown)
router_hidden_dim: 64
within_group_hidden_dim: 16
router_aggregation: "per_layer_mean"  # or "global_mean", "per_layer_max"
group_init_strategy: "random"  # or "layer_based", "kmeans"

# Optional: Add group sparsity loss
loss_metric_configs:
  - classname: "ImportanceMinimalityLoss"
    coeff: 1e-5
    pnorm: 2.0
    # ... other params

  - classname: "GroupSparsityLoss"  # NEW: Hierarchical loss
    coeff: 1e-4
    pnorm: 1.0
```

### Recommended Hyperparameters (from Plan Section 6.2)

**For TMS5-2:**
```yaml
num_groups: 8
router_hidden_dim: 32
within_group_hidden_dim: 8
group_sparsity_coeff: 1e-4  # Or use loss_metric_configs
```

**For TMS40-10:**
```yaml
num_groups: 50
router_hidden_dim: 64
within_group_hidden_dim: 16
group_sparsity_coeff: 1e-5
```

**For larger models:**
```yaml
num_groups: 120
router_hidden_dim: 128
within_group_hidden_dim: 32
group_sparsity_coeff: 1e-5
```

## Backward Compatibility

✅ **FULLY BACKWARD COMPATIBLE**

All existing tests pass without modification:
- `test_component_model.py`: 17/17 passed
- `test_spd_losses.py`: 28/28 passed
- `test_hierarchical.py`: 11/11 passed (new)

The original SPD behavior is preserved when `ci_fn_type` is not set to `"hierarchical"`.

## Architecture Details

### Key Design Decisions

1. **Additive, not replacement**: Hierarchical mode is a new option alongside existing CI function types
2. **Config-driven**: All parameters have sensible defaults
3. **Modular**: Hierarchical components are in separate module (`spd/hierarchical/`)
4. **Factorized importance**: `g_combined = g_group × g_within` enables both coarse and fine-grained control

### Data Flow in Hierarchical Mode

```
Input Data
    ↓
Target Model (frozen)
    ↓
Pre-weight Activations (cached)
    ↓
Components.get_inner_acts() → Inner Activations (per layer)
    ↓
HierarchicalImportanceFunction:
    1. GroupRouter(inner_acts) → group_importances (batch, G)
    2. For each group g:
        - Gather inner_acts for subcomponents in g
        - WithinGroupImportance(group_acts) → within_importances
        - Multiply: importance = group_importances[:, g] × within_importances
    ↓
CIOutputs with:
    - lower_leaky/upper_leaky (per module, per subcomponent)
    - group_importances (batch, num_groups)
    ↓
Loss computation:
    - Standard losses use lower_leaky/upper_leaky
    - GroupSparsityLoss uses group_importances
```

### File Structure

```
spd/
├── hierarchical/              # NEW MODULE
│   ├── __init__.py
│   ├── group_assignment.py    # GroupAssignment class
│   ├── router.py              # GroupRouter MLP
│   ├── within_group.py        # WithinGroupImportance MLPs
│   └── importance.py          # HierarchicalImportanceFunction
│
├── models/
│   ├── components.py          # MODIFIED: Added "hierarchical" to CiFnType
│   └── component_model.py     # MODIFIED: Hierarchical mode support
│
├── metrics/
│   ├── __init__.py            # MODIFIED: Export group_sparsity_loss
│   └── group_sparsity_loss.py # NEW: Group sparsity metric
│
├── configs.py                 # MODIFIED: Added hierarchical params & GroupSparsityLossConfig
├── losses.py                  # MODIFIED: Added GroupSparsityLoss case
├── run_spd.py                 # MODIFIED: Pass hierarchical params
│
└── tests/
    └── test_hierarchical.py   # NEW: 11 tests for hierarchical components
```

## Known Limitations

1. **No dynamic reassignment**: Groups are fixed after initialization
2. **Single-level hierarchy**: No nested groups (groups of groups)
3. **Hard group assignments**: Each subcomponent belongs to exactly one group
4. **No group-specific components**: All components still use V/U decomposition (groups only affect importance)

## Future Work Roadmap

### Priority 1: Dynamic Reassignment
- Implement `ImportanceHistoryTracker`
- Add reassignment logic to training loop
- Make reassignment optional via config flag

### Priority 2: Advanced Features
- Soft/probabilistic group assignments
- Multi-level hierarchies
- Group-specific loss terms (balance, size constraints)

### Priority 3: Validation
- Run TMS experiments with hierarchical mode
- Compare parameter efficiency vs standard SPD
- Benchmark training time and memory usage

## Integration with Existing Code

### Where ComponentModel is Created

All locations have been updated to pass hierarchical parameters:

1. **`spd/run_spd.py`** (lines 148-162)
   - Main training entry point

2. **`spd/models/component_model.py`** in `from_run_info()` (lines 547-561)
   - Loading checkpoints

### Where to Add New Loss Terms

To add a new hierarchical-specific loss:

1. Create metric function in `spd/metrics/your_loss.py`
2. Add config class in `spd/configs.py` inheriting from `LossMetricConfig`
3. Add to `LossMetricConfigType` union
4. Add case in `spd/losses.py::compute_total_loss()`
5. Export from `spd/metrics/__init__.py`

## Testing

Run hierarchical tests:
```bash
source .venv/bin/activate
python -m pytest tests/test_hierarchical.py -v
```

Run all tests (including backward compatibility):
```bash
python -m pytest tests/test_component_model.py tests/test_spd_losses.py tests/test_hierarchical.py -v
```

## Questions & Support

Refer to:
- Original plan: `hierarchical-spd-plan.txt`
- SPD paper: `papers/Stochastic_Parameter_Decomposition/spd_paper.md`
- APD paper (conceptual): `papers/Attribution_based_Parameter_Decomposition/apd_paper.md`

For technical details on specific components, see docstrings in the source code.

---

**Status**: Ready for experimental use. Backward compatibility verified. Future enhancements can be added incrementally.
