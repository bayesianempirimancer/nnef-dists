# Debug Scripts Directory

This directory contains focused debugging scripts to test specific components and resolve issues systematically.

## Available Debug Scripts

### `test_jacobian_theory.py`
**Purpose:** Debug the theoretical relationship between network Jacobian and covariance in exponential families.

**Issue Being Debugged:** 
- Original implementation incorrectly used `J @ J^T` for covariance estimation
- For exponential families: `Cov[T(X)] = ∇_η E[T(X)]` (Fisher Information Matrix)
- Need to resolve dimensional mismatch: Jacobian `[batch,9,12]` vs Covariance `[batch,9,9]`

**Usage:**
```bash
python scripts/debug/test_jacobian_theory.py
```

**Output:**
- Jacobian analysis and structure visualization
- Comparison of different covariance estimation methods
- Theoretical recommendations for fixing the geometric loss

### `test_depth_vs_width.py`
**Purpose:** Systematically test the depth vs width hypothesis with controlled parameter counts.

**Hypothesis:** Deep narrow networks outperform wide shallow networks for natural parameter mapping.

**Method:**
- Creates architectures with similar parameter counts (~50k)
- Varies depth/width ratios: 20×32, 12×64, 8×80, 4×128, 2×256
- Measures performance on identical datasets

**Usage:**
```bash
python scripts/debug/test_depth_vs_width.py
```

**Output:**
- Performance ranking by architecture
- Correlation analysis (depth/width vs performance)
- Visualization of architecture space
- Statistical hypothesis test results

### `test_individual_models.py`
**Purpose:** Quick verification that all standardized models work with the new framework.

**Tests:**
- Model initialization
- Forward pass
- Single training step
- Evaluation metrics
- Configuration compatibility

**Usage:**
```bash
python scripts/debug/test_individual_models.py
```

**Output:**
- Success/failure status for each model
- Parameter counts and basic metrics
- Error messages for failed models
- Integration test results

### `test_config_system.py`
**Purpose:** Comprehensive testing of the configuration system.

**Tests:**
- Basic config class creation
- Serialization/deserialization (dict and JSON)
- Predefined configurations
- Configuration modifications
- Model integration

**Usage:**
```bash
python scripts/debug/test_config_system.py
```

**Output:**
- Configuration system validation
- Available configs listing
- Serialization test results
- Integration verification

## Debug Workflow

1. **Start with config system:** `test_config_system.py`
2. **Verify individual models:** `test_individual_models.py`
3. **Test core hypothesis:** `test_depth_vs_width.py`
4. **Debug theoretical issues:** `test_jacobian_theory.py`

## Output Structure

All debug scripts save results to `artifacts/debug_*/`:

```
artifacts/
├── debug_config/
│   └── config_test_summary.txt
├── debug_individual_models/
│   └── individual_model_tests.json
├── debug_depth_width/
│   ├── depth_vs_width_analysis.png
│   └── depth_width_debug.json
└── debug_jacobian/
    ├── jacobian_covariance_debug.png
    └── debug_results.json
```

## Benefits of Systematic Debugging

1. **Isolated Testing** - Each script tests one specific component
2. **Fast Iteration** - Small datasets and short training for quick feedback
3. **Clear Documentation** - Each issue is clearly defined and tracked
4. **Reproducible Results** - Consistent setup and output format
5. **Visual Analysis** - Plots help understand complex relationships

## Next Steps

After debugging:
1. Fix identified issues in the main codebase
2. Re-run debug scripts to verify fixes
3. Proceed with comprehensive experiments
4. Use debug insights to optimize architectures
