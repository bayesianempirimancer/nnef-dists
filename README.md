# nnef-dists

Fast variational inference with non-named exponential family distributions using neural networks to learn the mapping between natural parameters and expected sufficient statistics. Features both direct neural approximation and novel geometric flow-based approaches.

## Overview

For an exponential family with natural parameters `eta` and sufficient statistics `T(x)`, inference by message passing requires the mapping `mu_T(eta) = E[T(x) | eta]`. Named exponential-family distributions have closed-form `mu_T`. This project learns `mu_T` for non-named families using two complementary approaches:

1. **Direct Neural Approximation**: Traditional neural networks trained on MCMC samples
2. **Geometric Flow Networks**: Novel flow-based approach using continuous dynamics that respects the geometric structure of exponential families

This repository uses JAX/Flax for modeling and BlackJAX for HMC sampling. It supports both 1D and multidimensional exponential family distributions with comprehensive training and visualization capabilities.

## Quickstart

1. Install dependencies:

```bash
pip install -e .
```

2. Generate training data for a 1D Gaussian model:

```bash
python src/generate_data.py --config configs/gaussian_1d_large.yaml
```

3. Train neural networks (multiple approaches available):

```bash
# Traditional ET Network (direct approximation)
python scripts/training/train_mlp_ET.py

# LogZ Network (gradient-based)
python scripts/training/train_mlp_logZ.py

# Geometric Flow Network (novel flow-based approach)
python scripts/training/train_geometric_flow_ET.py --save-dir artifacts/geometric_flow
```

4. For 3D multivariate Gaussian models:

```bash
# Generate data
python src/generate_data.py --config configs/multivariate_3d_large.yaml

# Train different model types
python scripts/training/train_mlp_ET.py
python scripts/training/train_geometric_flow_ET.py --save-dir artifacts/geometric_flow_3d
```

5. Generate plots only (if training history exists):

```bash
python scripts/train_large_1d.py --plot-only
python scripts/train_large_3d.py --plot-only
```

Artifacts (learned parameters, training history, and plots) will be saved under `artifacts/`.

## Project layout

- `src/`: Core library
  - `ef.py`: Generic `ExponentialFamily` interface with `GaussianNatural1D`, `MultivariateNormal` implementations, EF factory, and analytical reference point methods
  - `sampling.py`: BlackJAX HMC sampling utilities for arbitrary-shaped distributions
  - `models/`: Neural network architectures
    - `ET_Net.py`: Direct expectation networks (MLP, GLU, Quadratic ResNet, etc.)
    - `logZ_Net.py`: Log-normalizer networks with gradient/Hessian computation
    - `geometric_flow_net.py`: **NEW** Geometric flow networks using continuous dynamics
    - `glow_net_ET.py`: Glow networks using normalizing flows with affine coupling
  - `generate_data.py`: Data generation using HMC sampling with configurable parameters
- `scripts/`: Training and visualization scripts
  - `training/`: Training scripts for different model architectures
    - `train_mlp_ET.py`: Standard MLP ET networks
    - `train_geometric_flow_ET.py`: **NEW** Geometric flow networks
    - `train_*_logZ.py`: Various LogZ network architectures
  - `test_flow_math_fixed.py`: Mathematical verification of flow dynamics
  - `demo_flow_scaling.py`: Demonstration of flow efficiency with coarse discretization
- `src/utils/`: Utility modules
  - `performance.py`: Performance measurement utilities
  - `matrix_utils.py`: JAX utilities for matrix operations
  - `data_utils.py`: Data loading and preprocessing utilities
  - `exact_covariance.py`: Analytical covariance computation for known exponential families
- `configs/`: Configuration files
  - `gaussian_1d_large.yaml`: Large-scale 1D Gaussian configuration
  - `multivariate_3d_large.yaml`: Large-scale 3D multivariate Gaussian configuration
- `artifacts/`: Saved models, training history, and plots (organized by model type)
- `data/`: Generated training datasets (pickle files) - all training data stored here for reuse

## Supported Distributions

### 1D Gaussian Natural Parameters
- Natural parameters: `eta = [eta1, eta2]` where `eta2 < 0` for integrability
- Sufficient statistics: `T(x) = [x, x^2]`
- Configuration: `gaussian_1d_large.yaml`

### 3D Multivariate Normal
- Natural parameters: `eta = [eta1_0, eta1_1, eta1_2, eta2_00, eta2_01, ..., eta2_22]` (12 dimensions)
- Sufficient statistics: `T(x) = [x_0, x_1, x_2, x_0^2, x_0*x_1, x_0*x_2, x_1^2, x_1*x_2, x_2^2]` (9 dimensions)
- Configuration: `multivariate_3d_large.yaml`

## Neural Network Approaches

### Traditional Approaches
- **ET Networks**: Directly learn `μ_T(η) = E[T(x)|η]` using standard neural architectures
- **LogZ Networks**: Learn log-normalizer `A(η)`, then compute `μ_T = ∇A(η)` via automatic differentiation

### Geometric Flow Networks (Novel)
A breakthrough approach that learns flow dynamics to compute expectations:

```
du/dt = A(u,t,η_t) @ A(u,t,η_t)^T @ (η_target - η_init)
```

where:
- `u(t)` evolves from analytical reference point `μ_0` to target `μ_T(η_target)`
- `η_t = (1-t)η_init + t*η_target` (linear interpolation in parameter space)
- `A(u,t,η_t)` is learned via neural networks with sinusoidal time embeddings
- `A@A^T` structure ensures positive semidefinite flow matrices

**Key advantages**:
- Respects geometric structure of exponential families
- Uses analytical reference points via `find_nearest_analytical_point()`
- Requires minimal time steps (2-5) due to smooth dynamics
- Includes smoothness penalties for stable training

## Defining new exponential-family distributions

- Implement the `ExponentialFamily` interface in `src/ef.py`: define immutable `x_shape`, `eta_dim`, and the methods `compute_stats(x)`, `logdensity_fn(eta)`, and `flatten_stats_or_eta(dict)`.
- Add your EF to `ef_factory` in `src/ef.py` or pass an instance programmatically.
- For geometric flow networks, implement `find_nearest_analytical_point()` to provide analytical reference points.
- The system automatically handles flattening for HMC sampling and respects your shapes when computing moments.

## Configuration System

The project uses YAML configuration files to specify:
- **Exponential Family**: Type and parameters
- **Grid**: Number of training/validation points and parameter ranges
- **Sampling**: HMC parameters (samples, warmup, step size, integration steps)
- **Optimization**: Neural network architecture, learning rate, epochs, batch size

## Mathematical Verification

The geometric flow approach includes mathematical verification scripts:

```bash
# Verify flow dynamics on 3D multivariate Gaussian with exact covariance
python scripts/test_flow_math_fixed.py

# Demonstrate flow efficiency with coarse time discretization  
python scripts/demo_flow_scaling.py

# Test nearest analytical point computation
python scripts/test_nearest_analytical.py
```

## Visualization

The project includes comprehensive plotting capabilities:
- **1D Results**: Training curves, moment comparisons, error distributions
- **3D Results**: Multi-panel analysis including linear/quadratic term comparisons, covariance heatmaps, and component-wise MSE breakdowns
- **Flow Networks**: Flow trajectories, time embeddings, convergence analysis, and geometric flow visualizations

## Notes

- All implementations use JAX for efficient computation and automatic differentiation
- Training data is cached using configuration hashes to avoid regeneration
- The system supports both CPU and GPU execution via JAX backends
- Matrix operations use optimized JAX routines with `vmap` compatibility for batch processing


