# nnef-dists

Fast variational inference with non-named exponential family distributions using neural networks to learn the mapping between natural parameters and expected sufficient statistics.

## Overview

For an exponential family with natural parameters `eta` and sufficient statistics `T(x)`, inference by message passing requires the mapping `mu_T(eta) = E[T(x) | eta]`. Named exponential-family distributions have closed-form `mu_T`. This project learns `mu_T` for non-named families by training a neural network on MCMC samples generated for many `eta` values.

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

3. Train the neural moment network:

```bash
python scripts/train_large_1d.py
```

4. For 3D multivariate Gaussian models:

```bash
# Generate data
python src/generate_data.py --config configs/multivariate_3d_large.yaml

# Train the model
python scripts/train_large_3d.py
```

5. Generate plots only (if training history exists):

```bash
python scripts/train_large_1d.py --plot-only
python scripts/train_large_3d.py --plot-only
```

Artifacts (learned parameters, training history, and plots) will be saved under `artifacts/`.

## Project layout

- `src/`: Core library
  - `ef.py`: Generic `ExponentialFamily` interface, `GaussianNatural1D`, and `MultivariateNormal` implementations with EF factory
  - `sampling.py`: BlackJAX HMC sampling utilities for arbitrary-shaped distributions
  - `model.py`: Flax neural network models for moment mapping (eta -> E[T(x)])
  - `train.py`: Training loop and optimization utilities
  - `generate_data.py`: Data generation using HMC sampling with configurable parameters
  - `data_utils.py`: Data loading and preprocessing utilities
- `scripts/`: Training and visualization scripts
  - `train_large_1d.py`: Training script for 1D Gaussian models
  - `train_large_3d.py`: Training script for 3D multivariate Gaussian models
  - `plot_results.py`: Plotting utilities for 1D results
  - `plot_3d_results.py`: Specialized plotting for 3D multivariate results
  - `matrix_utils.py`: JAX utilities for matrix operations
  - `test_training.py`: Test script for small-scale training
- `configs/`: Configuration files
  - `gaussian_1d_large.yaml`: Large-scale 1D Gaussian configuration
  - `multivariate_3d_large.yaml`: Large-scale 3D multivariate Gaussian configuration
- `artifacts/`: Saved models, training history, and plots
- `data/`: Generated training datasets (pickle files)

## Supported Distributions

### 1D Gaussian Natural Parameters
- Natural parameters: `eta = [eta1, eta2]` where `eta2 < 0` for integrability
- Sufficient statistics: `T(x) = [x, x^2]`
- Configuration: `gaussian_1d_large.yaml`

### 3D Multivariate Normal
- Natural parameters: `eta = [eta1_0, eta1_1, eta1_2, eta2_00, eta2_01, ..., eta2_22]` (12 dimensions)
- Sufficient statistics: `T(x) = [x_0, x_1, x_2, x_0^2, x_0*x_1, x_0*x_2, x_1^2, x_1*x_2, x_2^2]` (9 dimensions)
- Configuration: `multivariate_3d_large.yaml`

## Defining new exponential-family distributions

- Implement the `ExponentialFamily` interface in `src/ef.py`: define immutable `x_shape`, `eta_dim`, and the methods `compute_stats(x)`, `logdensity_fn(eta)`, and `flatten_stats_or_eta(dict)`.
- Add your EF to `ef_factory` in `src/ef.py` or pass an instance programmatically.
- The system automatically handles flattening for HMC sampling and respects your shapes when computing moments.

## Configuration System

The project uses YAML configuration files to specify:
- **Exponential Family**: Type and parameters
- **Grid**: Number of training/validation points and parameter ranges
- **Sampling**: HMC parameters (samples, warmup, step size, integration steps)
- **Optimization**: Neural network architecture, learning rate, epochs, batch size

## Visualization

The project includes comprehensive plotting capabilities:
- **1D Results**: Training curves, moment comparisons, error distributions
- **3D Results**: Multi-panel analysis including linear/quadratic term comparisons, covariance heatmaps, and component-wise MSE breakdowns

## Notes

- All implementations use JAX for efficient computation and automatic differentiation
- Training data is cached using configuration hashes to avoid regeneration
- The system supports both CPU and GPU execution via JAX backends
- Matrix operations use optimized JAX routines with `vmap` compatibility for batch processing


