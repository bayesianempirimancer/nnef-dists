# nnef-dist

Fast variational inference with non-named exponential family distributions using neural networks to learn the mapping between natural parameters and expected sufficient statistics.

## Overview

For an exponential family with natural parameters `eta` and sufficient statistics `T(x)`, inference by message passing requires the mapping `mu_T(eta) = E[T(x) | eta]`. Named exponential-family distributions have closed-form `mu_T`. This project learns `mu_T` for non-named families by training a neural network on MCMC samples generated for many `eta` values.

This repository uses JAX/Flax for modeling and BlackJAX for HMC sampling.

## Quickstart

1. Install dependencies (CPU JAX):

```bash
pip install -e .[dev]
```

2. Train a simple 1D Gaussian-in-natural-params EF moment network:

```bash
python scripts/train_moment_net.py --config configs/gaussian_moment_net.yaml
```

Artifacts (learned parameters and metrics) will be saved under `artifacts/`.

## Project layout

- `src/nnef_dist/`: Core library
  - `ef.py`: Generic `ExponentialFamily` interface, `GaussianNatural1D` example, and EF factory
  - `sampling.py`: BlackJAX HMC sampling utilities (arbitrary-shaped x via flattening)
  - `model.py`: Flax models for moment mapping (eta -> E[T(x)])
  - `train.py`: Data generation (MCMC) and training loop
- `scripts/`: CLI entry points
- `configs/`: Example configuration files
- `artifacts/`: Saved models and metrics

## Defining new exponential-family distributions

- Implement the `ExponentialFamily` interface in `src/nnef_dist/ef.py` (or your own module): define immutable `x_shape`, `t_shape`, and the methods `sufficient_statistic(x)` and `log_unnormalized(x, eta)`. The shapes may be arbitrary tensors, not just vectors. The trainer flattens `x` for HMC and respects your shapes when computing moments.
- Add your EF to `ef_factory` or pass an instance programmatically.

## Notes

- The included example `GaussianNatural1D` uses `T(x) = [x, x^2]` with `eta = [eta1, eta2]` and `eta2 < 0` for integrability.
- Future extensions include higher-dimensional/tensor `x` and `T(x)`, alternative base measures, and invertible bi-directional models between `eta` and `E[T]`.


