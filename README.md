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

2. Train a simple 1D polynomial EF (Gaussian-like) moment network:

```bash
python scripts/train_moment_net.py --config configs/gaussian_moment_net.yaml
```

Artifacts (learned parameters and metrics) will be saved under `artifacts/`.

## Project layout

- `src/nnef_dist/`: Core library
  - `ef.py`: Exponential-family specs and log-density helpers
  - `sampling.py`: BlackJAX HMC sampling utilities
  - `model.py`: Flax models for moment mapping
  - `train.py`: Data generation and training loop
- `scripts/`: CLI entry points
- `configs/`: Example configuration files
- `artifacts/`: Saved models and metrics

## Notes

- The example configuration models a 1D polynomial EF with `T(x) = [x, x^2]` and `eta = [eta1, eta2]` where `eta2 < 0` to ensure integrability, corresponding to a Gaussian in natural parameterization.
- Future extensions include higher-dimensional `T(x)`, alternative base measures, and invertible bi-directional models between `eta` and `E[T]`.


