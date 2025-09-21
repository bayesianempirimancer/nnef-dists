# nnef-dists

Fast variational inference with non-named exponential family distributions using neural networks to learn the mapping between natural parameters and expected sufficient statistics. Features both direct neural approximation and novel geometric flow-based approaches.

## Overview

For an exponential family with natural parameters $\eta$ and sufficient statistics $T(x)$, inference by message passing requires the mapping $\mu_T(eta) = \left< T(x) | \eta\right>$. Named exponential-family distributions are particularly easy to work with because they have uniquely invertible closed-form expressions for $\mu_T(\eta)$.  More generally, exponential family distributions can be written

$$ \log p(x|\eta) = \eta\cdot T(x) - A(\eta)$$

and include many distributions for which $\mu_T(\eta)$ is unknown and must be obtained via a sampling procedure such as MCMC.  This project aims to massively expand the set of exponential family distributions that are as easy to work with as named distributions by learning the function $mu_T(\eta)$ for arbitrary exponential family distributions.  The basic approach is to train a set of relatively small neural networks using samples conditioned on $\eta$ for a particular choice of $T(x)$.  Each choice of $T(x)$ then gives us a new class of distributions for which algorithms like Variational Bayesian Expectation Maximization or Coordinate Ascent Variational Inference become trivially implementable via message passing.  Currently we are comparing three classes of neural network architectures all of which can be trained using $\left\{\eta, \mu_T\right\}$ pairs generated via MCMC smaples and some of which can be fit directly to samples from any distribution:

1. **Direct Neural Approximation**: Traditional neural networks trained on MCMC samples
2. **Geometric Network**: Traditional Neural Networks that approximate $A(\eta)$ and exploit the relationship $\mu_T(\eta) = \nabla A(\eta)$
3. **Geometric Flow Networks**: Novel flow-based approach using continuous dynamics that respects the geometric structure of exponential families, specificially $\Sigma_{TT}(\eta) = \nabla \mu_T(\eta)$

This repository uses JAX/Flax for modeling and BlackJAX for HMC sampling. It utilizes an arbitrary exponential family distribution class that allows for user specification of the vector of sufficient statistic function $T(x)$.  

## Quickstart

1. Install dependencies:

```bash
pip install -e .
```

2. Generate training data for a 3D Gaussian model:

```bash
python src/generate_data.py --config configs/gaussian_3d_large.yaml
```

3. Train neural networks (multiple approaches available):

```bash
# Traditional ET Network (direct approximation)
python scripts/training/train_mlp_ET.py

# LogZ Network (gradient-based)
python scripts/training/train_mlp_logZ.py

# Geometric Flow Network (novel flow-based approach)
python scripts/training/train_geometric_flow_ET.py --save-dir artifacts/geometric_flow

# Deep Flow Network (normalizing flows)
python scripts/training/train_glow_ET.py

# Quadratic ResNet (advanced architecture)
python scripts/training/train_quadratic_resnet_ET.py
```

4. Train individual models for comparison:

```bash
# Train specific model types
python scripts/train_comparison_models.py --model mlp_ET
python scripts/train_comparison_models.py --model geometric_flow_ET
python scripts/train_comparison_models.py --model convex_nn_logZ
```

5. Test all models comprehensively:

```bash
python scripts/test_all_models.py
```

Artifacts (learned parameters, training history, and plots) will be saved under `artifacts/`.

## Project Layout

- `src/`: Core library
  - `ef.py`: Generic `ExponentialFamily` interface with `GaussianNatural1D`, `MultivariateNormal`, and `MultivariateNormal_tril` implementations
  - `sampling.py`: BlackJAX HMC sampling utilities for arbitrary-shaped distributions
  - `config.py`: Configuration management system with `FullConfig`, `NetworkConfig`, and `TrainingConfig` classes
  - `models/`: Neural network architectures
    - `ET_Net.py`: Direct expectation networks (MLP, GLU, Quadratic ResNet, etc.)
    - `logZ_Net.py`: Log-normalizer networks with gradient/Hessian computation
    - `geometric_flow_net.py`: **NEW** Geometric flow networks using continuous dynamics
    - `glow_ET.py`: Glow networks using normalizing flows with affine coupling
    - `mlp_ET.py`: Standard MLP expectation networks
    - `mlp_logZ.py`: MLP-based log normalizer networks
    - `glu_network.py`: Gated Linear Unit networks
    - `quadratic_resnet.py`: Quadratic residual networks
    - `invertible_nn.py`: Invertible neural networks with coupling layers
    - `noprop_ct_ET.py`: No-propagation continuous-time networks
    - `convex_nn_logZ.py`: Input convex neural networks for log normalizers
  - `generate_data.py`: Data generation using HMC sampling with configurable parameters
  - `utils/`: Utility modules
    - `performance.py`: Performance measurement utilities
    - `matrix_utils.py`: JAX utilities for matrix operations
    - `data_utils.py`: Data loading and preprocessing utilities
    - `exact_covariance.py`: Analytical covariance computation for known exponential families
- `scripts/`: Training and visualization scripts
  - `training/`: Training scripts for different model architectures (see table below)
  - `test_flow_math_fixed.py`: Mathematical verification of flow dynamics
  - `demo_flow_scaling.py`: Demonstration of flow efficiency with coarse discretization
- `configs/`: Configuration files
  - `gaussian_1d_large.yaml`: Large-scale 1D Gaussian configuration
  - `multivariate_3d_large.yaml`: Large-scale 3D multivariate Gaussian configuration
- `plotting/`: Visualization and comparison utilities
- `artifacts/`: Saved models, training history, and plots (organized by model type)
- `data/`: Generated training datasets (pickle files) - all training data stored here for reuse
- `examples/`: Example usage scripts and notebooks
- `docs/`: Documentation files
- `papers/`: Research papers and manuscripts

## Training Scripts

The project includes comprehensive training scripts for various neural network architectures. Each script can be run independently and includes evaluation, plotting, and result saving capabilities.

| Script | Model Type | Description |
|--------|------------|-------------|
| `train_mlp_ET.py` | Direct ET | Standard MLP networks that directly learn the expectation mapping μ_T(η) = E[T(X)\|η]. Features multiple architecture sizes and comprehensive evaluation. |
| `train_standard_mlp_ET.py` | Direct ET | Simplified standard MLP implementation with official model integration and fallback capabilities. |
| `train_geometric_flow_ET.py` | Geometric Flow | **Novel** geometric flow networks using continuous dynamics: du/dt = A@A^T@(η_target - η_init). Respects exponential family geometry with minimal time steps. |
| `train_glow_ET.py` | Normalizing Flow | Deep flow networks using normalizing flows with affine coupling layers. Features 50+ flow layers with diffusion-based training. |
| `train_quadratic_resnet_ET.py` | Direct ET | Quadratic residual networks with adaptive quadratic mixing. Deep narrow architecture (10 layers x 96 units) optimized for complex mappings. |
| `train_glu_ET.py` | Direct ET | Gated Linear Unit networks with deep narrow architecture (10 layers x 80 units). Features gated activations for improved expressiveness. |
| `train_invertible_nn_ET.py` | Invertible NN | Invertible neural networks with additive coupling layers and ActNorm. 8 coupling layers x 128 units with strict gradient clipping. |
| `train_noprop_ct_ET.py` | Continuous Time | No-propagation continuous-time networks with ODE solvers. 8 CT layers x 96 units with Euler integration and noise scaling. |
| `train_mlp_logZ.py` | Log Normalizer | MLP networks that learn the log normalizer A(η) and compute expectations via ∇A(η). Multiple architecture comparisons included. |
| `train_glu_logZ.py` | Log Normalizer | GLU-based log normalizer networks with gated activations for learning A(η). |
| `train_quadratic_resnet_logZ.py` | Log Normalizer | Quadratic residual networks for log normalizer learning with residual connections and adaptive mixing. |
| `train_convex_nn_logZ.py` | Log Normalizer | **Novel** alternating convex neural networks with Type 1/Type 2 layers. Maintains convexity properties essential for exponential families. |
| `ET_training_template.py` | Template | Training script template for ET networks. Provides standardized structure and plotting utilities for creating new model training scripts. |
| `logZ_training_template.py` | Template | Training script template for LogZ networks. Provides standardized structure and plotting utilities for creating new LogZ model training scripts. |
| `train_comparison_models.py` | Multi-Model | Unified training script for standardized comparison across all model types. Supports individual model training with consistent evaluation. |
| `test_all_models.py` | Multi-Model | Comprehensive testing script that evaluates all available models on standardized datasets with performance comparison. |

### Model Categories

**Direct ET Networks**: Learn μ_T(η) directly from (η, μ_T) pairs
- MLP, GLU, Quadratic ResNet, Standard MLP variants
- Fast training, direct optimization of target mapping

**Log Normalizer Networks**: Learn A(η) and compute μ_T = ∇A(η)  
- MLP, GLU, Quadratic ResNet, Convex NN variants
- Leverages automatic differentiation, maintains mathematical properties

**Flow-Based Networks**: Use continuous dynamics or normalizing flows
- Geometric Flow (novel approach respecting exponential family geometry)
- Glow Networks (normalizing flows with affine coupling)
- Invertible Neural Networks (coupling layers)

**Continuous Time Networks**: Solve ODEs for expectation computation
- NoProp-CT (no propagation continuous time)
- ODE solvers with learned dynamics

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
An approach that learns flow dynamics to compute expectations:

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


