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

3. Test all models with small architectures (quick validation):

```bash
# Test all 11 models with small networks (20 epochs, ~1K parameters each)
python scripts/test_all_training_scripts.py
```

4. Train models individually (standardized approach):

```bash
# Traditional ET Networks (direct approximation)
python scripts/training/train_mlp_ET.py
python scripts/training/train_glu_ET.py
python scripts/training/train_quadratic_resnet_ET.py

# LogZ Networks (gradient-based)
python scripts/training/train_mlp_logZ.py
python scripts/training/train_glu_logZ.py
python scripts/training/train_quadratic_resnet_logZ.py

# Flow-based Networks
python scripts/training/train_geometric_flow_ET.py
python scripts/training/train_glow_ET.py
python scripts/training/train_invertible_nn_ET.py

# Specialized Networks
python scripts/training/train_noprop_ct_ET.py
python scripts/training/train_convex_nn_logZ.py
```

5. Run comprehensive model comparison (all 11 models with standardized architectures):

```bash
# Train all models with comparable architectures (12 layers, 128 units; Glow 24 layers)
python scripts/run_comprehensive_model_comparison.py --data data/easy_3d_gaussian.pkl --ef gaussian_3d

# Analyze comprehensive results
python scripts/create_comparison_analysis.py --mode full
```

6. Analyze test results:

```bash
# Analyze small architecture test results
python scripts/create_comparison_analysis.py --mode test
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
    - `glu_ET.py`: Gated Linear Unit networks for ET
    - `glu_logZ.py`: Gated Linear Unit networks for LogZ
    - `quadratic_resnet_ET.py`: Quadratic residual networks for ET
    - `quadratic_resnet_logZ.py`: Quadratic residual networks for LogZ
    - `invertible_nn_ET.py`: Invertible neural networks with coupling layers
    - `noprop_ct_ET.py`: No-propagation continuous-time networks
    - `convex_nn_logZ.py`: Input convex neural networks for log normalizers
  - `generate_data.py`: Data generation using HMC sampling with configurable parameters
  - `utils/`: Utility modules
    - `performance.py`: Performance measurement utilities
    - `matrix_utils.py`: JAX utilities for matrix operations
    - `data_utils.py`: Data loading and preprocessing utilities
    - `exact_covariance.py`: Analytical covariance computation for known exponential families
- `scripts/`: Training and analysis scripts
  - `training/`: Individual training scripts for all 11 model architectures
  - `test_all_training_scripts.py`: Quick validation with small architectures
  - `run_comprehensive_model_comparison.py`: Full-scale comparison of all models
  - `create_comparison_analysis.py`: Analysis and visualization of results
  - `generate_normal_data.py`: Data generation utilities
  - `list_available_models.py`: Model listing and configuration overview
  - `plot_training_results.py`: Standardized plotting functions
  - `debug/`: Outdated and experimental scripts
- `configs/`: Configuration files
  - `gaussian_1d_large.yaml`: Large-scale 1D Gaussian configuration
  - `multivariate_3d_large.yaml`: Large-scale 3D multivariate Gaussian configuration
- `artifacts/`: Saved models, training history, and plots
  - `ET_models/`: Results from comprehensive ET model comparison
  - `logZ_models/`: Results from comprehensive LogZ model comparison
  - `tests/`: Results from small architecture test runs
  - `comprehensive_comparison/`: Analysis results and comparison plots
- `data/`: Generated training datasets (pickle files)
  - `easy_3d_gaussian.pkl`: Standard dataset for model comparison
  - `easy_3d_gaussian_small.pkl`: Small dataset for quick testing
- `examples/`: Example usage scripts and notebooks
- `docs/`: Documentation files
- `papers/`: Research papers and manuscripts

## Training Scripts

The project includes comprehensive training scripts for various neural network architectures. Each script can be run independently and includes evaluation, plotting, and result saving capabilities.

| Script | Model Type | Description |
|--------|------------|-------------|
| `train_mlp_ET.py` | Direct ET | Standard MLP networks that directly learn the expectation mapping μ_T(η) = E[T(X)\|η]. Uses standardized data loading and plotting. |
| `train_glu_ET.py` | Direct ET | Gated Linear Unit networks with gated activations for improved expressiveness. Standardized architecture and training pipeline. |
| `train_quadratic_resnet_ET.py` | Direct ET | Quadratic residual networks with adaptive quadratic mixing. Deep narrow architecture optimized for complex mappings. |
| `train_invertible_nn_ET.py` | Invertible NN | Invertible neural networks with additive coupling layers and ActNorm. 9D output format for 3D Gaussian statistics. |
| `train_noprop_ct_ET.py` | Continuous Time | No-propagation continuous-time networks with ODE solvers. Converts 12D data to 9D format for model compatibility. |
| `train_geometric_flow_ET.py` | Geometric Flow | **Novel** geometric flow networks using continuous dynamics: du/dt = A@A^T@(η_target - η_init). Respects exponential family geometry with minimal time steps. |
| `train_glow_ET.py` | Normalizing Flow | Deep flow networks using normalizing flows with affine coupling layers. Features 50+ flow layers with 9D output format. |
| `train_mlp_logZ.py` | Log Normalizer | MLP networks that learn the log normalizer A(η) and compute expectations via ∇A(η). Uses MSE loss with L1 regularization. |
| `train_glu_logZ.py` | Log Normalizer | GLU-based log normalizer networks with gated activations for learning A(η). Standardized training pipeline. |
| `train_quadratic_resnet_logZ.py` | Log Normalizer | Quadratic residual networks for log normalizer learning with residual connections and adaptive mixing. |
| `train_convex_nn_logZ.py` | Log Normalizer | **Novel** alternating convex neural networks with Type 1/Type 2 layers. Maintains convexity properties essential for exponential families. |
| `ET_training_template.py` | Template | Training script template for ET networks. Provides standardized structure and plotting utilities for creating new model training scripts. |
| `logZ_training_template.py` | Template | Training script template for LogZ networks. Provides standardized structure and plotting utilities for creating new LogZ model training scripts. |
| `test_all_training_scripts.py` | Multi-Model | Tests all 11 models with small architectures (20 epochs, ~1K parameters). Creates compatible output for analysis. |
| `run_comprehensive_model_comparison.py` | Multi-Model | Comprehensive comparison of all 11 models with standardized architectures (12 layers, 128 units; Glow 24 layers). |
| `create_comparison_analysis.py` | Analysis | Creates comprehensive analysis plots and tables. Supports both full comparison and test result analysis modes. |

### Model Categories

**Direct ET Networks**: Learn μ_T(η) directly from (η, μ_T) pairs
- MLP, GLU, Quadratic ResNet, Invertible NN, NoProp-CT variants
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

## Workflow Overview

### 1. Quick Testing
```bash
# Test all models with small architectures
python scripts/test_all_training_scripts.py

# Analyze test results
python scripts/create_comparison_analysis.py --mode test
```

### 2. Comprehensive Comparison
```bash
# Train all models with standardized architectures
python scripts/run_comprehensive_model_comparison.py --data data/easy_3d_gaussian.pkl --ef gaussian_3d

# Analyze comprehensive results
python scripts/create_comparison_analysis.py --mode full
```

### 3. Individual Model Training
```bash
# Train specific models
python scripts/training/train_mlp_ET.py
python scripts/training/train_geometric_flow_ET.py
```

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
