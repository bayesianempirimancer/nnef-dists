# nnef-dists

Fast variational inference with non-named exponential family distributions using neural networks to learn the mapping between natural parameters and expected sufficient statistics. Features both direct neural approximation and novel geometric flow-based approaches.

## Overview

For an exponential family with natural parameters $\eta$ and sufficient statistics $T(x)$, inference by message passing requires the mapping $\mu_T(eta) = \left< T(x) | \eta\right>$. Named exponential-family distributions are particularly easy to work with because they have uniquely invertible closed-form expressions for $\mu_T(\eta)$.  More generally, exponential family distributions can be written

$$ \log p(x|\eta) = \eta\cdot T(x) - A(\eta)$$

and include many distributions for which $\mu_T(\eta)$ is unknown and must be obtained via a sampling procedure such as MCMC.  This project aims to massively expand the set of exponential family distributions that are as easy to work with as named distributions by learning the function $mu_T(\eta)$ for arbitrary exponential family distributions.  The basic approach is to train a set of relatively small neural networks using samples conditioned on $\eta$ for a particular choice of $T(x)$.  Each choice of $T(x)$ then gives us a new class of distributions for which algorithms like Variational Bayesian Expectation Maximization or Coordinate Ascent Variational Inference become trivially implementable via message passing.  Currently we are comparing three classes of neural network architectures all of which can be trained using $\left\{\eta, \mu_T\right\}$ pairs generated via MCMC smaples and some of which can be fit directly to samples from any distribution:

1. **Direct Neural Approximation**: Traditional neural networks trained on MCMC samples
2. **Geometric Network**: Traditional Neural Networks that approximate $A(\eta)$ and exploit the relationship $\mu_T(\eta) = \nabla A(\eta)$ to generate predictions
3. **Geometric Flow Networks**: Novel flow-based approach using continuous dynamics that respects the geometric structure of exponential families, specificially $\Sigma_{TT}(\eta) = \nabla \mu_T(\eta)$, which implies that $\frac{d\mu_T}{dt} = \Sigma_{TT}(\eta)\frac{d\eta}{dt}$.

This repository uses JAX/Flax for modeling and BlackJAX for HMC sampling. It utilizes an arbitrary exponential family distribution class that allows for user specification of the vector of sufficient statistic function $T(x)$.  

## Complete Pipeline: From Distribution Definition to Neural Network Training

This section outlines the complete workflow for creating a new exponential family distribution and training neural networks to learn the expectation mapping. We'll use the **LaplaceProduct** distribution as a concrete example.

### Step 1: Define the Exponential Family Distribution

First, specify the sufficient statistics `T(x)` in the exponential family distribution file `src/ef.py`:

```python
@dataclass(frozen=True)
class LaplaceProduct(ExponentialFamily):
    """ 
    Laplace product in natural parameterization with T(x) = -abs(x+1)-abs(x-1) where x is a vector.
    """
    x_shape: Tuple[int,]

    @cached_property
    def stat_specs(self) -> Dict[str, Tuple[int, ...]]:
        return {"xm1": (self.x_shape[-1],), "xp1": (self.x_shape[-1],)}

    def _compute_stats(self, x: Array) -> Dict[str, Array]:
        return {"xm1": -jnp.abs(x-1), "xp1": -jnp.abs(x+1)}
```

Add the distribution to the factory function:

```python
def ef_factory(name: str, **kwargs) -> ExponentialFamily:
    # ... existing distributions ...
    elif n in {"laplace_product", "product_laplace"}:
        x_shape = kwargs.get("x_shape", (1,))
        if isinstance(x_shape, list):
            x_shape = tuple(x_shape)
        return LaplaceProduct(x_shape=x_shape)
```



### Step 2: Generate MCMC Samples and compute expected statistics

Create a YAML configuration file for data generation:

```yaml
# configs/laplace_product_1d.yaml
ef:
  name: "laplace_product"
  x_shape: [1]

grid:
  num_train_points: 800
  num_val_points: 200
  eta_ranges:
    - [0.2, 4.0]  # eta_xm1 range
    - [0.2, 4.0]  # eta_xp1 range

sampling:
  num_samples: 1000
  num_warmup: 500
  step_size: 0.1
  num_integration_steps: 10
  initial_position: [0.0]

optim:
  seed: 42
```

**Run the data generation script:**
```
# Create the configuration file
cat > configs/laplace_product_1d.yaml << 'EOF'
ef:
  name: "laplace_product"
  x_shape: [1]

grid:
  num_train_points: 800
  num_test_points: 100
  num_val_points: 100
  eta_ranges:
    - [0.2, 4.0]  # eta_xm1 range
    - [0.2, 4.0]  # eta_xp1 range

sampling:
  num_samples: 1000
  num_warmup: 500
  step_size: 0.1
  num_integration_steps: 10
  initial_position: [0.0]

optim:
  seed: 42
EOF
```

### Step 3: Generate Training Data

Generate samples for many values of the natural parameters using BlackJAX MCMC:

```bash
# Optional: Force regeneration if data already exists
python src/generate_data.py --config configs/laplace_product_1d.yaml --force
```

This will:
- Generate 1000 different `eta` parameter combinations and associated `mu_T`
- Use HMC sampling to generate 1000 samples for each `eta`
- Compute expected sufficient statistics `μ_T(η) = ⟨T(x) | η⟩` from the samples
- Save timing information (total sampling time, per-eta timing)
- Output: `data/training_data_[hash].pkl`

### Step 4: Choose a Network and Specify Network Architectures

The repository provides multiple neural network approaches for learning the expectation mapping `μ_T(η) = E[T(x)|η]`. Choose from:

#### **Direct ET Networks** 
Learn `μ_T(η)` directly from `(η, μ_T)` pairs:

```bash
# Available architectures:
python scripts/training/train_mlp_ET.py          # Standard MLP
python scripts/training/train_glu_ET.py          # Gated Linear Units
python scripts/training/train_quadratic_resnet_ET.py  # Quadratic ResNet
```

#### **Log Normalizer Networks**
Learn `A(η)` and compute `μ_T = ∇A(η)` via automatic differentiation:

```bash
# Available architectures:
python scripts/training/train_mlp_logZ.py        # MLP for log normalizer
python scripts/training/train_glu_logZ.py        # GLU for log normalizer
python scripts/training/train_convex_nn_logZ.py  # Convex neural networks
```

#### **Flow-Based Networks** (Advanced)
Use continuous dynamics or normalizing flows:

```bash
# Available architectures:
python scripts/training/train_geometric_flow_ET.py  # Novel geometric flows
python scripts/training/train_glow_ET.py            # Invertible Networks (GLOW)
python scripts/training/train_invertible_nn_ET.py   # Invertible networks
```

#### **Architecture Configuration**

**To customize architectures:**

1. **Edit the training script directly**: Modify the `architectures` dictionary in the training script
2. **Add new architectures**: Add entries to the dictionary with your desired layer configurations
3. **Modify existing ones**: Change the layer sizes or depths as needed

**Example customizations:**

```python
# Custom architectures for different complexity levels
architectures = {
    "Shallow": [64, 64],                              # 2 layers, 64 units each
    "Medium": [128, 128, 128],                        # 3 layers, 128 units each  
    "Deep": [256, 256, 256, 256, 256],                # 5 layers, 256 units each
    "Wide": [512, 512],                               # 2 layers, 512 units each
}
```
### Step 5: Train Neural Networks

Train neural networks on the `(eta, mu_T)` data:

```bash
# Train MLP networks on the LaplaceProduct data
python scripts/training/train_mlp_ET.py \
    --data_file data/training_data_6e498cc8e69bc76f92e150200406bfa5.pkl \
    --save_dir artifacts/ET_models/laplace_product_mlp_ET \
    --epochs 300

# Alternative: Train with different architectures
python scripts/training/train_glu_ET.py \
    --data_file data/training_data_6e498cc8e69bc76f92e150200406bfa5.pkl \
    --save_dir artifacts/ET_models/laplace_product_glu_ET \
    --epochs 300
```

This will:
- Load the generated `(eta, mu_T)` pairs
- Train MLP networks to learn the mapping `eta → mu_T`
- Evaluate performance on test data
- Save model parameters, training history, and comparison plots


**Generated Files:**
- `data/training_data_6e498cc8e69bc76f92e150200406bfa5.pkl` - Training dataset with timing info
- `artifacts/ET_models/laplace_product_mlp_ET/` - Trained models and results
- `laplace_product_1d_test.png` - Validation plots

This demonstrates how to create a new exponential family distribution and train neural networks to learn the expectation mapping `μ_T(η) = E[T(x)|η]`, making previously intractable distributions as easy to work with as named distributions.

## Installation

1. Install dependencies:

```bash
pip install -e .
```



## Project Layout

- `src/`: Core library
  - `ef.py`: Generic `ExponentialFamily` interface with `GaussianNatural1D`, `MultivariateNormal`, `MultivariateNormal_tril`, and `LaplaceProduct` implementations
  - `sampling.py`: BlackJAX HMC sampling utilities for arbitrary-shaped distributions
  - `config.py`: Configuration management system with `FullConfig`, `NetworkConfig`, and `TrainingConfig` classes
  - `generate_data.py`: Data generation using HMC sampling with configurable parameters and timing information
  - `models/`: Neural network architectures
    - `ET_Net.py`: Direct expectation networks (MLP, GLU, Quadratic ResNet, etc.)
    - `logZ_Net.py`: Log-normalizer networks with gradient/Hessian computation
    - `geometric_flow_net.py`: **NEW** Geometric flow networks using continuous dynamics
    - `glow_net_ET.py`: Glow networks using normalizing flows with affine coupling
    - `mlp_ET.py`: Standard MLP expectation networks
    - `mlp_logZ.py`: MLP-based log normalizer networks
    - `glu_ET.py`: Gated Linear Unit networks for ET
    - `glu_logZ.py`: Gated Linear Unit networks for LogZ
    - `quadratic_resnet_ET.py`: Quadratic residual networks for ET
    - `quadratic_resnet_logZ.py`: Quadratic residual networks for LogZ
    - `noprop_ct_ET.py`: No-propagation continuous-time networks
    - `noprop_geometric_flow_ET.py`: No-propagation geometric flow networks
    - `convex_nn_logZ.py`: Input convex neural networks for log normalizers
  - `utils/`: Utility modules
    - `performance.py`: Performance measurement utilities
    - `matrix_utils.py`: JAX utilities for matrix operations
    - `data_utils.py`: Data loading and preprocessing utilities
    - `exact_covariance.py`: Analytical covariance computation for known exponential families
    - `generate_normal_data.py`: Data generation utilities
- `scripts/`: Training and analysis scripts
  - `training/`: Individual training scripts for all model architectures
    - `train_mlp_ET.py`, `train_glu_ET.py`, `train_quadratic_resnet_ET.py`: Direct ET networks
    - `train_mlp_logZ.py`, `train_glu_logZ.py`, `train_quadratic_resnet_logZ.py`: Log normalizer networks
    - `train_geometric_flow_ET.py`, `train_glow_ET.py`, `train_noprop_ct_ET.py`: Flow-based networks
    - `train_convex_nn_logZ.py`: Convex neural networks
    - `training_template_ET.py`, `train_template_logZ.py`: Training script templates
  - `test_all_training_scripts.py`: Quick validation with small architectures
  - `run_comprehensive_model_comparison.py`: Full-scale comparison of all models
  - `list_available_models.py`: Model listing and configuration overview
  - `plotting/`: Analysis and visualization scripts
    - `create_comparison_analysis.py`: Analysis and visualization of results
    - `plot_training_results.py`: Standardized plotting functions
  - `debug/`: Outdated and experimental scripts
- `configs/`: Configuration files for data generation
  - `laplace_product_1d.yaml`: Configuration for 1D LaplaceProduct distribution
- `data/configs/`: Additional configuration files
  - `gaussian_1d_large.yaml`, `gaussian_1d_medium.yaml`, `gaussian_1d_small.yaml`: 1D Gaussian configurations
  - `multivariate_3d_large.yaml`, `multivariate_3d_medium.yaml`: 3D multivariate Gaussian configurations
  - `multivariate_3d_tril_large_4x.yaml`: Lower triangular multivariate Gaussian configurations
- `artifacts/`: Saved models, training history, and plots
  - `ET_models/`: Results from comprehensive ET model comparison
  - `logZ_models/`: Results from comprehensive LogZ model comparison
  - `tests/`: Results from small architecture test runs
  - `comprehensive_comparison/`: Analysis results and comparison plots
- `data/`: Generated training datasets (pickle files)
  - `easy_3d_gaussian.pkl`: Standard dataset for model comparison
  - `easy_3d_gaussian_tril.pkl`: Lower triangular Gaussian dataset
  - `training_data_*.pkl`: Generated datasets with timing information
- `docs/`: Documentation files
- `papers/`: Research papers and manuscripts
- `debug/`: Debugging and experimental scripts

## Training Scripts

The project includes comprehensive training scripts for various neural network architectures. Each script can be run independently and includes evaluation, plotting, and result saving capabilities.

| Script | Model Type | Description |
|--------|------------|-------------|
| `train_mlp_ET.py` | Direct ET | Standard MLP networks that directly learn the expectation mapping μ_T(η) = E[T(X)\|η]. Uses standardized data loading and plotting. |
| `train_glu_ET.py` | Direct ET | Gated Linear Unit networks with gated activations for improved expressiveness. Standardized architecture and training pipeline. |
| `train_quadratic_resnet_ET.py` | Direct ET | Quadratic residual networks with adaptive quadratic mixing. Deep narrow architecture optimized for complex mappings. |
| `train_noprop_ct_ET.py` | Continuous Time | No-propagation continuous-time networks with ODE solvers. Converts 12D data to 9D format for model compatibility. |
| `train_noprop_geometric_flow_ET.py` | Geometric Flow | No-propagation geometric flow networks combining continuous-time dynamics with geometric flow principles. |
| `train_geometric_flow_ET.py` | Geometric Flow | **Novel** geometric flow networks using continuous dynamics: du/dt = A@A^T@(η_target - η_init). Respects exponential family geometry with minimal time steps. |
| `train_glow_ET.py` | Normalizing Flow | Deep flow networks using normalizing flows with affine coupling layers. Features 50+ flow layers with 9D output format. |
| `train_mlp_logZ.py` | Log Normalizer | MLP networks that learn the log normalizer A(η) and compute expectations via ∇A(η). Uses MSE loss with L1 regularization. |
| `train_glu_logZ.py` | Log Normalizer | GLU-based log normalizer networks with gated activations for learning A(η). Standardized training pipeline. |
| `train_quadratic_resnet_logZ.py` | Log Normalizer | Quadratic residual networks for log normalizer learning with residual connections and adaptive mixing. |
| `train_convex_nn_logZ.py` | Log Normalizer | **Novel** alternating convex neural networks with Type 1/Type 2 layers. Maintains convexity properties essential for exponential families. |
| `training_template_ET.py` | Template | Training script template for ET networks. Provides standardized structure and plotting utilities for creating new model training scripts. |
| `train_template_logZ.py` | Template | Training script template for LogZ networks. Provides standardized structure and plotting utilities for creating new LogZ model training scripts. |
| `test_all_training_scripts.py` | Multi-Model | Tests all models with small architectures (20 epochs, ~1K parameters). Creates compatible output for analysis. |
| `run_comprehensive_model_comparison.py` | Multi-Model | Comprehensive comparison of all models with standardized architectures (12 layers, 128 units; Glow 24 layers). |
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

## Supported Distributions

### 1D Gaussian Natural Parameters
- Natural parameters: `eta = [eta1, eta2]` where `eta2 < 0` for integrability
- Sufficient statistics: `T(x) = [x, x^2]`
- Configuration: `gaussian_1d_large.yaml`

### Multivariate Normal
- Natural parameters: `eta = [eta1_0, eta1_1, eta1_2, ... eta2_00, eta2_01, ...,]` ($N$ + $N^2$ dimensions)
- Sufficient statistics: `T(x) = [x_0, x_1, x_2, x_0^2, x_0*x_1, x_0*x_2, x_1^2, x_1*x_2, x_2^2]` 
- 3D Configuration: `multivariate_3d_large.yaml`

### Multivariate Normal_tril (Lower triangular for eta2 -- ensures full rank $\Sigma_{TT}(\eta)$)
- Natural parameters: `eta = [eta1_0, eta1_1, eta1_2, ... eta2_00, eta2_01, ...,]` ($N$ + $N(N+1)/2$ dimensions)
- Sufficient statistics: `T(x) = [x_0, x_1, x_2, x_0^2, x_0*x_1, x_0*x_2, x_1^2, x_1*x_2, x_2^2]` 
- Configuration: `multivariate_3d_large.yaml`

### 1D LaplaceProduct (Example of Custom Distribution)
- Natural parameters: `eta = [eta_xm1, eta_xp1]` where both parameters are positive
- Sufficient statistics: `T(x) = [-|x-1|, -|x+1|]`
- Distribution: `p(x|η) ∝ exp(η₁·(-|x-1|) + η₂·(-|x+1|))`
- Configuration: `laplace_product_1d.yaml`
- **Key features**: Non-standard distribution requiring numerical integration for normalization, demonstrates the full pipeline from definition to neural network training

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



## Visualization

The project includes comprehensive plotting capabilities:
- Training curves, moment comparisons, error distributions
- Multi-panel analysis including linear/quadratic term comparisons, covariance heatmaps, and component-wise MSE breakdowns

## Notes

- All implementations use JAX for efficient computation and automatic differentiation
- Training data is cached using configuration hashes to avoid regeneration
- The system supports both CPU and GPU execution via JAX backends
- Certain matrix operations use optimized JAX routines with `vmap` compatibility for batch processing
