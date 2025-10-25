# nnef-dists

Fast variational inference with non‑named exponential-family distributions using neural networks to learn the mapping between natural parameters and expected sufficient statistics. Features both direct ET networks and geometric-flow approaches, plus a unified data-generation and training pipeline.

## Recent Updates

- **NEW**: `flow_models` directory with unified flow-based training framework
- **NEW**: 3 training protocols (FM, CT, DF) × 7 model architectures = 21 combinations
- **NEW**: Flow wrappers for potential, geometric, natural, and convex flows
- **NEW**: Direct ResNet models (MLP, Convex, Bilinear) with conditional inputs
- **NEW**: `layers/gradnet_utils.py` with efficient gradient computation utilities  
- **IMPROVED**: Automatic broadcasting and parameter scoping for flow models
- **IMPROVED**: Unified training script supporting all protocol-architecture combinations

## Overview

For an exponential family with natural parameters $\eta$ and sufficient statistics $T(x)$, inference by message passing requires the mapping $\mu_T(\eta) = \langle T(x) \mid \eta \rangle$. Named exponential-family distributions are easy because they have closed-form, invertible expressions for $\mu_T(\eta)$. In general,

$$ \log p(x\mid\eta) = \eta\cdot T(x) - A(\eta) $$

and $\mu_T(\eta)$ may be unknown. This project learns $\mu_T(\eta)$ for arbitrary exponential families by training compact neural networks on samples conditioned on $\eta$.

## Project Structure (Updated)

Active code lives under `src/` with EF definitions, data generation, and production model trainers. Legacy trainers remain under `src/training/` and should be avoided.

```
src/
├── expfam/                     # Exponential family distributions and data generation
│   ├── ef_base.py             # ExponentialFamily base class
│   ├── ef.py                  # Distribution implementations
│   ├── data_generator.py      # BlackJAX-based sampling and expectations
│   └── generate_data.py       # CLI/script to generate datasets
├── layers/                     # Neural network layers and utilities
│   ├── gradnet_utils.py      # Gradient computation utilities for flow models
│   └── archive/               # Archived layer implementations
└── models/
    ├── __init__.py             # Model registry/imports
    ├── base_config.py          # BaseConfig
    ├── base_model.py           # BaseModel[T]
    ├── base_trainer.py         # BaseETTrainer (80-10-10 randomized splitter)
    ├── base_training_config.py # BaseTrainingConfig
    ├── mlp_et/                 # MLP ET model + train.py
    ├── glu_et/                 # GLU ET model + train.py
    ├── quadratic_et/           # Quadratic ET model + train.py
    ├── glow_et/                # Glow ET model + train.py
    ├── geometric_flow_et/      # Geometric Flow ET model + train.py
    ├── flow_models/            # Flow-based models (NEW)
    │   ├── crn.py             # Conditional ResNet architectures + flow wrappers
    │   ├── fm.py              # Flow Matching protocol (NoPropFM)
    │   ├── ct.py              # Continuous-Time protocol (NoPropCT)
    │   ├── df.py              # Diffusion protocol (NoPropDF)
    │   ├── train.py           # Unified training script (21 combinations)
    │   └── trainer.py         # Unified trainer (NoPropTrainer)
    ├── noprop/                # NoProp variants
    └── archive/               # Archived/legacy code (IGNORE)

```

- Data configs for sampling are in `data/configs/*.yaml`.
- Archive/legacy directories under `src/models/archive/` and `src/training/` should be ignored for new development.

## Flow Models (NEW)

The `flow_models` directory contains a unified framework for flow-based neural networks that learn the mapping between natural parameters and expected sufficient statistics. This framework supports **three training protocols** and **seven model architectures**, providing maximum flexibility for different learning scenarios.

### Training Protocols

The framework supports three distinct training protocols, each with different mathematical foundations:

1. **Flow Matching (FM)**: Direct learning of the flow field `dz/dt = v_θ(z,t)` where the model learns to match the true flow field
2. **Continuous-Time (CT)**: Continuous-time dynamics with learned drift `dz/dt = f_θ(z,t)` using neural ODEs
3. **Diffusion (DF)**: Diffusion-based learning with noise schedules, learning to reverse a diffusion process

### Model Architectures

The framework supports seven different model architectures, each with different mathematical properties:

#### Flow Wrapper Models (4 types)
1. **Potential Flow**: `dz/dt = -∇_z V(z,x,t)` where V is a learned convex potential function
2. **Geometric Flow**: `dz/dt = Σ @ x` where Σ is a learned matrix (often Hessian of a potential)
3. **Natural Flow**: `dz/dt = Σ @ Σ^T @ x` where Σ is learned and symmetric
4. **Convex Potential Flow**: Like potential flow but with guaranteed convexity using ICNN

#### Direct ResNet Models (3 types)
5. **Conditional ResNet MLP**: Direct neural network `dz/dt = f(z,x,t)` with standard MLP architecture
6. **Convex Conditional ResNet**: ResNet with guaranteed convexity using ICNN layers
7. **Bilinear Conditional ResNet**: ResNet with bilinear interactions between z and processed x

### Usage

The unified training script supports all combinations of training protocols and model architectures:

```bash
# Flow Matching (FM) examples
conda activate numpyro
python -m src.models.flow_models.train --data data/laplace_product_data_10000.pkl --training-protocol fm --model potential_flow --epochs 200 --learning-rate 0.00001
python -m src.models.flow_models.train --data data/laplace_product_data_10000.pkl --training-protocol fm --model geometric_flow --epochs 200
python -m src.models.flow_models.train --data data/laplace_product_data_10000.pkl --training-protocol fm --model natural_flow --epochs 200
python -m src.models.flow_models.train --data data/laplace_product_data_10000.pkl --training-protocol fm --model convex_potential_flow --epochs 200
python -m src.models.flow_models.train --data data/laplace_product_data_10000.pkl --training-protocol fm --model conditional_resnet_mlp --epochs 200
python -m src.models.flow_models.train --data data/laplace_product_data_10000.pkl --training-protocol fm --model convex_conditional_resnet --epochs 200
python -m src.models.flow_models.train --data data/laplace_product_data_10000.pkl --training-protocol fm --model bilinear_conditional_resnet --epochs 200

# Continuous-Time (CT) examples
python -m src.models.flow_models.train --data data/laplace_product_data_10000.pkl --training-protocol ct --model potential_flow --epochs 200 --learning-rate 0.00001
python -m src.models.flow_models.train --data data/laplace_product_data_10000.pkl --training-protocol ct --model geometric_flow --epochs 200
python -m src.models.flow_models.train --data data/laplace_product_data_10000.pkl --training-protocol ct --model natural_flow --epochs 200
python -m src.models.flow_models.train --data data/laplace_product_data_10000.pkl --training-protocol ct --model convex_potential_flow --epochs 200
python -m src.models.flow_models.train --data data/laplace_product_data_10000.pkl --training-protocol ct --model conditional_resnet_mlp --epochs 200
python -m src.models.flow_models.train --data data/laplace_product_data_10000.pkl --training-protocol ct --model convex_conditional_resnet --epochs 200
python -m src.models.flow_models.train --data data/laplace_product_data_10000.pkl --training-protocol ct --model bilinear_conditional_resnet --epochs 200

# Diffusion (DF) examples
python -m src.models.flow_models.train --data data/laplace_product_data_10000.pkl --training-protocol df --model potential_flow --epochs 200 --learning-rate 0.00001
python -m src.models.flow_models.train --data data/laplace_product_data_10000.pkl --training-protocol df --model geometric_flow --epochs 200
python -m src.models.flow_models.train --data data/laplace_product_data_10000.pkl --training-protocol df --model natural_flow --epochs 200
python -m src.models.flow_models.train --data data/laplace_product_data_10000.pkl --training-protocol df --model convex_potential_flow --epochs 200
python -m src.models.flow_models.train --data data/laplace_product_data_10000.pkl --training-protocol df --model conditional_resnet_mlp --epochs 200
python -m src.models.flow_models.train --data data/laplace_product_data_10000.pkl --training-protocol df --model convex_conditional_resnet --epochs 200
python -m src.models.flow_models.train --data data/laplace_product_data_10000.pkl --training-protocol df --model bilinear_conditional_resnet --epochs 200
```

**Total combinations: 3 protocols × 7 architectures = 21 different training configurations!**

### Key Features

- **Unified Training**: Single script supports all protocols and architectures
- **Gradient Utilities**: Efficient computation of gradients and Hessians
- **Broadcasting Support**: Handles batch dimensions automatically
- **Flow Wrappers**: Clean interface for different flow types
- **Automatic Plotting**: Learning curves, trajectory diagnostics, and flow visualizations

## API Reference

### Training Script Arguments

The unified training script `src/models/flow_models/train.py` accepts the following arguments:

#### Required Arguments
- `--data`: Path to training data file (.pkl format)

#### Training Protocol Selection
- `--training-protocol {fm,ct,df}`: Choose the training protocol
  - `fm`: Flow Matching - Direct learning of flow field
  - `ct`: Continuous-Time - Neural ODE-based dynamics
  - `df`: Diffusion - Diffusion-based learning with noise schedules

#### Model Architecture Selection
- `--model {potential_flow,geometric_flow,natural_flow,convex_potential_flow,conditional_resnet_mlp,convex_conditional_resnet,bilinear_conditional_resnet}`: Choose the model architecture

#### Training Parameters
- `--epochs`: Number of training epochs (default: 400)
- `--dropout-epochs`: Number of epochs with dropout (default: 300)
- `--learning-rate`: Learning rate (default: 0.001)
- `--batch-size`: Batch size (default: 256)
- `--eval-steps`: Steps between detailed evaluation (default: 10)

#### Loss and Regularization
- `--loss-type {mse,snr_weighted_mse}`: Loss function type (default: mse)
- `--reg-weight`: Regularization weight (default: 0.0)

#### Noise Schedule (for FM/DF protocols)
- `--noise-schedule {linear,sigmoid,cosine,simple_learnable,learnable}`: Noise schedule type (default: linear)
- `--sigma-t`: Noise level for FM/DF models (default: 0.1)

#### Output
- `--output-dir`: Output directory (default: auto-generated)

### Model Architecture Details

#### Flow Wrapper Models
These models use mathematical flow equations with learned components:

1. **`potential_flow`**: `dz/dt = -∇_z V(z,x,t)`
   - Uses a learned potential function V(z,x,t)
   - Suitable for conservative flows
   - Requires gradient computation of the potential
   - **⚠️ Learning Rate**: Requires lower learning rate (0.00001) for numerical stability

2. **`geometric_flow`**: `dz/dt = Σ @ x`
   - Uses a learned matrix Σ (often Hessian of a potential)
   - Suitable for geometric transformations
   - Efficient for linear-like flows

3. **`natural_flow`**: `dz/dt = Σ @ Σ^T @ x`
   - Uses a learned symmetric matrix Σ
   - Ensures positive definiteness through Σ @ Σ^T
   - Suitable for natural gradient flows

4. **`convex_potential_flow`**: Like potential flow but with guaranteed convexity
   - Uses ICNN (Input Convex Neural Networks) for the potential
   - Ensures the potential function is convex
   - Suitable for optimization-based flows

#### Direct ResNet Models
These models use direct neural network architectures:

5. **`conditional_resnet_mlp`**: Standard MLP with conditional inputs
   - Direct neural network: `dz/dt = f(z,x,t)`
   - Most flexible architecture
   - Standard MLP with residual connections

6. **`convex_conditional_resnet`**: ResNet with guaranteed convexity
   - Uses ICNN layers for convexity
   - Ensures output is convex in z
   - Suitable for optimization-based learning

7. **`bilinear_conditional_resnet`**: ResNet with bilinear interactions
   - Processes x with MLP, then combines with z via bilinear interaction
   - Uses ConcatSquash for time integration
   - Suitable for multiplicative interactions

### Training Protocol Details

#### Flow Matching (FM)
- **Mathematical Foundation**: Learns to match the true flow field `v_θ(z,t) ≈ v_true(z,t)`
- **Loss Function**: MSE between predicted and true flow fields
- **Advantages**: Direct, interpretable, stable training
- **Best For**: Most flow-based applications, especially with potential flows
- **Note**: `potential_flow` models require lower learning rates (0.00001) for numerical stability

#### Continuous-Time (CT)
- **Mathematical Foundation**: Neural ODEs with learned drift `dz/dt = f_θ(z,t)`
- **Loss Function**: ODE integration loss
- **Advantages**: Continuous dynamics, flexible time stepping
- **Best For**: Time-dependent problems, continuous optimization

#### Diffusion (DF)
- **Mathematical Foundation**: Learns to reverse a diffusion process
- **Loss Function**: Denoising loss with noise schedules
- **Advantages**: Robust to noise, good for generative modeling
- **Best For**: Noisy data, generative tasks, robust learning

### Command Line Examples

```bash
# Quick start with Flow Matching + Potential Flow (requires lower learning rate)
python -m src.models.flow_models.train --data data/laplace_product_data_10000.pkl --training-protocol fm --model potential_flow --epochs 100 --learning-rate 0.00001

# High-performance training with Continuous-Time + Geometric Flow
python -m src.models.flow_models.train --data data/laplace_product_data_10000.pkl --training-protocol ct --model geometric_flow --epochs 300 --learning-rate 0.0005 --batch-size 512

# Robust training with Diffusion + Bilinear ResNet
python -m src.models.flow_models.train --data data/laplace_product_data_10000.pkl --training-protocol df --model bilinear_conditional_resnet --epochs 250 --dropout-epochs 200 --learning-rate 0.0005

# Convex optimization with Flow Matching + Convex Potential Flow
python -m src.models.flow_models.train --data data/laplace_product_data_10000.pkl --training-protocol fm --model convex_potential_flow --epochs 200 --learning-rate 0.001
```

## End-to-End Pipeline

### Step 1: Define an Exponential-Family Distribution via Sufficient Statistics and Domain

Implement an `ExponentialFamily` subclass in `src/expfam/ef.py` by defining:
1. **Sufficient statistics T(x)** - the functions that define your distribution
2. **Domain constraints** - the allowed values for x

```python
@dataclass(frozen=True)
class YourDistribution(ExponentialFamily):
    x_shape: Tuple[int, ...]
    
    @cached_property
    def stat_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Define the shapes of sufficient statistics T(x)."""
        return {
            "x": (self.x_shape[-1],),           # First-order statistic: T₁(x) = x
            "xxT": (self.x_shape[-1], self.x_shape[-1])  # Second-order statistic: T₂(x) = x⊗x
        }
    
    def _compute_stats(self, x: Array) -> Dict[str, Array]:
        """Compute sufficient statistics T(x) from data x."""
        return {
            "x": x,                                    # T₁(x) = x
            "xxT": x[..., None] * x[..., None, :]      # T₂(x) = x⊗x (outer product)
        }
    
    def x_bounds(self) -> Optional[Tuple[Array, Array]]:
        """Define domain constraints for x (lower and upper bounds)."""
        # Example: x must be in [0, ∞) for some distributions
        lower = jnp.zeros(self.x_shape)
        upper = jnp.full(self.x_shape, jnp.inf)
        return (lower, upper)
    
    def x_constraint_fn(self, x: Array) -> Array:
        """Optional: Additional domain constraints (e.g., x > 0)."""
        # Example: enforce x > 0
        return jnp.maximum(x, 1e-8)  # Clip to small positive value
    
    def eta_bounds(self) -> Optional[Dict[str, Tuple[float, float]]]:
        """Define parameter constraints for natural parameters η."""
        return {
            "x": (-jnp.inf, jnp.inf),      # η₁ unbounded
            "xxT": (-jnp.inf, -0.1)        # η₂ < 0 for negative definiteness
        }
```

**Required methods:**
- `x_shape`: Input data dimensions
- `stat_shapes`: Dictionary mapping statistic names to their shapes
- `_compute_stats(x)`: Compute sufficient statistics T(x) from data

**Domain definition methods:**
- `x_bounds()`: Optional bounds (lower, upper) for x values
- `x_constraint_fn(x)`: Optional function to enforce domain constraints

**Parameter constraint methods:**
- `eta_bounds()`: Optional constraints on natural parameters η
- `reparam_eta()`: Optional reparameterization for numerical stability

**Example distributions:**
- **Gaussian**: T(x) = [x, x⊗x], domain: x ∈ ℝᵈ
- **Laplace**: T(x) = [|x|], domain: x ∈ ℝᵈ  
- **Gamma**: T(x) = [log(x), x], domain: x > 0

### Step 2: Create a Sampling Configuration

Create a YAML file in `data/configs/your_distribution.yaml`:

```yaml
ef:
  name: your_distribution
  x_shape: [10]  # Input dimensions

grid:
  num_points: 10000
  eta_range:
    stat1: [-5, 5]      # Per-dimension parameter ranges
    stat2: [-5, -0.1]   # Must satisfy eta_bounds()
  batch_size: 100

sampling:
  num_samples: 2000     # MCMC samples per eta
  num_warmup: 500      # Burn-in samples
  step_size: 0.1       # HMC step size
  num_integration_steps: 10
  num_chains: 1
  parallel_strategy: sequential

optim:
  seed: 42
```

**Key parameters:**
- `ef.name`: Must match your class name
- `ef.x_shape`: Input data dimensions
- `grid.eta_range`: Dictionary with per-dimension parameter bounds
- `sampling.*`: MCMC sampling parameters (see BlackJAX documentation)

### Step 3: Generate Training Data via MCMC Sampling

Use the data generation script to create training datasets:

```bash
conda activate numpyro
python -m src.expfam.generate_data --config data/configs/your_distribution.yaml --force
```

**Output:**
- Creates `data/your_distribution_data_10000.pkl`
- Contains: `{'eta': array, 'mu_T': array, 'cov_TT': array, 'ess': array}`
- Automatic train-val-test split (80-10-10) with randomization

**Troubleshooting:**
- If sampling fails: Reduce `num_samples`, increase `step_size`, or check `eta_bounds()`
- If memory issues: Reduce `num_points` or `batch_size`

### Step 4: Create a Training Script

Use the base trainer to create your training script in `src/models/your_model/train.py`:

```python
from ..base_trainer import BaseETTrainer
from ..base_training_config import BaseTrainingConfig
from .model import YourModel, Config

class TrainingConfig(BaseTrainingConfig):
    training_switches = {
        'dropout_epochs': 150,
        'batch_size': 256,
        'learning_rate': 0.001,
    }

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to training data')
    parser.add_argument('--epochs', type=int, default=200)
    # ... add model-specific arguments
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = BaseETTrainer(
        model_class=YourModel,
        model_config=Config(),
        training_config=TrainingConfig(),
        data_path=args.data,
        output_dir=f"artifacts/your_model_{timestamp}"
    )
    
    # Train
    trainer.train(epochs=args.epochs)

if __name__ == "__main__":
    main()
```

### Step 5: Train Your Model

#### Traditional ET Models

Run the training script:

```bash
conda activate numpyro
python src/models/your_model/train.py --data data/your_distribution_data_10000.pkl --epochs 200
```

#### Flow Models (NEW)

For flow-based models, use the unified training script:

```bash
conda activate numpyro
python -m src.models.flow_models.train --data data/your_distribution_data_10000.pkl --training-protocol fm --model potential_flow --epochs 200 --learning-rate 0.00001
```

**Training features:**
- Automatic data loading and validation
- Train-val-test split (80-10-10)
- Learning curves and model saving
- Configurable hyperparameters via command line
- **Flow Models**: Support for multiple training protocols (FM, CT, DF) and architectures

## Gradient Utilities and Flow Wrappers

The `src/layers/gradnet_utils.py` module provides efficient gradient computation utilities for flow models:

### Gradient Utilities

- **`GradNetUtility`**: Computes gradients using `jax.grad`
- **`ValueAndGradNetUtility`**: Computes both values and gradients
- **`HessNetUtility`**: Computes Hessians using `jax.hessian`
- **`GradAndHessNetUtility`**: Computes both gradients and Hessians
- **`ValueAndGradAndHessNetUtility`**: Computes values, gradients, and Hessians

### Flow Wrappers

The `src/models/flow_models/crn.py` module provides flow wrapper classes:

- **`PotentialFlowWrapper`**: Implements `dz/dt = -∇_z V(z,x,t)`
- **`GeometricFlowWrapper`**: Implements `dz/dt = Σ @ x` where Σ is the Hessian
- **`NaturalFlowWrapper`**: Implements `dz/dt = Σ @ Σ^T @ x` where Σ is learned

### Key Features

- **Automatic Broadcasting**: Handles batch dimensions and scalar inputs
- **Parameter Scoping**: Proper Flax parameter management
- **Efficient Computation**: Optimized gradient and Hessian computation
- **Unified Interface**: Consistent API across all flow types

## Recent Changes to Models Directory

### Directory Structure Changes

1. **New `flow_models` Directory**: Added unified framework for flow-based models
2. **New `layers` Directory**: Added gradient computation utilities
3. **Archive Directory**: Moved legacy code to `src/models/archive/`

### Model Usage Changes

1. **Unified Training Script**: Single script supports all training protocols and model architectures
2. **Flow Wrappers**: Clean interface for different flow types (potential, geometric, natural)
3. **Gradient Utilities**: Efficient computation of gradients and Hessians for flow models
4. **Broadcasting Support**: Automatic handling of batch dimensions and scalar inputs

### Migration Guide

- **Old**: Individual training scripts for each model type
- **New**: Unified training script with protocol and model selection
- **Old**: Direct gradient computation in model classes
- **New**: Gradient utilities with automatic broadcasting and parameter scoping

### Step 6: Use Your Model for Coordinate Ascent Variational Inference

Your trained networks enable **cheap and easy Bayesian inference** through coordinate ascent on factorized variational posteriors.

#### Mathematical Framework

Consider a model of the form:
```
log p(x,y) = T(x) · A S(y)
```

With a **factorized variational approximate posterior**:
```
q(x)q(y) = exp(ηₓ · T(x) - Aₓ(ηₓ)) × exp(ηᵧ · S(y) - Aᵧ(ηᵧ))
```

#### Coordinate Ascent Algorithm

**Coordinate ascent** works by iterating along natural parameters according to:
```
ηₓ = A ⟨S(y) | ηᵧ⟩
ηᵧ = Aᵀ ⟨T(x) | ηₓ⟩
```

**Your trained networks compute these expectations:**
- **T-network**: Computes `⟨T(x) | ηₓ⟩` 
- **S-network**: Computes `⟨S(y) | ηᵧ⟩`

#### Implementation

```python
import jax
import jax.numpy as jnp
from your_trained_models import load_t_network, load_s_network

# Load your trained networks
t_network, t_params = load_t_network("artifacts/t_network_20240101_120000")
s_network, s_params = load_s_network("artifacts/s_network_20240101_120000")

def coordinate_ascent_inference(eta_x_init, eta_y_init, A_matrix, max_iter=100, tol=1e-6):
    """
    Coordinate ascent variational inference using trained networks.
    
    Args:
        eta_x_init: Initial natural parameters for x
        eta_y_init: Initial natural parameters for y  
        A_matrix: Coupling matrix A
        max_iter: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        eta_x_final, eta_y_final: Converged natural parameters
    """
    eta_x, eta_y = eta_x_init, eta_y_init
    
    for iteration in range(max_iter):
        # Store previous values for convergence check
        eta_x_old, eta_y_old = eta_x, eta_y
        
        # Update ηₓ: ηₓ = A ⟨S(y) | ηᵧ⟩
        mu_S = s_network.apply(s_params, eta_y)  # ⟨S(y) | ηᵧ⟩
        eta_x = A_matrix @ mu_S
        
        # Update ηᵧ: ηᵧ = Aᵀ ⟨T(x) | ηₓ⟩  
        mu_T = t_network.apply(t_params, eta_x)   # ⟨T(x) | ηₓ⟩
        eta_y = A_matrix.T @ mu_T
        
        # Check convergence
        x_converged = jnp.allclose(eta_x, eta_x_old, atol=tol)
        y_converged = jnp.allclose(eta_y, eta_y_old, atol=tol)
        
        if x_converged and y_converged:
            print(f"Converged after {iteration + 1} iterations")
            break
    
    return eta_x, eta_y

# Example usage
eta_x_init = jnp.array([0.0, 0.0])  # Initial x parameters
eta_y_init = jnp.array([0.0, 0.0])  # Initial y parameters
A = jnp.array([[1.0, 0.5], [0.5, 1.0]])  # Coupling matrix

eta_x_final, eta_y_final = coordinate_ascent_inference(eta_x_init, eta_y_init, A)
print(f"Final parameters: η_x = {eta_x_final}, η_y = {eta_y_final}")
```

#### Key Advantages

- **Cheap Inference**: No expensive MCMC sampling during inference
- **Fast Convergence**: Coordinate ascent typically converges in about 20 iterations
- **Scalable**: Works for high-dimensional problems
- **Flexible**: Handles arbitrary exponential family distributions

**This enables cheap and easy Bayesian inference for complex models!** 🚀
**You can even let define `T(x)` by a static neural network if you like!** 😱


## Data Format (Single Source)

All generated datasets must be a dict with exactly:

```python
{
  'eta':  <array>,
  'mu_T': <array>,
  'cov_TT': <array>,
  'ess': <array>
}
```

Old pre-split formats are not supported.

## Notes on EF Implementations

- `flattened_log_density_fn(η)` returns a callable for batches of flattened `x`.
- Constraints (`x_bounds`, `x_constraint_fn`) are enforced inside `log_unnormalized` in a JAX‑compatible way.
- Multivariate normals share a unified `_reparam_eta` logic to ensure negative‑definite precision; `MultivariateNormal_tril` accepts `xxT_tril` at the interface, converts internally to full, reuses the same core logic, and returns tril format where appropriate.
