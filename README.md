# nnef-dists

Fast variational inference with non‑named exponential-family distributions using neural networks to learn the mapping between natural parameters and expected sufficient statistics. Features both direct ET networks and geometric-flow approaches, plus a unified data-generation and training pipeline.

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
    └── noprop_*                # NoProp variants

```

- Data configs for sampling are in `data/configs/*.yaml`.
- Archive/legacy directories under `src/models/archive/` and `src/training/` should be ignored for new development.

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

Run the training script:

```bash
conda activate numpyro
python src/models/your_model/train.py --data data/your_distribution_data_10000.pkl --epochs 200
```

**Training features:**
- Automatic data loading and validation
- Train-val-test split (80-10-10)
- Learning curves and model saving
- Configurable hyperparameters via command line

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
