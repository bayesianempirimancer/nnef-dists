# NoProp-CT Implementation for Exponential Family Moment Mapping

This directory contains a complete implementation of continuous-time NoProp (NoProp-CT) for learning the mapping from natural parameters to expected sufficient statistics in exponential family distributions.

## Overview

The NoProp-CT approach treats neural network training as a continuous-time denoising problem. Instead of traditional backpropagation, the network is modeled as a Neural ODE that evolves from noisy initial conditions toward the correct moment predictions through a learned vector field.

### Key Features

- **Neural ODE Framework**: Implements continuous-time dynamics using Euler and RK4 solvers
- **Denoising Training**: Learns to map from noisy inputs to clean moment predictions
- **Consistency Regularization**: Ensures similar inputs produce similar trajectories
- **Exponential Family Integration**: Works seamlessly with existing EF implementations
- **Comprehensive Visualization**: Tools for analyzing training dynamics and ODE trajectories

## Files Structure

```
src/
├── noprop_ct.py              # Core NoProp-CT implementation
scripts/
├── run_noprop_ct_demo.py     # End-to-end demonstration
├── compare_noprop_ct.py      # Comparison with standard MLP
├── visualize_noprop_dynamics.py  # Visualization tools
├── generate_comparison_data.py    # Data generation for comparisons
configs/
├── noprop_ct_comparison.yaml # Configuration for comparisons
```

## Quick Start

### 1. Run a Simple Demo

```bash
# 1D Gaussian demonstration
python scripts/run_noprop_ct_demo.py --ef-type gaussian_1d --num-samples 2000 --num-epochs 100

# 2D Multivariate Normal demonstration  
python scripts/run_noprop_ct_demo.py --ef-type multivariate_2d --num-samples 2000 --num-epochs 100
```

### 2. Compare with Standard MLP

```bash
# Generate comparison data
python scripts/generate_comparison_data.py --config configs/noprop_ct_comparison.yaml

# Run comparison
python scripts/compare_noprop_ct.py \
    --config configs/noprop_ct_comparison.yaml \
    --data-path data/noprop_ct_comparison.pkl \
    --epochs 200 \
    --output-dir artifacts/comparison
```

### 3. Visualize Training Dynamics

```bash
# After running comparison, visualize results
python scripts/visualize_noprop_dynamics.py \
    --results artifacts/comparison/comparison_results.pkl \
    --output-dir artifacts/noprop_visualizations \
    --create-animation
```

## Core Concepts

### Neural ODE Architecture

The NoProp-CT model defines a continuous-time dynamical system:

```
dx/dt = f(x, η, t; θ)
```

Where:
- `x(t)` is the network state at time t
- `η` are the natural parameters (input)
- `f` is a neural network (vector field)
- `θ` are the learnable parameters

### Training Process

1. **Initialization**: Start from noisy version of input: `x(0) = η + ε`
2. **Evolution**: Integrate ODE from t=0 to t=T using learned vector field
3. **Loss Computation**: 
   - Denoising loss: `||x(T) - μ(η)||²` where `μ(η)` are true moments
   - Consistency loss: Similar inputs should have similar trajectories
4. **Parameter Update**: Update vector field parameters to minimize total loss

### Configuration

The `NoPropCTConfig` class controls all aspects of the model:

```python
config = NoPropCTConfig(
    hidden_sizes=(64, 64, 32),      # Vector field network architecture
    activation="swish",              # Activation function
    noise_scale=0.1,                # Initial noise magnitude
    time_horizon=1.0,               # Integration time T
    num_time_steps=10,              # ODE solver steps
    ode_solver="euler",             # "euler" or "rk4"
    learning_rate=1e-3,             # Adam learning rate
    denoising_weight=1.0,           # Weight for denoising loss
    consistency_weight=0.1,         # Weight for consistency loss
)
```

## Usage Examples

### Basic Training

```python
from src.ef import GaussianNatural1D
from src.noprop_ct import NoPropCTMomentNet, NoPropCTConfig, train_noprop_ct_moment_net

# Create exponential family
ef = GaussianNatural1D()

# Configure model
config = NoPropCTConfig(
    hidden_sizes=(64, 32),
    time_horizon=1.0,
    num_time_steps=10,
)

# Train model
state, history = train_noprop_ct_moment_net(
    ef=ef,
    train_data=train_data,
    val_data=val_data,
    config=config,
    num_epochs=100,
    batch_size=64,
    seed=42,
)
```

### Custom Vector Field

```python
class CustomVectorField(nn.Module):
    @nn.compact
    def __call__(self, state, eta, time):
        # Custom vector field implementation
        x = jnp.concatenate([state, eta, time], axis=-1)
        # ... custom network architecture ...
        return dx_dt

# Use in NoPropCTMomentNet
model = NoPropCTMomentNet(ef=ef, config=config)
model.vector_field = CustomVectorField()
```

### Inference

```python
# Load trained model
model = NoPropCTMomentNet(ef=ef, config=config)

# Predict moments for new natural parameters
eta_new = jnp.array([[1.0, -0.5]])  # Example natural parameters
predicted_moments = model.apply(state.params, eta_new, training=False)
```

## Comparison Results

The comparison scripts will generate comprehensive analysis including:

- **Training Curves**: Loss evolution for both methods
- **Test Performance**: MSE, MAE, and component-wise analysis  
- **Training Time**: Computational efficiency comparison
- **Convergence Analysis**: Stability and convergence properties

### Expected Performance

Based on the NoProp-CT paper, you can expect:

- **Similar or better accuracy** compared to standard backpropagation
- **More stable training** due to the continuous-time formulation
- **Better generalization** from the denoising training process
- **Slightly higher computational cost** due to ODE integration

## Visualization Tools

### Training Dynamics

- Loss component evolution (denoising vs consistency)
- Training stability analysis
- Loss ratio analysis over time

### ODE Trajectories

- State space trajectory visualization
- Convergence analysis over time
- Vector field magnitude evolution

### 2D Vector Fields

- Vector field visualization at different time points
- Trajectory animations for multiple starting points
- Phase portrait analysis

## Advanced Usage

### Custom ODE Solvers

```python
class AdaptiveRK45Solver:
    @staticmethod
    def integrate(vector_field, initial_state, eta, time_span, **kwargs):
        # Custom adaptive solver implementation
        pass

# Use custom solver
NeuralODESolver.integrate = AdaptiveRK45Solver.integrate
```

### Multi-Scale Time Integration

```python
config = NoPropCTConfig(
    time_horizon=2.0,           # Longer integration
    num_time_steps=50,          # More steps for accuracy
    ode_solver="rk4",           # Higher-order method
)
```

### Regularization Techniques

```python
config = NoPropCTConfig(
    consistency_weight=0.5,     # Stronger consistency regularization
    noise_scale=0.2,            # Higher noise for robustness
)
```

## Troubleshooting

### Common Issues

1. **Numerical Instability**
   - Reduce `noise_scale`
   - Increase `num_time_steps`
   - Use `"rk4"` instead of `"euler"`

2. **Slow Convergence**
   - Increase `learning_rate`
   - Adjust `denoising_weight` vs `consistency_weight` ratio
   - Reduce `time_horizon`

3. **Memory Issues**
   - Reduce `batch_size`
   - Decrease `num_time_steps`
   - Use gradient checkpointing (not implemented)

### Performance Tips

- Start with `"euler"` solver for faster training, switch to `"rk4"` for final runs
- Use smaller `time_horizon` (0.5-1.0) for most problems
- Balance `denoising_weight` and `consistency_weight` based on your problem
- Monitor loss components separately to diagnose training issues

## References

- Li, Q., Teh, Y. W., & Pascanu, R. (2025). NoProp: Training Neural Networks without Full Back-propagation or Full Forward-propagation. arXiv:2503.24322
- Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). Neural ordinary differential equations. NeurIPS.

## Contributing

To extend the NoProp-CT implementation:

1. Add new ODE solvers in `NeuralODESolver`
2. Implement custom vector field architectures
3. Add new loss functions and regularization terms
4. Extend visualization tools for new analysis types

The modular design makes it easy to experiment with different components while maintaining compatibility with the existing exponential family framework.
