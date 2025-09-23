# Log Normalizer Neural Network Approach

## Overview

This document describes the implementation of a novel approach for learning exponential family distributions using neural networks that directly approximate the log normalizer function A(η). This approach leverages the fundamental relationship between the log normalizer and the moments of exponential family distributions through automatic differentiation.

## Theoretical Foundation

For exponential family distributions with natural parameters η and sufficient statistics T(x):

```
p(x|η) = h(x) exp(ηᵀT(x) - A(η))
```

The key relationships are:
- **Mean**: E[T(X)] = ∇A(η) (gradient of log normalizer)
- **Covariance**: Cov[T(X)] = ∇²A(η) (Hessian of log normalizer)
- **Convexity**: A(η) is convex (ensures positive definite covariance)

## Key Advantages

1. **Theoretical Soundness**: Directly learns the fundamental quantity A(η)
2. **Automatic Constraint Enforcement**: Convexity of A(η) ensures valid covariances
3. **Unified Representation**: Single network learns both mean and covariance relationships
4. **Potential for Better Generalization**: Shared representation may improve performance
5. **Enables Geometric Loss Functions**: q(T(X)|η) ~ N(∇A(η), ∇²A(η)) can be used with empirical distribution, p(T(X)), 
                                                                      and KL based loss functions  

## Implementation
    1. Parameterize a neural network that takes in η and returns a scalar value (may use convex NN)
    2. Use autograd to compute gradient and hessian
    3. Train with MSE or KL loss functions
    4. Compare with similiar networks that use MSE to estimate E[T(X)] directly

### Core Components

#### 1. LogNormalizerNetwork (`src/models/log_normalizer.py`)
- Neural network that outputs scalar log normalizer A(η)
- Supports feature engineering for complex mappings
- Configurable architecture with various activation functions

#### 2. Automatic Differentiation Functions
- `compute_log_normalizer_gradient()`: Computes ∇A(η) for mean prediction
- `compute_log_normalizer_hessian()`: Computes ∇²A(η) for covariance prediction
- Supports multiple Hessian computation methods (full, diagonal, block)

#### 3. Loss Functions
- `log_normalizer_loss_fn()`: Combines mean and covariance losses
- Supports weighted loss components
- Includes regularization for numerical stability

#### 4. Stable Implementation (`src/logZ_grads.py`)
- Enhanced numerical stability measures
- Adaptive loss weights during training
- Gradient clipping and regularization
- Multiple Hessian computation strategies

### Training Scripts

#### 1. Main Training Script (`scripts/training/train_log_normalizer.py`)
- Complete training pipeline for log normalizer networks
- Comparison with baseline models
- Comprehensive evaluation metrics

#### 2. Test Script (`scripts/test_log_normalizer.py`)
- Verification of basic functionality
- Tests on 1D and 3D Gaussian examples
- Training simulation

## Usage

### Basic Training

```python
from models.log_normalizer import LogNormalizerTrainer
from config import load_config

# Load configuration
config = load_config("data/configs/gaussian_1d.yaml")

# Create trainer
trainer = LogNormalizerTrainer(config, hessian_method='diagonal')

# Prepare data
train_data = prepare_log_normalizer_data(eta_data, mean_data, cov_data)

# Train model
params, history = trainer.train(train_data, val_data)
```

### Stable Training

```python
from logZ_grads import StableLogNormalizerTrainer

# Create stable trainer with adaptive weights
trainer = StableLogNormalizerTrainer(
    config, 
    hessian_method='diagonal',
    adaptive_weights=True
)

# Train with stability measures
params, history = trainer.train(train_data, val_data)
```

### Command Line Usage

```bash
# Train log normalizer model
python scripts/training/train_log_normalizer.py --config data/configs/gaussian_1d.yaml

# Use stable implementation
python scripts/training/train_log_normalizer.py --config data/configs/gaussian_1d.yaml --stable

# Compare with baseline
python scripts/training/train_log_normalizer.py --config data/configs/gaussian_1d.yaml --compare
```

## Numerical Considerations

### Potential Issues

1. **Hessian Computation**: Full Hessian computation is O(d³) where d is parameter dimension
2. **Numerical Stability**: Higher-order derivatives can be numerically unstable
3. **Training Complexity**: More complex loss function may require careful tuning

### Stability Measures

1. **Gradient Clipping**: Prevents extreme gradients during training
2. **Regularization**: Adds small values to ensure positive definiteness
3. **Adaptive Loss Weights**: Start with mean-only training, gradually add covariance loss
4. **Multiple Hessian Methods**: Diagonal approximation for efficiency, full Hessian for accuracy

## Experimental Results

### Expected Benefits

1. **Better Mean Estimation**: Direct learning of ∇A(η) may improve mean predictions
2. **Valid Covariance Structure**: Automatic enforcement of positive definiteness
3. **Improved Generalization**: Shared representation for mean and covariance

### Comparison with Current Approach

| Aspect | Current (Direct) | Log Normalizer |
|--------|------------------|----------------|
| Output | Mean + Covariance | Log Normalizer (scalar) |
| Derivatives | Network Jacobian | ∇A(η), ∇²A(η) |
| Constraints | Manual | Automatic (convexity) |
| Complexity | Medium | Higher (Hessian) |
| Stability | Good | Requires care |

## Future Directions

1. **Advanced Hessian Methods**: Implement Lanczos approximation for large-scale problems
2. **Distribution-Specific Optimizations**: Tailor the approach to specific exponential families
3. **Uncertainty Quantification**: Use the learned A(η) for confidence intervals
4. **Multi-Scale Training**: Curriculum learning from simple to complex distributions

## Files Created

- `src/models/log_normalizer.py`: Core implementation
- `src/logZ_grads.py`: Stable implementation with numerical measures
- `scripts/training/train_log_normalizer.py`: Training script
- `scripts/test_log_normalizer.py`: Test script
- `docs/log_normalizer_approach.md`: This documentation

## Testing

To test the implementation:

```bash
# Run basic tests
python scripts/test_log_normalizer.py

# Train on 1D Gaussian
python scripts/training/train_log_normalizer.py --config data/configs/gaussian_1d.yaml

# Train on 3D multivariate Gaussian
python scripts/training/train_log_normalizer.py --config data/configs/multivariate_3d.yaml
```

## Conclusion

The log normalizer approach represents a theoretically grounded alternative to the current direct prediction approach. While it introduces additional complexity in terms of Hessian computation and numerical stability, it offers several potential advantages:

1. **Theoretical Foundation**: Direct learning of the fundamental quantity A(η)
2. **Constraint Enforcement**: Automatic positive definiteness of covariances
3. **Unified Learning**: Single network for both mean and covariance

The implementation includes both basic and stable versions, comprehensive training scripts, and extensive testing capabilities. The approach is ready for experimental evaluation against the current baseline methods.
