# Quadratic ResNet Log Normalizer Approach

## Overview

This document describes the implementation of a Quadratic ResNet architecture for learning the log normalizer function A(η) of exponential families. This approach combines the theoretical foundation of log normalizer learning with advanced neural network architectures featuring quadratic residual connections.

## Architecture

### Quadratic Residual Block

The core innovation is the quadratic residual block that computes:
```
output = x + f(x) + g(x)²
```

Where:
- `x` is the input (residual connection)
- `f(x)` is a linear transformation with activation
- `g(x)²` is a quadratic term that captures curvature relationships

This design is particularly suited for learning log normalizers because:
1. **Residual connections** help with gradient flow in deep networks
2. **Quadratic terms** naturally capture the curvature needed for Hessian computation
3. **Terminal scalar projection** ensures clean output for log normalizer

### Network Structure

```
Input (η) → Feature Engineering → Input Projection → 
[Quadratic ResBlock₁ → Quadratic ResBlock₂ → ...] → 
Pre-final Layer → Final Scalar Output (A(η))
```

## Key Features

### 1. Quadratic Residual Connections
- **Linear term**: `f(x)` with activation function
- **Quadratic term**: `g(x)²` for curvature modeling
- **Residual connection**: `x + f(x) + g(x)²`

### 2. Terminal Scalar Projection
- Sophisticated final layers for smooth scalar output
- Pre-final layer with tanh activation
- Single scalar output representing log normalizer

### 3. Curriculum Learning
- Start with mean-only training (epochs 0-50)
- Gradual transition to include covariance loss (epochs 50-100)
- Full training with both losses (epochs 100+)

### 4. Gradient Stability
- Gradient clipping (-0.5, 0.5) for ResNet stability
- Adaptive loss weights during training
- Numerical stability measures

## Implementation

### Core Classes

#### `QuadraticResidualBlock`
- Implements the quadratic residual connection
- Configurable activation functions
- Optional dropout for regularization

#### `QuadraticResNetLogNormalizer`
- Main network architecture
- Multiple quadratic residual blocks
- Terminal scalar projection

#### `QuadraticResNetLogNormalizerTrainer`
- Specialized trainer with curriculum learning
- Gradient clipping for stability
- Comprehensive evaluation metrics

## Comparison with Basic LogNormalizer

| Aspect | Basic LogNormalizer | Quadratic ResNet |
|--------|-------------------|------------------|
| Architecture | Simple MLP | Quadratic residual blocks |
| Gradient Flow | Standard | Enhanced with residuals |
| Curvature Modeling | Implicit | Explicit quadratic terms |
| Training Stability | Good | Enhanced with clipping |
| Parameter Count | Lower | Higher (due to residual connections) |
| Training Complexity | Simple | Curriculum learning |

## Usage

### Basic Usage

```python
from models.quadratic_resnet_log_normalizer import QuadraticResNetLogNormalizerTrainer
from config import load_config

# Create trainer
trainer = QuadraticResNetLogNormalizerTrainer(
    config, 
    hessian_method='diagonal',
    use_curriculum=True
)

# Train model
params, history = trainer.train(train_data, val_data)
```

### Configuration

```python
config = NetworkConfig()
config.hidden_sizes = [96, 64, 32]  # ResNet block sizes
config.output_dim = 1  # Scalar log normalizer
config.use_feature_engineering = True
config.activation = "tanh"
config.dropout_rate = 0.0  # No dropout for stability
```

## Testing and Comparison

### Test Scripts

1. **`test_3d_gaussian_comparison.py`**: Simple comparison test
2. **`compare_log_normalizer_approaches.py`**: Full training comparison

### 3D Gaussian Case

The comparison focuses on 3D multivariate Gaussian distributions:
- **Input**: 12 natural parameters (3 mean + 9 covariance)
- **Output**: Expected sufficient statistics (12 values)
- **Evaluation**: Mean MSE and Hessian (covariance) MSE

### Expected Benefits

1. **Better Mean Estimation**: Quadratic terms may improve mean predictions
2. **Improved Hessian Learning**: Explicit curvature modeling for covariance
3. **Training Stability**: Residual connections help with deep networks
4. **Gradient Flow**: Better backpropagation through residual connections

## Files Created

- `src/models/quadratic_resnet_log_normalizer.py`: Core implementation
- `scripts/test_3d_gaussian_comparison.py`: Simple comparison test
- `scripts/compare_log_normalizer_approaches.py`: Full training comparison
- `docs/quadratic_resnet_log_normalizer.md`: This documentation

## Running the Comparison

### Simple Test
```bash
python scripts/test_3d_gaussian_comparison.py
```

### Full Training Comparison
```bash
python scripts/compare_log_normalizer_approaches.py
```

## Results and Analysis

The comparison script provides:

1. **Training Progress**: Loss curves for both approaches
2. **Mean Prediction Accuracy**: MSE comparison
3. **Covariance Prediction**: Hessian MSE comparison
4. **Training Efficiency**: Time vs accuracy trade-offs
5. **Model Complexity**: Parameter count comparison

## Future Directions

1. **Advanced Residual Blocks**: More sophisticated quadratic connections
2. **Attention Mechanisms**: For better parameter interaction modeling
3. **Multi-Scale Architecture**: Different resolutions for different parameter types
4. **Uncertainty Quantification**: Using learned A(η) for confidence intervals

## Conclusion

The Quadratic ResNet approach represents an advanced architecture specifically designed for log normalizer learning. By combining residual connections with quadratic terms, it aims to better capture the complex relationships between natural parameters and the moments of exponential family distributions.

The curriculum learning approach and gradient stability measures make it robust for training, while the explicit curvature modeling through quadratic terms may provide better performance on covariance prediction tasks.
