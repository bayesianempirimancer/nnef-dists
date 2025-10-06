# Model Interface Specification

All models in this directory (`src/models`) must inherit from `BaseETModel` and implement the following methods:

## Base Class

All models inherit from `BaseETModel[ConfigType]` which provides:
- Standard interface methods (`__call__`, `predict`, `loss`)
- Common functionality (`from_config`, `save_pretrained`, `get_config`)
- Type safety with generic configuration types

## Required Methods

### `__call__(self, eta, training=True, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]`
- **Purpose**: Main forward pass where the model is defined and executed
- **Input**: 
  - `eta`: Natural parameters of shape (batch_size, eta_dim)
  - `training`: Whether in training mode (affects dropout, etc.)
  - `**kwargs`: Additional model-specific arguments
- **Output**: Tuple of (predictions, internal_loss)
  - `predictions`: Model output of shape (batch_size, output_dim)
  - `internal_loss`: Internal regularization loss (scalar, usually 0.0)

### `predict(self, params, eta, rngs=None, **kwargs) -> jnp.ndarray`
- **Purpose**: Make predictions for inference
- **Input**:
  - `params`: Model parameters
  - `eta`: Natural parameters of shape (batch_size, eta_dim)
  - `rngs`: Random number generator keys for stochastic operations (optional)
  - `**kwargs`: Additional model-specific arguments
- **Output**: Predictions of shape (batch_size, output_dim)

### `loss(self, params, eta, targets, training=True, rngs=None, **kwargs) -> jnp.ndarray`
- **Purpose**: Compute training loss
- **Input**:
  - `params`: Model parameters
  - `eta`: Natural parameters of shape (batch_size, eta_dim)
  - `targets`: Target values of shape (batch_size, output_dim)
  - `training`: Whether in training mode
  - `rngs`: Random number generator keys for stochastic operations (optional)
  - `**kwargs`: Additional model-specific arguments
- **Output**: Loss value (scalar)

## Design Principles

1. **Simplicity**: No complex inheritance, just implement the three methods
2. **Self-contained**: Each model handles its own computation, loss, and prediction
3. **Consistent**: All models follow the same interface pattern
4. **Flexible**: `**kwargs` allows model-specific arguments as needed

## Example Structure

```python
from .base_model import BaseETModel

class MyETModel(BaseETModel[MyConfig]):
    config: MyConfig
    
    @nn.compact
    def __call__(self, eta, training=True, rngs=None, **kwargs):
        # Model definition and execution
        predictions = self._my_model_logic(eta, training, rngs, **kwargs)
        internal_loss = jnp.array(0.0)  # Usually 0.0
        return predictions, internal_loss
    
    def predict(self, params, eta, **kwargs):
        # Inference method
        predictions, _ = self.apply(params, eta, training=False, **kwargs)
        return predictions
    
    def loss(self, params, eta, targets, training=True, rngs=None, **kwargs):
        # Training loss computation
        predictions, internal_loss = self.apply(params, eta, training=training, rngs=rngs, **kwargs)
        mse_loss = jnp.mean((predictions - targets) ** 2)
        return mse_loss + internal_loss
```
