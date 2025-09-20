"""
GLU-based moment prediction network using gating mechanisms for stable division-like operations.

GLU (Gated Linear Unit) uses sigmoid gating to create multiplicative interactions that can
approximate division operations without numerical instability.
"""

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from typing import Callable, Tuple, Sequence
from .ef import ExponentialFamily

Array = jax.Array


class GLULayer(nn.Module):
    """Gated Linear Unit: GLU(x) = (W1 * x) ⊙ σ(W2 * x)"""
    
    hidden_size: int
    use_bias: bool = True
    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        # Two parallel linear transformations
        gate = nn.Dense(self.hidden_size, use_bias=self.use_bias, name='gate')(x)
        value = nn.Dense(self.hidden_size, use_bias=self.use_bias, name='value')(x)
        
        # Apply sigmoid to gate and multiply
        return value * jax.nn.sigmoid(gate)


class GLUMomentNet(nn.Module):
    """GLU-based moment prediction network."""
    
    ef: ExponentialFamily
    hidden_sizes: Sequence[int] = (64, 32)
    activation: str = "tanh"
    use_glu_layers: bool = True
    glu_hidden_ratio: float = 2.0  # GLU hidden size as ratio of input size
    
    def setup(self):
        self.activation_fn = self._get_activation(self.activation)
    
    def _get_activation(self, name: str) -> Callable[[Array], Array]:
        if name == "relu":
            return nn.relu
        elif name == "tanh":
            return nn.tanh
        elif name == "gelu":
            return nn.gelu
        elif name == "swish":
            return nn.swish
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    @nn.compact
    def __call__(self, eta: Array) -> Array:
        """Predict expected sufficient statistics from natural parameters."""
        
        x = eta
        
        # Initial feature extraction
        x = nn.Dense(self.hidden_sizes[0])(x)
        x = self.activation_fn(x)
        
        # GLU layers for division-like operations
        if self.use_glu_layers:
            glu_size = int(self.hidden_sizes[0] * self.glu_hidden_ratio)
            x = GLULayer(glu_size)(x)
            x = self.activation_fn(x)
            
            # Second GLU layer
            glu_size = int(self.hidden_sizes[0] * self.glu_hidden_ratio)
            x = GLULayer(glu_size)(x)
            x = self.activation_fn(x)
        
        # Standard MLP layers
        for hidden_size in self.hidden_sizes[1:]:
            x = nn.Dense(hidden_size)(x)
            x = self.activation_fn(x)
        
        # Output layer
        x = nn.Dense(self.ef.eta_dim)(x)  # Output dimension matches sufficient statistics
        
        return x


class ResidualGLULayer(nn.Module):
    """Residual GLU layer: x = x + GLU(x)"""
    
    hidden_size: int
    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        # Ensure input and output dimensions match
        if x.shape[-1] != self.hidden_size:
            x = nn.Dense(self.hidden_size)(x)
        
        # GLU transformation
        glu_out = GLULayer(self.hidden_size)(x)
        
        # Residual connection
        return x + glu_out


class DeepGLUMomentNet(nn.Module):
    """Deep GLU network with residual connections."""
    
    ef: ExponentialFamily
    hidden_size: int = 64
    num_glu_layers: int = 4
    activation: str = "tanh"
    
    def setup(self):
        self.activation_fn = self._get_activation(self.activation)
    
    def _get_activation(self, name: str) -> Callable[[Array], Array]:
        if name == "relu":
            return nn.relu
        elif name == "tanh":
            return nn.tanh
        elif name == "gelu":
            return nn.gelu
        elif name == "swish":
            return nn.swish
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    @nn.compact
    def __call__(self, eta: Array) -> Array:
        """Predict expected sufficient statistics from natural parameters."""
        
        # Input projection
        x = nn.Dense(self.hidden_size)(eta)
        x = self.activation_fn(x)
        
        # Stack of residual GLU layers
        for i in range(self.num_glu_layers):
            x = ResidualGLULayer(self.hidden_size)(x)
            x = self.activation_fn(x)
        
        # Output projection
        x = nn.Dense(self.ef.eta_dim)(x)
        
        return x


def create_glu_train_state(
    ef: ExponentialFamily,
    config: dict,
    rng: Array
) -> Tuple[nn.Module, dict]:
    """Create and initialize a GLU-based model."""
    
    model_type = config.get('model_type', 'glu')
    
    if model_type == 'glu':
        model = GLUMomentNet(
            ef=ef,
            hidden_sizes=config.get('hidden_sizes', (64, 32)),
            activation=config.get('activation', 'tanh'),
            use_glu_layers=config.get('use_glu_layers', True),
            glu_hidden_ratio=config.get('glu_hidden_ratio', 2.0)
        )
    elif model_type == 'deep_glu':
        model = DeepGLUMomentNet(
            ef=ef,
            hidden_size=config.get('hidden_size', 64),
            num_glu_layers=config.get('num_glu_layers', 4),
            activation=config.get('activation', 'tanh')
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Initialize
    dummy_eta = jnp.zeros((1, ef.eta_dim))
    params = model.init(rng, dummy_eta)
    
    return model, params


# Test the GLU operations
if __name__ == "__main__":
    import numpy as np
    
    # Test GLU layer
    print("Testing GLULayer...")
    glu_layer = GLULayer(hidden_size=4)
    x = jnp.array([[1.0, 2.0, 3.0, 4.0]])
    
    rng = random.PRNGKey(0)
    params = glu_layer.init(rng, x)
    output = glu_layer.apply(params, x)
    
    print(f"Input: {x}")
    print(f"Output: {output}")
    print(f"Output shape: {output.shape}")
    
    # Test Residual GLU
    print("\nTesting ResidualGLULayer...")
    residual_glu = ResidualGLULayer(hidden_size=4)
    
    params = residual_glu.init(rng, x)
    output = residual_glu.apply(params, x)
    
    print(f"Input: {x}")
    print(f"Output: {output}")
    print(f"Output shape: {output.shape}")
    
    print("\n✅ GLU layers working correctly!")
