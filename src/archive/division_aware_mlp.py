"""
Division-aware MLP that can easily learn functions like 1/x and x/y.

This addresses the architectural bias against division operations in standard neural networks.
"""

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from typing import Callable, Tuple
from .ef import ExponentialFamily

Array = jax.Array


class DivisionLayer(nn.Module):
    """A layer that can learn division operations: output = a / (b + epsilon)."""
    
    epsilon: float = 1e-6
    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        # Split input into numerator and denominator components
        n_features = x.shape[-1]
        assert n_features % 2 == 0, "Input dimension must be even for division layer"
        
        mid = n_features // 2
        numerator = x[..., :mid]
        denominator = x[..., mid:]
        
        # Learn division: output = numerator / (denominator + epsilon)
        # Add learnable bias to denominator for flexibility
        bias = self.param('bias', nn.initializers.zeros, (mid,))
        denominator_safe = denominator + bias + self.epsilon
        
        return numerator / denominator_safe


class ReciprocalLayer(nn.Module):
    """A layer that can learn reciprocal operations: output = 1 / (x + epsilon)."""
    
    epsilon: float = 1e-6
    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        # Learn reciprocal with learnable bias
        bias = self.param('bias', nn.initializers.zeros, x.shape[-1:])
        x_safe = x + bias + self.epsilon
        return 1.0 / x_safe


class DivisionAwareMLP(nn.Module):
    """MLP with explicit division and reciprocal operations."""
    
    hidden_sizes: Tuple[int, ...]
    activation: str = "tanh"
    use_division_layers: bool = True
    use_reciprocal_layers: bool = True
    
    def setup(self):
        self.activation_fn = self._get_activation(self.activation)
    
    def _get_activation(self, name: str) -> Callable[[Array], Array]:
        if name == "relu":
            return nn.relu
        elif name == "tanh":
            return nn.tanh
        elif name == "gelu":
            return nn.gelu
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        # Standard MLP layers
        for hidden_size in self.hidden_sizes:
            x = nn.Dense(hidden_size)(x)
            x = self.activation_fn(x)
        
        # Division-aware layers
        if self.use_division_layers:
            # Ensure input dimension is even for division
            if x.shape[-1] % 2 != 0:
                x = nn.Dense(x.shape[-1] + 1)(x)  # Add one dimension if odd
            
            x = DivisionLayer()(x)
            x = self.activation_fn(x)
        
        if self.use_reciprocal_layers:
            x = ReciprocalLayer()(x)
            x = self.activation_fn(x)
        
        # Final output layer
        x = nn.Dense(2)(x)  # For Gaussian 1D: [E[x], E[x^2]]
        
        return x


class DivisionAwareMomentNet(nn.Module):
    """Division-aware moment prediction network."""
    
    ef: ExponentialFamily
    hidden_sizes: Tuple[int, ...] = (64, 32)
    activation: str = "tanh"
    use_division_layers: bool = True
    use_reciprocal_layers: bool = True
    
    @nn.compact
    def __call__(self, eta: Array) -> Array:
        """Predict expected sufficient statistics from natural parameters."""
        # Use division-aware MLP
        mlp = DivisionAwareMLP(
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
            use_division_layers=self.use_division_layers,
            use_reciprocal_layers=self.use_reciprocal_layers
        )
        
        return mlp(eta)


def create_division_aware_train_state(
    ef: ExponentialFamily,
    config: dict,
    rng: Array
) -> Tuple[nn.Module, dict]:
    """Create and initialize a division-aware model."""
    
    model = DivisionAwareMomentNet(
        ef=ef,
        hidden_sizes=config.get('hidden_sizes', (64, 32)),
        activation=config.get('activation', 'tanh'),
        use_division_layers=config.get('use_division_layers', True),
        use_reciprocal_layers=config.get('use_reciprocal_layers', True)
    )
    
    # Initialize
    dummy_eta = jnp.zeros((1, ef.eta_dim))
    params = model.init(rng, dummy_eta)
    
    return model, params


# Test the division operations
if __name__ == "__main__":
    import numpy as np
    
    # Test division layer
    print("Testing DivisionLayer...")
    division_layer = DivisionLayer()
    x = jnp.array([[1.0, 2.0, 3.0, 4.0]])  # [a, b, c, d] -> [a/c, b/d]
    
    rng = random.PRNGKey(0)
    params = division_layer.init(rng, x)
    output = division_layer.apply(params, x)
    
    print(f"Input: {x}")
    print(f"Output: {output}")
    print(f"Expected: [{1.0/3.0}, {2.0/4.0}] = [0.333, 0.5]")
    
    # Test reciprocal layer
    print("\nTesting ReciprocalLayer...")
    reciprocal_layer = ReciprocalLayer()
    x = jnp.array([[2.0, 4.0, 8.0]])
    
    params = reciprocal_layer.init(rng, x)
    output = reciprocal_layer.apply(params, x)
    
    print(f"Input: {x}")
    print(f"Output: {output}")
    print(f"Expected: [0.5, 0.25, 0.125]")
