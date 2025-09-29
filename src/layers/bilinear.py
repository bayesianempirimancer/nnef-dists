"""
Bilinear layers and blocks for neural networks.

This module provides bilinear layer implementations and residual blocks
that use bilinear transformations.
"""

import jax.numpy as jnp
import flax.linen as nn
from typing import Callable


class BilinearLayer(nn.Module):
    """Bilinear layer that computes x^T W y for input tensors x and y."""
    features: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        # Initialize bilinear weight matrix
        Wxy = self.param('W', nn.initializers.lecun_normal(), use_bias=False, shape=(x.shape[-1], y.shape[-1], self.features))
        Wx = self.param('Wx', nn.initializers.lecun_normal(), use_bias=False, shape=(x.shape[-1], self.features))
        Wy = self.param('Wy', nn.initializers.lecun_normal(), use_bias=False, shape=(y.shape[-1], self.features))
        bias = self.param('bias', nn.initializers.zeros, (self.features,))
        
        x_broadcast = x
        y_broadcast = y
            
        bilinear_term = jnp.einsum('...bi,ijc,...bj->...bc', x, Wxy, y)
        linear_x = x @ Wx
        linear_y = y @ Wy

        return bilinear_term + linear_x + linear_y + bias


class BilinearProjectionLayer(nn.Module):
    """Bilinear projection layer that computes (Wx x) * (Wy y)."""
    features: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        # Project both inputs to the same feature space with LeCun normal initialization

        kernel_init = lambda key, shape, dtype: nn.initializers.lecun_normal()(key, shape, dtype) / jnp.sqrt(x.shape[-1])
        x_proj = nn.Dense(self.features, kernel_init=kernel_init, use_bias=False, name='x_proj')(x)
        y_proj = nn.Dense(self.features, kernel_init=nn.initializers.lecun_normal(), use_bias=False, name='y_proj')(y)
        # Element-wise multiplication (linear terms handled by bias from nn.Dense)
        x_linear = nn.Dense(self.features, kernel_init=nn.initializers.lecun_normal(), name='x_linear')(x)
        y_linear = nn.Dense(self.features, kernel_init=nn.initializers.lecun_normal(), use_bias=False, name='y_linear')(y)
        return x_proj * y_proj + x_linear + y_linear


class BilinearResidualBlock(nn.Module):
    """Bilinear residual block with multiple layers and optional layer normalization."""
    hidden_sizes: tuple[int, ...]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish
    use_layer_norm: bool = True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        in_features = x.shape[-1]
        residual = x
        
        # Multiple bilinear layers with different hidden sizes
        for i, hidden_size in enumerate(self.hidden_sizes):
            x = BilinearLayer(hidden_size, name=f'bilinear_layer_{i}')(x, y)
            x = self.activation(x)
            
            # Optional layer normalization after each layer
            if self.use_layer_norm:
                x = nn.LayerNorm(name=f'layer_norm_{i}')(x)
        
        # Project back to input dimension and add residual
        return nn.Dense(in_features)(x) + residual

