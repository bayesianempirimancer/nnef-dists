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
        Wxy = self.param('W', nn.initializers.lecun_normal(), (x.shape[-1], y.shape[-1], self.features))
        Wx = self.param('Wx', nn.initializers.lecun_normal(), (x.shape[-1], self.features))
        Wy = self.param('Wy', nn.initializers.lecun_normal(), (y.shape[-1], self.features))
        bias = self.param('bias', nn.initializers.zeros, (self.features,))
        
        # Compute bilinear transformation: x^T W y + Wx x + Wy y
        # Handle both 2D and higher-dimensional inputs
        if x.ndim == 2 and y.ndim == 2:
            bilinear_term = jnp.einsum('bi,ijc,bj->bc', x, Wxy, y)
            linear_x = x @ Wx
            linear_y = y @ Wy
        else:
            # Reshape to 2D, compute, then reshape back
            x_original_shape = x.shape
            y_original_shape = y.shape
            
            # Ensure both inputs have the same batch shape for einsum
            if x.shape[:-1] != y.shape[:-1]:
                # For now, we'll require the same batch shape
                # This is a design choice - bilinear layers typically expect matching batch shapes
                raise ValueError(f'Batch shapes must match for bilinear layer. Got x.shape={x.shape}, y.shape={y.shape}')
            else:
                x_broadcast = x
                y_broadcast = y
            
            x_flat = x_broadcast.reshape(-1, x_broadcast.shape[-1])
            y_flat = y_broadcast.reshape(-1, y_broadcast.shape[-1])
            
            bilinear_term = jnp.einsum('bi,ijc,bj->bc', x_flat, Wxy, y_flat)
            linear_x = x_flat @ Wx
            linear_y = y_flat @ Wy
            
            # Reshape back to original batch shape
            output_shape = x_original_shape[:-1] + (self.features,)
            bilinear_term = bilinear_term.reshape(output_shape)
            linear_x = linear_x.reshape(output_shape)
            linear_y = linear_y.reshape(output_shape)

        return bilinear_term + linear_x + linear_y + bias


class BilinearProjectionLayer(nn.Module):
    """Bilinear projection layer that computes (Wx x) * (Wy y)."""
    features: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        # Project both inputs to the same feature space with LeCun normal initialization
        x_proj = nn.Dense(self.features, kernel_init=nn.initializers.lecun_normal(), name='x_proj')(x)
        y_proj = nn.Dense(self.features, kernel_init=nn.initializers.lecun_normal(), name='y_proj')(y)
        # Element-wise multiplication (linear terms handled by bias from nn.Dense)
        return x_proj * y_proj


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


class BilinearProjectionResidualBlock(nn.Module):
    """Bilinear projection residual block with multiple layers using element-wise multiplication."""
    hidden_sizes: tuple[int, ...]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish
    use_layer_norm: bool = True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        in_features = x.shape[-1]
        residual = x
        
        # Multiple bilinear projection layers with different hidden sizes
        for i, hidden_size in enumerate(self.hidden_sizes):
            x = BilinearProjectionLayer(hidden_size, name=f'bilinear_proj_layer_{i}')(x, y)
            x = self.activation(x)
            
            # Optional layer normalization after each layer
            if self.use_layer_norm:
                x = nn.LayerNorm(name=f'layer_norm_{i}')(x)
        
        # Project back to input dimension and add residual
        return nn.Dense(in_features)(x) + residual
