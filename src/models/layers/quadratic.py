"""
Quadratic layers and blocks for polynomial neural networks.

This module provides quadratic layer implementations that enable polynomial
transformations beyond standard linear layers, allowing for more expressive
neural network architectures.
"""

import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Optional


class QuadraticLayer(nn.Module):
    """
    Quadratic layer that computes x^T W x for input tensor x.
    
    This is the quadratic equivalent of BilinearLayer but with (x, x) inputs.
    Implements the transformation: x^T W x + Wx x + c
    """
    features: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through quadratic layer.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            training: Whether in training mode
            
        Returns:
            Output tensor [batch_size, features]
        """
        # Initialize quadratic weight matrix (mimics BilinearLayer's Wxy)
        Wxx = self.param('W', nn.initializers.lecun_normal(), 
                        (x.shape[-1], x.shape[-1], self.features))
        # Initialize linear weight matrix (mimics BilinearLayer's Wx)
        Wx = self.param('Wx', nn.initializers.lecun_normal(), 
                       (x.shape[-1], self.features))
        bias = self.param('bias', nn.initializers.zeros, (self.features,))
        
        # Compute quadratic transformation: x^T W x + Wx x
        # Using @ operator for better performance on linear terms
        # Handle both 2D and 3D inputs
        if x.ndim == 2:
            quadratic_term = jnp.einsum('bi,ijc,bj->bc', x, Wxx, x)
        else:  # 3D or higher
            # Reshape to 2D, compute, then reshape back
            original_shape = x.shape
            x_flat = x.reshape(-1, x.shape[-1])
            quadratic_term = jnp.einsum('bi,ijc,bj->bc', x_flat, Wxx, x_flat)
            quadratic_term = quadratic_term.reshape(original_shape[:-1] + (self.features,))
        
        # Compute linear term
        if x.ndim == 2:
            linear_term = x @ Wx
        else:  # 3D or higher
            original_shape = x.shape
            x_flat = x.reshape(-1, x.shape[-1])
            linear_term = x_flat @ Wx
            linear_term = linear_term.reshape(original_shape[:-1] + (self.features,))
        
        return quadratic_term + linear_term + bias


class QuadraticResidualBlock(nn.Module):
    """
    Quadratic residual block with multiple layers and optional layer normalization.
    
    This mimics BilinearResidualBlock but uses QuadraticLayer instead of BilinearLayer.
    """
    hidden_sizes: tuple[int, ...]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish
    use_layer_norm: bool = True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through quadratic residual block.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            training: Whether in training mode
            
        Returns:
            Output tensor [batch_size, input_dim]
        """
        in_features = x.shape[-1]
        residual = x
        
        # Multiple quadratic layers with different hidden sizes
        for i, hidden_size in enumerate(self.hidden_sizes):
            x = QuadraticLayer(hidden_size, name=f'quadratic_layer_{i}')(x, training=training)
            x = self.activation(x)
            
            # Optional layer normalization after each layer
            if self.use_layer_norm:
                x = nn.LayerNorm(name=f'layer_norm_{i}')(x)
        
        # Project back to input dimension and add residual
        return nn.Dense(in_features)(x) + residual


class QuadraticProjectionLayer(nn.Module):
    """
    Quadratic projection layer that computes (Wx1 x) * (Wx2 x).
    
    This mimics BilinearProjectionLayer but uses (x, x) instead of (x, y).
    """
    features: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through quadratic projection layer.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            training: Whether in training mode
            
        Returns:
            Output tensor [batch_size, features]
        """
        # Project input to the same feature space with LeCun normal initialization
        x_proj1 = nn.Dense(self.features, kernel_init=nn.initializers.lecun_normal(), name='x_proj1')(x)
        x_proj2 = nn.Dense(self.features, kernel_init=nn.initializers.lecun_normal(), name='x_proj2')(x)
        # Element-wise multiplication.  Bias terms in nn.Dense handle the linear terms.
        return x_proj1 * x_proj2


class QuadraticProjectionResidualBlock(nn.Module):
    """
    Quadratic projection residual block with multiple layers using element-wise multiplication.
    
    This mimics BilinearProjectionResidualBlock but uses QuadraticProjectionLayer instead.
    """
    hidden_sizes: tuple[int, ...]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish
    use_layer_norm: bool = True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through quadratic projection residual block.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            training: Whether in training mode
            
        Returns:
            Output tensor [batch_size, input_dim]
        """
        in_features = x.shape[-1]
        residual = x
        
        # Multiple quadratic projection layers with different hidden sizes
        for i, hidden_size in enumerate(self.hidden_sizes):
            x = QuadraticProjectionLayer(hidden_size, name=f'quadratic_proj_layer_{i}')(x, training=training)
            x = self.activation(x)
            
            # Optional layer normalization after each layer
            if self.use_layer_norm:
                x = nn.LayerNorm(name=f'layer_norm_{i}')(x)
        
        # Project back to input dimension and add residual
        return nn.Dense(in_features)(x) + residual

