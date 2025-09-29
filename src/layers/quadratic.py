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
    use_quadratic_norm: bool = False
    dropout_rate: float = 0.0
    
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
        # Initialize quadratic weight matrix with smaller scale for numerical stability
        # Use much smaller scale for quadratic terms to prevent explosion
        quad_scale = 0.01 / jnp.sqrt(x.shape[-1])  # Much smaller scale
        Wxx = self.param('W', nn.initializers.normal(quad_scale), 
                        (x.shape[-1], x.shape[-1], self.features))
        # Initialize linear weight matrix with scaled xavier normal
        Wx = self.param('Wx', nn.initializers.xavier_normal() * 0.1, 
                       (x.shape[-1], self.features))
        bias = self.param('bias', nn.initializers.zeros, (self.features,))
        
        # Compute quadratic transformation: x^T W x + Wx x
        # Using @ operator for better performance on linear terms
        # Handle both 2D and 3D inputs
        linear_term = x @ Wx
        if self.use_quadratic_norm:    
            x = x / jnp.sqrt(x.shape[-1])
        quadratic_term = jnp.einsum('...bi,...ijc,...bj->...bc', x, Wxx, x)

        output = quadratic_term + linear_term + bias
        if self.dropout_rate > 0:
            output = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(output)
        return output
        
class QuadraticBlock(nn.Module):
    """
    Quadratic block with multiple layers and optional layer normalization.
    
    This mimics BilinearBlock but uses QuadraticLayer instead of BilinearLayer.
    Uses 'features' attribute to match ResNet wrapper interface.
    """
    features: tuple[int, ...]  # Changed from hidden_sizes to features for ResNet compatibility
    activation: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x
    use_quadratic_norm: bool = False
    use_layer_norm: bool = False
    dropout_rate: float = 0.0
    
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
        
        # Multiple quadratic layers with different hidden sizes
        for i, feat in enumerate(self.features):
            x = QuadraticLayer(
                features=feat, 
                use_quadratic_norm=self.use_quadratic_norm,
                dropout_rate=self.dropout_rate,
                name=f'quadratic_layer_{i}'
            )(x, training=training)
            
            # Apply activation if enabled
            x = self.activation(x)
            
            # Apply layer normalization if enabled
            if self.use_layer_norm:
                x = nn.LayerNorm(name=f'layer_norm_{i}')(x)
        
        return x


class QuadraticProjectionLayer(nn.Module):
    """
    Quadratic projection layer that computes (Wx1 x) * (Wx2 x).
    
    This mimics BilinearProjectionLayer but uses (x, x) instead of (x, y).
    """
    features: int
    use_quadratic_norm: bool = True
    dropout_rate: float = 0.0

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
    
        # Use scaled initialization for better numerical stability
        scaled_init = lambda key, shape, dtype: nn.initializers.xavier_normal()(key, shape, dtype) / x.shape[-1]
        
        linear = nn.Dense(self.features, kernel_init=nn.initializers.xavier_normal(), name='linear')(x)
        
        x_proj1 = nn.Dense(self.features, kernel_init=scaled_init, name='x_proj1', use_bias=False)(x)
        x_proj2 = nn.Dense(self.features, kernel_init=scaled_init, name='x_proj2', use_bias=False)(x)

        output = x_proj1 * jnp.clip(x_proj2, a_min=0, a_max=None)
        output = linear - output

        # Apply dropout if enabled
        if self.dropout_rate > 0:
            output = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(output)
        
        return output


class QuadraticProjectionBlock(nn.Module):
    """
    Quadratic projection residual block with multiple layers using element-wise multiplication.
    
    This mimics BilinearProjectionBlock but uses QuadraticProjectionLayer instead.
    """
    features: tuple[int, ...]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x
    use_layer_norm: bool = False
    use_quadratic_norm: bool = False
    dropout_rate: float = 0.0
    
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
        # Multiple quadratic projection layers with different hidden sizes
        for i, feat in enumerate(self.features):
            x = QuadraticProjectionLayer(
                features=feat, 
                use_quadratic_norm=self.use_quadratic_norm,
                dropout_rate=self.dropout_rate,
                name=f'quadratic_proj_layer_{i}'
            )(x, training=training)

            # Apply activation
            x = self.activation(x)
            
            # Apply layer normalization if enabled
            if self.use_layer_norm:
                x = nn.LayerNorm(name=f'layer_norm_{i}')(x)

        return x

