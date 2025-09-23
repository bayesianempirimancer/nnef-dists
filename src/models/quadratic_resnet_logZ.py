"""
Quadratic ResNet LogZ implementation with dedicated architecture.

This module provides a standalone Quadratic ResNet-based LogZ model for learning log normalizers.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import random, grad, hessian, jacfwd
from typing import Dict, Any, Tuple, Optional, Union, List

from ..base_model import BaseNeuralNetwork
from .logZ_Net import LogZTrainer
from ..config import FullConfig


class Quadratic_ResNet_LogZ_Network(BaseNeuralNetwork):
    """
    Quadratic ResNet-based LogZ Network for learning log normalizers.
    
    This network uses a Quadratic ResNet architecture with residual connections
    to learn the log normalizer A(η) of exponential family distributions.
    """
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through the Quadratic ResNet LogZ network.
        
        Args:
            eta: Natural parameters of shape (batch_size, input_dim)
            training: Whether in training mode (affects dropout)
            
        Returns:
            Log normalizer A(η) of shape (batch_size,)
        """
        x = eta
        
        # Input projection
        if len(self.config.hidden_sizes) > 0:
            x = nn.Dense(self.config.hidden_sizes[0], name='quad_input_proj')(x)
        else:
            x = nn.Dense(64, name='quad_input_proj')(x)
        
        # Quadratic residual blocks
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            x = self._quadratic_residual_block(x, hidden_size, i, training)
        
        # Final projection to scalar log normalizer
        x = nn.Dense(1, name='logZ_output')(x)
        return jnp.squeeze(x, axis=-1)
    
    def compute_internal_loss(self, params: Dict, eta: jnp.ndarray, 
                            predicted_logZ: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Compute internal losses (e.g., smoothness penalties, regularization).
        
        Args:
            params: Model parameters
            eta: Natural parameters
            predicted_logZ: Predicted log normalizer
            training: Whether in training mode
            
        Returns:
            Internal loss value
        """
        return 0.0
    
    def _quadratic_residual_block(self, x: jnp.ndarray, hidden_size: int, 
                                 block_idx: int, training: bool) -> jnp.ndarray:
        """Single quadratic residual block."""
        # Store input for residual connection
        residual = x
        if residual.shape[-1] != hidden_size:
            residual = nn.Dense(hidden_size, name=f'quad_residual_proj_{block_idx}')(residual)
        
        # Linear transformation
        linear_out = nn.Dense(hidden_size, name=f'quad_linear_{block_idx}')(x)
        linear_out = nn.swish(linear_out)
        
        # Quadratic transformation with smaller initialization
        quadratic_out = nn.Dense(hidden_size, 
                                kernel_init=nn.initializers.normal(stddev=0.01),
                                name=f'quad_quadratic_{block_idx}')(x)
        quadratic_out = nn.swish(quadratic_out)
        
        # Combine: y = residual + Ax + (Bx)x (updated formula)
        output = residual + linear_out - residual * quadratic_out
        
        # Layer normalization
        if getattr(self.config, 'use_layer_norm', True):
            output = nn.LayerNorm(name=f'quad_layer_norm_{block_idx}')(output)
        
        return output


class Quadratic_ResNet_LogZ_Trainer(LogZTrainer):
    """Trainer for Quadratic ResNet LogZ Network."""
    
    def __init__(self, config: FullConfig, hessian_method='diagonal', adaptive_weights=True):
        model = Quadratic_ResNet_LogZ_Network(config=config.network)
        super().__init__(model, config, hessian_method=hessian_method, adaptive_weights=adaptive_weights)
    


def create_model_and_trainer(config: FullConfig, hessian_method='diagonal', adaptive_weights=True):
    """Factory function to create Quadratic ResNet LogZ model and trainer."""
    return Quadratic_ResNet_LogZ_Trainer(config, hessian_method=hessian_method, adaptive_weights=adaptive_weights)