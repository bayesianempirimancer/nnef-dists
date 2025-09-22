"""
Quadratic ResNet ET implementation with dedicated architecture.

This module provides a standalone Quadratic ResNet-based ET model for directly predicting expected statistics.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any, Tuple, Optional, Union, List

from ..base_model import BaseNeuralNetwork
from .ET_Net import ETTrainer
from ..config import FullConfig


class Quadratic_ResNet_ET_Network(BaseNeuralNetwork):
    """
    Quadratic ResNet-based ET Network for directly predicting expected statistics.
    
    This network uses a Quadratic ResNet architecture with residual connections
    to directly predict expected sufficient statistics E[T(X)|η] from natural parameters η.
    """
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through the Quadratic ResNet ET network.
        
        Args:
            eta: Natural parameters of shape (batch_size, input_dim)
            training: Whether in training mode (affects dropout)
            
        Returns:
            Predicted expected statistics of shape (batch_size, output_dim)
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
        
        # Final output layer - sufficient statistics
        x = nn.Dense(self.config.output_dim, name='et_output')(x)
        return x
    
    def compute_internal_loss(self, params: Dict, eta: jnp.ndarray, 
                            predicted_stats: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Compute internal losses (e.g., smoothness penalties, regularization).
        
        Args:
            params: Model parameters
            eta: Natural parameters
            predicted_stats: Predicted statistics
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


class Quadratic_ResNet_ET_Trainer(ETTrainer):
    """Trainer for Quadratic ResNet ET Network."""
    
    def __init__(self, config: FullConfig):
        model = Quadratic_ResNet_ET_Network(config=config.network)
        super().__init__(model, config)


def create_model_and_trainer(config: FullConfig) -> Quadratic_ResNet_ET_Trainer:
    """Factory function to create Quadratic ResNet ET model and trainer."""
    return Quadratic_ResNet_ET_Trainer(config)