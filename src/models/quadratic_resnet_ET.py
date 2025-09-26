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
from .layers.quadratic import QuadraticResidualBlock


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
        
        # Quadratic residual blocks using standardized components
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            quadratic_block = QuadraticResidualBlock(
                features=hidden_size,
                activation=nn.swish,
                use_layer_norm=getattr(self.config, 'use_layer_norm', True),
                name=f'quadratic_block_{i}'
            )
            x = quadratic_block(x, training=training)
        
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
    
# Note: _quadratic_residual_block is now replaced by QuadraticResidualBlock from layers.quadratic


class Quadratic_ResNet_ET_Trainer(ETTrainer):
    """Trainer for Quadratic ResNet ET Network."""
    
    def __init__(self, config: FullConfig):
        model = Quadratic_ResNet_ET_Network(config=config.network)
        super().__init__(model, config)


def create_model_and_trainer(config: FullConfig) -> Quadratic_ResNet_ET_Trainer:
    """Factory function to create Quadratic ResNet ET model and trainer."""
    return Quadratic_ResNet_ET_Trainer(config)