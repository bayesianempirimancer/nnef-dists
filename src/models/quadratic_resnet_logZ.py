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
from .layers.quadratic import QuadraticResidualBlock


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
        
        # Quadratic residual blocks using standardized components
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            quadratic_block = QuadraticResidualBlock(
                features=hidden_size,
                activation=nn.swish,
                use_layer_norm=getattr(self.config, 'use_layer_norm', True),
                name=f'quadratic_block_{i}'
            )
            x = quadratic_block(x, training=training)
        
        # Final projection to scalar log normalizer
        x = nn.Dense(1, name='logZ_output')(x)
        return x  # Return (batch_size, 1) shape for gradient/hessian computation
    
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
    
# Note: _quadratic_residual_block is now replaced by QuadraticResidualBlock from layers.quadratic


class Quadratic_ResNet_LogZ_Trainer(LogZTrainer):
    """Trainer for Quadratic ResNet LogZ Network."""
    
    def __init__(self, config: FullConfig, hessian_method='full', adaptive_weights=True):
        model = Quadratic_ResNet_LogZ_Network(config=config.network)
        super().__init__(model, config, hessian_method=hessian_method, adaptive_weights=adaptive_weights)
    


def create_model_and_trainer(config: FullConfig, hessian_method='full', adaptive_weights=True):
    """Factory function to create Quadratic ResNet LogZ model and trainer."""
    return Quadratic_ResNet_LogZ_Trainer(config, hessian_method=hessian_method, adaptive_weights=adaptive_weights)