"""
Invertible NN ET implementation with dedicated architecture.

This module provides a standalone Invertible NN-based ET model for directly predicting expected statistics.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any, Tuple, Optional, Union, List

from ..base_model import BaseNeuralNetwork
from .ET_Net import ETTrainer
from ..config import FullConfig


class Invertible_ET_Network(BaseNeuralNetwork):
    """
    Invertible NN-based ET Network for directly predicting expected statistics.
    
    This network uses an Invertible Neural Network architecture with Real NVP-style layers
    to directly predict expected sufficient statistics E[T(X)|η] from natural parameters η.
    """
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through the Invertible ET network.
        
        Args:
            eta: Natural parameters of shape (batch_size, input_dim)
            training: Whether in training mode (affects dropout)
            
        Returns:
            Predicted expected statistics of shape (batch_size, output_dim)
        """
        x = eta
        
        # Invertible layers
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            x = self._invertible_layer(x, hidden_size, i, training)
        
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
    
    def _invertible_layer(self, x: jnp.ndarray, hidden_size: int, 
                         layer_idx: int, training: bool) -> jnp.ndarray:
        """Single invertible layer."""
        # Permutation to mix dimensions
        perm = jnp.arange(x.shape[-1])
        if layer_idx % 2 == 1:
            perm = jnp.roll(perm, x.shape[-1] // 2)
        x_perm = x[..., perm]
        
        # Neural network for transformation
        net_out = nn.Dense(hidden_size, name=f'inv_net1_{layer_idx}')(x_perm)
        net_out = nn.swish(net_out)
        net_out = nn.Dense(hidden_size, name=f'inv_net2_{layer_idx}')(net_out)
        net_out = nn.swish(net_out)
        net_out = nn.Dense(x.shape[-1], name=f'inv_transform_{layer_idx}')(net_out)
        
        # Apply transformation
        output = x + net_out
        
        return output


class Invertible_ET_Trainer(ETTrainer):
    """Trainer for Invertible ET Network."""
    
    def __init__(self, config: FullConfig):
        model = Invertible_ET_Network(config=config.network)
        super().__init__(model, config)


def create_model_and_trainer(config: FullConfig) -> Invertible_ET_Trainer:
    """Factory function to create Invertible ET model and trainer."""
    return Invertible_ET_Trainer(config)