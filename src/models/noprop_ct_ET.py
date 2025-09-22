"""
NoProp-CT ET implementation with dedicated architecture.

This module provides a standalone NoProp-CT-based ET model for directly predicting expected statistics.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any, Tuple, Optional, Union, List

from ..base_model import BaseNeuralNetwork
from .ET_Net import ETTrainer
from ..config import FullConfig


class NoProp_CT_ET_Network(BaseNeuralNetwork):
    """
    NoProp-CT-based ET Network for directly predicting expected statistics.
    
    This network uses a Non-propagating Continuous-Time architecture inspired by
    Neural ODEs to directly predict expected sufficient statistics E[T(X)|η] from natural parameters η.
    """
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through the NoProp-CT ET network.
        
        Args:
            eta: Natural parameters of shape (batch_size, input_dim)
            training: Whether in training mode (affects dropout)
            
        Returns:
            Predicted expected statistics of shape (batch_size, output_dim)
        """
        x = eta
        
        # Continuous-time inspired transformations
        dt = getattr(self.config, 'dt', 0.2)  # Default dt = 0.2, configurable
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            x = self._ct_layer(x, hidden_size, i, training, dt)
        
        # Final output layer - sufficient statistics
        x = nn.Dense(self.config.output_dim, name='et_output')(x)
        return x
    
    def _ct_layer(self, x: jnp.ndarray, hidden_size: int, 
                  layer_idx: int, training: bool, dt: float = 0.2) -> jnp.ndarray:
        """Continuous-time inspired layer."""
        # Time-like parameter
        t = jnp.ones((x.shape[0], 1)) * (layer_idx + 1)
        
        # Concatenate time with input
        x_with_time = jnp.concatenate([x, t], axis=-1)
        
        # Neural ODE-like transformation
        net_out = nn.Dense(hidden_size, name=f'ct_net1_{layer_idx}')(x_with_time)
        net_out = nn.swish(net_out)
        net_out = nn.Dense(hidden_size, name=f'ct_net2_{layer_idx}')(net_out)
        net_out = nn.swish(net_out)
        net_out = nn.Dense(x.shape[-1], name=f'ct_ode_{layer_idx}')(net_out)
        
        # Euler step integration with configurable step size
        output = x + dt * net_out
        
        return output
    
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


class NoProp_CT_ET_Trainer(ETTrainer):
    """Trainer for NoProp-CT ET Network."""
    
    def __init__(self, config: FullConfig):
        model = NoProp_CT_ET_Network(config=config.network)
        super().__init__(model, config)


def create_model_and_trainer(config: FullConfig) -> NoProp_CT_ET_Trainer:
    """Factory function to create NoProp-CT ET model and trainer."""
    return NoProp_CT_ET_Trainer(config)