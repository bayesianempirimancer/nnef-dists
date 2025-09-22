"""
Glow ET implementation with dedicated architecture.

This module provides a standalone Glow-based ET model for directly predicting expected statistics.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any, Tuple, Optional, Union, List

from ..base_model import BaseNeuralNetwork
from .ET_Net import ETTrainer
from ..config import FullConfig


class Glow_ET_Network(BaseNeuralNetwork):
    """
    Glow-based ET Network for directly predicting expected statistics.
    
    This network uses a Glow architecture with affine coupling layers
    to directly predict expected sufficient statistics E[T(X)|η] from natural parameters η.
    """
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through the Glow ET network.
        
        Args:
            eta: Natural parameters of shape (batch_size, input_dim)
            training: Whether in training mode (affects dropout)
            
        Returns:
            Predicted expected statistics of shape (batch_size, output_dim)
        """
        x = eta
        
        # Flow-based transformations
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            # Affine coupling layer
            x = self._affine_coupling_layer(x, hidden_size, i, training)
        
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
    
    def _affine_coupling_layer(self, x: jnp.ndarray, hidden_size: int, 
                              layer_idx: int, training: bool) -> jnp.ndarray:
        """Affine coupling layer for flow-based architecture."""
        # Split input (handle odd dimensions by rounding)
        input_dim = x.shape[-1]
        split_idx = input_dim // 2  # This handles odd dimensions by taking floor
        x1, x2 = x[..., :split_idx], x[..., split_idx:]
        
        # Neural network for scaling and translation
        net_out = nn.Dense(hidden_size, name=f'flow_net_{layer_idx}')(x1)
        net_out = nn.swish(net_out)
        net_out = nn.Dense(x2.shape[-1] * 2, name=f'flow_params_{layer_idx}')(net_out)
        
        # Split into scale and translation
        params_dim = net_out.shape[-1]
        params_split_idx = params_dim // 2
        log_scale, translation = net_out[..., :params_split_idx], net_out[..., params_split_idx:]
        
        # Apply transformation
        x2_transformed = x2 * jnp.exp(log_scale) + translation
        
        # Concatenate back
        output = jnp.concatenate([x1, x2_transformed], axis=-1)
        
        return output


class Glow_ET_Trainer(ETTrainer):
    """Trainer for Glow ET Network."""
    
    def __init__(self, config: FullConfig):
        model = Glow_ET_Network(config=config.network)
        super().__init__(model, config)


def create_glow_et_model_and_trainer(config: FullConfig) -> Glow_ET_Trainer:
    """Factory function to create Glow ET model and trainer."""
    return Glow_ET_Trainer(config)