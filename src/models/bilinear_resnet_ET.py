"""
Bilinear ResNet ET implementation with dedicated architecture.

This module provides a standalone Bilinear ResNet-based ET model for directly predicting expected statistics.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any, Tuple, Optional, Union, List, Callable

from ..base_model import BaseNeuralNetwork
from .ET_Net import ETTrainer
from ..config import FullConfig
from .layers.bilinear import BilinearResidualBlock, BilinearProjectionResidualBlock

class BilinearResNetETNetwork(BaseNeuralNetwork):
    """
    Bilinear ResNet-based ET Network for directly predicting expected statistics.
    
    This network uses a Bilinear ResNet architecture with residual connections
    to directly predict expected sufficient statistics E[T(X)|η] from natural parameters η.
    """
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through the Bilinear ResNet ET network.
        
        Args:
            eta: Natural parameters of shape (batch_size, input_dim)
            training: Whether in training mode (affects dropout)
            
        Returns:
            Predicted expected statistics of shape (batch_size, output_dim)
        """
        x = eta/jnp.sum(eta**2, axis=-1, keepdims=True)
        
        # Input projection
        if len(self.config.hidden_sizes) > 0:
            x = nn.Dense(self.config.hidden_sizes[0], name='input_proj')(x)
        else:
            x = nn.Dense(64, name='input_proj')(x)
        
        # Bilinear residual blocks
        use_projection_blocks = getattr(self.config, 'use_projection_blocks', False)
        
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            # Create hidden sizes tuple for this block
            num_layers_per_block = getattr(self.config, 'num_layers_per_block', 2)
            block_hidden_sizes = (hidden_size,) * num_layers_per_block
            
            if use_projection_blocks:
                # Use bilinear projection residual block with (x, eta) inputs
                x = BilinearProjectionResidualBlock(
                    hidden_sizes=block_hidden_sizes,
                    activation=nn.swish,
                    use_layer_norm=getattr(self.config, 'use_layer_norm', True)
                )(x, eta, training=training)
            else:
                # Use standard bilinear residual block with (x, eta) inputs
                x = BilinearResidualBlock(
                    hidden_sizes=block_hidden_sizes,
                    activation=nn.swish,
                    use_layer_norm=getattr(self.config, 'use_layer_norm', True)
                )(x, eta, training=training)
        
        # Final output layer - sufficient statistics
        x = nn.Dense(self.config.output_dim, name='output')(x)
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

class BilinearResNetETTrainer(ETTrainer):
    """Trainer for Bilinear ResNet ET Network."""
    
    def __init__(self, config: FullConfig):
        model = BilinearResNetETNetwork(config=config.network)
        super().__init__(model, config)


def create_model_and_trainer(config: FullConfig) -> BilinearResNetETTrainer:
    """Factory function to create Bilinear ResNet ET model and trainer."""
    return BilinearResNetETTrainer(config)