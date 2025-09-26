"""
MLP ET implementation with dedicated architecture.

This module provides a standalone MLP-based ET model for directly predicting expected statistics.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any, Tuple, Optional, Union, List

from ..base_model import BaseNeuralNetwork
from .ET_Net import ETTrainer
from ..config import FullConfig
from .layers.mlp import MLPBlock
from .layers.resnet_wrapper import ResNetWrapper


class MLP_ET_Network(BaseNeuralNetwork):
    """
    MLP-based ET Network for directly predicting expected statistics.
    
    This network uses a standard multi-layer perceptron architecture to directly
    predict expected sufficient statistics E[T(X)|η] from natural parameters η.
    """
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through the MLP ET network.
        
        Args:
            eta: Natural parameters of shape (batch_size, input_dim)
            training: Whether in training mode (affects dropout)
            
        Returns:
            Predicted expected statistics of shape (batch_size, output_dim)
        """
        x = eta
        
        # MLP layers with residual connections using standardized components
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            # Create MLP block
            mlp_block = MLPBlock(
                features=hidden_size,
                use_bias=True,
                activation=nn.swish,
                use_layer_norm=getattr(self.config, 'use_layer_norm', True),
                dropout_rate=getattr(self.config, 'dropout_rate', 0.0),
                name=f'mlp_block_{i}'
            )
            
            # Wrap with ResNet for residual connections
            mlp_resnet = ResNetWrapper(
                base_module=mlp_block,
                num_blocks=1,
                use_projection=True,
                activation=None,  # Activation is handled by MLPBlock
                name=f'mlp_resnet_{i}'
            )
            
            x = mlp_resnet(x, training=training)
        
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


class MLP_ET_Trainer(ETTrainer):
    """Trainer for MLP ET Network."""
    
    def __init__(self, config: FullConfig):
        model = MLP_ET_Network(config=config.network)
        super().__init__(model, config)

