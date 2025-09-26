"""
MLP LogZ implementation with dedicated architecture.

This module provides a standalone MLP-based LogZ model for learning log normalizers.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import random
from typing import Dict, Any, Tuple, Optional, Union, List

from ..base_model import BaseNeuralNetwork
from .logZ_Net import LogZTrainer
from ..config import FullConfig
from .layers.mlp import MLPBlock
from .layers.resnet_wrapper import ResNetWrapper


class MLP_LogZ_Network(BaseNeuralNetwork):
    """
    MLP-based LogZ Network for learning log normalizers.
    
    This network uses a standard multi-layer perceptron architecture to learn
    the log normalizer A(η) of exponential family distributions.
    """
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through the MLP LogZ network.
        
        Args:
            eta: Natural parameters of shape (batch_size, input_dim)
            training: Whether in training mode (affects dropout)
            
        Returns:
            Log normalizer A(η) of shape (batch_size,)
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


class MLP_LogZ_Trainer(LogZTrainer):
    """Trainer for MLP LogZ Network."""
    
    def __init__(self, config: FullConfig, hessian_method='full', adaptive_weights=True):
        model = MLP_LogZ_Network(config=config.network)
        super().__init__(model, config, hessian_method=hessian_method, adaptive_weights=adaptive_weights)


def create_model_and_trainer(config: FullConfig, hessian_method='full', adaptive_weights=True):
    """Factory function to create MLP LogZ model and trainer."""
    return MLP_LogZ_Trainer(config, hessian_method=hessian_method, adaptive_weights=adaptive_weights)