"""
GLU ET implementation with dedicated architecture.

This module provides a standalone GLU-based ET model for directly predicting expected statistics.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any, Tuple, Optional, Union, List

from ..base_model import BaseNeuralNetwork
from .ET_Net import ETTrainer
from ..config import FullConfig
from .layers.glu import GLUBlock
from .layers.resnet_wrapper import ResNetWrapper


class GLU_ET_Network(BaseNeuralNetwork):
    """
    GLU-based ET Network for directly predicting expected statistics.
    
    This network uses a Gated Linear Unit (GLU) architecture with residual connections
    to directly predict expected sufficient statistics E[T(X)|η] from natural parameters η.
    """
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through the GLU ET network.
        
        Args:
            eta: Natural parameters of shape (batch_size, input_dim)
            training: Whether in training mode (affects dropout)
            
        Returns:
            Predicted expected statistics of shape (batch_size, output_dim)
        """
        x = eta
        
        # Input projection
        x = nn.Dense(self.config.hidden_sizes[0], name='glu_input_proj')(x)
        x = nn.swish(x)
        
        # GLU blocks with residual connections using standardized components
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            # Create GLU block
            glu_block = GLUBlock(
                features=hidden_size,
                use_bias=True,
                use_layer_norm=getattr(self.config, 'use_layer_norm', True),
                dropout_rate=getattr(self.config, 'dropout_rate', 0.0),
                gate_activation=nn.sigmoid,
                activation=nn.swish,
                name=f'glu_block_{i}'
            )
            
            # Wrap with ResNet for residual connections
            glu_resnet = ResNetWrapper(
                base_module=glu_block,
                num_blocks=1,
                use_projection=True,
                activation=None,  # Activation is handled by GLUBlock
                name=f'glu_resnet_{i}'
            )
            
            x = glu_resnet(x, training=training)
        
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


class GLU_ET_Trainer(ETTrainer):
    """Trainer for GLU ET Network using ET-specific trainer."""
    
    def __init__(self, config: FullConfig):
        model = GLU_ET_Network(config=config.network)
        super().__init__(model, config)


def create_model_and_trainer(config: FullConfig) -> GLU_ET_Trainer:
    """Factory function to create GLU ET model and trainer."""
    return GLU_ET_Trainer(config)