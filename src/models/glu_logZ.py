"""
GLU LogZ implementation with dedicated architecture.

This module provides a standalone GLU-based LogZ model for learning log normalizers.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import random, grad, hessian, jacfwd
from typing import Dict, Any, Tuple, Optional, Union, List

from ..base_model import BaseNeuralNetwork
from .logZ_Net import LogZTrainer
from ..config import FullConfig
from .layers.glu import GLUBlock
from .layers.resnet_wrapper import ResNetWrapper


class GLU_LogZ_Network(BaseNeuralNetwork):
    """
    GLU-based LogZ Network for learning log normalizers.
    
    This network uses a Gated Linear Unit (GLU) architecture with residual connections
    to learn the log normalizer A(η) of exponential family distributions.
    """
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through the GLU LogZ network.
        
        Args:
            eta: Natural parameters of shape (batch_size, input_dim)
            training: Whether in training mode (affects dropout)
            
        Returns:
            Log normalizer A(η) of shape (batch_size,)
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


class GLU_LogZ_Trainer(LogZTrainer):
    """Trainer for GLU LogZ Network."""
    
    def __init__(self, config: FullConfig, hessian_method='full', adaptive_weights=True):
        model = GLU_LogZ_Network(config=config.network)
        super().__init__(model, config, hessian_method=hessian_method, adaptive_weights=adaptive_weights)
    


def create_model_and_trainer(config: FullConfig, hessian_method='full', adaptive_weights=True):
    """Factory function to create GLU LogZ model and trainer."""
    return GLU_LogZ_Trainer(config, hessian_method=hessian_method, adaptive_weights=adaptive_weights)