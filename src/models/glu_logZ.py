"""
GLU LogZ implementation with dedicated architecture.

This module provides a standalone GLU-based LogZ model for learning log normalizers.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import random, grad, hessian, jacfwd
from typing import Dict, Any, Tuple, Optional, Union, List

from ..base_model import BaseNeuralNetwork, BaseTrainer
from ..config import FullConfig


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
        
        # GLU blocks
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            residual = x
            if residual.shape[-1] != hidden_size:
                residual = nn.Dense(hidden_size, name=f'glu_residual_proj_{i}')(residual)
            
            # GLU layer
            linear1 = nn.Dense(hidden_size, name=f'glu_linear1_{i}')(x)
            linear2 = nn.Dense(hidden_size, name=f'glu_linear2_{i}')(x)
            gate = nn.sigmoid(linear1)
            glu_out = gate * linear2
            
            # Residual connection
            x = residual + glu_out
            
            # Layer normalization
            if getattr(self.config, 'use_layer_norm', True):
                x = nn.LayerNorm(name=f'glu_layer_norm_{i}')(x)
            
            dropout_rate = getattr(self.config, 'dropout_rate', 0.0)
            if dropout_rate > 0:
                x = nn.Dropout(rate=dropout_rate, deterministic=not training)(x)
        
        # Final projection to scalar log normalizer
        x = nn.Dense(1, name='logZ_output')(x)
        return jnp.squeeze(x, axis=-1)


class GLU_LogZ_Trainer(BaseTrainer):
    """Trainer for GLU LogZ Network."""
    
    def __init__(self, config: FullConfig, hessian_method='diagonal', adaptive_weights=True):
        model = GLU_LogZ_Network(config=config.network)
        super().__init__(model, config)
        self.hessian_method = hessian_method
        self.adaptive_weights = adaptive_weights
        
        # Compile gradient and Hessian functions
        self._compiled_gradient_fn = jax.jit(grad(self.model.apply, argnums=1))
        if hessian_method == 'diagonal':
            self._compiled_hessian_fn = jax.jit(jax.hessian(self.model.apply, argnums=1))
        elif hessian_method == 'full':
            self._compiled_hessian_fn = jax.jit(hessian(self.model.apply, argnums=1))
        else:
            raise ValueError(f"Unknown hessian_method: {hessian_method}")
    
    def predict_mean(self, params: Dict, eta: jnp.ndarray) -> jnp.ndarray:
        """Predict mean statistics using gradients of log normalizer."""
        return self._compiled_gradient_fn(params, eta)
    
    def predict_covariance(self, params: Dict, eta: jnp.ndarray) -> jnp.ndarray:
        """Predict covariance using Hessian of log normalizer."""
        if self.hessian_method == 'diagonal':
            hess = self._compiled_hessian_fn(params, eta)
            return jnp.diagonal(hess, axis1=-2, axis2=-1)
        else:  # full
            return self._compiled_hessian_fn(params, eta)


def create_model_and_trainer(config: FullConfig, hessian_method='diagonal', adaptive_weights=True):
    """Factory function to create GLU LogZ model and trainer."""
    return GLU_LogZ_Trainer(config, hessian_method=hessian_method, adaptive_weights=adaptive_weights)