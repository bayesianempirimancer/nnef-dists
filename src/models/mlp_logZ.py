"""
MLP LogZ implementation with dedicated architecture.

This module provides a standalone MLP-based LogZ model for learning log normalizers.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import random, grad, hessian, jacfwd
from typing import Dict, Any, Tuple, Optional, Union, List

from ..base_model import BaseNeuralNetwork
from .logZ_Net import LogZTrainer
from ..config import FullConfig


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
        
        # MLP layers
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            x = nn.Dense(hidden_size, name=f'mlp_hidden_{i}')(x)
            x = nn.swish(x)
            if getattr(self.config, 'use_layer_norm', True):
                x = nn.LayerNorm(name=f'mlp_layer_norm_{i}')(x)
            
            dropout_rate = getattr(self.config, 'dropout_rate', 0.0)
            if dropout_rate > 0:
                x = nn.Dropout(rate=dropout_rate, deterministic=not training)(x)
        
        # Final projection to scalar log normalizer
        x = nn.Dense(1, name='logZ_output')(x)
        return jnp.squeeze(x, axis=-1)
    
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
    
    def __init__(self, config: FullConfig, hessian_method='diagonal', adaptive_weights=True):
        model = MLP_LogZ_Network(config=config.network)
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
    """Factory function to create MLP LogZ model and trainer."""
    return MLP_LogZ_Trainer(config, hessian_method=hessian_method, adaptive_weights=adaptive_weights)