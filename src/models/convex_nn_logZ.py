"""
Convex NN LogZ implementation with dedicated architecture.

This module provides a standalone Convex NN-based LogZ model for learning log normalizers.
It implements an Input Convex Neural Network (ICNN) that parameterizes convex functions,
ensuring that the learned log normalizer A(η) maintains convexity properties essential 
for exponential family distributions.

Key features:
- Non-negative weights in hidden layers to maintain convexity
- Convex activation functions (ReLU, Softplus)
- Skip connections from input to all hidden layers
- Guaranteed convex output with respect to input η

Theoretical foundation:
- For exponential family: p(x|η) = h(x) exp(ηᵀT(x) - A(η))
- A(η) must be convex to ensure valid probability distribution
- E[T(X)] = ∇A(η) (gradient of log normalizer)
- Cov[T(X)] = ∇²A(η) (Hessian of log normalizer, positive definite)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import random, grad, hessian, jacfwd
from typing import Dict, Any, Tuple, Optional, Union, List

from ..base_model import BaseNeuralNetwork
from .logZ_Net import LogZTrainer
from ..config import FullConfig, NetworkConfig
from .layers.convex import ConvexHiddenLayer, SimpleConvexBlock, ICNNBlock, ConvexResNetWrapper


# Note: ConvexLayer is now replaced by ConvexHiddenLayer from layers.convex


class ConvexNeuralNetwork(nn.Module):
    """
    Generic Input Convex Neural Network (ICNN) implementation using standardized components.
    
    This network ensures convexity through:
    - Non-negative weights between hidden layers (via softplus)
    - Skip connections from input to all layers
    - Proper composition of convex functions
    
    Can be used for any task requiring convex neural networks,
    not just log normalizers.
    """
    
    config: Any
    output_dim: int = 1
    
    @nn.compact
    def __call__(self, x_input: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through convex neural network.
        
        Args:
            x_input: Input features [batch_size, input_dim]
            training: Whether in training mode
            
        Returns:
            Network output [batch_size, output_dim]
        """
        # Use ICNNBlock for the main convex network
        if len(self.config.hidden_sizes) > 0:
            convex_block = ICNNBlock(
                features=self.output_dim,
                hidden_sizes=tuple(self.config.hidden_sizes),
                activation=getattr(self.config, 'activation', 'softplus'),
                use_bias=True,
                name='icnn_block'
            )
            output = convex_block(x_input, training=training)
        else:
            # If no hidden layers, use simple convex layer
            output = ConvexHiddenLayer(
                features=self.output_dim,
                activation="linear",
                use_bias=True,
                name='convex_output_layer'
            )(None, x_input, training=training)
        
        return output


class Convex_LogZ_Network(BaseNeuralNetwork):
    """
    Convex NN-based LogZ Network for learning log normalizers using standardized components.
    
    This network ensures convexity of the log normalizer A(η) by using
    non-negative weights and convex activation functions.
    """
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through convex network.
        
        Args:
            eta: Natural parameters of shape (batch_size, input_dim)
            training: Whether in training mode (affects dropout)
            
        Returns:
            Log normalizer A(η) of shape (batch_size,)
        """
        # Use ICNNBlock for the main convex network
        if len(self.config.hidden_sizes) > 0:
            convex_block = ICNNBlock(
                features=1,  # Output scalar log normalizer
                hidden_sizes=tuple(self.config.hidden_sizes),
                activation=getattr(self.config, 'activation', 'softplus'),
                use_bias=True,
                name='icnn_block'
            )
            output = convex_block(eta, training=training)
        else:
            # If no hidden layers, use simple convex layer
            output = ConvexHiddenLayer(
                features=1,
                activation="linear",
                use_bias=True,
                name='convex_output_layer'
            )(None, eta, training=training)
        
        return jnp.squeeze(output, axis=-1)
    
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

class Convex_LogZ_Trainer(LogZTrainer):
    """Trainer for Convex LogZ Network."""
    
    def __init__(self, config: FullConfig, hessian_method='full', adaptive_weights=True):
        model = Convex_LogZ_Network(config=config.network)
        super().__init__(model, config, hessian_method=hessian_method, adaptive_weights=adaptive_weights)


def create_model_and_trainer(config: FullConfig, hessian_method='full', adaptive_weights=True):
    """Factory function to create Convex LogZ model and trainer."""
    return Convex_LogZ_Trainer(config, hessian_method=hessian_method, adaptive_weights=adaptive_weights)
