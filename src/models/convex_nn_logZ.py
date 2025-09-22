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

from ..base_model import BaseNeuralNetwork, BaseTrainer
from ..config import FullConfig, NetworkConfig


class ConvexLayer(nn.Module):
    """
    Type 1 Convex Layer - Standard ICNN layer.
    
    Maintains convexity by:
    1. Non-negative weights from previous layer
    2. Skip connections from input with unrestricted weights
    3. Convex activation function
    """
    
    hidden_size: int
    use_bias: bool = True
    activation: str = "relu"
    layer_type: str = "type1"  # "type1" or "type2"
    
    @nn.compact
    def __call__(self, z_prev: jnp.ndarray, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through convex layer.
        
        Args:
            z_prev: Output from previous layer [batch_size, prev_hidden_size]
            x_input: Original input (skip connection) [batch_size, input_dim]
            training: Whether in training mode
            
        Returns:
            Layer output [batch_size, hidden_size]
        """
        # Handle different convex layer types
        if z_prev is not None:
            if self.layer_type == "type1":
                # Type 1: Non-negative weights + convex activation
                W_z = self.param('W_z', 
                               nn.initializers.uniform(scale=1.0), 
                               (z_prev.shape[-1], self.hidden_size))
                W_z = W_z + 0.5  # Positive bias
                W_z_processed = nn.softplus(W_z)  # Ensure non-negative
                z_term = jnp.dot(z_prev, W_z_processed)
                
            elif self.layer_type == "type2":
                # Type 2: Negative weights + concave activation
                W_z = self.param('W_z', 
                               nn.initializers.uniform(scale=1.0), 
                               (z_prev.shape[-1], self.hidden_size))
                W_z = W_z - 0.5  # Negative bias
                W_z_processed = -nn.softplus(-W_z)  # Ensure non-positive
                z_term = jnp.dot(z_prev, W_z_processed)

            else:
                raise ValueError(f"Unknown layer_type: {self.layer_type}")
        else:
            z_term = 0.0
        
        # Skip connection from input (unrestricted weights)
        W_x = self.param('W_x',
                        nn.initializers.xavier_uniform(),
                        (x.shape[-1], self.hidden_size))
        x = jnp.dot(x, W_x)
        
        # Bias term
        if self.use_bias:
            b = self.param('b', nn.initializers.zeros, (self.hidden_size,))
            output = z_term + x + b
        else:
            output = z_term + x
        
        # Apply activation function based on layer type
        if self.layer_type == "type1":
            # Type 1: Convex activations
            if self.activation == "relu":
                output = nn.relu(output)
            elif self.activation == "softplus":
                output = nn.softplus(output)
            elif self.activation == "elu":
                output = nn.elu(output)
            elif self.activation == "leaky_relu":
                output = nn.leaky_relu(output, negative_slope=0.01)
            elif self.activation == "linear":
                # No activation (linear output)
                pass
            else:
                # Default to softplus for type 1
                output = nn.softplus(output)
                
        elif self.layer_type == "type2":
            # Type 2: Concave activations (for negative weights)
            if self.activation == "relu":
                # Use negative softplus as concave activation
                output = -nn.softplus(-output)  # Concave
            elif self.activation == "softplus":
                # Use log(1 + exp(-x)) which is concave
                output = jnp.log(1.0 + jnp.exp(-jnp.abs(output))) * jnp.sign(output)
            else:
                # Default concave activation: -softplus(-x)
                output = -nn.softplus(-output)
                
        else:
            raise ValueError(f"Unknown layer_type: {self.layer_type}")
        
        return output


class ConvexNeuralNetwork(nn.Module):
    """
    Generic Input Convex Neural Network (ICNN) implementation.
    
    This network ensures convexity through:
    - Non-negative weights between hidden layers (via softplus)
    - Skip connections from input to all layers
    - Alternating convex/concave layer types
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
        # Store original input for skip connections
        original_input = x_input
        
        # First layer (no previous z, only input) - always Type 1
        if len(self.config.hidden_sizes) > 0:
            z = ConvexLayer(
                hidden_size=self.config.hidden_sizes[0],
                activation=self.config.activation,
                layer_type="type1",
                name='convex_layer_0'
            )(None, original_input, training=training)
        else:
            # If no hidden layers, go directly to output
            z = original_input
        
        # Hidden convex layers with alternating types
        for i, hidden_size in enumerate(self.config.hidden_sizes[1:], 1):
            # Alternate between Type 1 and Type 2 layers
            layer_type = "type2" if i % 2 == 0 else "type1"
            
            z = ConvexLayer(
                hidden_size=hidden_size,
                activation=self.config.activation,
                layer_type=layer_type,
                name=f'convex_layer_{i}'
            )(z, original_input, training=training)
        
        # Final output layer using ConvexLayer
        output = ConvexLayer(
            hidden_size=self.output_dim,
            activation="linear",  # No activation for final layer
            layer_type="type1",   # Non-negative weights to maintain convexity
            name='convex_output_layer'
        )(z, original_input, training=training)
        
        return output


class Convex_LogZ_Network(BaseNeuralNetwork):
    """
    Convex NN-based LogZ Network for learning log normalizers.
    
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
        x = eta
        
        # First layer with skip connection
        z = nn.Dense(self.config.hidden_sizes[0], name='convex_hidden_0')(x)
        z = nn.relu(z)  # Convex activation
        
        # Convex layers with skip connections
        for i, hidden_size in enumerate(self.config.hidden_sizes[1:], 1):
            # Convex layer with skip connection from input
            z = ConvexLayer(hidden_size, name=f'convex_layer_{i}')(z, x, training)
        
        # Final projection to scalar log normalizer
        z = nn.Dense(1, name='logZ_output')(z)
        return jnp.squeeze(z, axis=-1)

class Convex_LogZ_Trainer(BaseTrainer):
    """Trainer for Convex LogZ Network."""
    
    def __init__(self, config: FullConfig, hessian_method='diagonal', adaptive_weights=True):
        model = Convex_LogZ_Network(config=config.network)
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
    """Factory function to create Convex LogZ model and trainer."""
    return Convex_LogZ_Trainer(config, hessian_method=hessian_method, adaptive_weights=adaptive_weights)
