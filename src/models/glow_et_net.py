"""
Glow ET Network implementation compatible with Hugging Face.

This module provides a Hugging Face compatible Glow-based ET model for directly 
predicting expected statistics using affine coupling layers.
"""

from typing import Optional
import jax
import jax.numpy as jnp
import flax.linen as nn
from ..configs.glow_et_config import Glow_ET_Config
from ..layers.affine import AffineCouplingLayer
from ..embeddings.eta_embedding import EtaEmbedding
from ..utils.activation_utils import get_activation_function


class Glow_ET_Network(nn.Module):
    """
    Glow-based ET Network for directly predicting expected statistics.
    
    This network uses a Glow architecture with affine coupling layers
    to directly predict expected sufficient statistics E[T(X)|η] from natural parameters η.
    """
    config: Glow_ET_Config

    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True, rngs: dict = None, **kwargs) -> jnp.ndarray:
        """
        Forward pass through the Glow ET network with proper affine coupling layers.
        
        Args:
            eta: Natural parameters of shape (batch_size, input_dim)
            training: Whether in training mode (affects dropout)
            rngs: RNG keys for stochastic operations (e.g., dropout)
            **kwargs: Additional arguments (ignored for now)
            
        Returns:
            Predicted expected statistics of shape (batch_size, output_dim)
        """
        # Start with eta as input
        x = eta
        
        # Convert activation string to callable function
        activation_fn = get_activation_function(self.config.activation)
        
        # Apply all affine coupling layers
        for i in range(self.config.num_flow_layers):
            # Create each layer with a unique name to ensure separate parameters
            x, _ = AffineCouplingLayer(
                features=self.config.features,
                activation=activation_fn,
                use_residual=self.config.use_residual,
                use_actnorm=self.config.use_actnorm,
                residual_weight=self.config.residual_weight,
                log_scale_clamp=self.config.log_scale_clamp,
                translation_clamp=self.config.translation_clamp,
                dropout_rate=self.config.dropout_rate,
                seed=i,  # Use layer index as seed for reproducible permutations
                name=f'affine_coupling_{i}'
            )(x, layer_idx=i, training=training, rngs=rngs)
        
        # Final output layer        
        return x
    
    def forward(self, eta: jnp.ndarray, training: bool = True, rngs: dict = None, **kwargs) -> jnp.ndarray:
        """
        Forward pass (alias for __call__ for HF compatibility).
        
        Args:
            eta: Natural parameters of shape (batch_size, input_dim)
            training: Whether in training mode
            **kwargs: Additional arguments
            
        Returns:
            Expected sufficient statistics E[T(X)|η] of shape (batch_size, output_dim)
        """
        return self.__call__(eta, training=training, rngs=rngs, **kwargs)
    
    def compute_internal_loss(self, params: dict, eta: jnp.ndarray, 
                            predicted_mu: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Compute internal losses (e.g., smoothness penalties, regularization).
        
        Args:
            params: Model parameters
            eta: Natural parameters
            predicted_mu: Predicted expected sufficient statistics
            training: Whether in training mode
            
        Returns:
            Internal loss value
        """
        return 0.0

