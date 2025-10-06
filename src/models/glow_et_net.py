"""
Glow ET Network implementation compatible with Hugging Face.

This module provides a Hugging Face compatible Glow-based ET model for directly 
predicting expected statistics using affine coupling layers.
"""

from typing import Optional, Dict, Any, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
from .base_model import BaseETModel
from ..configs.glow_et_config import Glow_ET_Config
from ..layers.affine import AffineCouplingLayer
from ..embeddings.eta_embedding import EtaEmbedding
from ..utils.activation_utils import get_activation_function


class Glow_ET_Network(BaseETModel[Glow_ET_Config]):
    """
    Glow-based ET Network for directly predicting expected statistics.
    
    This network uses a Glow architecture with affine coupling layers
    to directly predict expected sufficient statistics E[T(X)|η] from natural parameters η.
    """
    config: Glow_ET_Config

    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True, rngs: dict = None, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass through the Glow ET network with proper affine coupling layers.
        
        Args:
            eta: Natural parameters of shape (batch_size, input_dim)
            training: Whether in training mode (affects dropout)
            rngs: RNG keys for stochastic operations (e.g., dropout)
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (predictions, internal_loss):
            - predictions: Expected sufficient statistics of shape (batch_size, output_dim)
            - internal_loss: Internal regularization loss (scalar, usually 0.0)
        """
        # Apply eta embedding if specified
        if hasattr(self.config, 'embedding_type') and self.config.embedding_type is not None:
            eta_embedding = EtaEmbedding(
                embedding_type=self.config.embedding_type,
                eta_dim=self.config.input_dim
            )
            eta = eta_embedding(eta)
        
        # Apply affine coupling layers
        x = eta
        for i in range(self.config.num_flow_layers):
            coupling_layer = AffineCouplingLayer(
                features=self.config.features,
                activation=get_activation_function(self.config.activation),
                dropout_rate=self.config.dropout_rate,
                name=f'coupling_layer_{i}'
            )
            x, _ = coupling_layer(x, layer_idx=i, training=training, rngs=rngs)
        
        # Final projection to expected statistics
        x = nn.Dense(self.config.output_dim, name='et_output')(x)
        
        # Return predictions and internal loss (usually 0.0 for standard models)
        internal_loss = jnp.array(0.0)
        return x, internal_loss
    
    
    def predict(self, params: dict, eta: jnp.ndarray, rngs: dict = None, **kwargs) -> jnp.ndarray:
        """
        Make predictions using the model.
        
        Args:
            params: Model parameters
            eta: Natural parameters of shape (batch_size, eta_dim)
            rngs: Random number generator keys for stochastic operations
            **kwargs: Additional arguments
            
        Returns:
            Predicted expected sufficient statistics of shape (batch_size, output_dim)
        """
        predictions, _ = self.apply(params, eta, training=False, rngs=rngs, **kwargs)
        return predictions
    
    def loss(self, params: Dict, eta: jnp.ndarray, targets: jnp.ndarray, 
             training: bool = True, rngs: dict = None, **kwargs) -> jnp.ndarray:
        """
        Compute training loss.
        
        Args:
            params: Model parameters
            eta: Natural parameters of shape (batch_size, input_dim)
            targets: Target expected sufficient statistics of shape (batch_size, output_dim)
            training: Whether in training mode
            rngs: Random number generator keys for stochastic operations
            **kwargs: Additional arguments
            
        Returns:
            Loss value (scalar)
        """
        predictions, internal_loss = self.apply(params, eta, training=training, rngs=rngs, **kwargs)
        
        # Primary loss (MSE)
        primary_loss = jnp.mean((predictions - targets) ** 2)
        
        # Total loss
        total_loss = primary_loss + internal_loss
        
        return total_loss
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """Load model from pretrained configuration."""
        config = Glow_ET_Config.from_pretrained(model_name_or_path)
        return cls.from_config(config, **kwargs)
    
