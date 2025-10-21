"""
Glow ET Network implementation compatible with Hugging Face.

This module provides a Hugging Face compatible Glow-based ET model for directly 
predicting expected statistics using affine coupling layers.
"""

from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
import flax.linen as nn
from ..base_model import BaseModel
from ..base_config import BaseConfig
from ...layers.affine import AffineCouplingLayer
from ...embeddings.eta_embedding import EtaEmbedding
from ...utils.activation_utils import get_activation_function


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

@dataclass
class Config(BaseConfig):
    """
    Configuration class for Glow ET networks.
    
    Inherits all common parameters from BaseConfig.
    """
    
    # === MODEL IDENTIFICATION ===
    model_name: str = "glow_et_net"
    
    # === GLOW-SPECIFIC PARAMETERS ===
    num_flow_layers: int = 8  # Number of affine coupling layers
    features: Tuple[int, ...] = (32, 32)  # Hidden layer sizes for coupling networks
    
    # Affine coupling layer parameters
    use_residual: bool = False  # Whether to use residual connections in coupling networks
    use_actnorm: bool = True  # Whether to use activation normalization
    log_scale_clamp: Tuple[float, float] = (-4.0, 4.0)  # Clamping range for log_scale
    translation_clamp: Tuple[float, float] = (-100.0, 100.0)  # Clamping range for translation
    
    # Override some defaults for Glow models
    use_resnet: bool = False  # Glow doesn't use ResNet blocks
    num_resnet_blocks: int = 0
    




# ============================================================================
# MODEL IMPLEMENTATION
# ============================================================================


class Glow_ET_Net(BaseModel[Config]):
    """
    Glow-based ET Network for directly predicting expected statistics.
    
    This network uses a Glow architecture with affine coupling layers
    to directly predict expected sufficient statistics E[T(X)|η] from natural parameters η.
    """
    config: Config

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
    
    
