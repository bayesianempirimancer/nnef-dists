"""
Hugging Face compatible Quadratic ET model implementation.

This module provides a Hugging Face compatible quadratic ET model
for directly predicting expected sufficient statistics from natural parameters.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

from ..base_model import BaseModel
from ..base_config import BaseConfig
from ...layers.quadratic import QuadraticProjectionBlock
from ...layers.resnet_wrapper import ResNetWrapper
from ...embeddings.eta_embedding import EtaEmbedding
from ...utils.activation_utils import get_activation_function


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

@dataclass
class Config(BaseConfig):
    """
    Configuration class for Quadratic ET networks.
    
    Inherits all common parameters from BaseConfig.
    """
    
    # === MODEL IDENTIFICATION ===
    model_name: str = "quadratic_et_net"
    
    # === QUADRATIC-SPECIFIC PARAMETERS ===
    use_quadratic_norm: bool = True  # Whether to use quadratic normalization
    # Override some defaults for quadratic models
    hidden_sizes: Tuple[int, ...] = (16,)  # Smaller hidden sizes work better for quadratic
    activation: str = "none"  # No activation for quadratic blocks
    num_resnet_blocks: int = 5  # More blocks for quadratic models
    




# ============================================================================
# MODEL IMPLEMENTATION
# ============================================================================


class Quadratic_ET_Net(BaseModel[Config]):
    """
    Hugging Face compatible Quadratic ET Network.
    
    This network uses quadratic transformations with residual connections
    to directly predict expected sufficient statistics E[T(X)|η] from natural parameters η.
    
    Compatible with Hugging Face model loading/saving patterns.
    """
    
    config: Config
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True, rngs: dict = None, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass through the Quadratic ET network.
        
        Args:
            eta: Natural parameters of shape (batch_size, input_dim)
            training: Whether in training mode (affects dropout)
            **kwargs: Additional arguments (for HF compatibility)
            
        Returns:
            Expected sufficient statistics E[T(X)|η] of shape (batch_size, output_dim)
        """
        # Apply eta embedding if specified
        if hasattr(self.config, 'embedding_type') and self.config.embedding_type is not None:
            eta_embedding = EtaEmbedding(
                embedding_type=self.config.embedding_type,
                eta_dim=self.config.input_dim
            )
            x = eta_embedding(eta)
        else:
            x = eta
        
        if self.config.num_resnet_blocks > 0:

            if self.config.hidden_sizes[0] != self.config.hidden_sizes[-1]:
                raise ValueError("hidden_sizes[0] must be equal to hidden_sizes[-1] when using ResNet blocks")
            # Project to the first hidden dimension of the quadratic block (one-time expansion)        
            if x.shape[-1] != self.config.hidden_sizes[0]:
                x = nn.Dense(self.config.hidden_sizes[0], name='initial_projection')(x)

            # Wrap with ResNet for residual connections
            quadratic_resnet = ResNetWrapper(
                base_module_class=QuadraticProjectionBlock,
                base_module_kwargs={
                    'features': tuple(self.config.hidden_sizes),
                    'activation': get_activation_function(self.config.activation),
                    'use_layer_norm': self.config.use_layer_norm,
                    'use_quadratic_norm': getattr(self.config, 'use_quadratic_norm', False),
                    'dropout_rate': self.config.dropout_rate,
                },
                num_blocks=self.config.num_resnet_blocks,
                activation=None,  # Activation is handled by QuadraticResidualBlock
                share_parameters=self.config.share_parameters,
                weight_residual=self.config.weight_residual,
                residual_weight=self.config.residual_weight,
                name=f'quadratic_resnet'
            )
            x = quadratic_resnet(x, training=training)

        else:
            # No ResNet blocks - use simple quadratic block without residual connections
            quadratic_block = QuadraticProjectionBlock(
                features=tuple(self.config.hidden_sizes),  # Use all hidden sizes in the block
                activation=get_activation_function(self.config.activation),  
                use_layer_norm=self.config.use_layer_norm,
                use_quadratic_norm=getattr(self.config, 'use_quadratic_norm', False),
                dropout_rate=self.config.dropout_rate,
                name=f'quadratic_projection_block'
            )
            x = quadratic_block(x, training=training)
        # Final projection to expected statistics
        x = nn.Dense(self.config.output_dim, name='et_output')(x)
        
        # Return predictions and internal loss (usually 0.0 for standard models)
        internal_loss = jnp.array(0.0)
        return x, internal_loss
    
    def predict(self, params: dict, eta: jnp.ndarray, rngs: dict = None, **kwargs) -> jnp.ndarray:
        """
        Make predictions (for trainer compatibility).
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
    
    
