"""
Unified GLU ET model implementation.

This module provides a self-contained GLU-based ET model that implements
the unified interface for directly predicting expected sufficient statistics
from natural parameters.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any, Tuple, Optional, Union, List
from flax.core import FrozenDict

from .base_model import BaseETModel
from ..configs.glu_et_config import GLU_ET_Config
from ..layers.glu import GLUBlock
from ..layers.resnet_wrapper import ResNetWrapper
from ..embeddings.eta_embedding import EtaEmbedding
from ..utils.activation_utils import get_activation_function


class GLU_ET_Network(BaseETModel[GLU_ET_Config]):
    """
    Hugging Face compatible GLU-based ET Network.
    
    This network uses a Gated Linear Unit (GLU) architecture with residual connections
    to directly predict expected sufficient statistics E[T(X)|η] from natural parameters η.
    
    Compatible with Hugging Face model loading/saving patterns.
    """
    
    config: GLU_ET_Config
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True, rngs: dict = None, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass through the GLU ET network.
        
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
        
        # Pass throught GLU blocks with optional residual connections
        if self.config.num_resnet_blocks > 0:

            if self.config.hidden_sizes[0] != self.config.hidden_sizes[-1]:
                raise ValueError("hidden_sizes[0] must be equal to hidden_sizes[-1] when using ResNet blocks")
            # Project to the first hidden dimension of the GLU Residual block (one-time expansion)        
            x = nn.Dense(self.config.hidden_sizes[0], name='initial_projection')(x)

            # GLU block for resnet wrapper
            glu_block = GLUBlock(
                features=tuple(self.config.hidden_sizes),  # Use all hidden sizes in the block
                use_bias=True,
                activation=get_activation_function(self.config.activation),
                use_layer_norm=self.config.use_layer_norm,
                dropout_rate=self.config.dropout_rate,
                gate_activation=get_activation_function(self.config.gate_activation),
                name=f'glu_block'
            )

            # Wrap with ResNet for residual connections
            glu_resnet = ResNetWrapper(
                base_module=glu_block,
                num_blocks=self.config.num_resnet_blocks,
                activation=None,  # Activation is handled by GLUBlock
                share_parameters=self.config.share_parameters,
                weight_residual=self.config.weight_residual,
                residual_weight=self.config.residual_weight,
                name=f'glu_resnet'
            )

            x = glu_resnet(x, training=training)
        else:
            # No ResNet blocks - use simple GLU without residual connections
            glu_block = GLUBlock(
                features=self.config.hidden_sizes,  # Use all hidden sizes in the block
                use_bias=True,
                activation=get_activation_function(self.config.activation),
                use_layer_norm=self.config.use_layer_norm,
                dropout_rate=self.config.dropout_rate,
                gate_activation=get_activation_function(self.config.gate_activation),
                name=f'glu_block'
            )            
            x = glu_block(x, training=training)
        
        # Final projection to expected statistics
        x = nn.Dense(self.config.output_dim, name='et_output')(x)
        
        # Return predictions and internal loss (usually 0.0 for standard models)
        internal_loss = jnp.array(0.0)
        return x, internal_loss
    
    def predict(self, params: Dict, eta: jnp.ndarray, rngs: dict = None, **kwargs) -> jnp.ndarray:
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
        config = GLU_ET_Config.from_pretrained(model_name_or_path)
        return cls.from_config(config, **kwargs)
    
