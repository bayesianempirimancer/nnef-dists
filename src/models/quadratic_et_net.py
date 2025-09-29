"""
Hugging Face compatible Quadratic ET model implementation.

This module provides a Hugging Face compatible quadratic ET model
for directly predicting expected sufficient statistics from natural parameters.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any, Tuple, Optional, Union, List
from flax.core import FrozenDict

from ..configs.quadratic_et_config import Quadratic_ET_Config
from ..layers.quadratic import QuadraticProjectionBlock
from ..layers.resnet_wrapper import ResNetWrapper
from ..embeddings.eta_embedding import EtaEmbedding
from ..utils.activation_utils import get_activation_function


class Quadratic_ET_Network(nn.Module):
    """
    Hugging Face compatible Quadratic ET Network.
    
    This network uses quadratic transformations with residual connections
    to directly predict expected sufficient statistics E[T(X)|η] from natural parameters η.
    
    Compatible with Hugging Face model loading/saving patterns.
    """
    
    config: Quadratic_ET_Config
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True, **kwargs) -> jnp.ndarray:
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

            # Quadratic block for resnet wrapper
            quadratic_block = QuadraticProjectionBlock(
                features=tuple(self.config.hidden_sizes),  # Use all hidden sizes in the block
                activation=get_activation_function(self.config.activation),  
                use_layer_norm=self.config.use_layer_norm,
                use_quadratic_norm=getattr(self.config, 'use_quadratic_norm', False),
                dropout_rate=self.config.dropout_rate,
                name=f'quadratic_projection_block'
            )

            # Wrap with ResNet for residual connections
            quadratic_resnet = ResNetWrapper(
                base_module=quadratic_block,
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
        return x  # Return (batch_size, output_dim) shape
    
    def forward(self, eta: jnp.ndarray, training: bool = True, **kwargs) -> jnp.ndarray:
        """
        Forward pass (alias for __call__ for HF compatibility).
        """
        return self.__call__(eta, training=training, **kwargs)
    
    def compute_internal_loss(self, params: Dict, eta: jnp.ndarray, 
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
    
    @classmethod
    def from_config(cls, config: Quadratic_ET_Config, **kwargs):
        """
        Create model from configuration.
        
        Args:
            config: Model configuration
            **kwargs: Additional arguments
            
        Returns:
            Initialized model
        """
        return cls(config=config, **kwargs)
    
    def save_pretrained(self, save_directory: str, params: Optional[Dict] = None):
        """
        Save model and configuration to directory.
        
        Args:
            save_directory: Directory to save to
            params: Model parameters to save
        """
        import os
        import pickle
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save configuration
        self.config.save_pretrained(save_directory)
        
        # Save model parameters if provided
        if params is not None:
            params_path = os.path.join(save_directory, "model_params.pkl")
            with open(params_path, "wb") as f:
                pickle.dump(params, f)
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """
        Load model from directory or model name.
        
        Args:
            model_name_or_path: Path to model directory or model name
            **kwargs: Additional arguments
            
        Returns:
            Model instance (without parameters)
        """
        # Load configuration
        config = Quadratic_ET_Config.from_pretrained(model_name_or_path)
        
        # Create model from config
        model = cls.from_config(config, **kwargs)
        
        return model
    
    def get_config(self) -> Quadratic_ET_Config:
        """Get model configuration."""
        return self.config
    
    def get_input_embeddings(self):
        """Get input embeddings (for HF compatibility)."""
        return None  # This model doesn't use embeddings
    
    def set_input_embeddings(self, value):
        """Set input embeddings (for HF compatibility)."""
        pass  # This model doesn't use embeddings
    
    def get_output_embeddings(self):
        """Get output embeddings (for HF compatibility)."""
        return None  # This model doesn't use embeddings in the HF sense
    
    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings (for HF compatibility)."""
        pass  # Not applicable for this model
    
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None):
        """Resize token embeddings (for HF compatibility)."""
        return self.get_output_embeddings()
    
    def tie_weights(self):
        """Tie weights (for HF compatibility)."""
        pass  # Not applicable for this model
    
    def init_weights(self, rng: jax.random.PRNGKey):
        """Initialize model weights."""
        pass  # This is handled by Flax's initialization
    
    def _init_weights(self, module):
        """Initialize weights for a module (for HF compatibility)."""
        pass  # This is handled by Flax's initialization
