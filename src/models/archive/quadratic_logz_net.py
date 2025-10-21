"""
Hugging Face compatible Quadratic LogZ model implementation.

This module provides a Hugging Face compatible quadratic LogZ model
for learning log normalizers using quadratic transformations.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any, Tuple, Optional, Union, List
from flax.core import FrozenDict

from ..configs.quadratic_logz_config import Quadratic_LogZ_Config
from ..layers.quadratic import QuadraticBlock
from ..utils.activation_utils import get_activation_function


class Quadratic_LogZ_Network(nn.Module):
    """
    Hugging Face compatible Quadratic LogZ Network.
    
    This network uses quadratic transformations to learn
    the log normalizer A(η) of exponential family distributions.
    
    Compatible with Hugging Face model loading/saving patterns.
    """
    
    config: Quadratic_LogZ_Config
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True, **kwargs) -> jnp.ndarray:
        """
        Forward pass through the Quadratic LogZ network.
        
        Args:
            eta: Natural parameters of shape (batch_size, input_dim)
            training: Whether in training mode (affects dropout)
            **kwargs: Additional arguments (for HF compatibility)
            
        Returns:
            Log normalizer A(η) of shape (batch_size, output_dim)
        """
        x = eta
        
        # Pass through quadratic blocks
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            quadratic_block = QuadraticBlock(
                features=(hidden_size,),  # Single hidden layer
                use_activation=True,
                activation=get_activation_function(self.config.activation),  # Use configurable activation function
                use_layer_norm=True,  # Default to True for stability
                dropout_rate=0.0,  # No dropout for LogZ networks
                name=f'quadratic_block_{i}'
            )
            x = quadratic_block(x, training=training)
        
        # Final projection to log normalizer
        x = nn.Dense(self.config.output_dim, name='logZ_output')(x)
        return x  # Return (batch_size, output_dim) shape
    
    def forward(self, eta: jnp.ndarray, training: bool = True, **kwargs) -> jnp.ndarray:
        """
        Forward pass (alias for __call__ for HF compatibility).
        
        Args:
            eta: Natural parameters of shape (batch_size, input_dim)
            training: Whether in training mode
            **kwargs: Additional arguments
            
        Returns:
            Log normalizer A(η) of shape (batch_size, output_dim)
        """
        return self.__call__(eta, training=training, **kwargs)
    
    def compute_internal_loss(self, params: Dict, eta: jnp.ndarray, 
                            predicted_logZ: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Compute internal losses (e.g., regularization).
        
        Args:
            params: Model parameters
            eta: Natural parameters
            predicted_logZ: Predicted log normalizer
            training: Whether in training mode
            
        Returns:
            Internal loss value
        """
        return 0.0
    
    @classmethod
    def from_config(cls, config: Quadratic_LogZ_Config, **kwargs):
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
        config = Quadratic_LogZ_Config.from_pretrained(model_name_or_path)
        
        # Create model from config
        model = cls.from_config(config, **kwargs)
        
        return model
    
    def get_config(self) -> Quadratic_LogZ_Config:
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
