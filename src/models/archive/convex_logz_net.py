"""
Hugging Face compatible Convex LogZ model implementation.

This module provides a Hugging Face compatible convex LogZ model
for learning log normalizers while maintaining convexity constraints.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any, Tuple, Optional, Union, List
from flax.core import FrozenDict

from ..configs.convex_logz_config import Convex_LogZ_Config
from ..layers.convex import ICNNBlock, SimpleConvexBlock, ConvexResNetWrapper


class Convex_LogZ_Network(nn.Module):
    """
    Hugging Face compatible Convex LogZ Network.
    
    This network maintains convexity constraints while learning
    the log normalizer A(η) of exponential family distributions.
    
    Compatible with Hugging Face model loading/saving patterns.
    """
    
    config: Convex_LogZ_Config
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True, **kwargs) -> jnp.ndarray:
        """
        Forward pass through the Convex LogZ network.
        
        Args:
            eta: Natural parameters of shape (batch_size, input_dim)
            training: Whether in training mode (affects dropout)
            **kwargs: Additional arguments (for HF compatibility)
            
        Returns:
            Log normalizer A(η) of shape (batch_size, output_dim)
        """
        x = eta
        
        # Pass through convex blocks
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            if self.config.convex_architecture == "icnn":
                convex_block = ICNNBlock(
                    features=hidden_size,
                    hidden_sizes=(hidden_size,),  # Single hidden layer
                    activation=self.config.activation,
                    use_bias=self.config.use_bias,
                    name=f'icnn_block_{i}'
                )
            else:  # simple_convex
                convex_block = SimpleConvexBlock(
                    features=hidden_size,
                    activation=self.config.activation,
                    weight_scale=self.config.weight_scale,
                    use_bias=self.config.use_bias,
                    name=f'simple_convex_block_{i}'
                )
            
            x = convex_block(x, training=training)
        
        # Final convex output
        if self.config.convex_architecture == "icnn":
            output_layer = ICNNBlock(
                features=self.config.output_dim,
                hidden_sizes=(self.config.output_dim,),  # Single hidden layer
                activation="linear",  # Linear for final output
                use_bias=self.config.use_bias,
                name='convex_output_layer'
            )
        else:
            output_layer = SimpleConvexBlock(
                features=self.config.output_dim,
                activation="linear",  # Linear for final output
                weight_scale=self.config.weight_scale,
                use_bias=self.config.use_bias,
                name='convex_output_layer'
            )
        
        x = output_layer(x, training=training)
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
        Compute internal losses (e.g., convexity validation).
        
        Args:
            params: Model parameters
            eta: Natural parameters
            predicted_logZ: Predicted log normalizer
            training: Whether in training mode
            
        Returns:
            Internal loss value
        """
        if self.config.validate_convexity and training:
            # Add convexity validation loss if needed
            # This could include penalties for non-convex behavior
            return 0.0
        return 0.0
    
    @classmethod
    def from_config(cls, config: Convex_LogZ_Config, **kwargs):
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
        config = Convex_LogZ_Config.from_pretrained(model_name_or_path)
        
        # Create model from config
        model = cls.from_config(config, **kwargs)
        
        return model
    
    def get_config(self) -> Convex_LogZ_Config:
        """Get model configuration."""
        return self.config
    
