"""
Base model class with common functionality for all ET models.

This provides the standard interface and common methods that all models should implement.
"""

import os
import pickle
from abc import ABC, abstractmethod
from typing import Dict, Optional, Type, TypeVar, Generic
import jax.numpy as jnp
from flax import linen as nn

# Type variable for configuration classes
ConfigType = TypeVar('ConfigType')

class BaseETModel(nn.Module, ABC, Generic[ConfigType]):
    """
    Base class for all Exponential Family (ET) models.
    
    This class provides the standard interface and common functionality
    that all ET models should implement.
    """
    
    config: ConfigType
    
    @nn.compact
    @abstractmethod
    def __call__(self, eta: jnp.ndarray, training: bool = True, rngs: dict = None, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass through the model.
        
        Args:
            eta: Natural parameters of shape (batch_size, eta_dim)
            training: Whether in training mode
            rngs: Random number generator keys for stochastic operations
            **kwargs: Additional model-specific arguments
            
        Returns:
            Tuple of (predictions, internal_loss):
            - predictions: Model output of shape (batch_size, output_dim)
            - internal_loss: Internal regularization loss (scalar, usually 0.0)
        """
        pass
    
    @abstractmethod
    def predict(self, params: Dict, eta: jnp.ndarray, rngs: dict = None, **kwargs) -> jnp.ndarray:
        """
        Make predictions for inference.
        
        Args:
            params: Model parameters
            eta: Natural parameters of shape (batch_size, eta_dim)
            rngs: Random number generator keys for stochastic operations (optional)
            **kwargs: Additional model-specific arguments
            
        Returns:
            Predictions of shape (batch_size, output_dim)
        """
        pass
    
    @abstractmethod
    def loss(self, params: Dict, eta: jnp.ndarray, targets: jnp.ndarray, 
             training: bool = True, rngs: dict = None, **kwargs) -> jnp.ndarray:
        """
        Compute training loss.
        
        Args:
            params: Model parameters
            eta: Natural parameters of shape (batch_size, eta_dim)
            targets: Target values of shape (batch_size, output_dim)
            training: Whether in training mode
            rngs: Random number generator keys for stochastic operations (optional)
            **kwargs: Additional model-specific arguments
            
        Returns:
            Loss value (scalar)
        """
        pass
    
    @classmethod
    def from_config(cls, config: ConfigType, **kwargs):
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
            params: Model parameters to save (optional)
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(save_directory, "config.json")
        self.config.save_pretrained(config_path)
        
        # Save parameters if provided
        if params is not None:
            params_path = os.path.join(save_directory, "model_params.pkl")
            with open(params_path, "wb") as f:
                pickle.dump(params, f)
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """
        Load model from pretrained configuration.
        
        Args:
            model_name_or_path: Path to model directory or model name
            **kwargs: Additional arguments
            
        Returns:
            Model instance (without parameters)
        """
        # This method should be overridden by subclasses to specify their config type
        raise NotImplementedError("Subclasses must implement from_pretrained with their specific config type")
    
    def get_config(self) -> ConfigType:
        """Get model configuration."""
        return self.config
