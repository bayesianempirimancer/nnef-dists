"""
Hugging Face compatible MLP LogZ model implementation.

This module provides a Hugging Face compatible MLP-based LogZ model
for learning log normalizers of exponential family distributions.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any, Tuple, Optional, Union, List, Callable
from flax.core import FrozenDict

from ..configs.mlp_logz_config import MLP_LogZ_Config
from ..layers.mlp import MLPBlock
from ..layers.resnet_wrapper import ResNetWrapper
from ..utils.activation_utils import get_activation_function


class MLP_LogZ_Network(nn.Module):
    """
    Hugging Face compatible MLP-based LogZ Network.
    
    This network uses a standard multi-layer perceptron architecture to learn
    the log normalizer A(η) of exponential family distributions.
    
    Compatible with Hugging Face model loading/saving patterns.
    """
    
    config: MLP_LogZ_Config
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True, **kwargs) -> jnp.ndarray:
        """
        Forward pass through the MLP LogZ network that returns gradients.
        
        This method computes the log normalizer A(η) and then returns its gradient
        ∇A(η) = μ_T, which represents the expected sufficient statistics.
        Uses the sum approach for correct parameter handling.
        
        Args:
            eta: Natural parameters of shape (batch_size, input_dim)
            training: Whether in training mode (affects dropout)
            **kwargs: Additional arguments (for HF compatibility)
            
        Returns:
            Expected sufficient statistics μ_T of shape (batch_size, input_dim)
        """
        # Build the neural network layers (this will use the existing parameters)
        x = eta
        
        # Pass through MLP blocks with residual connections
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            # Create MLP block
            mlp_block = MLPBlock(
                features=hidden_size,
                use_bias=True,
                activation=get_activation_function(self.config.activation),
                use_layer_norm=self.config.use_layer_norm,
                dropout_rate=self.config.dropout_rate,
                name=f'mlp_block_{i}'
            )
            
            # Wrap with ResNet for residual connections
            mlp_resnet = ResNetWrapper(
                base_module=mlp_block,
                num_blocks=1,
                use_projection=True,
                activation=None,  # Activation is handled by MLPBlock
                name=f'mlp_resnet_{i}'
            )
            
            x = mlp_resnet(x, training=training)
        
        # Final projection to log normalizer
        x = nn.Dense(self.config.output_dim, name='logZ_output')(x)
        
        # For gradient computation, we need to use the sum approach since vmap with @nn.compact
        # is problematic for parameter handling. The sum approach works correctly with parameters.
        def log_normalizer_batch(eta_batch: jnp.ndarray) -> jnp.ndarray:
            """Compute log normalizer for a batch of eta vectors."""
            # We need to recompute the network output for the batch
            x_batch = eta_batch
            
            # Pass through MLP blocks with residual connections
            for i, hidden_size in enumerate(self.config.hidden_sizes):
                # Create MLP block
                mlp_block = MLPBlock(
                    features=hidden_size,
                    use_bias=True,
                    activation=get_activation_function(self.config.activation),
                    use_layer_norm=self.config.use_layer_norm,
                    dropout_rate=self.config.dropout_rate,
                    name=f'mlp_block_{i}'
                )
                
                # Wrap with ResNet for residual connections
                mlp_resnet = ResNetWrapper(
                    base_module=mlp_block,
                    num_blocks=1,
                    use_projection=True,
                    activation=None,  # Activation is handled by MLPBlock
                    name=f'mlp_resnet_{i}'
                )
                
                x_batch = mlp_resnet(x_batch, training=training)
            
            # Final projection to log normalizer
            x_batch = nn.Dense(self.config.output_dim, name='logZ_output')(x_batch)
            # Output should be (batch_size, 1), we want to sum to get scalar
            return jnp.sum(x_batch)
        
        # Compute gradients directly on the batch
        gradients = jax.grad(log_normalizer_batch)(eta)
        return gradients
    
    def forward(self, eta: jnp.ndarray, training: bool = True, **kwargs) -> jnp.ndarray:
        """
        Forward pass (alias for __call__ for HF compatibility).
        
        Args:
            eta: Natural parameters of shape (batch_size, input_dim)
            training: Whether in training mode
            **kwargs: Additional arguments
            
        Returns:
            Expected sufficient statistics μ_T of shape (batch_size, input_dim)
        """
        return self.__call__(eta, training=training, **kwargs)
    
    def compute_internal_loss(self, params: Dict, eta: jnp.ndarray, 
                            predicted_mu: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Compute internal losses (e.g., smoothness penalties, regularization).
        
        Args:
            params: Model parameters
            eta: Natural parameters
            predicted_mu: Predicted expected sufficient statistics μ_T
            training: Whether in training mode
            
        Returns:
            Internal loss value
        """
        return 0.0
    
    @classmethod
    def from_config(cls, config: MLP_LogZ_Config, **kwargs):
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
        config = MLP_LogZ_Config.from_pretrained(model_name_or_path)
        
        # Create model from config
        model = cls.from_config(config, **kwargs)
        
        return model
    
    def get_config(self) -> MLP_LogZ_Config:
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
        # Return None since this model doesn't use embeddings in the HF sense
        return None
    
    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings (for HF compatibility)."""
        # Not applicable for this model
        pass
    
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None):
        """Resize token embeddings (for HF compatibility)."""
        # Not applicable for this model
        return self.get_output_embeddings()
    
    def tie_weights(self):
        """Tie weights (for HF compatibility)."""
        # Not applicable for this model
        pass
    
    def init_weights(self, rng: jax.random.PRNGKey):
        """Initialize model weights."""
        # This is handled by Flax's initialization
        pass
    
    def _init_weights(self, module):
        """Initialize weights for a module (for HF compatibility)."""
        # This is handled by Flax's initialization
        pass
    
    def predict_mean(self, params: Dict, eta: jnp.ndarray) -> jnp.ndarray:
        """
        Predict mean sufficient statistics E[T(X)|η] using gradient of log normalizer.
        
        This is the main prediction method for LogZ models, computing the expected
        sufficient statistics via automatic differentiation of the log normalizer.
        
        Args:
            params: Model parameters
            eta: Natural parameters of shape (batch_size, input_dim)
            
        Returns:
            Expected sufficient statistics μ_T of shape (batch_size, input_dim)
        """
        return self.apply(params, eta, training=False)
