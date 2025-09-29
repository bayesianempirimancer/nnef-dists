"""
MLP LogZ Network implementation.

This module implements a neural network that learns the log normalizer A(η) of an
exponential family distribution and returns its gradient ∇A(η) = μ_T (expected
sufficient statistics).

The network uses a multi-layer perceptron (MLP) architecture with residual connections
and returns gradients computed using automatic differentiation.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any, Optional
from ..configs.mlp_logz_config import MLP_LogZ_Config
from ..layers.mlp import MLPBlock
from ..layers.resnet_wrapper import ResNetWrapper
from ..utils.activation_utils import get_activation_function


class MLP_LogZ_Network(nn.Module):
    """
    MLP LogZ Network that learns log normalizer A(η) and returns ∇A(η) = μ_T.
    
    This network computes the log normalizer A(η) of an exponential family distribution
    and returns its gradient with respect to the natural parameters η, which gives us
    the expected sufficient statistics μ_T.
    
    Architecture:
    - Multi-layer perceptron with residual connections
    - Swish activation functions
    - Optional layer normalization and dropout
    - Final dense layer with output_dim=1 for log normalizer
    
    Compatible with Hugging Face model loading/saving patterns.
    """
    
    config: MLP_LogZ_Config
    
    @nn.compact
    def _compute_log_normalizer(self, eta: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Compute the log normalizer A(η) - returns scalar values.
        
        This method computes the log normalizer A(η) using the neural network.
        It's designed to work with gradient_hessian_utils.
        
        Args:
            eta: Natural parameters of shape (batch_size, input_dim)
            training: Whether in training mode (affects dropout)
            
        Returns:
            Log normalizer A(η) of shape (batch_size, 1)
        """
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
        return x  # Return (batch_size, 1) shape
    
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
            predicted_mu: Predicted expected sufficient statistics
            training: Whether in training mode
            
        Returns:
            Internal loss value
        """
        # For now, return zero loss (no internal losses)
        return jnp.array(0.0)
    
    def predict_mean(self, params: Dict, eta: jnp.ndarray) -> jnp.ndarray:
        """
        Predict expected sufficient statistics (mean parameters).
        
        Args:
            params: Model parameters
            eta: Natural parameters
            
        Returns:
            Expected sufficient statistics μ_T
        """
        return self.apply(params, eta, training=False)
    
    def predict_covariance(self, params: Dict, eta: jnp.ndarray) -> jnp.ndarray:
        """
        Predict covariance matrix of sufficient statistics.
        
        This would require computing the Hessian of the log normalizer.
        For now, returns identity matrix as placeholder.
        
        Args:
            params: Model parameters
            eta: Natural parameters
            
        Returns:
            Covariance matrix
        """
        # Placeholder implementation - returns identity matrix
        batch_size = eta.shape[0]
        input_dim = eta.shape[-1]
        return jnp.tile(jnp.eye(input_dim)[None, ...], (batch_size, 1, 1))
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save the model to a directory (HF compatibility).
        
        Args:
            save_directory: Directory to save the model
            **kwargs: Additional arguments
        """
        # Placeholder implementation
        pass
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Load a model from a directory (HF compatibility).
        
        Args:
            pretrained_model_name_or_path: Path to the pretrained model
            **kwargs: Additional arguments
            
        Returns:
            Loaded model instance
        """
        # Placeholder implementation
        pass