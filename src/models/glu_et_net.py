"""
Hugging Face compatible GLU ET model implementation.

This module provides a Hugging Face compatible GLU-based ET model
for directly predicting expected sufficient statistics from natural parameters.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any, Tuple, Optional, Union, List
from flax.core import FrozenDict

from ..configs.glu_et_config import GLU_ET_Config
from ..layers.glu import GLUBlock
from ..layers.resnet_wrapper import ResNetWrapper
from ..embeddings.eta_embedding import EtaEmbedding
from ..utils.activation_utils import get_activation_function


class GLU_ET_Network(nn.Module):
    """
    Hugging Face compatible GLU-based ET Network.
    
    This network uses a Gated Linear Unit (GLU) architecture with residual connections
    to directly predict expected sufficient statistics E[T(X)|η] from natural parameters η.
    
    Compatible with Hugging Face model loading/saving patterns.
    """
    
    config: GLU_ET_Config
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True, **kwargs) -> jnp.ndarray:
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
        return x  # Return (batch_size, output_dim) shape
    
    def forward(self, eta: jnp.ndarray, training: bool = True, **kwargs) -> jnp.ndarray:
        """
        Forward pass (alias for __call__ for compatibility).
        """
        return self.__call__(eta, training=training, **kwargs)
    
    def loss_fn(self, params: Dict, eta: jnp.ndarray, targets: jnp.ndarray, 
                training: bool = True, rngs: dict = None) -> jnp.ndarray:
        """
        Compute model-specific loss function.
        
        This method is called by the trainer when loss_function='model_specific'.
        It should compute the loss in a single forward pass for efficiency.
        
        Args:
            params: Model parameters
            eta: Natural parameters of shape (batch_size, input_dim)
            targets: Target expected sufficient statistics of shape (batch_size, output_dim)
            training: Whether in training mode
            rngs: Random number generator keys for stochastic operations
            
        Returns:
            Loss value (scalar)
        """
        # Forward pass to get predictions
        predictions = self.apply(params, eta, training=training, rngs=rngs)
        
        # Primary loss (MSE)
        primary_loss = jnp.mean((predictions - targets) ** 2)
        
        # Internal loss (e.g., smoothness penalties, regularization)
        internal_loss = self.compute_internal_loss(params, eta, predictions, training=training)
        
        # Total loss
        total_loss = primary_loss + internal_loss
        
        return total_loss
        
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
    def from_config(cls, config: GLU_ET_Config, **kwargs):
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
        config = GLU_ET_Config.from_pretrained(model_name_or_path)
        
        # Create model from config
        model = cls.from_config(config, **kwargs)
        
        return model
    
    def get_config(self) -> GLU_ET_Config:
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
