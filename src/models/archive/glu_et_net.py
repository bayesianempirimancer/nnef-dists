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
from dataclasses import dataclass
from flax.core import FrozenDict

from .base_model import BaseModel
from .base_config import BaseConfig
from ..layers.glu import GLUBlock
from ..layers.resnet_wrapper import ResNetWrapper
from ..embeddings.eta_embedding import EtaEmbedding
from ..utils.activation_utils import get_activation_function


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

@dataclass
class GLU_ET_Config(BaseConfig):
    """
    Streamlined configuration class for GLU ET networks.
    
    Inherits all common parameters from BaseConfig and only adds
    GLU-specific parameters that are unique to this model type.
    """
    
    # === MODEL IDENTIFICATION ===
    model_type: str = "glu_et"
    model_name: str = "glu_et_network"
    
    # === INPUT/OUTPUT DIMENSIONS ===
    input_dim: int = 0  # Must be set by caller based on actual data
    output_dim: int = 0  # Must be set by caller based on actual data
    
    # === GLU-SPECIFIC PARAMETERS ===
    gate_activation: str = "sigmoid"  # Gate activation function (only GLU-specific parameter)
    
    def _validate_model_specific(self) -> None:
        """GLU-specific validation."""
        # Validate gate activation function
        valid_activations = ['relu', 'gelu', 'swish', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'linear', 'none']
        if self.gate_activation not in valid_activations:
            raise ValueError(f"gate_activation must be one of {valid_activations}, got {self.gate_activation}")


def create_glu_et_config(
    input_dim: int,
    output_dim: int,
    hidden_sizes: List[int],
    activation: str,
    gate_activation: str,
    use_resnet: bool,
    num_resnet_blocks: int,
    **kwargs
) -> GLU_ET_Config:
    """
    Create a GLU ET configuration with specified parameters.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_sizes: List of hidden layer sizes
        activation: Activation function for the main path
        gate_activation: Activation function for the gate
        use_resnet: Whether to use ResNet blocks
        num_resnet_blocks: Number of ResNet blocks
        **kwargs: Additional parameters to override
        
    Returns:
        GLU_ET_Config instance
    """
    config = GLU_ET_Config(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
        gate_activation=gate_activation,
        use_resnet=use_resnet,
        num_resnet_blocks=num_resnet_blocks,
        **kwargs
    )
    
    config.validate()
    return config


# ============================================================================
# MODEL IMPLEMENTATION
# ============================================================================


class GLU_ET_Network(BaseModel[GLU_ET_Config]):
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
    
