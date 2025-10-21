"""
Unified MLP ET model implementation.

This module provides a self-contained MLP-based ET model that implements
the unified interface for directly predicting expected sufficient statistics
from natural parameters.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

from ..base_model import BaseModel
from ..base_config import BaseConfig
from ...layers.mlp import MLPBlock
from ...layers.resnet_wrapper import ResNetWrapper
from ...embeddings.eta_embedding import EtaEmbedding
from ...utils.activation_utils import get_activation_function


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

@dataclass
class Config(BaseConfig):
    """
    Configuration class for MLP ET networks.
    
    Inherits all common parameters from BaseConfig.
    """
    
    # === MODEL IDENTIFICATION ===
    model_name: str = "mlp_et_net"
    
    # === MLP-SPECIFIC PARAMETERS ===
    hidden_sizes: Tuple[int, ...] = (64, 64, 64)
    activation: str = "swish"
    use_resnet: bool = True
    num_resnet_blocks: int = 3
    residual_weight: float = 1.0
    weight_residual: bool = False
    share_parameters: bool = False
    

# ============================================================================
# MODEL IMPLEMENTATION
# ============================================================================


class MLP_ET_Net(BaseModel[Config]):
    """
    Hugging Face compatible MLP-based ET Network.
    
    This network uses a standard multi-layer perceptron architecture to directly
    predict expected sufficient statistics E[T(X)|η] from natural parameters η.
    
    Compatible with Hugging Face model loading/saving patterns.
    """
    
    config: Config
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True, rngs: dict = None, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Main forward pass through the MLP ET network.
        
        Args:
            eta: Natural parameters of shape (batch_size, eta_dim)
            training: Whether in training mode (affects dropout)
            **kwargs: Additional arguments (for HF compatibility)
            
        Returns:
            Tuple of (predictions, internal_loss) where:
            - predictions: Expected sufficient statistics E[T(X)|η] of shape (batch_size, output_dim)
            - internal_loss: Internal regularization loss (scalar, usually 0.0)
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
        
        # Pass through MLP blocks with optional residual connections
        if self.config.num_resnet_blocks > 0:

            if self.config.hidden_sizes[0] != self.config.hidden_sizes[-1]:
                raise ValueError("hidden_sizes[0] must be equal to hidden_sizes[-1] when using ResNet blocks")
            # Project to the first hidden dimension of the MLP Residual block (one-time expansion)        
            x = nn.Dense(self.config.hidden_sizes[0], name='initial_projection')(x)

            # Wrap with ResNet for residual connections
            mlp_resnet = ResNetWrapper(
                base_module_class=MLPBlock,
                base_module_kwargs={
                    'features': self.config.hidden_sizes,
                    'use_bias': True,
                    'activation': get_activation_function(self.config.activation),
                    'use_layer_norm': self.config.use_layer_norm,
                    'dropout_rate': self.config.dropout_rate,
                },
                num_blocks=self.config.num_resnet_blocks,
                share_parameters=self.config.share_parameters,
                weight_residual=self.config.weight_residual,
                residual_weight=self.config.residual_weight,
                name=f'mlp_resnet'
            )
            x = mlp_resnet(x, training=training, rngs=rngs)
        else:
            # No ResNet blocks - use simple MLP without residual connections
            # Create a single MLP that goes through all hidden layers
            mlp_block = MLPBlock(
                features = self.config.hidden_sizes,  # Single-layer block: current_dim -> current_dim
                use_bias=True,
                activation=nn.swish,
                use_layer_norm=self.config.use_layer_norm,
                dropout_rate=self.config.dropout_rate,
                name=f'mlp_block'
            )       
            x = mlp_block(x, training=training, rngs=rngs)
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
    
    
