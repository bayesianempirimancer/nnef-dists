"""
Image-based Conditional ResNet architectures for NoProp implementations.

This module provides CNN-based ResNet wrappers that can be used with the NoProp algorithm
for image-like data (e.g., MNIST, CIFAR, etc.). The wrappers handle the specific 
input/output requirements for each NoProp variant with image inputs.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import jax.numpy as jnp
import flax.linen as nn

from ...embeddings.time_embeddings import create_time_embedding
from ...layers.concatsquash import ConcatSquash
from ..base_config import BaseConfig


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

@dataclass(frozen=True)
class Config(BaseConfig):
    """Configuration for Image-based Conditional ResNet."""
    
    hidden_dims: Tuple[int, ...] = (64, 128, 64)
    time_embed_dim: int = 64
    time_embed_method: str = "sinusoidal"
    dropout_rate: float = 0.1
    eta_embed_type: Optional[str] = None  # Not used by CNN model
    eta_embed_dim: Optional[int] = None  # Not used by CNN model
    activation_fn: str = "relu"  # CNN uses relu by default
    use_batch_norm: bool = False  # Not used by CNN model
    init_scale: float = 1.0  # Not used by CNN model


# ============================================================================
# MODEL IMPLEMENTATIONS
# ============================================================================

class ConditionalResNet_CNNx(nn.Module):
    """Simple CNN Conditional ResNet for smaller datasets like MNIST."""
    
    output_dim: Optional[int] = None
    hidden_dims: Tuple[int, ...] = (64, 128, 64)
    time_embed_dim: int = 64
    time_embed_method: str = "sinusoidal"
    dropout_rate: float = 0.1
    eta_embed_type: Optional[str] = None  # Not used by CNN model
    eta_embed_dim: Optional[int] = None  # Not used by CNN model
    activation_fn: str = "relu"  # CNN uses relu by default
    use_batch_norm: bool = False  # Not used by CNN model
    init_scale: float = 1.0  # Not used by CNN model
    
    def setup(self):
        # Create time embedding module
        self.t_embed = create_time_embedding(
            embed_dim=self.time_embed_dim,
            method=self.time_embed_method
        )
    
    @nn.compact
    def __call__(
        self, 
        z: jnp.ndarray, 
        x: jnp.ndarray, 
        t: Optional[jnp.ndarray] = None, 
        training: bool = True
    ) -> jnp.ndarray:
        """Forward pass through CNN Conditional ResNet.
        
        Args:
            z: Current state [batch_size, z_dim]
            x: Image input [batch_size, height, width, channels]
            t: Time values [batch_size] or scalar (optional, defaults to 0.0)
            training: Whether in training mode (affects dropout)
            
        Returns:
            Updated state [batch_size, z_dim] (same shape as input z)
        """
        batch_size = z.shape[0]
        if self.output_dim is None:
            output_dim = z.shape[-1]
        else:
            output_dim = self.output_dim
        
        # CNN for x
        x = nn.Conv(32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = nn.Conv(64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = x.reshape(batch_size, -1)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        
        # Time embedding - handle t=None case
        if t is None:
            t_embedding = jnp.zeros((1,))  # Default to t=0 when None
            x = ConcatSquash(self.hidden_dims[0])(z, x)
        else:
            t = create_time_embedding(embed_dim=self.time_embed_dim, method=self.time_embed_method)(t)
            t = jnp.broadcast_to(t, z.shape[:-1] + t.shape[-1:])
            x = ConcatSquash(self.hidden_dims[0])(z, x, t)

        # Fusion
        z = ConcatSquash(self.hidden_dims[0])(z, x, t)

        # Processing layers
        for hidden_dim in self.hidden_dims[1:]:
            z = nn.Dense(hidden_dim)(z)
            z = nn.relu(z)
            z = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(z)

        # Output projection to match input
        return nn.Dense(output_dim)(z)
