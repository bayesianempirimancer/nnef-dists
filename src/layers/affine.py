"""
Affine coupling layers for normalizing flows.

This module provides affine coupling layer implementations following Glow network
conventions and Flax naming standards.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple, Callable


class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer for normalizing flows following Glow conventions.
    
    Implements the transformation:
        y1 = x1
        y2 = x2 * exp(log_scale) + translation
    
    where log_scale and translation are computed from x1 using a neural network.
    
    Key features:
    - Random permutation for each layer to improve expressiveness
    - Conservative initialization for numerical stability
    - Residual connections for training stability
    - Proper inverse permutation handling
    - Dropout support for regularization
    """
    features: [int, ...]
    activation: Callable = nn.swish
    use_residual: bool = False  # Whether to use residual connections
    use_actnorm: bool = False  # Whether to use activation normalization
    residual_weight: float = 1.0  # Weight for residual connection (1.0 for proper ResNet skip connections)
    log_scale_clamp: Tuple[float, float] = (-2.0, 2.0)  # Clamping range for log_scale
    translation_clamp: Tuple[float, float] = (-1.0, 1.0)  # Clamping range for translation
    dropout_rate: float = 0.0  # Dropout rate for regularization
    seed: int = 37  # Seed for reproducible permutations

    @nn.compact
    def __call__(self, x: jnp.ndarray, layer_idx: int = 0, training: bool = True, rngs: dict = None) -> jnp.ndarray:
        """
        Forward pass through the affine coupling layer.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            layer_idx: Index of the layer (used for reproducible permutations)
            training: Whether in training mode
            rngs: RNG keys for stochastic operations (e.g., dropout)
            
        Returns:
            Transformed tensor of same shape as input
        """
        input_dim = x.shape[-1]
        
        # Apply activation normalization (if enabled)
        if self.use_actnorm:
            bias = self.param('bias', nn.initializers.zeros, (x.shape[-1],))
            scale = self.param('scale', nn.initializers.ones, (x.shape[-1],))
            x = (x - bias) * scale
    
        # Create reproducible permutation for this layer
        rng = jax.random.PRNGKey(layer_idx + self.seed)
        perm = jax.random.permutation(rng, input_dim)

        # Apply permutation
        x_permuted = x[..., perm]
        
        # Split the permuted input (handle odd dimensions)
        split_idx = round(input_dim / 2)  # Round to nearest integer for balanced split
        x1, x2 = x_permuted[..., :split_idx], x_permuted[..., split_idx:]
        
        # Neural network for computing scale and translation parameters
        # Use consistent naming that doesn't depend on layer_idx for parameter reuse
        net_out = x1

        for i, feat in enumerate(self.features):
            net_out = nn.Dense(feat,
                          kernel_init=nn.initializers.lecun_normal(),
                          name=f'coupling_layer_{layer_idx}_{i}')(net_out)
            
            net_out = self.activation(net_out)
            # Apply dropout if enabled
            if self.dropout_rate > 0:
                net_out = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(net_out)

        net_out = nn.Dense(x2.shape[-1] * 2,
                          kernel_init=nn.initializers.lecun_normal(),
                          name=f'coupling_layer_{layer_idx}_final')(net_out)

        log_scale, translation = net_out[..., :x2.shape[-1]], net_out[..., x2.shape[-1]:]

        log_scale = jnp.clip(log_scale, self.log_scale_clamp[0], self.log_scale_clamp[1])
        scale = jnp.exp(log_scale)
        log_det_jacobian = jnp.sum(log_scale, axis=-1)
        
        # Apply translation activation function                
        x2 = x2 * scale + translation
        
        # Add residual connection for stability
        if self.use_residual:
            x2_final = x2 + self.residual_weight * x2
        else:
            x2_final = x2
        
        # Concatenate back in permuted space
        x_permuted_out = jnp.concatenate([x1, x2_final], axis=-1)
        
        # Apply inverse permutation to restore original order
        inv_perm = jnp.argsort(perm)
        output = x_permuted_out[..., inv_perm]
        
        return output, log_det_jacobian

