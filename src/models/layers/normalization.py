"""
Non-standard normalization layers for neural networks.

This module provides various normalization methods beyond the standard
LayerNorm, including RMSNorm, GroupNorm, InstanceNorm, and other
specialized normalization techniques.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple, Union, Callable

class ScaleMLP(nn.Module):
    """
    MLP-based scale factor computation.
    
    Learns an adaptive scaling factor using an MLP that operates on the input.
    The scaling factor is constrained to be positive definite (positive) to ensure
    numerical stability and meaningful scaling.
    """
    
    features: tuple[int, ...]
    eps: float = 1e-6
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    pd_activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.softplus  # positive definite activation
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute MLP-based adaptive scaling factor.
        
        Args:
            x: Input tensor [..., features]
            
        Returns:
            Scaling factor tensor [..., 1]
        """
        # Apply MLP to learn adaptive scaling factors
        scale = x
        for feat in self.features[:-1]:
            scale = nn.Dense(feat)(scale)
            scale = self.activation(scale)
        
        # Final layer to produce scaling factor
        scale = nn.Dense(self.features[-1], name='scale_final')(scale)
        return self.pd_activation(scale) + self.eps  # Ensure positive scaling
        
class MLPscaler(nn.Module):
    """
    MLP-based adaptive scaling layer.
    
    Learns an adaptive scaling factor using an MLP that operates on the input.
    The scaling factor is constrained to be positive definite (positive) to ensure
    numerical stability and meaningful scaling.
    """
    
    features: tuple[int, ...]
    eps: float = 1e-6
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    pd_activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.softplus  # positive definite activation
    use_norm: bool = True
    global_scale: bool = True
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply MLP-based adaptive scaling.
        
        Args:
            x: Input tensor [..., features]
            
        Returns:
            Scaled tensor [..., features]
        """
        # Compute scaling factor and apply it
        if self.global_scale:
            features = self.features + (1,)
        else:
            features = self.features + (x.shape[-1],)  # Add input dimension for per-dimension scaling

        scale_factor = ScaleMLP(
            features=features, 
            activation=self.activation, 
            pd_activation=self.pd_activation)(x)

        if self.use_norm:    
            x_norm = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + self.eps)
            return x * (scale_factor / x_norm)
        else:
            return x * scale_factor
        


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    A simpler and more efficient alternative to LayerNorm that
    normalizes by the root mean square of the inputs. This is
    widely used in modern architectures like LLaMA and Mamba.
    """
    
    features: int
    eps: float = 1e-6
    use_scale: bool = True
    
    def setup(self):
        """Initialize the RMSNorm parameters."""
        if self.use_scale:
            self.scale = self.param('scale', nn.initializers.ones, (self.features,))
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor [..., features]
            
        Returns:
            Normalized tensor [..., features]
        """
        # Compute RMS
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        
        # Normalize
        x_norm = x / rms
        
        # Apply scale if enabled
        if self.use_scale:
            x_norm = x_norm * self.scale
        
        return x_norm


class GroupNorm(nn.Module):
    """
    Group Normalization.
    
    Normalizes features within groups, making it independent of batch size
    and suitable for both training and inference.
    """
    
    num_groups: int
    features: int
    eps: float = 1e-6
    use_bias: bool = True
    use_scale: bool = True
    
    def setup(self):
        """Initialize the GroupNorm parameters."""
        if self.use_scale:
            self.scale = self.param('scale', nn.initializers.ones, (self.features,))
        if self.use_bias:
            self.bias = self.param('bias', nn.initializers.zeros, (self.features,))
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply group normalization.
        
        Args:
            x: Input tensor [batch, ..., features]
            
        Returns:
            Normalized tensor [batch, ..., features]
        """
        # Reshape to [batch, ..., num_groups, features_per_group]
        batch_size = x.shape[0]
        features_per_group = self.features // self.num_groups
        
        # Reshape for group processing
        x_reshaped = x.reshape(batch_size, -1, self.num_groups, features_per_group)
        
        # Compute mean and variance over groups and features
        mean = jnp.mean(x_reshaped, axis=(-2, -1), keepdims=True)
        var = jnp.var(x_reshaped, axis=(-2, -1), keepdims=True)
        
        # Normalize
        x_norm = (x_reshaped - mean) / jnp.sqrt(var + self.eps)
        
        # Reshape back
        x_norm = x_norm.reshape(x.shape)
        
        # Apply scale and bias
        if self.use_scale:
            x_norm = x_norm * self.scale
        if self.use_bias:
            x_norm = x_norm + self.bias
        
        return x_norm


class InstanceNorm(nn.Module):
    """
    Instance Normalization.
    
    Normalizes each instance independently, commonly used in
    style transfer and generative models.
    """
    
    features: int
    eps: float = 1e-6
    use_bias: bool = True
    use_scale: bool = True
    
    def setup(self):
        """Initialize the InstanceNorm parameters."""
        if self.use_scale:
            self.scale = self.param('scale', nn.initializers.ones, (self.features,))
        if self.use_bias:
            self.bias = self.param('bias', nn.initializers.zeros, (self.features,))
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply instance normalization.
        
        Args:
            x: Input tensor [batch, ..., features]
            
        Returns:
            Normalized tensor [batch, ..., features]
        """
        # Compute mean and variance over spatial dimensions
        # Assuming x has shape [batch, ..., features]
        spatial_dims = tuple(range(1, x.ndim - 1))
        
        mean = jnp.mean(x, axis=spatial_dims, keepdims=True)
        var = jnp.var(x, axis=spatial_dims, keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / jnp.sqrt(var + self.eps)
        
        # Apply scale and bias
        if self.use_scale:
            x_norm = x_norm * self.scale
        if self.use_bias:
            x_norm = x_norm + self.bias
        
        return x_norm


class WeightNorm(nn.Module):
    """
    Weight Normalization.
    
    Normalizes the weight vectors of a linear layer instead of
    the activations, often used for training stability.
    """
    
    features: int
    eps: float = 1e-6
    
    def setup(self):
        """Initialize the WeightNorm parameters."""
        self.weight = self.param('weight', nn.initializers.normal(), (self.features,))
        self.bias = self.param('bias', nn.initializers.zeros, (self.features,))
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply weight normalization.
        
        Args:
            x: Input tensor [..., features]
            
        Returns:
            Output tensor [..., features]
        """
        # Normalize weight vector
        weight_norm = jnp.linalg.norm(self.weight, axis=-1, keepdims=True)
        weight_normalized = self.weight / (weight_norm + self.eps)
        
        # Apply normalized weights
        output = x * weight_normalized + self.bias
        
        return output


class SpectralNorm(nn.Module):
    """
    Spectral Normalization.
    
    Normalizes the spectral norm of weight matrices, commonly used
    in GANs and other generative models for training stability.
    """
    
    features: int
    eps: float = 1e-6
    n_power_iterations: int = 1
    
    def setup(self):
        """Initialize the SpectralNorm parameters."""
        self.weight = self.param('weight', nn.initializers.normal(), (self.features, self.features))
        self.bias = self.param('bias', nn.initializers.zeros, (self.features,))
        
        # Initialize u and v for power iteration
        self.u = self.param('u', nn.initializers.normal(), (self.features,))
        self.v = self.param('v', nn.initializers.normal(), (self.features,))
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply spectral normalization.
        
        Args:
            x: Input tensor [..., features]
            
        Returns:
            Output tensor [..., features]
        """
        # Power iteration to estimate spectral norm
        u = self.u
        v = self.v
        
        for _ in range(self.n_power_iterations):
            v = self.weight.T @ u
            v = v / (jnp.linalg.norm(v) + self.eps)
            u = self.weight @ v
            u = u / (jnp.linalg.norm(u) + self.eps)
        
        # Compute spectral norm
        sigma = u.T @ self.weight @ v
        
        # Normalize weights
        weight_normalized = self.weight / (sigma + self.eps)
        
        # Apply normalized weights
        output = x @ weight_normalized + self.bias
        
        return output


class AdaptiveNorm(nn.Module):
    """
    Adaptive Normalization.
    
    Learns the normalization parameters adaptively based on the input,
    providing more flexibility than fixed normalization schemes.
    """
    
    features: int
    eps: float = 1e-6
    use_bias: bool = True
    use_scale: bool = True
    
    def setup(self):
        """Initialize the AdaptiveNorm parameters."""
        if self.use_scale:
            self.scale = self.param('scale', nn.initializers.ones, (self.features,))
        if self.use_bias:
            self.bias = self.param('bias', nn.initializers.zeros, (self.features,))
        
        # Adaptive parameters
        self.adaptive_scale = nn.Dense(self.features, name='adaptive_scale')
        self.adaptive_bias = nn.Dense(self.features, name='adaptive_bias')
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply adaptive normalization.
        
        Args:
            x: Input tensor [..., features]
            
        Returns:
            Normalized tensor [..., features]
        """
        # Compute mean and variance
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / jnp.sqrt(var + self.eps)
        
        # Compute adaptive parameters
        adaptive_scale = self.adaptive_scale(x_norm)
        adaptive_bias = self.adaptive_bias(x_norm)
        
        # Apply adaptive normalization
        x_norm = x_norm * adaptive_scale + adaptive_bias
        
        # Apply fixed scale and bias if enabled
        if self.use_scale:
            x_norm = x_norm * self.scale
        if self.use_bias:
            x_norm = x_norm + self.bias
        
        return x_norm


class LayerScale(nn.Module):
    """
    Layer Scale normalization.
    
    A simple scaling mechanism that multiplies the output of a layer
    by a learnable scalar, often used in Vision Transformers.
    """
    
    features: int
    init_value: float = 1e-6
    
    def setup(self):
        """Initialize the LayerScale parameters."""
        self.scale = self.param('scale', 
                               lambda rng, shape: jnp.full(shape, self.init_value),
                               (self.features,))
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply layer scaling.
        
        Args:
            x: Input tensor [..., features]
            
        Returns:
            Scaled tensor [..., features]
        """
        return x * self.scale


class DropPath(nn.Module):
    """
    DropPath (Stochastic Depth) regularization.
    
    Randomly drops entire paths through the network during training,
    providing regularization and improving generalization.
    """
    
    drop_prob: float = 0.0
    deterministic: Optional[bool] = None
    
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Apply drop path regularization.
        
        Args:
            x: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor (possibly with dropped paths)
        """
        if self.drop_prob == 0.0 or not training:
            return x
        
        # Generate random mask
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = jax.random.bernoulli(
            jax.random.PRNGKey(0), keep_prob, shape
        )
        
        # Scale by keep probability
        output = x * random_tensor / keep_prob
        
        return output


class PreNorm(nn.Module):
    """
    Pre-normalization wrapper.
    
    Applies normalization before the main layer, commonly used
    in modern transformer architectures.
    """
    
    norm_layer: nn.Module
    main_layer: nn.Module
    
    def __call__(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        """
        Apply pre-normalization.
        
        Args:
            x: Input tensor
            *args, **kwargs: Arguments for the main layer
            
        Returns:
            Output tensor
        """
        # Apply normalization first
        x_norm = self.norm_layer(x)
        
        # Apply main layer
        return self.main_layer(x_norm, *args, **kwargs)


class PostNorm(nn.Module):
    """
    Post-normalization wrapper.
    
    Applies normalization after the main layer, the traditional
    approach in transformer architectures.
    """
    
    norm_layer: nn.Module
    main_layer: nn.Module
    
    def __call__(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        """
        Apply post-normalization.
        
        Args:
            x: Input tensor
            *args, **kwargs: Arguments for the main layer
            
        Returns:
            Output tensor
        """
        # Apply main layer first
        x_out = self.main_layer(x, *args, **kwargs)
        
        # Apply normalization
        return self.norm_layer(x_out)
