"""
Vector-based Conditional ResNet architectures for NoProp implementations.

This module provides MLP and flow-based ResNet wrappers that can be used with the NoProp algorithm
for vector inputs (e.g., natural parameters, expected sufficient statistics). The wrappers handle 
the specific input/output requirements for each NoProp variant.

For image-based models, see crn_image.py.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from ...embeddings.time_embeddings import create_time_embedding
from ...embeddings.eta_embedding import create_eta_embedding
from ...layers.concatsquash import ConcatSquash
from ...utils.activation_utils import get_activation_function
from ..base_config import BaseConfig
# from ...layers.gradnet_wrappers import GeometricFlowWrapper  # This is now defined locally


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

@dataclass(frozen=True)
class Config(BaseConfig):
    """Configuration for Conditional ResNet."""
    
    # Set model_name from config_dict
    model_name: str = "conditional_resnet"
    output_dir_parent: str = "artifacts"
    
    # Hierarchical configuration structure
    config_dict = {
        "output_dim": None,
        "hidden_dims": (128, 128, 128),
        "time_embed_dim": 64,
        "time_embed_method": "sinusoidal",
        "dropout_rate": 0.1,
        "eta_embed_type": "default",
        "eta_embed_dim": None,
        "activation_fn": "swish",
        "use_batch_norm": False,
    }


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_cond_resnet(model_type: str, model_config: dict) -> nn.Module:
    """
    Factory function to create model instances.
    
    Args:
        model_type: Type of model to create ("conditional_resnet_mlp", "geometric_flow", "potential_flow")
        model_config: Dictionary containing model configuration parameters
        
    Returns:
        Instantiated model
    """
    if model_type == "conditional_resnet_mlp":
        return ConditionalResnet_MLP(**model_config)
    elif model_type == "geometric_flow":
        return GeometricFlow(**model_config)
    elif model_type == "potential_flow":
        return PotentialFlow(**model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class ConditionalResnet_MLP(nn.Module):
    """Simple MLP for NoProp with vector inputs.
    
    Args:
        hidden_dims: Tuple of hidden layer dimensions
        time_embed_dim: Dimension of time embedding
        time_embed_method: Method for time embedding
        activation_fn: Activation function to use (string)
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate for regularization
    """
    output_dim: Optional[int] = None
    hidden_dims: Tuple[int, ...] = (128, 128, 128)
    time_embed_dim: int = 64
    time_embed_method: str = "sinusoidal"
    eta_embed_type: Optional[str] = "default"
    eta_embed_dim: Optional[int] = None
    activation_fn: str = "swish"
    use_batch_norm: bool = False
    dropout_rate: float = 0.1
        
    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, training: bool = True) -> jnp.ndarray:
        """Forward pass through simple MLP.
        
        Args:
            z: Current state [batch_size, z_dim]
            x: Conditional input [batch_size, x_dim]
            t: Time values [batch_size] or scalar (optional, defaults to 0.0)
            
        Returns:
            Updated state [batch_size, z_dim] (same shape as input z)
        """        
        # Convert string activation function to callable
        activation_fn = get_activation_function(self.activation_fn)

        if self.output_dim is None:
            output_dim = z.shape[-1]
        else:
            output_dim = self.output_dim
        
        # 1. eta embedding and preprocessing
        if self.eta_embed_type is not None:
            eta_embedding = create_eta_embedding(embedding_type=self.eta_embed_type, eta_dim=x.shape[-1])
            x = eta_embedding(x)
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim, kernel_init=jax.nn.initializers.xavier_normal())(x)
            if self.use_batch_norm:
                x = nn.BatchNorm(use_running_average=True)(x)
            x = activation_fn(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        if self.eta_embed_dim is not None:
            x = nn.Dense(self.eta_embed_dim, kernel_init=jax.nn.initializers.xavier_normal())(x)
        
        # 2. time embeddings - handle t=None case
        if t is None:
            t_embedding = jnp.zeros((1,))  # Default to t=0 when None
            x = ConcatSquash(self.hidden_dims[0])(z, x)
        else:
            t = create_time_embedding(embed_dim=self.time_embed_dim, method=self.time_embed_method)(t)
            t = jnp.broadcast_to(t, z.shape[:-1] + t.shape[-1:])
            x = ConcatSquash(self.hidden_dims[0])(z, x, t)
        
        # 3. processing layers
        for hidden_dim in self.hidden_dims[1:]:
            x = nn.Dense(hidden_dim, kernel_init=jax.nn.initializers.xavier_normal())(x)
            if self.use_batch_norm:
                x = nn.BatchNorm(use_running_average=True)(x)
            x = activation_fn(x)            
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
                
        # 5. output projection to match input
        return nn.Dense(output_dim, kernel_init=jax.nn.initializers.xavier_normal())(x)


class GeometricFlow(nn.Module):
    """
    dz/dt parameterized to be consistent with a Geometric Flow, i.e. dmu_T/dt = Sigma(\\eta)\\eta
    
    Args:
        hidden_dims: Tuple of hidden layer dimensions
        time_embed_dim: Dimension of time embedding
        time_embed_method: Method for time embedding
        activation: Activation function to use
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate for regularization
    """
    
    output_dim: Optional[int] = None  # Not used by GeometricFlow (outputs dz/dt)
    hidden_dims: Tuple[int, ...] = (128, 128, 128)
    time_embed_dim: int = 64
    time_embed_method: str = "sinusoidal"
    eta_embed_type: Optional[str] = "default"
    eta_embed_dim: Optional[int] = None
    activation_fn: str = "swish"
    use_batch_norm: bool = False
    dropout_rate: float = 0.1
        
    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """This is intended for flow models which expect the output to be dz/dt, i.e. flow models 
        
        Args:
            z: Current state [batch_size, z_dim]
            x: Conditional input [batch_size, x_dim]
            t: Time values [batch_size] or scalar
            
        Returns:
            Updated state [batch_size, z_dim] (same shape as input z)
        """        
        # Convert string activation function to callable
        activation_fn = get_activation_function(self.activation_fn)

        input_dim = z.shape[-1]
        x_raw = x 
        
        # 1. eta embedding and preprocessing
        if self.eta_embed_type is not None:
            eta_embedding = create_eta_embedding(embedding_type=self.eta_embed_type, eta_dim=x.shape[-1])
            x = eta_embedding(x)
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim, kernel_init=jax.nn.initializers.xavier_normal())(x)
            if self.use_batch_norm:
                x = nn.BatchNorm(use_running_average=True)(x)
            x = activation_fn(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        if self.eta_embed_dim is not None:
            x = nn.Dense(self.eta_embed_dim, kernel_init=jax.nn.initializers.xavier_normal())(x)
        
        # 2. time embeddings
        t_embedding = create_time_embedding(
            embed_dim=self.time_embed_dim, 
            method=self.time_embed_method
        )
        t = t_embedding(t)
        
        # 3. fusion - broadcast t to match batch shape of z
        t_broadcast = jnp.broadcast_to(t, z.shape[:-1] + t.shape[-1:])
        x = ConcatSquash(self.hidden_dims[0])(z, x, t_broadcast)

        # 4. processing layers
        for hidden_dim in self.hidden_dims[1:]:
            x = nn.Dense(hidden_dim, kernel_init=jax.nn.initializers.xavier_normal())(x)
            if self.use_batch_norm:
                x = nn.BatchNorm(use_running_average=True)(x)
            x = activation_fn(x)            
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
                
        # 5. output projection to match input
        Sigma_flat = nn.Dense(input_dim**2)(x)
        # Reshape to get the Sigma matrix [batch_size, input_dim, input_dim]
        Sigma = Sigma_flat.reshape(-1, input_dim, input_dim)
        return jnp.einsum("...ij, ...j -> ...i", Sigma@Sigma.mT, x_raw)/x_raw.shape[-1]

import jax

class PotentialFlow(nn.Module):
    """
    For Flow models that expect the output to be dz_dt = -\\nabla_z V(z,x,t)
    
    This module computes the gradient of a potential function V(z,x,t) with respect to z.
    
    Args:
        hidden_dims: Tuple of hidden layer dimensions
        time_embed_dim: Dimension of time embedding
        time_embed_method: Method for time embedding
        activation: Activation function to use
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate for regularization
    """
    output_dim: Optional[int] = None  # Not used by PotentialFlow (outputs gradient)
    hidden_dims: Tuple[int, ...] = (128, 128, 128)
    time_embed_dim: int = 64
    time_embed_method: str = "sinusoidal"
    eta_embed_type: Optional[str] = "default"
    eta_embed_dim: Optional[int] = None
    activation_fn: str = "swish"
    use_batch_norm: bool = False
    dropout_rate: float = 0.1
        
    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the gradient and value of the potential function with respect to z.
        
        Args:
            z: Current state [batch_size, z_dim]
            x: Conditional input [batch_size, x_dim]
            t: Time values [batch_size] or scalar
            
        Returns:
            Tuple of (gradient, value) where:
            - gradient: Gradient of potential w.r.t. z [batch_size, z_dim]
            - value: Potential values [batch_size]
        """        
        # Convert string activation function to callable
        activation_fn = get_activation_function(self.activation_fn)

        def potential_fn(z_inner, x_inner, t_inner):
            z_inner = z_inner[None, :]
            x_inner = x_inner[None, :]
            t_inner = jnp.asarray(t_inner)[None]
            # 1. eta embedding and preprocessing
            if self.eta_embed_type is not None:
                x_inner = create_eta_embedding(embedding_type=self.eta_embed_type, eta_dim=x_inner.shape[-1])(x_inner)
                
            for hidden_dim in self.hidden_dims:
                x_inner = nn.Dense(hidden_dim)(x_inner)
                if self.use_batch_norm:
                    x_inner = nn.BatchNorm(use_running_average=True)(x_inner)
                x_inner = activation_fn(x_inner)
                if self.dropout_rate > 0:
                    x_inner = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x_inner)
            if self.eta_embed_dim is not None:
                x_inner = nn.Dense(self.eta_embed_dim)(x_inner)
            
            # 2. time embeddings
            t_embedding = create_time_embedding(
                embed_dim=self.time_embed_dim, 
                method=self.time_embed_method
            )
            t_processed = t_embedding(t_inner)
            
            # 3. fusion - broadcast t to match batch shape of z_inner
            t_broadcast = jnp.broadcast_to(t_processed, z_inner.shape[:-1] + t_processed.shape[-1:])
            z_fused = ConcatSquash(self.hidden_dims[0])(z_inner, x_inner, t_broadcast)

            # 4. processing layers
            for hidden_dim in self.hidden_dims[1:]:
                z_fused = nn.Dense(hidden_dim)(z_fused)
                if self.use_batch_norm:
                    z_fused = nn.BatchNorm(use_running_average=True)(z_fused)
                z_fused = activation_fn(z_fused)            
                if self.dropout_rate > 0:
                    z_fused = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(z_fused)
                
            # 5. output to a scalar (size is batch shape)
            return nn.Dense(1)(z_fused).squeeze()

        # Compute both gradient and value of potential with respect to z
        # Use vmap to ensure proper batching through the gradient computation
        value_and_grad_fn = jax.value_and_grad(potential_fn, argnums=0)  # gradient w.r.t. first argument (z)
        vmapped_value_and_grad_fn = jax.vmap(value_and_grad_fn, in_axes=(0, 0, 0))  # vmap over batch dimension
        
        # Get both gradient and value
        potential_values, gradients = vmapped_value_and_grad_fn(z, x, t)
        
        # Return gradient first, then value (as tuple)
        return -gradients, potential_values


class NaturalFlowWrapper(nn.Module):
    """
    Wrapper that converts any conditional ResNet into a natural parameter flow. 
    
    Takes a conditional ResNet (function of z,x,t) with output_dim = z_dim**2 and:
    1. Reshapes ResNet output to get Sigma matrix
    2. Computes dz/dt = Sigma @ Sigma.T @ x_input
    
    Args:
        resnet: The underlying conditional ResNet module (must output z_dim**2 values)
    """
    resnet: nn.Module
    resnet_config: Config
    
    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray, training: bool = True, rngs=None) -> jnp.ndarray:
        """Compute the natural flow dz/dt = Sigma @ Sigma.T @ x.
        
        Args:
            z: Current state [batch_size, z_dim]
            x: Conditional input [batch_size, x_dim] 
            t: Time values [batch_size] or scalar
            
        Returns:
            Flow dz/dt [batch_size, z_dim]
        """
        
        # Apply the ResNet to get the matrix elements
        try:
            resnet_output = self.resnet(z, x, t, training=training, rngs=rngs)
        except TypeError:
            # ResNet doesn't support rngs, call without it
            resnet_output = self.resnet(z, x, t, training=training)
        
        # Reshape to get Sigma matrix [batch_size, z_dim, z_dim]
        z_dim = z.shape[-1]
        Sigma = resnet_output.reshape(-1, z_dim, z_dim)
        
        # Compute dz/dt = Sigma @ Sigma.T @ x
        # x needs to be broadcast to match z_dim if needed
        if x.shape[-1] != z_dim:
            # If x has different dimension, we need to handle this
            # For now, assume x should be broadcast or repeated
            x_broadcast = jnp.broadcast_to(x, z.shape)
        else:
            x_broadcast = x
            
        # Compute the geometric flow: Sigma @ Sigma.T @ x
        dz_dt = jnp.einsum("...ij, ...jk, ...k -> ...i", Sigma, Sigma.transpose(0, 2, 1), x_broadcast) / z_dim
        
        return dz_dt
        

class GeometricFlowWrapper(nn.Module):
    """
    Wrapper that converts any conditional ResNet into a geometric flow using Hessian.
    
    This wrapper uses the Hessian of a potential function (computed via PotentialFlowHessianWrapper)
    as the Sigma matrix to compute dz/dt = Sigma @ x, where Sigma is the Hessian of some potential.
    
    Args:
        resnet: The underlying conditional ResNet module (used to create the potential)
    """
    resnet: nn.Module
    resnet_config: Config
    
    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, training: bool = True, rngs=None) -> jnp.ndarray:
        """Compute the geometric flow dz/dt = Sigma @ x, where Sigma is the Hessian of some potential.
        
        Args:
            z: Current state [batch_size, z_dim]
            x: Conditional input [batch_size, x_dim] 
            t: Time values [batch_size] or scalar (optional)
            training: Whether in training mode
            rngs: Random number generator keys
            
        Returns:
            Flow dz/dt [batch_size, z_dim]
        """
        
        # Use the generalized GeometricFlowWrapper
        geometric_wrapper = GeometricFlowWrapper(resnet=self.resnet)
        dz_dt = geometric_wrapper(z, x, t, training=training, rngs=rngs)
        
        return dz_dt


# Alias for backward compatibility
create_model = create_cond_resnet
