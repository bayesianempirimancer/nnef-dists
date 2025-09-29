"""
Geometric Flow ET Network implementation compatible with Hugging Face.

This module provides a Hugging Face compatible Geometric Flow ET model
that learns flow dynamics for exponential families.
"""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
from ..configs.geometric_flow_et_config import Geometric_Flow_ET_Config
from ..utils.activation_utils import get_activation_function
from ..layers.normalization import get_normalization_layer


class Geometric_Flow_ET_Network(nn.Module):
    """
    Geometric Flow ET Network that learns flow dynamics for exponential families.
    
    The network learns A(u, t, η_t) such that:
        du/dt = A(u, t, η_t) @ A(u, t, η_t)^T @ (η_target - η_init)
    
    Key features:
    - Geometric flow dynamics with PSD constraints
    - Smoothness penalties for stable dynamics
    - Minimal time steps due to expected smoothness
    """
    config: Geometric_Flow_ET_Config

    def generate_initial_values(self, eta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate eta_init and mu_init from eta using the exponential family distribution.
        
        Args:
            eta: Target natural parameters [batch_shape, eta_dim]
            
        Returns:
            Tuple of (eta_init, mu_init):
                - eta_init: Initial natural parameters [batch_shape, eta_dim]
                - mu_init: Initial expectations [batch_shape, mu_dim]
        """
        # Get the exponential family distribution from config
        dist = self.config.get_ef_distribution()
        eta_init, mu_init = dist.find_nearest_analytical_point(eta)
        return eta_init, mu_init

    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True, **kwargs) -> jnp.ndarray:
        """
        Geometric flow computation using inherited ET network architecture.
        
        Args:
            eta: Target natural parameters [batch_shape, eta_dim]
            training: Whether in training mode
            
        Returns:
            mu_target: Predicted expectations at eta [batch_shape, mu_dim]
        """
        # Generate initial values
        eta_init, mu_init = self.generate_initial_values(eta)
        
        # Simple forward Euler integration
        dt = 1.0 / self.config.n_time_steps
        u_current = mu_init
        derivative_norms_squared = []
        
        for i in range(self.config.n_time_steps):
            t = i * dt
            du_dt = self.compute_flow_field(u_current, t, eta, eta_init)
            
            # Store norm squared for smoothness penalty
            du_dt_norm_squared = jnp.sum(du_dt ** 2, axis=-1)
            derivative_norms_squared.append(du_dt_norm_squared)
            
            # Forward Euler step
            u_current = u_current + dt * du_dt
        
        # Store derivative norms for smoothness penalty
        if training:
            self.sow('intermediates', 'derivative_norms_squared', jnp.array(derivative_norms_squared))
        
        return u_current
    
    def predict(self, eta: jnp.ndarray, params: dict = None, **kwargs) -> jnp.ndarray:
        """
        Predict method for regular geometric flow (same as __call__).
        
        Args:
            eta: Target natural parameters [batch_shape, eta_dim]
            params: Model parameters (if None, will be initialized)
            **kwargs: Additional arguments
            
        Returns:
            mu_target: Predicted expectations at eta [batch_shape, mu_dim]
        """
        if params is None:
            # Initialize parameters if not provided
            key = jax.random.PRNGKey(0)
            params = self.init(key, eta)
        return self.apply(params, eta, training=False, **kwargs)
    
    @nn.compact
    def compute_flow_field(self, u: jnp.ndarray, t: float, eta: jnp.ndarray, 
                          eta_init: jnp.ndarray) -> jnp.ndarray:
        """
        Compute flow field du/dt using the ET network architecture.
        
        Args:
            u: Current state [batch_shape, mu_dim]
            t: Current time
            eta: Target natural parameters [batch_shape, eta_dim]
            eta_init: Initial natural parameters [batch_shape, eta_dim]
            
        Returns:
            du_dt: Flow field [batch_shape, mu_dim]
        """
        batch_shape = u.shape[:-1]
        eta_dim = eta.shape[-1]
        mu_dim = u.shape[-1]
        matrix_rank = self.config.matrix_rank if self.config.matrix_rank is not None else eta_dim
        
        # Time embedding
        time_embed_dim = self.config.time_embed_dim if self.config.time_embed_dim is not None else eta_dim
        time_embed_dim = min(time_embed_dim, 16)
        
        from ..embeddings.time_embeddings import SimpleTimeEmbedding
        from ..layers.normalization import WeakLayerNorm
        
        t_embed = SimpleTimeEmbedding(embed_dim=time_embed_dim)(t)
        t_embed_batch = jnp.broadcast_to(t_embed, batch_shape + (time_embed_dim,))
        
        # Network input: [u, sin/cos(t), η_init, η_target]
        net_input = jnp.concatenate([u, t_embed_batch, eta_init, eta], axis=-1)
        
        # Use ET network architecture to predict matrix A
        if self.config.architecture == "mlp":
            x = net_input
            for j, hidden_size in enumerate(self.config.hidden_sizes):
                x = nn.Dense(hidden_size, name=f'mlp_{j}',
                           kernel_init=lambda key, shape, dtype: nn.initializers.xavier_normal()(key, shape, dtype) / jnp.sqrt(matrix_rank),
                           bias_init=nn.initializers.zeros)(x)
                x = get_activation_function(self.config.activation)(x)
                if self.config.layer_norm_type != "none":
                    norm_layer = get_normalization_layer(self.config.layer_norm_type, features=hidden_size, name=f'norm_mlp_{j}')
                    if norm_layer is not None:
                        x = norm_layer(x)
                # Note: dropout not used in geometric flow model
            
            A_flat = nn.Dense(eta_dim * matrix_rank, name='matrix_A_output',
                            kernel_init=lambda key, shape, dtype: nn.initializers.xavier_normal()(key, shape, dtype) / jnp.sqrt(matrix_rank),
                            bias_init=nn.initializers.zeros)(x)
        elif self.config.architecture == "glu":
            x = net_input
            for j, hidden_size in enumerate(self.config.hidden_sizes):
                gate = nn.Dense(hidden_size, name=f'glu_gate_{j}',
                              kernel_init=nn.initializers.xavier_normal,
                              bias_init=nn.initializers.zeros)(x)
                value = nn.Dense(hidden_size, name=f'glu_value_{j}',
                               kernel_init=nn.initializers.xavier_normal,
                               bias_init=nn.initializers.zeros)(x)
                
                gate = get_activation_function('sigmoid')(gate)
                value = get_activation_function(self.config.activation)(value)
                x = gate * value
                if self.config.layer_norm_type != "none":
                    norm_layer = get_normalization_layer(self.config.layer_norm_type, features=hidden_size, name=f'norm_glu_{j}')
                    if norm_layer is not None:
                        x = norm_layer(x)
            
            A_flat = nn.Dense(eta_dim * matrix_rank, name='matrix_A_output',
                            kernel_init=lambda key, shape, dtype: nn.initializers.xavier_normal()(key, shape, dtype) / jnp.sqrt(matrix_rank),
                            bias_init=nn.initializers.zeros)(x)
        else:
            raise ValueError(f"Architecture {self.config.architecture} not supported")
        
        A = A_flat.reshape(batch_shape + (eta_dim, matrix_rank))
        Sigma = A@A.mT        
        # # Add numerical stability: scale A by 1/n where n is the matrix dimension
        # # This ensures the matrix product A @ A.T has bounded eigenvalues
        # scale_factor = 1.0 / jnp.sqrt(matrix_rank)
        # A_scaled = A * scale_factor
        # Sigma = A_scaled @ A_scaled.mT

        # Flow field: du/dt = (η_target - η_init) @ Σ
        delta_eta = eta - eta_init
        du_dt = (delta_eta[..., None, :] @ Sigma).squeeze(-2)
        
        return du_dt
    
    def compute_internal_loss(self, params: dict, eta: jnp.ndarray, 
                            predicted_stats: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Compute smoothness penalty for geometric flow dynamics.
        
        This implements the smoothness penalty that encourages stable flow dynamics
        by penalizing large derivatives du/dt during the ODE integration.
        """
        if not training or self.config.smoothness_weight <= 0:
            return 0.0
            
        # For geometric flow, we need to compute smoothness penalty
        # This requires running a forward pass to collect intermediate derivatives
        try:
            # Run forward pass with intermediate collection
            _, intermediates = self.apply(
                params, eta,  # Only eta_target needed now
                training=True, mutable=['intermediates']
            )
            
            # Smoothness penalty: penalize large derivatives du/dt
            if 'derivative_norms_squared' in intermediates and self.config.smoothness_weight > 0:
                derivative_norms_squared = intermediates['derivative_norms_squared']  # tuple of (n_time_steps,) + batch_shape
                
                # Convert tuple to array and compute penalty for large derivatives
                derivative_norms_squared_array = jnp.array(derivative_norms_squared)
                smoothness_loss = jnp.mean(derivative_norms_squared_array)
                
                return self.config.smoothness_weight * smoothness_loss
            
        except Exception:
            # If intermediate collection fails, return 0 (graceful degradation)
            pass
            
        return 0.0

