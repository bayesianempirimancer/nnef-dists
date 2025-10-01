"""
Geometric Flow ET Network implementation compatible with Hugging Face.

This module provides a Hugging Face compatible Geometric Flow ET model
that learns flow dynamics for exponential families.
"""

from typing import Optional, Tuple
import jax.numpy as jnp
import flax.linen as nn
from ..configs.geometric_flow_et_config import Geometric_Flow_ET_Config
from ..layers.flow_field_net import FisherFlowFieldMLP
from ..utils.activation_utils import get_activation_function
from ..embeddings.time_embeddings import SimpleTimeEmbedding, ConstantTimeEmbedding

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

    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True, rngs: dict = None, **kwargs) -> jnp.ndarray:
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
        deta_dt = eta - eta_init
        u = mu_init
        
        # Create Fisher flow field network ONCE (not in compute_flow_field)
        eta_dim = eta.shape[-1]
        matrix_rank = self.config.matrix_rank if self.config.matrix_rank is not None else eta_dim
        # Handle temporal embedding: None or 0 means disable (use constant), otherwise use specified dim
        if self.config.time_embed_dim is None or self.config.time_embed_dim == 0:
            time_embed_dim = 1  # Use dimension 1 for constant embedding
            time_embedding_fn = lambda embed_dim: ConstantTimeEmbedding(embed_dim=embed_dim)
        else:
            time_embed_dim = min(self.config.time_embed_dim, 16)
            time_embedding_fn = SimpleTimeEmbedding
        
        # Get activation function
        activation_fn = get_activation_function(self.config.activation)
        
        # Create Fisher flow field network (shared across all time steps)
        fisher_flow = FisherFlowFieldMLP(
            dim=eta_dim,  # Output dimension (mu_dim)
            features=self.config.hidden_sizes,
            t_embed_dim=time_embed_dim,
            t_embedding_fn=time_embedding_fn,
            matrix_rank=matrix_rank,
            activation=activation_fn,
            use_layer_norm=self.config.layer_norm_type != "none",
            dropout_rate=self.config.dropout_rate
        )
        
        # Simple forward Euler integration
        dt = 1.0 / self.config.n_time_steps
        internal_loss = 0.0
        for i in range(self.config.n_time_steps):
            t = i * dt
            # Use the shared fisher_flow network
            # Note: deta_dt should have the same batch shape as u and eta
            du_dt = fisher_flow(u, eta, t, deta_dt, training=training, rngs=rngs)
            # Forward Euler step
            u = u + dt * du_dt            
            internal_loss += jnp.sum(du_dt ** 2, axis=-1)
        return u, internal_loss

    def loss_fn(self, params: dict, eta: jnp.ndarray, mu_T: jnp.ndarray, training: bool = True, rngs: dict = None) -> jnp.ndarray:
        """Legacy loss method - kept for backward compatibility."""
        mu_hat, internal_loss = self.apply(params, eta, training=training, rngs=rngs)
        primary_loss = jnp.mean((mu_hat - mu_T) ** 2)
        smoothness_loss = jnp.mean(internal_loss)
        return primary_loss + self.config.smoothness_weight * smoothness_loss

    def compute_internal_loss(self, params: dict, eta: jnp.ndarray, 
                            predicted_mu: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Helper that just computes internal losses independently.  Not used.  
        """
        return 0.0

    @nn.compact
    def forward(self, eta: jnp.ndarray, training: bool = True, **kwargs) -> jnp.ndarray:
        """
        Forward pass (alias for __call__ for compatibility).
        """
        return self.__call__(eta, training=training, **kwargs)[0]  # Return only mu_hat, not internal_loss

    def predict(self, params: dict, eta: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self.apply(params, eta, training=False, **kwargs)[0]

    def generate_initial_values(self, eta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Get the exponential family distribution from config
        dist = self.config.get_ef_distribution()
        eta_init, mu_init = dist.find_nearest_analytical_point(eta)
        return eta_init, mu_init