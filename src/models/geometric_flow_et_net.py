"""
Geometric Flow ET Network implementation compatible with Hugging Face.

This module provides a Hugging Face compatible Geometric Flow ET model
that learns flow dynamics for exponential families.
"""

from typing import Optional, Tuple, Dict, Any
import jax.numpy as jnp
import flax.linen as nn
from .base_model import BaseETModel
from ..configs.geometric_flow_et_config import Geometric_Flow_ET_Config
from ..layers.flow_field_net import FisherFlowFieldMLP
from ..utils.activation_utils import get_activation_function
from ..embeddings.time_embeddings import LogFreqTimeEmbedding, ConstantTimeEmbedding

class Geometric_Flow_ET_Network(BaseETModel[Geometric_Flow_ET_Config]):
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
    def __call__(self, eta: jnp.ndarray, training: bool = True, rngs: dict = None, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
        
        # Create Fisher flow field network ONCE (not in compute_flow_field)
        eta_dim = eta.shape[-1]
        matrix_rank = self.config.matrix_rank if self.config.matrix_rank is not None else eta_dim
        # Handle temporal embedding: None or 0 means disable (use constant), otherwise use specified dim
        if self.config.time_embed_dim is None or self.config.time_embed_dim == 0:
            time_embed_dim = 1  # Use dimension 1 for constant embedding
            time_embedding_fn = lambda embed_dim: ConstantTimeEmbedding(embed_dim=embed_dim)
        else:
            time_embed_dim = min(self.config.time_embed_dim, 16)
            # Ensure even dimension for log frequency embedding
            if time_embed_dim % 2 != 0:
                time_embed_dim += 1
            time_embedding_fn = lambda embed_dim: LogFreqTimeEmbedding(
                embed_dim=embed_dim,
                min_freq=self.config.time_embed_min_freq,
                max_freq=self.config.time_embed_max_freq
            )
        
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
        u = mu_init
        for i in range(self.config.n_time_steps):
            t = i * dt
            # Use the shared fisher_flow network
            # Note: deta_dt should have the same batch shape as u and eta
            du_dt = dt*fisher_flow(u, eta, t, deta_dt, training=training, rngs=rngs)
            # Forward Euler step
            u = u + du_dt            
            internal_loss += jnp.sum(du_dt ** 2, axis=-1)
        # Return predictions and internal loss
        internal_loss = jnp.array(0.0)  # For now, no internal loss
        return u, internal_loss

    def loss(self, params: dict, eta: jnp.ndarray, mu_T: jnp.ndarray, training: bool = True, rngs: dict = None) -> jnp.ndarray:
        mu_hat, internal_loss = self.apply(params, eta, training=training, rngs=rngs)
        primary_loss = jnp.mean((mu_hat - mu_T) ** 2)
        # For now, we'll compute internal loss separately if needed
        # This maintains compatibility with the unified interface
        return primary_loss

    def compute_internal_loss(self, params: dict, eta: jnp.ndarray, 
                            predicted_mu: jnp.ndarray) -> jnp.ndarray:
        """
        Helper that just computes internal losses independently.  Not used for training.  
        """
        return self.apply(params, eta, training=False, rngs={})[1]

    def predict(self, params: dict, eta: jnp.ndarray, rngs: dict = None, **kwargs) -> jnp.ndarray:
        if rngs is None:
            rngs = {}
        predictions, _ = self.apply(params, eta, training=False, rngs=rngs, **kwargs)
        return predictions
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """Load model from pretrained configuration."""
        config = Geometric_Flow_ET_Config.from_pretrained(model_name_or_path)
        return cls.from_config(config, **kwargs)
    
    def generate_initial_values(self, eta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Get the exponential family distribution from config
        dist = self.config.get_ef_distribution()
        eta_init, mu_init = dist.find_nearest_analytical_point(eta)
        return eta_init, mu_init