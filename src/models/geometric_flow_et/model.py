"""
Geometric Flow ET Network implementation compatible with Hugging Face.

This module provides a Hugging Face compatible Geometric Flow ET model
that learns flow dynamics for exponential families.
"""

from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import jax.numpy as jnp
import flax.linen as nn
from ..base_model import BaseModel
from ..base_config import BaseConfig
from ...layers.flow_field_net import FisherFlowFieldMLP
from ...utils.activation_utils import get_activation_function
from ...embeddings.time_embeddings import LogFreqTimeEmbedding, ConstantTimeEmbedding


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

@dataclass
class Config(BaseConfig):
    """
    Configuration for Geometric Flow ET models.
    
    Inherits all common parameters from BaseConfig.
    """
    
    # === MODEL IDENTIFICATION ===
    model_name: str = "geometric_flow_et_net"
    
    # === GEOMETRIC FLOW SPECIFIC PARAMETERS ===
    n_time_steps: int = 10  # Number of time steps for ODE integration
    smoothness_weight: float = 0.0  # Weight for smoothness penalty
    matrix_rank: Optional[int] = None  # Rank of the flow matrix (None = use eta_dim)
    architecture: str = "mlp"  # "mlp" or "glu"
    layer_norm_type: str = "weak_layer_norm"  # Type of layer normalization to use
    
    # === EXPONENTIAL FAMILY DISTRIBUTION ===
    ef_distribution_name: str = "LaplaceProduct"  # Default distribution
    x_shape: tuple = (1,)  # Default x shape for the distribution
    
    
    
    def disable_temporal_embedding(self) -> None:
        """Disable temporal embedding by setting time_embed_dim to 1."""
        self.time_embed_dim = 1
    
    def get_ef_distribution(self):
        """Get the exponential family distribution instance."""
        from ...ef import LaplaceProduct
        
        if self.ef_distribution_name == "LaplaceProduct":
            return LaplaceProduct(x_shape=self.x_shape)
        else:
            raise ValueError(f"Unsupported distribution: {self.ef_distribution_name}")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Config("
                f"model_type='{self.model_type}', "
                f"input_dim={self.input_dim}, "
                f"output_dim={self.output_dim}, "
                f"n_time_steps={self.n_time_steps}, "
                f"architecture='{self.architecture}', "
                f"hidden_sizes={self.hidden_sizes}, "
                f"activation='{self.activation}')")




# ============================================================================
# MODEL IMPLEMENTATION
# ============================================================================

class Geometric_Flow_ET_Net(BaseModel[Config]):
    """
    Geometric Flow ET Network that learns flow dynamics for exponential families.
    
    The network learns A(u, t, η_t) such that:
        du/dt = A(u, t, η_t) @ A(u, t, η_t)^T @ (η_target - η_init)
    
    Key features:
    - Geometric flow dynamics with PSD constraints
    - Smoothness penalties for stable dynamics
    - Minimal time steps due to expected smoothness
    """
    config: Config

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
    
    
    def generate_initial_values(self, eta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Get the exponential family distribution from config
        dist = self.config.get_ef_distribution()
        eta_init, mu_init = dist.find_nearest_analytical_point(eta)
        return eta_init, mu_init