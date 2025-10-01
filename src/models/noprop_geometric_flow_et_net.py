"""
NoProp Geometric Flow ET Network implementation compatible with Hugging Face.

This module provides a Hugging Face compatible NoProp Geometric Flow ET model
that combines geometric flow dynamics with NoProp continuous-time training protocols.
"""

from typing import Optional, List, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
from ..configs.noprop_geometric_flow_et_config import NoProp_Geometric_Flow_ET_Config
from ..embeddings.eta_embedding import EtaEmbedding
from ..utils.activation_utils import get_activation_function
from ..layers.normalization import get_normalization_layer
from ..layers.flow_field_net import FisherFlowFieldMLP
from ..embeddings.time_embeddings import LogFreqTimeEmbedding, ConstantTimeEmbedding


class NoProp_Geometric_Flow_ET_Network(nn.Module):
    """
    NoProp Geometric Flow ET Network that learns flow dynamics using NoProp training.
    
    This network learns the geometric flow dynamics:
        du/dt = A(u, t, η_t) @ A(u, t, η_t)^T @ (η_target - η_init)
    
    where A is learned using NoProp continuous-time training protocols.
    
    Key differences from regular geometric flow:
    - __call__ doesn't integrate over time (for training)
    - predict integrates over time (for inference)
    - Includes NoProp-specific noise schedules and loss functions
    """
    config: NoProp_Geometric_Flow_ET_Config
    
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
    
    def get_gamma_at_time(self, t: float) -> float:
        """Get γ(t) at continuous time t ∈ [0, 1]."""
        if self.config.noise_schedule == "noprop_ct":
            return jnp.array(t)
        elif self.config.noise_schedule == "flow_matching":
            return jnp.array(0.0)  # Not directly used in flow matching
        else:
            raise ValueError(f"Unknown noise schedule: {self.config.noise_schedule}")
    
    def get_noise_at_time(self, t: float) -> float:
        """Get noise level at continuous time t ∈ [0, 1] using ᾱ_t = σ(-γ(t))."""
        gamma_t = self.get_gamma_at_time(t)
        alpha_bar_t = jax.nn.sigmoid(-gamma_t)
        return alpha_bar_t
    
    def get_snr_at_time(self, t: float) -> float:
        """Get signal-to-noise ratio at continuous time t ∈ [0, 1]."""
        alpha_bar_t = self.get_noise_at_time(t)
        return alpha_bar_t / (1 - alpha_bar_t)
    
    def get_snr_derivative_at_time(self, t: float) -> float:
        """Get derivative of SNR at continuous time t ∈ [0, 1]."""
        gamma_t = self.get_gamma_at_time(t)
        gamma_prime_t = 1.0  # For linear γ(t) = t
        alpha_bar_t = self.get_noise_at_time(t)
        alpha_bar_prime = -gamma_prime_t * alpha_bar_t * (1 - alpha_bar_t)
        snr_derivative = alpha_bar_prime / ((1 - alpha_bar_t) ** 2)
        return snr_derivative
    
    def add_noise(self, x: jnp.ndarray, noise: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Add noise to input at continuous time t ∈ [0, 1] using variance-preserving OU process."""
        # Handle both scalar and batch t values
        if t.ndim == 0:
            # Scalar t
            alpha_bar_t = self.get_noise_at_time(t)
            sqrt_alpha_bar = jnp.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar = jnp.sqrt(1 - alpha_bar_t)
            return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
        else:
            # Batch t values
            alpha_bar_t = jax.vmap(self.get_noise_at_time)(t)
            sqrt_alpha_bar = jnp.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar = jnp.sqrt(1 - alpha_bar_t)
            # Broadcast to match x and noise shapes
            sqrt_alpha_bar = sqrt_alpha_bar[..., None]
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar[..., None]
            return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
    
    @nn.compact
    def __call__(self, eta_init: jnp.ndarray, eta_target: jnp.ndarray, 
                 mu_current: jnp.ndarray, t: float, training: bool = True, rngs: dict = None) -> jnp.ndarray:
        """
        NoProp training call - computes du/dt at a single time point (no integration).
        
        Args:
            eta_init: Initial natural parameters [batch_shape, eta_dim]
            eta_target: Target natural parameters [batch_shape, eta_dim]
            mu_current: Current mu values [batch_shape, mu_dim]
            t: Continuous time t ∈ [0, 1]
            training: Whether in training mode
            rngs: Random number generators for dropout
            
        Returns:
            du_dt: Flow field at time t [batch_shape, mu_dim]
        """
        eta_dim = eta_target.shape[-1]
        mu_dim = mu_current.shape[-1]
        matrix_rank = self.config.matrix_rank if self.config.matrix_rank is not None else eta_dim
        
        # Handle temporal embedding: None or 0 means disable (use constant), otherwise use specified dim
        if self.config.time_embed_dim is None or self.config.time_embed_dim == 0:
            time_embed_dim = 1  # Use dimension 1 for constant embedding
            time_embedding_fn = lambda embed_dim: ConstantTimeEmbedding(embed_dim=embed_dim)
        else:
            time_embed_dim = min(self.config.time_embed_dim, 16)
            time_embedding_fn = LogFreqTimeEmbedding
        
        # Create Fisher flow field network (shared across all time steps)
        fisher_flow = FisherFlowFieldMLP(
            dim=mu_dim,  # Output dimension (mu_dim)
            features=self.config.hidden_sizes,
            t_embed_dim=time_embed_dim,
            t_embedding_fn=time_embedding_fn,
            matrix_rank=matrix_rank,
            activation=get_activation_function(self.config.activation),
            use_layer_norm=self.config.layer_norm_type != "none",
            dropout_rate=self.config.dropout_rate
        )
        
        # Apply eta embedding if specified
        if hasattr(self.config, 'embedding_type') and self.config.embedding_type is not None:
            eta_embedding = EtaEmbedding(
                embedding_type=self.config.embedding_type,
                eta_dim=eta_target.shape[-1]
            )
            eta_embedded = eta_embedding(eta_target)
            eta_init_embedded = eta_embedding(eta_init)
        else:
            eta_embedded = eta_target
            eta_init_embedded = eta_init
        
        # Compute deta_dt = eta_target - eta_init
        deta_dt = eta_target - eta_init
        
        # Use Fisher flow field: du/dt = F(eta) @ deta_dt
        # where F(eta) is the Fisher information matrix parameterized by the network
        du_dt = fisher_flow(
            z=mu_current,  # Current state
            x=eta_embedded,  # Target eta (embedded)
            t=t,  # Time (can be scalar or batch)
            deta_dt=deta_dt,  # Natural parameter derivative
            training=training,
            rngs=rngs
        )
        
        return du_dt
    
    
    def predict(self, eta: jnp.ndarray, params: dict = None, **kwargs) -> jnp.ndarray:
        """
        Predict method for NoProp geometric flow - integrates over time for inference.
        
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
            eta_init, mu_init = self.generate_initial_values(eta)
            t = 0.5  # Dummy time for initialization
            params = self.init(key, eta_init, eta, mu_init, t)
        
        # Generate initial values
        eta_init, mu_init = self.generate_initial_values(eta)
        
        # Simple forward Euler integration (same as regular geometric flow)
        dt = 1.0 / self.config.n_time_steps
        u_current = mu_init
        
        for i in range(self.config.n_time_steps):
            t = i * dt
            # Use the __call__ method directly (no need for method parameter)
            du_dt = self.apply(params, eta_init, eta, u_current, t, training=False)
            u_current = u_current + dt * du_dt
        
        return u_current
    
    def compute_noprop_loss(self, params: dict, eta_init: jnp.ndarray, eta_target: jnp.ndarray,
                           mu_init: jnp.ndarray, t: jnp.ndarray, training: bool = True, rngs: dict = None) -> jnp.ndarray:
        """
        Compute NoProp-specific loss functions.
        
        Args:
            params: Model parameters
            eta_init: Initial natural parameters [batch_shape, eta_dim]
            eta_target: Target natural parameters [batch_shape, eta_dim]
            mu_init: Initial mu values [batch_shape, mu_dim]
            t: Continuous time t ∈ [0, 1] [batch_shape]
            training: Whether in training mode
            
        Returns:
            loss: NoProp loss value
        """
        if self.config.loss_type == "flow_matching":
            return self._compute_flow_matching_loss(params, eta_init, eta_target, mu_init, t, rngs)
        elif self.config.loss_type == "geometric_flow":
            return self._compute_geometric_flow_loss(params, eta_init, eta_target, mu_init, t, rngs)
        elif self.config.loss_type == "simple_target":
            return self._compute_simple_target_loss(params, eta_init, eta_target, mu_init, t, rngs)
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
    
    def _compute_flow_matching_loss(self, params: dict, eta_init: jnp.ndarray, eta_target: jnp.ndarray,
                                   mu_init: jnp.ndarray, t: jnp.ndarray, rngs: dict = None) -> jnp.ndarray:
        """Compute flow matching loss."""
        # Sample random noise for each sample in batch
        noise = jax.random.normal(jax.random.PRNGKey(0), mu_init.shape)
        
        # Add noise to mu_init for each sample
        mu_current = self.add_noise(mu_init, noise, t)
        
        # Compute predicted flow field using the new __call__ method
        du_dt_predicted = self.apply(params, eta_init, eta_target, mu_current, t, training=True, rngs=rngs)
        
        # Compute target flow field (simplified for now)
        du_dt_target = eta_target - eta_init  # Simplified target
        
        # Flow matching loss
        loss = jnp.mean((du_dt_predicted - du_dt_target) ** 2)
        return loss
    
    def _compute_geometric_flow_loss(self, params: dict, eta_init: jnp.ndarray, eta_target: jnp.ndarray,
                                    mu_init: jnp.ndarray, t: jnp.ndarray, rngs: dict = None) -> jnp.ndarray:
        """Compute geometric flow loss."""
        # Use the flow field computation with the new __call__ method
        du_dt = self.apply(params, eta_init, eta_target, mu_init, t, training=True, rngs=rngs)
        
        # Geometric flow loss (penalize large derivatives)
        loss = jnp.mean(du_dt ** 2)
        return loss
    
    def _compute_simple_target_loss(self, params: dict, eta_init: jnp.ndarray, eta_target: jnp.ndarray,
                                   mu_init: jnp.ndarray, t: jnp.ndarray, rngs: dict = None) -> jnp.ndarray:
        """Compute simple target loss."""
        # Predict final mu
        mu_predicted = self.predict(eta_target, params)
        
        # Simple target loss
        loss = jnp.mean((mu_predicted - eta_target) ** 2)
        return loss