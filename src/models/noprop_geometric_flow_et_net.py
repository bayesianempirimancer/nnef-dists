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
    def compute_flow_field(self, u: jnp.ndarray, t: jnp.ndarray, eta: jnp.ndarray, 
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
        embed_dim = min(time_embed_dim, 16)
        
        from ..embeddings.time_embeddings import SimpleTimeEmbedding
        
        # Handle batch time values
        if t.ndim == 0:
            # Scalar t
            t_embed = SimpleTimeEmbedding(embed_dim=embed_dim)(t)
            t_embed_batch = jnp.broadcast_to(t_embed, batch_shape + (embed_dim,))
        else:
            # Batch t values - use vmap for time embedding
            t_embed_batch = jax.vmap(lambda t_val: SimpleTimeEmbedding(embed_dim=embed_dim)(t_val))(t)
        
        # Apply eta embedding if specified
        if hasattr(self.config, 'embedding_type') and self.config.embedding_type is not None:
            eta_embedding = EtaEmbedding(
                embedding_type=self.config.embedding_type,
                eta_dim=eta.shape[-1]
            )
            eta_embedded = eta_embedding(eta)
            eta_init_embedded = eta_embedding(eta_init)
        else:
            eta_embedded = eta
            eta_init_embedded = eta_init
        
        # Network input: [u, sin/cos(t), η_init, η_target]
        net_input = jnp.concatenate([u, t_embed_batch, eta_init_embedded, eta_embedded], axis=-1)
        
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
        Sigma = A @ A.mT
        
        # Flow field: du/dt = (η_target - η_init) @ Σ
        delta_eta = eta - eta_init
        du_dt = (delta_eta[..., None, :] @ Sigma).squeeze(-2)
        
        return du_dt
    
    @nn.compact
    def __call__(self, eta_init: jnp.ndarray, eta_target: jnp.ndarray, 
                 mu_current: jnp.ndarray, t: float, training: bool = True) -> jnp.ndarray:
        """
        NoProp training call - computes du/dt at a single time point (no integration).
        
        Args:
            eta_init: Initial natural parameters [batch_shape, eta_dim]
            eta_target: Target natural parameters [batch_shape, eta_dim]
            mu_current: Current mu values [batch_shape, mu_dim]
            t: Continuous time t ∈ [0, 1]
            training: Whether in training mode
            
        Returns:
            du_dt: Flow field at time t [batch_shape, mu_dim]
        """
        # Use inherited compute_flow_field method
        du_dt = self.compute_flow_field(mu_current, t, eta_target, eta_init)
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
            # Broadcast scalar t to batch shape
            t_batch = jnp.full(u_current.shape[:-1], t)
            du_dt = self.apply(params, u_current, t_batch, eta, eta_init, method=self.compute_flow_field)
            u_current = u_current + dt * du_dt
        
        return u_current
    
    def compute_noprop_loss(self, params: dict, eta_init: jnp.ndarray, eta_target: jnp.ndarray,
                           mu_init: jnp.ndarray, t: jnp.ndarray, training: bool = True) -> jnp.ndarray:
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
            return self._compute_flow_matching_loss(params, eta_init, eta_target, mu_init, t)
        elif self.config.loss_type == "geometric_flow":
            return self._compute_geometric_flow_loss(params, eta_init, eta_target, mu_init, t)
        elif self.config.loss_type == "simple_target":
            return self._compute_simple_target_loss(params, eta_init, eta_target, mu_init, t)
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
    
    def _compute_flow_matching_loss(self, params: dict, eta_init: jnp.ndarray, eta_target: jnp.ndarray,
                                   mu_init: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Compute flow matching loss."""
        # Sample random noise for each sample in batch
        noise = jax.random.normal(jax.random.PRNGKey(0), mu_init.shape)
        
        # Add noise to mu_init for each sample
        mu_current = self.add_noise(mu_init, noise, t)
        
        # Compute predicted flow field
        du_dt_predicted = self.apply(params, mu_current, t, eta_target, eta_init, method=self.compute_flow_field)
        
        # Compute target flow field (simplified for now)
        du_dt_target = eta_target - eta_init  # Simplified target
        
        # Flow matching loss
        loss = jnp.mean((du_dt_predicted - du_dt_target) ** 2)
        return loss
    
    def _compute_geometric_flow_loss(self, params: dict, eta_init: jnp.ndarray, eta_target: jnp.ndarray,
                                    mu_init: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Compute geometric flow loss."""
        # Use the flow field computation
        du_dt = self.apply(params, mu_init, t, eta_target, eta_init, method=self.compute_flow_field, training=True)
        
        # Geometric flow loss (penalize large derivatives)
        loss = jnp.mean(du_dt ** 2)
        return loss
    
    def _compute_simple_target_loss(self, params: dict, eta_init: jnp.ndarray, eta_target: jnp.ndarray,
                                   mu_init: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Compute simple target loss."""
        # Predict final mu
        mu_predicted = self.predict(eta_target, params)
        
        # Simple target loss
        loss = jnp.mean((mu_predicted - eta_target) ** 2)
        return loss