"""
NoProp Geometric Flow ET Network implementation compatible with Hugging Face.

This module provides a Hugging Face compatible NoProp Geometric Flow ET model
that combines geometric flow dynamics with NoProp continuous-time training protocols.
"""

from typing import Optional, List, Tuple, Union
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
import flax.linen as nn
from .base_config import BaseConfig
from ..embeddings.eta_embedding import EtaEmbedding
from ..utils.activation_utils import get_activation_function
from ..layers.flow_field_net import FisherFlowFieldMLP
from ..embeddings.time_embeddings import LogFreqTimeEmbedding, ConstantTimeEmbedding


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

@dataclass
class NoProp_Geometric_Flow_ET_Config(BaseConfig):
    """
    Configuration for NoProp Geometric Flow ET models.
    
    NoProp Geometric Flow ET models use ODE integration with flow dynamics
    and NoProp continuous-time training protocols to directly predict expected 
    sufficient statistics from natural parameters.
    """
    model_type: str = "noprop_geometric_flow_et"
    model_name: str = "noprop_geometric_flow_et_network"
    input_dim: int = 0  # Must be set by caller based on actual data
    output_dim: int = 0  # Must be set by caller based on actual data
    
    # Geometric flow specific parameters
    n_time_steps: int = 10  # Number of time steps for ODE integration
    smoothness_weight: float = 0.1  # Weight for smoothness penalty
    matrix_rank: Optional[int] = None  # Rank of the flow matrix (None = use eta_dim)
    time_embed_dim: Optional[int] = 6  # Time embedding dimension (default 6, None/0 = disable)
    time_embed_min_freq: float = 0.25  # Minimum frequency for log frequency time embedding
    time_embed_max_freq: float = 1.0  # Maximum frequency for log frequency time embedding
    
    # Network architecture parameters
    architecture: str = "mlp"  # "mlp" or "glu"
    hidden_sizes: Tuple[int, ...] = (32,32,32) 
    activation: str = "swish"  # Activation function
    use_layer_norm: bool = False  # Whether to use layer normalization (deprecated, use layer_norm_type)
    layer_norm_type: str = "weak_layer_norm"  # Type of layer normalization to use
    
    # Embedding parameters
    embedding_type: Optional[str] = "default"
    
    # Regularization parameters
    dropout_rate: float = 0.1  # Dropout rate for Fisher flow field
    
    # Model capabilities
    supports_dropout: bool = True  # Fisher flow field supports dropout
    supports_batch_norm: bool = False
    supports_layer_norm: bool = True
    
    # Exponential family distribution
    ef_distribution_name: str = "LaplaceProduct"  # Default distribution
    x_shape: tuple = (1,)  # Default x shape for the distribution
    
    # NoProp specific parameters
    noise_schedule: str = "cosine"  # Noise schedule type: "cosine", "linear", "quadratic"
    min_snr: float = 0.01  # Minimum SNR value
    max_snr: float = 10.0  # Maximum SNR value
    num_noise_steps: int = 1000  # Number of noise steps for training
    
    # Validation parameters
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.n_time_steps <= 0:
            raise ValueError("n_time_steps must be positive")
        
        if self.smoothness_weight < 0:
            raise ValueError("smoothness_weight must be non-negative")
        
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes list cannot be empty")
        
        if any(h <= 0 for h in self.hidden_sizes):
            raise ValueError("All hidden sizes must be positive")
        
        if self.architecture not in ["mlp", "glu"]:
            raise ValueError(f"architecture must be 'mlp' or 'glu', got {self.architecture}")
        
        # Validate layer norm type
        valid_layer_norm_types = [
            "none", "weak_layer_norm", "rms_norm", "group_norm", "instance_norm",
            "weight_norm", "spectral_norm", "adaptive_norm", "pre_norm", "post_norm"
        ]
        if self.layer_norm_type not in valid_layer_norm_types:
            raise ValueError(f"layer_norm_type must be one of {valid_layer_norm_types}, got {self.layer_norm_type}")
        
        # Validate noise schedule
        valid_noise_schedules = ["cosine", "linear", "quadratic"]
        if self.noise_schedule not in valid_noise_schedules:
            raise ValueError(f"noise_schedule must be one of {valid_noise_schedules}, got {self.noise_schedule}")
        
        if self.min_snr <= 0 or self.max_snr <= 0:
            raise ValueError("min_snr and max_snr must be positive")
        
        if self.min_snr >= self.max_snr:
            raise ValueError("min_snr must be less than max_snr")
    
    def _validate_model_specific(self) -> None:
        """Validate NoProp geometric flow specific parameters."""
        # NoProp geometric flow specific validation is handled in __post_init__
        pass
    
    def disable_temporal_embedding(self) -> None:
        """Disable temporal embedding by setting time_embed_dim to 1."""
        self.time_embed_dim = 1
    
    def get_architecture_summary(self) -> str:
        """Get a summary of the model architecture."""
        hidden_str = " -> ".join(map(str, self.hidden_sizes))
        resnet_info = f" + NoProp Flow({self.n_time_steps} steps)"
        
        return f"NoProp Geometric Flow: {self.input_dim} -> {hidden_str} -> {self.output_dim} ({self.activation}){resnet_info}"
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'n_time_steps': self.n_time_steps,
            'smoothness_weight': self.smoothness_weight,
            'matrix_rank': self.matrix_rank,
            'time_embed_dim': self.time_embed_dim,
            'time_embed_min_freq': self.time_embed_min_freq,
            'time_embed_max_freq': self.time_embed_max_freq,
            'architecture': self.architecture,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'use_layer_norm': self.use_layer_norm,
            'layer_norm_type': self.layer_norm_type,
            'embedding_type': self.embedding_type,
            'dropout_rate': self.dropout_rate,
            'supports_dropout': self.supports_dropout,
            'noise_schedule': self.noise_schedule,
            'min_snr': self.min_snr,
            'max_snr': self.max_snr,
            'num_noise_steps': self.num_noise_steps
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'NoProp_Geometric_Flow_ET_Config':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"NoProp_Geometric_Flow_ET_Config(type={self.model_type}, arch={self.get_architecture_summary()})"
    
    def get_ef_distribution(self):
        """Get the exponential family distribution instance."""
        from ...expfam.ef import LaplaceProduct
        
        if self.ef_distribution_name == "LaplaceProduct":
            return LaplaceProduct(x_shape=self.x_shape)
        else:
            raise ValueError(f"Unsupported distribution: {self.ef_distribution_name}")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"NoProp_Geometric_Flow_ET_Config("
                f"model_type='{self.model_type}', "
                f"input_dim={self.input_dim}, "
                f"output_dim={self.output_dim}, "
                f"n_time_steps={self.n_time_steps}, "
                f"architecture='{self.architecture}', "
                f"hidden_sizes={self.hidden_sizes}, "
                f"activation='{self.activation}')")


def create_noprop_geometric_flow_et_config(
    input_dim: int,
    output_dim: int,
    n_time_steps: int = 10,
    smoothness_weight: float = 0.1,
    matrix_rank: Optional[int] = None,
    time_embed_dim: Optional[int] = 6,
    time_embed_min_freq: float = 0.25,
    time_embed_max_freq: float = 1.0,
    architecture: str = "mlp",
    hidden_sizes: List[int] = [32, 32],
    activation: str = "swish",
    use_layer_norm: bool = False,
    layer_norm_type: str = "weak_layer_norm",
    embedding_type: Optional[str] = "default",
    noise_schedule: str = "cosine",
    min_snr: float = 0.01,
    max_snr: float = 10.0,
    num_noise_steps: int = 1000,
    **kwargs
) -> NoProp_Geometric_Flow_ET_Config:
    """
    Create a NoProp Geometric Flow ET configuration with specified parameters.
    
    Args:
        input_dim: Input dimension (natural parameters)
        output_dim: Output dimension (target statistics)
        n_time_steps: Number of time steps for ODE integration
        smoothness_weight: Weight for smoothness penalty
        matrix_rank: Rank of the flow matrix (None = use eta_dim)
        time_embed_dim: Time embedding dimension (default 6, None/0 = disable)
        time_embed_min_freq: Minimum frequency for log frequency time embedding (default 0.25)
        time_embed_max_freq: Maximum frequency for log frequency time embedding (default 1.0)
        architecture: Network architecture ("mlp" or "glu")
        hidden_sizes: Hidden layer sizes
        activation: Activation function
        use_layer_norm: Whether to use layer normalization (deprecated)
        layer_norm_type: Type of layer normalization to use
        embedding_type: Type of embedding to use
        noise_schedule: Noise schedule type
        min_snr: Minimum SNR value
        max_snr: Maximum SNR value
        num_noise_steps: Number of noise steps for training
        **kwargs: Additional configuration parameters
        
    Returns:
        NoProp Geometric Flow ET configuration object
    """
    if hidden_sizes is None:
        raise ValueError("hidden_sizes cannot be None")
    
    config = NoProp_Geometric_Flow_ET_Config(
        input_dim=input_dim,
        output_dim=output_dim,
        n_time_steps=n_time_steps,
        smoothness_weight=smoothness_weight,
        matrix_rank=matrix_rank,
        time_embed_dim=time_embed_dim,
        time_embed_min_freq=time_embed_min_freq,
        time_embed_max_freq=time_embed_max_freq,
        architecture=architecture,
        hidden_sizes=hidden_sizes,
        activation=activation,
        use_layer_norm=use_layer_norm,
        layer_norm_type=layer_norm_type,
        embedding_type=embedding_type,
        noise_schedule=noise_schedule,
        min_snr=min_snr,
        max_snr=max_snr,
        num_noise_steps=num_noise_steps,
        **kwargs
    )
    
    return config


# ============================================================================
# MODEL IMPLEMENTATION
# ============================================================================

# JIT compilation removed for debugging


class NoProp_Geometric_Flow_ET_Network(nn.Module):
    """
    NoProp Geometric Flow ET Network that learns flow dynamics using NoProp training.
    
    This network learns the flow dynamics:
        du/dt = F(z, η, t)
    
    where F is learned using NoProp continuous-time training protocols.  Moreover, because 
    this is a denoising/diffusion-like model we do not require precise initial conditions 
    for the dynamics.  Conceptually, this is like starting with a flat distribution 
    and denoising to one associated with the natural parameter η.  
    
    Key differences from regular geometric flow:
    - __call__ doesn't integrate over time (for training)
    - predict integrates over time (for inference)
    - Includes NoProp-specific noise schedules and loss functions
    - du/dt is a direct function of (z, η, t) without deta_dt dependency
    - no need to generate initial values and so works with any ef distribution specified solely by T(x)
    """
    config: NoProp_Geometric_Flow_ET_Config
    

    #########################################################
    # Noise and SNR scheduling functions.  In principle these are learnable but are fixed for now.
    # Required functions are gamma, gamma_prime, alpha_bar, snr, snr_prime, added_noise
    #########################################################
    '''
    In the original noprop paper, the authors parameterized SNR directly via a real valued gamma(t) so that
    SNR(t) = exp(-gamma(t)), where gamma(t) is a real value decreasing function.  Here we choose a different
    parameterization that seem more consistent with the continuous time formulation.  Specifically, we define
    delta(t)dt to be the variance of the weiner process noise added during the backward proccess.  
    In terms of the original no-prop paper this is equivalent to defining 1-alpha(t) = dt*delta(t) and 
    1-alpha_bar(t) = Delta(t) = Delta(T=1) + int_t^T delta(t)dt which imlies alpha_bar_prime(t) = delta(t).  
    This parameterization is easier to constrain in a manner that is consistent with the continuous time formulation.
    For example, The cumulative noise added to the backward process from t = 1 to t = 0, should be Delta(0) = 1.0.  
    This suggests the parameterization Delta(t) = exp(-gamma(t)) where gamma(0) = 0 and gamma(t) is an increasing function, 
    terminating at gamma(1) = -log(Delta(1)) > 0.  
    This lead to the following identities

        Delta(t) = exp(-gamma(t))
        1-alpha_bar(t) = Delta(t) 
        1-alpha(t) = -Delta'(t)dt = gamma'(t)exp(-gamma(t)) dt 
        SNR(t) = alpha_bar(t)/(1-alpha_bar(t)) = (1-Delta(t))/Delta(t) = exp(gamma(t)) - 1
        SNR'(t) = gamma'(t)exp(gamma(t))

    With forward prcess identities a,b,c for mu_t = a(t) u_net(z(t-dt), x , t) + b(t) z(t-dt) + c(t) noise, given by 
       a(t) = -sqrt(1-Delta(t))/Delta(t)*Delta'(t) * dt  = sqrt(1-Delta(t)) * gamma'(t) dt
       b(t) = 1 + 1/2*Delta'(t) * dt + Delta'(t)/Delta(t) dt   = 1 - gamma'(t) (Delta(t)/2 + 1) dt  
       c(t) =     ...

    '''

    def gamma_prime(self, gamma_rate: jnp.ndarray) -> jnp.ndarray:  # positive valued function so that gamma is increasing.
        return nn.softplus(gamma_rate)

    def gamma(self, t: jnp.ndarray, gamma_rate: jnp.ndarray) -> jnp.ndarray:
        return self.gamma_prime(gamma_rate) * t

    def Delta(self, t: jnp.ndarray, gamma_rate: jnp.ndarray) -> jnp.ndarray:
#        return jnp.exp(-self.gamma(t, gamma_rate))
        return 1.0-t

    def Delta_prime(self, t: jnp.ndarray, gamma_rate: jnp.ndarray) -> jnp.ndarray:
#        return -self.gamma_prime(gamma_rate) * jnp.exp(-self.gamma(t, gamma_rate))
        return -1.0

    def SNR(self, t: jnp.ndarray, gamma_rate: jnp.ndarray) -> jnp.ndarray:
#        return jnp.exp(self.gamma(t, gamma_rate)) - 1.0
        return 1.0/self.Delta(t, gamma_rate)-1
    
    def SNR_prime(self, t: jnp.ndarray, gamma_rate: jnp.ndarray) -> jnp.ndarray:
#        return jnp.exp(self.gamma(t, gamma_rate)) * self.gamma_prime(gamma_rate)
        return -self.Delta_prime(t, gamma_rate)/self.Delta(t, gamma_rate)**2

    def alpha_bar(self, t: jnp.ndarray, gamma_rate: jnp.ndarray) -> jnp.ndarray:
        # this is called alpha_bar in the paper: recall that noise added from time t to T is 1 - alpha_bar(t)
        return 1.0 - self.Delta(t, gamma_rate)

    def one_minus_alpha_over_dt(self, t: jnp.ndarray, gamma_rate: jnp.ndarray) -> jnp.ndarray:  
        return -self.Delta_prime(t, gamma_rate)

    def get_a_b_minus_1(self, t: jnp.ndarray, gamma_rate: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        Delta = self.Delta(t, gamma_rate)
        Delta_prime = self.Delta_prime(t, gamma_rate)
        gamma_prime = self.gamma_prime(gamma_rate)

        a = -jnp.sqrt(1-Delta)*Delta_prime/Delta
        b_minus_1 = (Delta/2.0 + 1)*Delta_prime/Delta
        return a, b_minus_1

    @nn.compact
    def __call__(self, z: jnp.ndarray, eta: jnp.ndarray, t: float, training: bool = True, rngs: dict = None) -> jnp.ndarray:
        """
        NoProp Network.  Builds u(z,eta,t) for use in forward pass using using the equation:
        z(t+dt) = z(t) + dt*a*u(z(t),eta,t) + dt*(b-1)*z(t) + dt*c*noise (note noise is not added in forward pass during inference)

        Args:
            z: network activity used to build estimate noisy estimate, u(z(t),eta,t)[batch_shape, eta_dim]
            eta: Natural parameters serving as input to the network [batch_shape, eta_dim]
            t: Continuous time t ∈ [0, 1]
            training: Whether in training mode
            rngs: Random number generators for dropout
            
        Returns:
            u_t: Flow field at time t [batch_shape, mu_dim]
        """
        # Define gamma_rate parameter
        gamma_rate = self.param('gamma_rate', nn.initializers.constant(4.0), ())
        
        # Get eta_dim from input
        eta_dim = eta.shape[-1]
        matrix_rank = self.config.matrix_rank if self.config.matrix_rank is not None else eta_dim
        
        # Handle temporal embedding: None or 0 means disable (use constant), otherwise use specified dim
        if self.config.time_embed_dim is None or self.config.time_embed_dim == 0:
            time_embed_dim = 1  # Use dimension 1 for constant embedding
            time_embedding_fn = lambda embed_dim: ConstantTimeEmbedding(embed_dim=embed_dim)
        else:
            time_embed_dim = max(self.config.time_embed_dim, 4)
            time_embedding_fn = LogFreqTimeEmbedding
        
        # Create Fisher flow field network
        fisher_flow = FisherFlowFieldMLP(
            dim=eta_dim,  # Output dimension (mu_dim)
            features=self.config.hidden_sizes,
            t_embed_dim=time_embed_dim,
            t_embedding_fn=time_embedding_fn,
            matrix_rank=matrix_rank,
            activation=get_activation_function(self.config.activation),
            use_layer_norm=self.config.layer_norm_type != "none",
            dropout_rate=self.config.dropout_rate
        )
        
        # Create eta embedding if specified
        if hasattr(self.config, 'embedding_type') and self.config.embedding_type is not None:
            eta_embedding = EtaEmbedding(
                embedding_type=self.config.embedding_type,
                eta_dim=eta_dim
            )
            eta_embedded = eta_embedding(eta)
        else:
            eta_embedded = eta
        
        # Use Fisher flow field: du/dt = F(z, eta, t) @ deta_dt
        # where F is the Fisher information matrix parameterized by the network
        # In NoProp context, we can use eta as deta_dt since we want du/dt = F @ eta
        u_t = fisher_flow(
            z=z,  # Current state
            x=eta_embedded,  # Target eta (embedded)
            t=t,  # Time (can be scalar or have shape that matches the batch shape)
            deta_dt=eta,  # Use eta as the "derivative" in NoProp context since eta_init\approx 0 be assumption
            training=training,
            rngs=rngs
        )
        return u_t  
    
    def _compute_noprop_loss_impl(self, params: dict, eta: jnp.ndarray, mu_T: jnp.ndarray, training: bool = True, rngs: dict = None) -> jnp.ndarray:
        """Implementation of NoProp loss (no JIT compilation for debugging)."""
        # Generate random times t ~ Uniform(0, 1) for each sample in the batch
        if rngs is not None and 'noise' in rngs:
            noise_rng = rngs['noise']
        else:
            noise_rng = jax.random.PRNGKey(0)
        t_rng, noise_rng = jax.random.split(noise_rng, 2)
        
        # Sample time for each sample in the batch (shape: eta.shape[:-1])
        batch_shape = eta.shape[:-1]
        t = jax.random.uniform(t_rng, batch_shape)  # Batch of time values
        
        # Use the flow field computation with the new __call__ method
        gamma_rate = params['params']['gamma_rate']
        Delta_t = self.Delta(t, gamma_rate)  # Now Delta_t has batch_shape
        
        z_t = mu_T*jnp.sqrt(1-Delta_t[..., None]) + jax.random.normal(noise_rng, mu_T.shape)*jnp.sqrt(Delta_t[..., None])
        u_t = self.apply(params, z_t, eta, t, training=training, rngs=rngs)
        
        # NoProp loss (penalize large derivatives)
        # SNR_prime now has batch_shape, so we need to broadcast it properly
        SNR_prime_t = self.SNR_prime(t, gamma_rate)  # Shape: batch_shape
        loss = jnp.mean(SNR_prime_t[..., None] * (u_t - mu_T) ** 2)
        return loss

    def _compute_noprop_loss(self, params: dict, eta: jnp.ndarray, mu_T: jnp.ndarray, training: bool = True, rngs: dict = None) -> jnp.ndarray:
        """Compute NoProp loss (no JIT compilation for debugging)."""
        return self._compute_noprop_loss_impl(params, eta, mu_T, training, rngs)


    def _noprop_predict(self, params: dict, eta: jnp.ndarray, n_time_steps: int = 10) -> jnp.ndarray:
        """
        Internal NoProp prediction method.
        
        Args:
            params: Model parameters
            eta: Target natural parameters [batch_shape, eta_dim]
            gamma_rate: Gamma rate parameter
            n_time_steps: Number of integration steps
            
        Returns:
            mu_target: Predicted expectations at eta [batch_shape, mu_dim]
        """
        
        # Simple forward Euler integration starting at z = 0
        dt = jnp.array(1.0 / n_time_steps)
        z_current = jnp.zeros_like(eta)
        gamma_rate = params['params']['gamma_rate']
        for i in range(n_time_steps):
            t = i * dt
            # Use the __call__ method directly
            u_t = self.apply(params, z_current, eta, t, training=False)
            a, b_minus_1 = self.get_a_b_minus_1(i * dt, gamma_rate)
            z_current = z_current + dt * (a*u_t + (b_minus_1)*z_current)
        
        return z_current
    
    def _compute_flow_matching_loss_impl(self, params: dict, eta: jnp.ndarray, mu_T: jnp.ndarray, training: bool = True, rngs: dict = None) -> jnp.ndarray:
        """Implementation of flow matching loss (no JIT compilation for debugging)."""
        # Use the provided rngs or create a default one
        if rngs is not None and 'noise' in rngs:
            noise_rng = rngs['noise']
        else:
            noise_rng = jax.random.PRNGKey(0)
        
        # Generate random times t ~ Uniform(0, 1) for each sample in the batch
        t_rng, z_0_rng, z_t_rng = jax.random.split(noise_rng, 3)
        
        # Sample time for each sample in the batch (shape: eta.shape[:-1])
        batch_shape = eta.shape[:-1]
        t = jax.random.uniform(t_rng, batch_shape)  # Batch of time values
        z_0 = jax.random.normal(z_0_rng, eta.shape)
        
        # Linear interpolation with batch-wise time values
        z_t = t[..., None]*mu_T + (1-t[..., None])*z_0
        z_t = z_t + jax.random.normal(z_t_rng, eta.shape)*self.config.flow_matching_sigma
        
        # Compute predicted flow field using the new __call__ method
        du_dt_predicted = self.apply(params, z_t, eta, t, training=training, rngs=rngs)
        
        # Compute target flow field (simplified for now)
        loss = jnp.mean((du_dt_predicted -(mu_T-z_0)) ** 2)
        return loss

    def _compute_flow_matching_loss(self, params: dict, eta: jnp.ndarray, mu_T: jnp.ndarray, training: bool = True, rngs: dict = None) -> jnp.ndarray:
        """Compute flow matching loss (no JIT compilation for debugging)."""
        return self._compute_flow_matching_loss_impl(params, eta, mu_T, training, rngs)
    
    def _flow_matching_predict(self, params: dict, eta: jnp.ndarray, n_time_steps: int = 10) -> jnp.ndarray:
        """Compute flow matching prediction."""
        # Simple forward Euler integration starting at z = 0
        dt = jnp.array(1.0 / n_time_steps)
        z_current = jnp.zeros_like(eta)
        
        for i in range(n_time_steps):
            t = i * dt
            # Use the __call__ method directly
            du_dt = self.apply(params, z_current, eta, t, training=False)
            z_current = z_current + dt * du_dt
        return z_current

    def predict(self, params: dict, eta: jnp.ndarray, n_time_steps: int = None) -> jnp.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            params: Model parameters
            eta: Natural parameters [batch_shape, eta_dim]
            n_time_steps: Number of integration steps (default from config)
            
        Returns:
            Predicted expectations [batch_shape, mu_dim]
        """
        if n_time_steps is None:
            n_time_steps = self.config.n_time_steps
            
        if self.config.loss_type == "flow_matching":
            return self._flow_matching_predict(params, eta, n_time_steps)
        elif self.config.loss_type == "noprop":
            return self._noprop_predict(params, eta, n_time_steps)
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}. Use 'flow_matching' or 'noprop'")

    def loss_fn(self, params: dict, eta: jnp.ndarray, mu_T: jnp.ndarray, training: bool = True, rngs: dict = None) -> jnp.ndarray:
        """
        Compute NoProp-specific loss functions.
        
        Args:
            params: Model parameters
            eta: Natural parameters [batch_shape, eta_dim]
            mu_T: Target expectations [batch_shape, mu_dim]
            training: Whether in training mode
            rngs: Random number generators for dropout
            
        Returns:
            loss: NoProp loss value
        """
        if self.config.loss_type == "flow_matching":
            return self._compute_flow_matching_loss(params, eta, mu_T, training=training, rngs=rngs)
        elif self.config.loss_type == "noprop":
            return self._compute_noprop_loss(params, eta, mu_T, training=training, rngs=rngs)
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}. Use 'flow_matching' or 'noprop'")
