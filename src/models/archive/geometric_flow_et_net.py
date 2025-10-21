"""
Geometric Flow ET Network implementation compatible with Hugging Face.

This module provides a Hugging Face compatible Geometric Flow ET model
that learns flow dynamics for exponential families.
"""

from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field
import jax.numpy as jnp
import flax.linen as nn
from .base_model import BaseModel
from .base_config import BaseConfig
from ..layers.flow_field_net import FisherFlowFieldMLP
from ..utils.activation_utils import get_activation_function
from ..embeddings.time_embeddings import LogFreqTimeEmbedding, ConstantTimeEmbedding


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

@dataclass
class Geometric_Flow_ET_Config(BaseConfig):
    """
    Configuration for Geometric Flow ET models.
    
    Geometric Flow ET models use ODE integration with flow dynamics
    to directly predict expected sufficient statistics from natural parameters.
    """
    model_type: str = "geometric_flow_et"
    model_name: str = "geometric_flow_et_network"
    input_dim: int = 0  # Must be set by caller based on actual data
    output_dim: int = 0  # Must be set by caller based on actual data
    
    # Geometric flow specific parameters
    n_time_steps: int = 10  # Number of time steps for ODE integration
    smoothness_weight: float = 0.0  # Weight for smoothness penalty
    matrix_rank: Optional[int] = None  # Rank of the flow matrix (None = use eta_dim)
    time_embed_dim: Optional[int] = 6  # Time embedding dimension (default 6, None/0 = disable)
    time_embed_min_freq: float = 0.25  # Minimum frequency for log frequency time embedding
    time_embed_max_freq: float = 4.0  # Maximum frequency for log frequency time embedding
    
    # Network architecture parameters
    architecture: str = "mlp"  # "mlp" or "glu"
    hidden_sizes: List[int] = field(default_factory=lambda: [32, 32, 32])  # Hidden layer sizes
    activation: str = "swish"  # Activation function
    use_layer_norm: bool = False  # Whether to use layer normalization (deprecated, use layer_norm_type)
    layer_norm_type: str = "weak_layer_norm"  # Type of layer normalization to use (None, weak_layer_norm, rms_norm, group_norm, instance_norm, weight_norm, spectral_norm, adaptive_norm, pre_norm, post_norm)
    
    # Embedding parameters
    embedding_type: Optional[str] = "default"
    
    # Regularization parameters
    dropout_rate: float = 0.1  # Dropout rate for regularization
    
    # Model capabilities
    supports_dropout: bool = True  # Geometric flow supports dropout
    supports_batch_norm: bool = False
    supports_layer_norm: bool = True
    
    # Exponential family distribution
    ef_distribution_name: str = "LaplaceProduct"  # Default distribution
    x_shape: tuple = (1,)  # Default x shape for the distribution
    
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
    
    def _validate_model_specific(self) -> None:
        """Validate geometric flow specific parameters."""
        # Geometric flow specific validation is handled in __post_init__
        pass
    
    def disable_temporal_embedding(self) -> None:
        """Disable temporal embedding by setting time_embed_dim to 1."""
        self.time_embed_dim = 1
    
    def get_architecture_summary(self) -> str:
        """Get a summary of the model architecture."""
        hidden_str = " -> ".join(map(str, self.hidden_sizes))
        resnet_info = f" + Flow({self.n_time_steps} steps)"
        
        return f"Geometric Flow: {self.input_dim} -> {hidden_str} -> {self.output_dim} ({self.activation}){resnet_info}"
    
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
            'supports_dropout': self.supports_dropout
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Geometric_Flow_ET_Config':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Geometric_Flow_ET_Config(type={self.model_type}, arch={self.get_architecture_summary()})"
    
    def get_ef_distribution(self):
        """Get the exponential family distribution instance."""
        from ...expfam.ef import LaplaceProduct
        
        if self.ef_distribution_name == "LaplaceProduct":
            return LaplaceProduct(x_shape=self.x_shape)
        else:
            raise ValueError(f"Unsupported distribution: {self.ef_distribution_name}")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Geometric_Flow_ET_Config("
                f"model_type='{self.model_type}', "
                f"input_dim={self.input_dim}, "
                f"output_dim={self.output_dim}, "
                f"n_time_steps={self.n_time_steps}, "
                f"architecture='{self.architecture}', "
                f"hidden_sizes={self.hidden_sizes}, "
                f"activation='{self.activation}')")


def create_geometric_flow_et_config(
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
    **kwargs
) -> Geometric_Flow_ET_Config:
    """
    Create a Geometric Flow ET configuration with specified parameters.
    
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
        **kwargs: Additional configuration parameters
        
    Returns:
        Geometric Flow ET configuration object
    """
    if hidden_sizes is None:
        raise ValueError("hidden_sizes cannot be None")
    
    config = Geometric_Flow_ET_Config(
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
        **kwargs
    )
    
    return config


# ============================================================================
# MODEL IMPLEMENTATION
# ============================================================================

class Geometric_Flow_ET_Network(BaseModel[Geometric_Flow_ET_Config]):
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