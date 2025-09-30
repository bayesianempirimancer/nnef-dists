"""
NoProp Geometric Flow ET Configuration

This module provides configuration classes for NoProp Geometric Flow ET models,
following the same patterns as regular Geometric Flow ET configurations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from .base_model_config import BaseModelConfig


@dataclass
class NoProp_Geometric_Flow_ET_Config(BaseModelConfig):
    """
    Configuration for NoProp Geometric Flow ET models.
    
    NoProp Geometric Flow ET models use ODE integration with flow dynamics
    and NoProp continuous-time training protocols to directly predict expected 
    sufficient statistics from natural parameters.
    """
    model_type: str = "noprop_geometric_flow_et"
    model_name: str = "noprop_geometric_flow_et_network"
    input_dim: int = 4
    output_dim: int = 4
    
    # Geometric flow specific parameters
    n_time_steps: int = 10  # Number of time steps for ODE integration
    smoothness_weight: float = 0.1  # Weight for smoothness penalty
    matrix_rank: Optional[int] = None  # Rank of the flow matrix (None = use eta_dim)
    time_embed_dim: Optional[int] = None  # Time embedding dimension (None = use eta_dim)
    
    # Network architecture parameters
    architecture: str = "mlp"  # "mlp" or "glu"
    hidden_sizes: List[int] = field(default_factory=lambda: [32, 32])  # Hidden layer sizes
    activation: str = "swish"  # Activation function
    use_layer_norm: bool = False  # Whether to use layer normalization (deprecated, use layer_norm_type)
    layer_norm_type: str = "weak_layer_norm"  # Type of layer normalization to use
    
    # Embedding parameters
    embedding_type: Optional[str] = "default"
    
    # Regularization parameters
    dropout_rate: float = 0.0  # Geometric flow doesn't use dropout by default
    
    # Model capabilities
    supports_dropout: bool = False  # Geometric flow doesn't support dropout
    supports_batch_norm: bool = False
    supports_layer_norm: bool = True
    
    # NoProp-specific parameters
    noise_schedule: str = "noprop_ct"  # "noprop_ct" or "flow_matching"
    loss_type: str = "geometric_flow"  # "flow_matching", "geometric_flow", or "simple_target"
    
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
        
        # Validate NoProp-specific parameters
        if self.noise_schedule not in ["noprop_ct", "flow_matching"]:
            raise ValueError(f"noise_schedule must be 'noprop_ct' or 'flow_matching', got {self.noise_schedule}")
        
        if self.loss_type not in ["flow_matching", "geometric_flow", "simple_target"]:
            raise ValueError(f"loss_type must be one of ['flow_matching', 'geometric_flow', 'simple_target'], got {self.loss_type}")
    
    def _validate_model_specific(self) -> None:
        """Validate NoProp geometric flow specific parameters."""
        # NoProp-specific validation is handled in __post_init__
        pass
    
    def get_architecture_summary(self) -> str:
        """Get a summary of the model architecture."""
        hidden_str = " -> ".join(map(str, self.hidden_sizes))
        resnet_info = f" + Flow({self.n_time_steps} steps)"
        
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
            'architecture': self.architecture,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'use_layer_norm': self.use_layer_norm,
            'layer_norm_type': self.layer_norm_type,
            'embedding_type': self.embedding_type,
            'noise_schedule': self.noise_schedule,
            'loss_type': self.loss_type,
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
        from ..ef import LaplaceProduct
        
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
                f"activation='{self.activation}', "
                f"noise_schedule='{self.noise_schedule}', "
                f"loss_type='{self.loss_type}')")


def create_noprop_geometric_flow_et_config(
    input_dim: int,
    output_dim: int,
    n_time_steps: int = 10,
    smoothness_weight: float = 0.1,
    matrix_rank: Optional[int] = None,
    time_embed_dim: Optional[int] = None,
    architecture: str = "mlp",
    hidden_sizes: List[int] = [32, 32],
    activation: str = "swish",
    use_layer_norm: bool = False,
    layer_norm_type: str = "weak_layer_norm",
    embedding_type: Optional[str] = "default",
    noise_schedule: str = "noprop_ct",
    loss_type: str = "geometric_flow",
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
        time_embed_dim: Time embedding dimension (None = use eta_dim)
        architecture: Network architecture ("mlp" or "glu")
        hidden_sizes: Hidden layer sizes
        activation: Activation function
        use_layer_norm: Whether to use layer normalization (deprecated)
        layer_norm_type: Type of layer normalization to use
        embedding_type: Type of embedding to use
        noise_schedule: NoProp noise schedule ("noprop_ct" or "flow_matching")
        loss_type: Loss function type ("flow_matching", "geometric_flow", or "simple_target")
        **kwargs: Additional configuration parameters
        
    Returns:
        NoProp Geometric Flow ET configuration object
    """
    if hidden_sizes is None:
        raise ValueError("hidden_sizes cannot be None")
    
    return NoProp_Geometric_Flow_ET_Config(
        input_dim=input_dim,
        output_dim=output_dim,
        n_time_steps=n_time_steps,
        smoothness_weight=smoothness_weight,
        matrix_rank=matrix_rank,
        time_embed_dim=time_embed_dim,
        architecture=architecture,
        hidden_sizes=hidden_sizes,
        activation=activation,
        use_layer_norm=use_layer_norm,
        layer_norm_type=layer_norm_type,
        embedding_type=embedding_type,
        noise_schedule=noise_schedule,
        loss_type=loss_type,
        **kwargs
    )
