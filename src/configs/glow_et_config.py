"""
GLOW ET Configuration

This module provides configuration classes for GLOW ET models,
following the same patterns as MLP and GLU ET configurations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from .base_model_config import BaseModelConfig


@dataclass
class Glow_ET_Config(BaseModelConfig):
    """
    Configuration for GLOW ET models.
    
    GLOW (Glow-based) ET models use affine coupling layers for normalizing flows
    to directly predict expected sufficient statistics from natural parameters.
    """
    model_type: str = "glow_et"
    model_name: str = "glow_et_network"
    input_dim: int = 4
    output_dim: int = 4
    
    # GLOW-specific architecture parameters
    num_flow_layers: int = 4  # Number of affine coupling layers
    features: List[int] = field(default_factory=lambda: [64, 64])  # Hidden layer sizes for coupling networks
    activation: str = "swish"  # Activation function for coupling networks
    
    # Affine coupling layer parameters
    use_residual: bool = False  # Whether to use residual connections in coupling networks
    use_actnorm: bool = False  # Whether to use activation normalization
    residual_weight: float = 1.0  # Weight for residual connections
    log_scale_clamp: Tuple[float, float] = (-2.0, 2.0)  # Clamping range for log_scale
    translation_clamp: Tuple[float, float] = (-1.0, 1.0)  # Clamping range for translation
    dropout_rate: float = 0.1  # Dropout rate for regularization
    
    # Embedding parameters (for future use)
    embedding_type: Optional[str] = 'default'
    
    # Validation parameters
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Validate GLOW-specific parameters
        if self.num_flow_layers < 1:
            raise ValueError("num_flow_layers must be at least 1")
        
        if not self.features:
            raise ValueError("features list cannot be empty")
        
        if any(f <= 0 for f in self.features):
            raise ValueError("All feature sizes must be positive")
        
        if self.dropout_rate < 0.0 or self.dropout_rate > 1.0:
            raise ValueError("dropout_rate must be between 0.0 and 1.0")
        
        if self.residual_weight <= 0.0:
            raise ValueError("residual_weight must be positive")
        
        if len(self.log_scale_clamp) != 2 or self.log_scale_clamp[0] >= self.log_scale_clamp[1]:
            raise ValueError("log_scale_clamp must be a tuple (min, max) with min < max")
        
        if len(self.translation_clamp) != 2 or self.translation_clamp[0] >= self.translation_clamp[1]:
            raise ValueError("translation_clamp must be a tuple (min, max) with min < max")
    
    def _validate_model_specific(self) -> None:
        """Validate GLOW-specific parameters."""
        # GLOW-specific validation is handled in __post_init__
        pass
    
    def get_architecture_summary(self) -> str:
        """Get a summary of the model architecture."""
        features_str = " -> ".join(map(str, self.features))
        flow_info = f" + Flow({self.num_flow_layers} layers)"
        
        return f"GLOW: {self.input_dim} -> [{features_str}] -> {self.output_dim} ({self.activation}){flow_info}"
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_flow_layers': self.num_flow_layers,
            'features': self.features,
            'activation': self.activation,
            'use_residual': self.use_residual,
            'use_actnorm': self.use_actnorm,
            'residual_weight': self.residual_weight,
            'log_scale_clamp': self.log_scale_clamp,
            'translation_clamp': self.translation_clamp,
            'dropout_rate': self.dropout_rate,
            'embedding_type': self.embedding_type
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Glow_ET_Config':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Glow_ET_Config(type={self.model_type}, arch={self.get_architecture_summary()})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Glow_ET_Config("
                f"model_type='{self.model_type}', "
                f"input_dim={self.input_dim}, "
                f"output_dim={self.output_dim}, "
                f"num_flow_layers={self.num_flow_layers}, "
                f"features={self.features}, "
                f"activation='{self.activation}', "
                f"dropout_rate={self.dropout_rate})")


def create_glow_et_config(
    input_dim: int,
    output_dim: int,
    num_flow_layers: int = 4,
    features: List[int] = None,
    activation: str = "swish",
    use_residual: bool = False,
    use_actnorm: bool = False,
    residual_weight: float = 1.0,
    log_scale_clamp: Tuple[float, float] = (-2.0, 2.0),
    translation_clamp: Tuple[float, float] = (-1.0, 1.0),
    dropout_rate: float = 0.1,
    embedding_type: Optional[str] = 'default',
    **kwargs
) -> Glow_ET_Config:
    """
    Create a GLOW ET configuration with specified parameters.
    
    Args:
        input_dim: Input dimension (natural parameters)
        output_dim: Output dimension (target statistics)
        num_flow_layers: Number of affine coupling layers
        features: Hidden layer sizes for coupling networks
        activation: Activation function for coupling networks
        use_residual: Whether to use residual connections
        use_actnorm: Whether to use activation normalization
        residual_weight: Weight for residual connections
        log_scale_clamp: Clamping range for log_scale
        translation_clamp: Clamping range for translation
        dropout_rate: Dropout rate for regularization
        embedding_type: Type of embedding to use
        **kwargs: Additional configuration parameters
        
    Returns:
        GLOW ET configuration object
    """
    if features is None:
        features = [64, 64]
    
    return Glow_ET_Config(
        input_dim=input_dim,
        output_dim=output_dim,
        num_flow_layers=num_flow_layers,
        features=features,
        activation=activation,
        use_residual=use_residual,
        use_actnorm=use_actnorm,
        residual_weight=residual_weight,
        log_scale_clamp=log_scale_clamp,
        translation_clamp=translation_clamp,
        dropout_rate=dropout_rate,
        embedding_type=embedding_type,
        **kwargs
    )
