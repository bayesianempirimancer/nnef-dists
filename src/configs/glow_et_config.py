"""
Streamlined configuration for Glow ET models.

This module contains the streamlined configuration class for Glow ET networks.
Most parameters are inherited from BaseModelConfig, only Glow-specific parameters are added.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
from .base_model_config import BaseModelConfig


@dataclass
class Glow_ET_Config(BaseModelConfig):
    """
    Streamlined configuration class for Glow ET networks.
    
    Inherits all common parameters from BaseModelConfig and only adds
    Glow-specific parameters that are unique to this model type.
    """
    
    # === MODEL IDENTIFICATION ===
    model_type: str = "glow_et"
    model_name: str = "glow_et_network"
    
    # === INPUT/OUTPUT DIMENSIONS ===
    input_dim: int = 0  # Must be set by caller based on actual data
    output_dim: int = 0  # Must be set by caller based on actual data
    
    # === GLOW-SPECIFIC PARAMETERS ===
    num_flow_layers: int = 8  # Number of affine coupling layers
    features: List[int] = field(default_factory=lambda: [32, 32])  # Hidden layer sizes for coupling networks
    
    # Affine coupling layer parameters
    use_residual: bool = False  # Whether to use residual connections in coupling networks
    use_actnorm: bool = True  # Whether to use activation normalization
    log_scale_clamp: Tuple[float, float] = (-4.0, 4.0)  # Clamping range for log_scale
    translation_clamp: Tuple[float, float] = (-100.0, 100.0)  # Clamping range for translation
    
    # Override some defaults for Glow models
    use_resnet: bool = False  # Glow doesn't use ResNet blocks
    num_resnet_blocks: int = 0
    
    def _validate_model_specific(self) -> None:
        """Glow-specific validation."""
        # Validate GLOW-specific parameters
        if self.num_flow_layers < 1:
            raise ValueError("num_flow_layers must be at least 1")
        
        if not self.features:
            raise ValueError("features list cannot be empty")
        
        if not all(isinstance(size, int) and size > 0 for size in self.features):
            raise ValueError("All features must be positive integers")
        
        # Validate clamping ranges
        if self.log_scale_clamp[0] >= self.log_scale_clamp[1]:
            raise ValueError("log_scale_clamp[0] must be less than log_scale_clamp[1]")
        
        if self.translation_clamp[0] >= self.translation_clamp[1]:
            raise ValueError("translation_clamp[0] must be less than translation_clamp[1]")


def create_glow_et_config(
    input_dim: int,
    output_dim: int,
    num_flow_layers: int,
    features: List[int],
    activation: str,
    **kwargs
) -> Glow_ET_Config:
    """
    Create a Glow ET configuration with specified parameters.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        num_flow_layers: Number of flow layers
        features: List of hidden layer sizes for coupling networks
        activation: Activation function
        **kwargs: Additional parameters to override
        
    Returns:
        Glow_ET_Config instance
    """
    
    config = Glow_ET_Config(
        input_dim=input_dim,
        output_dim=output_dim,
        num_flow_layers=num_flow_layers,
        features=features,
        activation=activation,
        **kwargs
    )
    
    config.validate()
    return config
