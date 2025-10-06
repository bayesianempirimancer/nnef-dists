"""
Streamlined configuration for Quadratic ET models.

This module contains the streamlined configuration class for Quadratic ET networks.
Most parameters are inherited from BaseModelConfig, only Quadratic-specific parameters are added.
"""

from dataclasses import dataclass, field
from typing import List
from .base_model_config import BaseModelConfig


@dataclass
class Quadratic_ET_Config(BaseModelConfig):
    """
    Streamlined configuration class for Quadratic ET networks.
    
    Inherits all common parameters from BaseModelConfig and only adds
    Quadratic-specific parameters that are unique to this model type.
    """
    
    # === MODEL IDENTIFICATION ===
    model_type: str = "quadratic_et"
    model_name: str = "quadratic_et_network"
    
    # === INPUT/OUTPUT DIMENSIONS ===
    input_dim: int = 0  # Must be set by caller based on actual data
    output_dim: int = 0  # Must be set by caller based on actual data
    
    # === QUADRATIC-SPECIFIC PARAMETERS ===
    use_quadratic_norm: bool = True  # Whether to use quadratic normalization
    # Override some defaults for quadratic models
    hidden_sizes: List[int] = field(default_factory=lambda: [16])  # Smaller hidden sizes work better for quadratic
    activation: str = "none"  # No activation for quadratic blocks
    num_resnet_blocks: int = 5  # More blocks for quadratic models
    
    def _validate_model_specific(self) -> None:
        """Quadratic-specific validation."""
        # Quadratic models typically work better with smaller hidden sizes
        if any(size > 32 for size in self.hidden_sizes):
            print(f"Warning: Quadratic models typically work better with smaller hidden sizes (<=32), got {self.hidden_sizes}")


def create_quadratic_et_config(
    input_dim: int,
    output_dim: int,
    hidden_sizes: List[int],
    use_quadratic_norm: bool,
    num_resnet_blocks: int,
    **kwargs
) -> Quadratic_ET_Config:
    """
    Create a Quadratic ET configuration with specified parameters.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_sizes: List of hidden layer sizes
        use_quadratic_norm: Whether to use quadratic normalization
        num_resnet_blocks: Number of ResNet blocks
        **kwargs: Additional parameters to override
        
    Returns:
        Quadratic_ET_Config instance
    """
    config = Quadratic_ET_Config(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_sizes,
        use_quadratic_norm=use_quadratic_norm,
        num_resnet_blocks=num_resnet_blocks,
        **kwargs
    )
    
    config.validate()
    return config