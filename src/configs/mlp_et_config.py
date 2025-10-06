"""
Streamlined configuration for MLP ET models.

This module contains the streamlined configuration class for MLP-based ET networks.
Most parameters are inherited from BaseModelConfig, only MLP-specific parameters are added.
"""

from dataclasses import dataclass
from typing import List
from .base_model_config import BaseModelConfig


@dataclass
class MLP_ET_Config(BaseModelConfig):
    """
    Streamlined configuration class for MLP ET networks.
    
    Inherits all common parameters from BaseModelConfig and only adds
    MLP-specific parameters that are unique to this model type.
    """
    
    # === MODEL IDENTIFICATION ===
    model_type: str = "mlp_et"
    model_name: str = "mlp_et_network"
    
    # === INPUT/OUTPUT DIMENSIONS ===
    input_dim: int = 0  # Must be set by caller based on actual data
    output_dim: int = 0  # Must be set by caller based on actual data
    
    # === MLP-SPECIFIC PARAMETERS ===
    # (All other parameters inherited from BaseModelConfig)
    
    def _validate_model_specific(self) -> None:
        """MLP-specific validation - no additional validation needed."""
        pass


def create_mlp_et_config(
    input_dim: int,
    output_dim: int,
    hidden_sizes: List[int],
    activation: str,
    use_resnet: bool,
    num_resnet_blocks: int,
    **kwargs
) -> MLP_ET_Config:
    """
    Create an MLP ET configuration with specified parameters.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_sizes: List of hidden layer sizes
        activation: Activation function
        use_resnet: Whether to use ResNet blocks
        num_resnet_blocks: Number of ResNet blocks
        **kwargs: Additional parameters to override
        
    Returns:
        MLP_ET_Config instance
    """
    if hidden_sizes is None:
        hidden_sizes = [32, 32]
    
    config = MLP_ET_Config(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
        use_resnet=use_resnet,
        num_resnet_blocks=num_resnet_blocks,
        **kwargs
    )
    
    config.validate()
    return config