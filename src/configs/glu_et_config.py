"""
Streamlined configuration for GLU ET models.

This module contains the streamlined configuration class for GLU-based ET networks.
Most parameters are inherited from BaseModelConfig, only GLU-specific parameters are added.
"""

from dataclasses import dataclass
from typing import List
from .base_model_config import BaseModelConfig


@dataclass
class GLU_ET_Config(BaseModelConfig):
    """
    Streamlined configuration class for GLU ET networks.
    
    Inherits all common parameters from BaseModelConfig and only adds
    GLU-specific parameters that are unique to this model type.
    """
    
    # === MODEL IDENTIFICATION ===
    model_type: str = "glu_et"
    model_name: str = "glu_et_network"
    
    # === INPUT/OUTPUT DIMENSIONS ===
    input_dim: int = 0  # Must be set by caller based on actual data
    output_dim: int = 0  # Must be set by caller based on actual data
    
    # === GLU-SPECIFIC PARAMETERS ===
    gate_activation: str = "sigmoid"  # Gate activation function (only GLU-specific parameter)
    
    def _validate_model_specific(self) -> None:
        """GLU-specific validation."""
        # Validate gate activation function
        valid_activations = ['relu', 'gelu', 'swish', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'linear', 'none']
        if self.gate_activation not in valid_activations:
            raise ValueError(f"gate_activation must be one of {valid_activations}, got {self.gate_activation}")


def create_glu_et_config(
    input_dim: int,
    output_dim: int,
    hidden_sizes: List[int],
    activation: str,
    gate_activation: str,
    use_resnet: bool,
    num_resnet_blocks: int,
    **kwargs
) -> GLU_ET_Config:
    """
    Create a GLU ET configuration with specified parameters.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_sizes: List of hidden layer sizes
        activation: Activation function for the main path
        gate_activation: Activation function for the gate
        use_resnet: Whether to use ResNet blocks
        num_resnet_blocks: Number of ResNet blocks
        **kwargs: Additional parameters to override
        
    Returns:
        GLU_ET_Config instance
    """
    config = GLU_ET_Config(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
        gate_activation=gate_activation,
        use_resnet=use_resnet,
        num_resnet_blocks=num_resnet_blocks,
        **kwargs
    )
    
    config.validate()
    return config