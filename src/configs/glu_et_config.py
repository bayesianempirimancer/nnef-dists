"""
Configuration classes for GLU ET models.

This module contains the configuration classes for GLU-based ET networks
that directly predict expected sufficient statistics from natural parameters.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import jax.numpy as jnp
from .base_model_config import BaseModelConfig


@dataclass
class GLU_ET_Config(BaseModelConfig):
    """
    Configuration class for GLU ET networks.
    
    This configuration defines the GLU-specific architecture parameters
    for GLU-based ET networks that directly predict expected sufficient
    statistics from natural parameters.
    """
    
    # === MODEL IDENTIFICATION ===
    model_type: str = "glu_et"
    model_name: str = "glu_et_network"
    
    # === INPUT/OUTPUT DIMENSIONS ===
    input_dim: int = 4  # Override BaseModelConfig default
    output_dim: int = 4  # Override BaseModelConfig default
    
    # === GLU ARCHITECTURE PARAMETERS ===
    hidden_sizes: List[int] = field(default_factory=lambda: [32, 32])
    activation: str = "swish"  # 'relu', 'gelu', 'swish', 'tanh', 'sigmoid'
    gate_activation: str = "sigmoid"  # Gate activation function
    
    # === ETA EMBEDDING CONFIGURATION ===
    embedding_type: Optional[str] = 'default'  # None = no embedding, 'default' = default embedding, etc.
    
    # === REGULARIZATION ===
    dropout_rate: float = 0.1  # Dropout rate for model architecture
    use_layer_norm: bool = False
    use_batch_norm: bool = False
    
    # === RESNET CONFIGURATION ===
    use_resnet: bool = True
    num_resnet_blocks: int = 5
    residual_weight: float = 1.0
    weight_residual: bool = False  # Whether to weight the residual connection
    share_parameters: bool = False  # Whether to share parameters between ResNet blocks
    
    # === MODEL-SPECIFIC LOSS FUNCTIONS ===
    model_specific_loss_functions: List[str] = field(default_factory=lambda: ["default", "smoothness_penalty", "regularized_mse"])
    default_loss_function: str = "default"
    
    # === MODEL CAPABILITIES ===
    supports_dropout: bool = True
    supports_batch_norm: bool = True
    supports_layer_norm: bool = True
    supports_residual_connections: bool = True
    supports_attention: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        base_dict = super().to_dict()
        glu_dict = {
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'gate_activation': self.gate_activation,
            'embedding_type': self.embedding_type,
            'dropout_rate': self.dropout_rate,
            'use_layer_norm': self.use_layer_norm,
            'use_batch_norm': self.use_batch_norm,
            'use_resnet': self.use_resnet,
            'num_resnet_blocks': self.num_resnet_blocks,
            'residual_weight': self.residual_weight,
            'weight_residual': self.weight_residual,
            'share_parameters': self.share_parameters,
        }
        base_dict.update(glu_dict)
        return base_dict
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get only the model-specific parameters."""
        base_params = super().get_model_params()
        glu_params = {
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'gate_activation': self.gate_activation,
            'embedding_type': self.embedding_type,
            'dropout_rate': self.dropout_rate,
            'use_layer_norm': self.use_layer_norm,
            'use_batch_norm': self.use_batch_norm,
            'use_resnet': self.use_resnet,
            'num_resnet_blocks': self.num_resnet_blocks,
            'residual_weight': self.residual_weight,
            'weight_residual': self.weight_residual,
            'share_parameters': self.share_parameters,
        }
        base_params.update(glu_params)
        return base_params
    
    def _validate_model_specific(self) -> None:
        """GLU-specific validation."""
        super()._validate_model_specific()
        
        # Validate hidden sizes
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes cannot be empty")
        
        if not all(isinstance(size, int) and size > 0 for size in self.hidden_sizes):
            raise ValueError("All hidden_sizes must be positive integers")
        
        # Validate ResNet configuration
        if self.use_resnet and self.num_resnet_blocks > 0:
            if self.hidden_sizes[0] != self.hidden_sizes[-1]:
                raise ValueError(
                    f"hidden_sizes[0] ({self.hidden_sizes[0]}) must equal hidden_sizes[-1] "
                    f"({self.hidden_sizes[-1]}) when using ResNet blocks"
                )
        
        # Validate activation functions
        valid_activations = ['relu', 'gelu', 'swish', 'tanh', 'sigmoid', 'linear', 'none']
        if self.activation not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}, got {self.activation}")
        
        if self.gate_activation not in valid_activations:
            raise ValueError(f"gate_activation must be one of {valid_activations}, got {self.gate_activation}")
        
        # Validate dropout rate
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be between 0.0 and 1.0, got {self.dropout_rate}")
        
        # Validate ResNet parameters
        if self.num_resnet_blocks < 0:
            raise ValueError(f"num_resnet_blocks must be non-negative, got {self.num_resnet_blocks}")
        
        if self.residual_weight <= 0:
            raise ValueError(f"residual_weight must be positive, got {self.residual_weight}")
    
    def get_architecture_summary(self) -> str:
        """Get a summary of the architecture."""
        if self.use_resnet and self.num_resnet_blocks > 0:
            resnet_info = f" + ResNet({self.num_resnet_blocks} blocks)"
        else:
            resnet_info = " + ResNet(0 blocks)"
        
        hidden_str = " -> ".join(map(str, self.hidden_sizes))
        return f"GLU: {self.input_dim} -> {hidden_str} -> {self.output_dim} ({self.activation}, gate: {self.gate_activation}){resnet_info}"
    
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"GLU_ET_Config(type={self.model_type}, arch={self.get_architecture_summary()})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"GLU_ET_Config("
                f"model_type='{self.model_type}', "
                f"model_name='{self.model_name}', "
                f"input_dim={self.input_dim}, "
                f"output_dim={self.output_dim}, "
                f"hidden_sizes={self.hidden_sizes}, "
                f"activation='{self.activation}', "
                f"gate_activation='{self.gate_activation}', "
                f"num_resnet_blocks={self.num_resnet_blocks})")


def create_glu_et_config(
    input_dim: int = 4,
    output_dim: int = 4,
    hidden_sizes: List[int] = [32, 32],
    activation: str = "swish",
    gate_activation: str = "sigmoid",
    use_resnet: bool = True,
    num_resnet_blocks: int = 3,
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
    if hidden_sizes is None:
        hidden_sizes = [32, 32]
    
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
