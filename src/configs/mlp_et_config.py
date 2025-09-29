"""
Configuration classes for MLP ET models.

This module contains the configuration classes for MLP-based ET networks
that directly predict expected sufficient statistics from natural parameters.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import jax.numpy as jnp
from .base_model_config import BaseModelConfig


@dataclass
class MLP_ET_Config(BaseModelConfig):
    """
    Configuration class for MLP ET networks.
    
    This configuration defines the MLP-specific architecture parameters
    for MLP-based ET networks that directly predict expected sufficient
    statistics from natural parameters.
    """
    
    # === MODEL IDENTIFICATION ===
    model_type: str = "mlp_et"
    model_name: str = "mlp_et_network"
    
    # === INPUT/OUTPUT DIMENSIONS ===
    input_dim: int = 4  # Override BaseModelConfig default
    output_dim: int = 4  # Override BaseModelConfig default
    
    # === MLP ARCHITECTURE PARAMETERS ===
    hidden_sizes: List[int] = field(default_factory=lambda: [32, 32])
    activation: str = "swish"  # 'relu', 'gelu', 'swish', 'tanh', 'sigmoid'
    
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
        mlp_dict = {
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
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
        base_dict.update(mlp_dict)
        return base_dict
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get only the model-specific parameters."""
        base_params = super().get_model_params()
        mlp_params = {
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
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
        base_params.update(mlp_params)
        return base_params
    
    def _validate_model_specific(self) -> None:
        """MLP-specific validation."""
        # Validate hidden sizes
        if not all(size > 0 for size in self.hidden_sizes):
            raise ValueError("All hidden_sizes must be positive")
        
        # Validate activation function
        valid_activations = ['relu', 'gelu', 'swish', 'tanh', 'sigmoid', 'leaky_relu', 'elu']
        if self.activation not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}, got {self.activation}")
        
        # Validate embedding type
        if self.embedding_type is not None and self.embedding_type not in ['default', 'learned', 'fixed']:
            raise ValueError(f"embedding_type must be None or one of ['default', 'learned', 'fixed'], got {self.embedding_type}")
        
        # Validate dropout rate
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be in [0, 1]")
        
        # Validate ResNet parameters
        if self.num_resnet_blocks < 0:
            raise ValueError("num_resnet_blocks must be non-negative")
        if self.residual_weight <= 0:
            raise ValueError("residual_weight must be positive")
        
        # Validate that ResNet is only used with residual connections
        if self.use_resnet and not self.supports_residual_connections:
            raise ValueError("use_resnet=True requires supports_residual_connections=True")
        
        # Validate that batch norm and layer norm are not both used
        if self.use_batch_norm and self.use_layer_norm:
            raise ValueError("Cannot use both batch_norm and layer_norm simultaneously")
    
    def get_architecture_summary(self) -> str:
        """Get a summary of the MLP architecture."""
        layers = [str(self.input_dim)] + [str(size) for size in self.hidden_sizes] + [str(self.output_dim)]
        arch_str = " -> ".join(layers)
        
        features = []
        if self.use_resnet:
            features.append(f"ResNet({self.num_resnet_blocks} blocks)")
        if self.use_batch_norm:
            features.append("BatchNorm")
        if self.use_layer_norm:
            features.append("LayerNorm")
        if self.dropout_rate > 0:
            features.append(f"Dropout({self.dropout_rate})")
        
        feature_str = f" + {', '.join(features)}" if features else ""
        
        return f"MLP: {arch_str} ({self.activation}){feature_str}"
    
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"MLP_ET_Config(type={self.model_type}, arch={self.get_architecture_summary()})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"MLP_ET_Config("
                f"model_type='{self.model_type}', "
                f"model_name='{self.model_name}', "
                f"input_dim={self.input_dim}, "
                f"output_dim={self.output_dim}, "
                f"hidden_sizes={self.hidden_sizes}, "
                f"activation='{self.activation}', "
                f"use_resnet={self.use_resnet}, "
                f"num_resnet_blocks={self.num_resnet_blocks})")


def create_mlp_et_config(
    input_dim: int = 4,
    output_dim: int = 4,
    hidden_sizes: List[int] = [32,32],
    activation: str = "swish",
    use_resnet: bool = True,
    num_resnet_blocks: int = 3,
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