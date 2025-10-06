"""
Simplified base model configuration for model architecture parameters.

This module defines the core model parameters that are specific to the model
architecture, independent of training. Streamlined to focus on essential parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from abc import ABC, abstractmethod


@dataclass
class BaseModelConfig(ABC):
    """
    Base model configuration containing common parameters used by most models.
    
    This configuration includes the most commonly used parameters across different
    model types, with model-specific configs only adding unique parameters.
    """
    
    # === MODEL IDENTIFICATION ===
    model_type: str = "base_model"
    model_name: str = "base_model_network"
    
    # === INPUT/OUTPUT DIMENSIONS ===
    input_dim: int = 0
    output_dim: int = 0
    
    # === COMMON ARCHITECTURE PARAMETERS ===
    hidden_sizes: List[int] = field(default_factory=lambda: [32, 32])
    activation: str = "swish"
    dropout_rate: float = 0.1
    
    # === COMMON REGULARIZATION ===
    use_layer_norm: bool = False
    use_batch_norm: bool = False
    
    # === COMMON RESNET PARAMETERS ===
    use_resnet: bool = True
    num_resnet_blocks: int = 3
    residual_weight: float = 1.0
    weight_residual: bool = False
    share_parameters: bool = False
    
    # === MODEL INITIALIZATION ===
    initialization_method: str = "lecun_normal"
    initialization_scale: float = 1.0
    
    # === MODEL CAPABILITIES (for trainer compatibility) ===
    supports_dropout: bool = True
    supports_batch_norm: bool = True
    supports_layer_norm: bool = True
    supports_residual_connections: bool = True
    supports_attention: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'use_layer_norm': self.use_layer_norm,
            'use_batch_norm': self.use_batch_norm,
            'use_resnet': self.use_resnet,
            'num_resnet_blocks': self.num_resnet_blocks,
            'residual_weight': self.residual_weight,
            'weight_residual': self.weight_residual,
            'share_parameters': self.share_parameters,
            'initialization_method': self.initialization_method,
            'initialization_scale': self.initialization_scale,
            'supports_dropout': self.supports_dropout,
            'supports_batch_norm': self.supports_batch_norm,
            'supports_layer_norm': self.supports_layer_norm,
            'supports_residual_connections': self.supports_residual_connections,
            'supports_attention': self.supports_attention,
        }
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        # Basic dimension validation
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if self.initialization_scale <= 0:
            raise ValueError("initialization_scale must be positive")
        
        # Initialization method validation
        valid_init_methods = ['xavier_uniform', 'xavier_normal', 'he_uniform', 'he_normal', 'glorot_uniform', 'glorot_normal', 'lecun_normal']
        if self.initialization_method not in valid_init_methods:
            raise ValueError(f"initialization_method must be one of {valid_init_methods}, got {self.initialization_method}")
        
        # Common architecture validation
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes cannot be empty")
        if not all(isinstance(size, int) and size > 0 for size in self.hidden_sizes):
            raise ValueError("All hidden_sizes must be positive integers")
        
        # Activation validation
        valid_activations = ['relu', 'gelu', 'swish', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'linear', 'none']
        if self.activation not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}, got {self.activation}")
        
        # Regularization validation
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be in [0, 1]")
        if self.use_batch_norm and self.use_layer_norm:
            raise ValueError("Cannot use both batch_norm and layer_norm simultaneously")
        
        # ResNet validation
        if self.num_resnet_blocks < 0:
            raise ValueError("num_resnet_blocks must be non-negative")
        if self.residual_weight <= 0:
            raise ValueError("residual_weight must be positive")
        
        # ResNet compatibility validation
        if self.use_resnet and self.num_resnet_blocks > 0 and len(self.hidden_sizes) >= 2:
            if self.hidden_sizes[0] != self.hidden_sizes[-1]:
                raise ValueError(
                    f"hidden_sizes[0] ({self.hidden_sizes[0]}) must equal hidden_sizes[-1] "
                    f"({self.hidden_sizes[-1]}) when using ResNet blocks"
                )
        
        # Call model-specific validation
        self._validate_model_specific()
    
    @abstractmethod
    def _validate_model_specific(self) -> None:
        """Validate model-specific parameters. Must be implemented by subclasses."""
        pass
    
    def get_architecture_summary(self) -> str:
        """Get a summary of the architecture."""
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
        
        return f"{self.model_name.upper()}: {arch_str} ({self.activation}){feature_str}"
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"{self.__class__.__name__}(type={self.model_type}, input_dim={self.input_dim}, output_dim={self.output_dim})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"{self.__class__.__name__}("
                f"model_type='{self.model_type}', "
                f"model_name='{self.model_name}', "
                f"input_dim={self.input_dim}, "
                f"output_dim={self.output_dim})")
