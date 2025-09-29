"""
Base model configuration for model architecture and model-specific parameters.

This module defines the core model parameters that are specific to the model
architecture, independent of training. This allows for clean separation between
model architecture and training configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod


@dataclass
class BaseModelConfig(ABC):
    """
    Base model configuration containing all model architecture parameters.
    
    This configuration is training-agnostic and contains only the parameters
    needed for model architecture, initialization, and model-specific functionality.
    Training parameters should be defined in BaseTrainingConfig.
    """
    
    # === MODEL IDENTIFICATION ===
    model_type: str = "base_model"
    model_name: str = "base_model_network"
    
    # === INPUT/OUTPUT DIMENSIONS ===
    input_dim: int = 0
    output_dim: int = 0
    
    # === MODEL-SPECIFIC LOSS FUNCTIONS ===
    model_specific_loss_functions: List[str] = field(default_factory=lambda: ["default"])
    default_loss_function: str = "default"  # Which loss function to use by default
    
    # === MODEL INITIALIZATION ===
    initialization_method: str = "lecun_normal"  # 'xavier_uniform', 'xavier_normal', 'he_uniform', 'he_normal', 'glorot_uniform', 'glorot_normal', 'lecun_normal'
    initialization_scale: float = 1.0  # Scale factor for initialization
    
    # === MODEL-SPECIFIC PARAMETERS ===
    # These will be overridden by specific model configs
    model_specific_params: Dict[str, Any] = field(default_factory=dict)
    
    # === MODEL CAPABILITIES ===
    supports_dropout: bool = True
    supports_batch_norm: bool = True
    supports_layer_norm: bool = True
    supports_residual_connections: bool = False
    supports_attention: bool = False
    
    # === MODEL INTERFACE REQUIREMENTS ===
    # These define what methods the model must implement
    required_methods: List[str] = field(default_factory=lambda: ["apply", "init"])
    optional_methods: List[str] = field(default_factory=lambda: ["loss_fn", "predict"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'model_specific_loss_functions': self.model_specific_loss_functions,
            'default_loss_function': self.default_loss_function,
            'initialization_method': self.initialization_method,
            'initialization_scale': self.initialization_scale,
            'model_specific_params': self.model_specific_params,
            'supports_dropout': self.supports_dropout,
            'supports_batch_norm': self.supports_batch_norm,
            'supports_layer_norm': self.supports_layer_norm,
            'supports_residual_connections': self.supports_residual_connections,
            'supports_attention': self.supports_attention,
            'required_methods': self.required_methods,
            'optional_methods': self.optional_methods,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseModelConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get only the model-specific parameters."""
        return {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'model_specific_params': self.model_specific_params,
        }
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if self.initialization_scale <= 0:
            raise ValueError("initialization_scale must be positive")
        if self.default_loss_function not in self.model_specific_loss_functions:
            raise ValueError(f"default_loss_function '{self.default_loss_function}' must be in model_specific_loss_functions {self.model_specific_loss_functions}")
        if self.initialization_method not in ['xavier_uniform', 'xavier_normal', 'he_uniform', 'he_normal', 'glorot_uniform', 'glorot_normal', 'lecun_normal']:
            raise ValueError(f"initialization_method must be one of ['xavier_uniform', 'xavier_normal', 'he_uniform', 'he_normal', 'glorot_uniform', 'glorot_normal', 'lecun_normal'], got {self.initialization_method}")
        
        # Call model-specific validation
        self._validate_model_specific()
    
    @abstractmethod
    def _validate_model_specific(self) -> None:
        """Validate model-specific parameters. Must be implemented by subclasses."""
        pass
    
    def has_loss_fn(self) -> bool:
        """Check if model supports model-specific loss functions."""
        return len(self.model_specific_loss_functions) > 0 and self.model_specific_loss_functions != ["default"]
    
    def get_available_loss_functions(self) -> List[str]:
        """Get list of available loss functions for this model."""
        return self.model_specific_loss_functions.copy()
    
    def supports_feature(self, feature: str) -> bool:
        """Check if model supports a specific feature."""
        feature_map = {
            'dropout': self.supports_dropout,
            'batch_norm': self.supports_batch_norm,
            'layer_norm': self.supports_layer_norm,
            'residual_connections': self.supports_residual_connections,
            'attention': self.supports_attention,
        }
        return feature_map.get(feature, False)
    
    def get_required_methods(self) -> List[str]:
        """Get list of methods the model must implement."""
        return self.required_methods.copy()
    
    def get_optional_methods(self) -> List[str]:
        """Get list of methods the model may implement."""
        return self.optional_methods.copy()
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"BaseModelConfig(type={self.model_type}, input_dim={self.input_dim}, output_dim={self.output_dim})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"BaseModelConfig("
                f"model_type='{self.model_type}', "
                f"model_name='{self.model_name}', "
                f"input_dim={self.input_dim}, "
                f"output_dim={self.output_dim}, "
                f"supports_dropout={self.supports_dropout}, "
                f"supports_residual_connections={self.supports_residual_connections})")


def create_default_model_config() -> BaseModelConfig:
    """Create a default model configuration."""
    # This will return a concrete implementation
    # For now, we'll create a simple concrete class
    class DefaultModelConfig(BaseModelConfig):
        def _validate_model_specific(self) -> None:
            """Default validation - no model-specific parameters."""
            pass
    
    return DefaultModelConfig()


def create_model_config_from_dict(config_dict: Dict[str, Any]) -> BaseModelConfig:
    """Create model configuration from dictionary with validation."""
    # This would need to be implemented based on the model_type
    # For now, return default
    config = create_default_model_config()
    config.update_from_dict(config_dict)
    config.validate()
    return config
