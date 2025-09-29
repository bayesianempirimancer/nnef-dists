"""
Base configuration class with common methods for all model configurations.

This module provides a base configuration class that contains common methods
for serialization, validation, and Hugging Face compatibility that can be
inherited by specific model configuration classes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List
import json
import os


@dataclass
class BaseConfig(ABC):
    """
    Base configuration class with common methods for all model configurations.
    
    This provides common functionality for all model configurations including
    serialization, validation, and Hugging Face compatibility.
    """
    
    # Common model parameters
    model_type: str = "base_model"
    model_name: str = "base_model_network"
    
    # Common architecture parameters
    input_dim: int = 12  # Example case for 3D Multivariate Normal
    output_dim: int = 1
    
    # Common training parameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 1e-4
    
    # Common optimizer parameters
    optimizer_type: str = "adam"  # 'adam', 'adamw', 'sgd', 'rmsprop'
    beta1: float = 0.9  # First moment decay rate for Adam/AdamW
    beta2: float = 0.999  # Second moment decay rate for Adam/AdamW
    epsilon: float = 1e-8  # Numerical stability parameter
    momentum: float = 0.9  # Momentum for SGD/RMSprop
    nesterov: bool = False  # Nesterov momentum for SGD
    rmsprop_decay: float = 0.9  # Decay rate for RMSprop
    
    # Common regularization parameters
    l1_reg_weight: float = 0.0  # L1 regularization weight
    
    # Loss function parameters
    loss_function: str = "mse"  # Loss function type: 'mse', 'mae', 'huber', 'model_specific'
    loss_alpha: float = 1.0  # Weight for primary loss
    loss_beta: float = 0.0  # Weight for secondary loss (if applicable)
    huber_delta: float = 1.0  # Delta parameter for Huber loss
    
    # Training control parameters
    dropout_epochs: int = 0  # Number of epochs to use dropout
    eval_steps: int = 10  # Steps between evaluations
    save_steps: Optional[int] = None  # Steps between model saves (None = no saves)
    early_stopping_patience: int = 50  # Epochs to wait before early stopping
    early_stopping_min_delta: float = 1e-4  # Minimum change to qualify as improvement
    use_mini_batching: bool = True  # Whether to use mini-batching (False = process entire dataset at once)
    random_sampling: bool = True  # Whether to use random sampling (True) or sequential batching (False)
    
    # Data processing parameters
    train_split: float = 0.8  # Fraction of data for training
    val_split: float = 0.1  # Fraction of data for validation
    test_split: float = 0.1  # Fraction of data for testing
    data_augmentation: bool = False  # Whether to use data augmentation
    normalize_data: bool = False  # Whether to normalize input data
    
    # Inference parameters
    inference_batch_size: int = 100  # Batch size for inference
    inference_temperature: float = 1.0  # Temperature for sampling
    num_inference_samples: int = 1  # Number of samples for inference
    
    # Logging and monitoring
    log_frequency: int = 10  # Steps between logging
    track_metrics: List[str] = field(default_factory=lambda: ['loss', 'accuracy'])  # Metrics to track
    save_predictions: bool = False  # Whether to save predictions
    plot_training: bool = True  # Whether to plot training curves
    
    # System parameters
    random_seed: int = 42  # Random seed for reproducibility
    device: str = "auto"  # Device to use ('auto', 'cpu', 'gpu')
    compile_model: bool = True  # Whether to compile model with JAX
    memory_efficient: bool = False  # Whether to use memory-efficient training
    
    # Common LogZ parameters
    hessian_method: str = "full"  # "full" or "diagonal"
    compile_functions: bool = True
    
    # Exponential family distribution
    ef_distribution: str = "multivariate_normal"  # Distribution type
    ef_params: Dict[str, Any] = field(default_factory=dict, metadata={"help": "Parameters for the exponential family distribution"})
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Basic validation that applies to all configs
        if hasattr(self, 'input_dim') and self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hasattr(self, 'output_dim') and self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if hasattr(self, 'learning_rate') and self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if hasattr(self, 'batch_size') and self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if hasattr(self, 'dropout_rate') and (self.dropout_rate < 0 or self.dropout_rate > 1):
            raise ValueError("dropout_rate must be between 0 and 1")
        
        # Call any model-specific validation
        self._validate_model_specific()
    
    @abstractmethod
    def _validate_model_specific(self):
        """Model-specific validation. Override in subclasses."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        # Get all attributes from the dataclass
        result = {}
        for field_name, field_value in self.__dict__.items():
            # Skip private attributes
            if not field_name.startswith('_'):
                result[field_name] = field_value
        return result
    
    def transmit_properties_to_dict(self, target_dict: Dict[str, Any], 
                                  property_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Transmit specific properties from this config to a target dictionary.
        
        Args:
            target_dict: Dictionary to update with properties
            property_names: List of property names to transmit. If None, transmits all properties.
            
        Returns:
            Updated target dictionary
        """
        if property_names is None:
            # Transmit all properties
            for field_name, field_value in self.__dict__.items():
                if not field_name.startswith('_'):
                    target_dict[field_name] = field_value
        else:
            # Transmit only specified properties
            for prop_name in property_names:
                if hasattr(self, prop_name):
                    target_dict[prop_name] = getattr(self, prop_name)
                else:
                    raise AttributeError(f"Property '{prop_name}' not found in config")
        
        return target_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save_pretrained(self, save_directory: str):
        """Save configuration to directory."""
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "config.json")
        
        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        """Load configuration from directory or model name."""
        if os.path.isdir(model_name_or_path):
            config_path = os.path.join(model_name_or_path, "config.json")
        else:
            # For now, assume it's a local path
            config_path = model_name_or_path
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def update_from_dict(self, update_dict: Dict[str, Any]):
        """
        Update configuration from a dictionary.
        
        Args:
            update_dict: Dictionary containing new values for configuration parameters
        """
        for key, value in update_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Configuration parameter '{key}' not found")
    
    def get_property(self, property_name: str, default: Any = None) -> Any:
        """
        Get a configuration property with optional default value.
        
        Args:
            property_name: Name of the property to get
            default: Default value if property doesn't exist
            
        Returns:
            Property value or default
        """
        return getattr(self, property_name, default)
    
    def set_property(self, property_name: str, value: Any):
        """
        Set a configuration property.
        
        Args:
            property_name: Name of the property to set
            value: Value to set
        """
        if hasattr(self, property_name):
            setattr(self, property_name, value)
        else:
            raise AttributeError(f"Configuration parameter '{property_name}' not found")
    
    def copy(self):
        """Create a copy of this configuration."""
        return self.__class__(**self.to_dict())
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in self.to_dict().items())})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the configuration."""
        return self.__str__()
