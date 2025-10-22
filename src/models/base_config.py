"""
Simplified base model configuration for model architecture parameters.

This module defines the core model parameters that are specific to the model
architecture, independent of training. Streamlined to focus on essential parameters.
"""

from dataclasses import dataclass, fields
from typing import Dict, Any, Tuple


@dataclass(frozen=True)
class BaseConfig:
    """
    Base model configuration containing common parameters used by most models.
    
    This configuration includes the most commonly used parameters across different
    model types, with model-specific configs only adding unique parameters.
    """
    
    # === MODEL IDENTIFICATION ===
    model_name: str = "base_model_network"
    
    # === INPUT/OUTPUT DIMENSIONS ===
    input_dim: int = 0
    output_dim: int = 0
    
    # === COMMON ARCHITECTURE PARAMETERS ===
    activation: str = "swish"
    dropout_rate: float = 0.1
    
    # === COMMON REGULARIZATION ===
    use_layer_norm: bool = False
    use_batch_norm: bool = False
        
    # === MODEL INITIALIZATION ===
    initialization_method: str = "lecun_normal"
    initialization_scale: float = 1.0
    
    # === EMBEDDING PARAMETERS ===
    embedding_type: str = "default"  # Type of eta embedding to use
        
    # === MODEL CAPABILITIES (for trainer compatibility) ===
    supports_dropout: bool = True
    supports_batch_norm: bool = True
    supports_layer_norm: bool = True
    supports_residual_connections: bool = True
    supports_attention: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary using introspection."""
        result = {}
        for field_info in fields(self):
            result[field_info.name] = getattr(self, field_info.name)
        return result
    
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
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """
        Load configuration from a pretrained model directory.
        
        Args:
            model_name_or_path: Path to the pretrained model directory
            **kwargs: Additional parameters to override the loaded config
            
        Returns:
            Config instance loaded from the pretrained model
        """
        import json
        from pathlib import Path
        
        model_path = Path(model_name_or_path)
        config_path = model_path / "config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Override with any additional kwargs
        config_dict.update(kwargs)
        
        return cls(**config_dict)
    
    @classmethod
    def create(cls, **kwargs):
        """
        Create a configuration with specified parameters.
        
        Args:
            **kwargs: Configuration parameters to override defaults
            
        Returns:
            Config instance
        """
        return cls(**kwargs)
    
    @classmethod
    def create_from_args(cls, args):
        """
        Create a configuration instance from command line arguments.
        
        Args:
            args: argparse.Namespace object containing all configuration parameters
            
        Returns:
            Config instance
        """
        # Convert args to dictionary
        args_dict = vars(args)
        
        # Add model-specific args to model_kwargs
        model_attributes = cls.get_model_attributes()
        filtered_kwargs = {}
        
        for attr_name, attr_value in args_dict.items():
            if attr_name in model_attributes and attr_value is not None:
                filtered_kwargs[attr_name] = attr_value
        
        # Convert hidden_sizes from list to tuple if provided as list
        if 'hidden_sizes' in filtered_kwargs and isinstance(filtered_kwargs['hidden_sizes'], list):
            filtered_kwargs['hidden_sizes'] = tuple(filtered_kwargs['hidden_sizes'])
        
        return cls(**filtered_kwargs)
    
    @classmethod
    def get_model_attributes(cls):
        """
        Get the list of model-specific attributes that can be set via command line arguments.
        
        Subclasses can override this to specify which parameters they support.
        """
        return [
            'input_dim', 'output_dim', 'hidden_sizes', 'activation', 'dropout_rate', 
            'num_resnet_blocks', 'initialization_method', 'embedding_type'
        ]
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"{self.__class__.__name__}(name={self.model_name}, input_dim={self.input_dim}, output_dim={self.output_dim})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"{self.__class__.__name__}("
                f"model_name='{self.model_name}', "
                f"input_dim={self.input_dim}, "
                f"output_dim={self.output_dim})")
