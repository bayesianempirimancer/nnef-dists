"""
Quadratic ET Configuration

This module provides configuration classes for Quadratic ET models,
following the same patterns as MLP, GLU, and GLOW ET configurations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from .base_model_config import BaseModelConfig
from .base_training_config import BaseTrainingConfig


@dataclass
class QuadraticTrainingConfig(BaseTrainingConfig):
    """
    Training configuration specifically optimized for Quadratic ET models.
    
    Uses RMSprop as the default optimizer since it works best with quadratic layers
    and mini-batching.
    """
    # Override default optimizer to use RMSprop for quadratic models
    optimizer: str = "rmsprop"  # RMSprop works best for quadratic layers


@dataclass
class Quadratic_ET_Config(BaseModelConfig):
    """
    Configuration for Quadratic ET models.
    
    Quadratic ET models use quadratic transformations with residual connections
    to directly predict expected sufficient statistics from natural parameters.
    """
    model_type: str = "quadratic_et"
    model_name: str = "quadratic_et_network"
    input_dim: int = 4
    output_dim: int = 4
    
    # Quadratic-specific architecture parameters
    hidden_sizes: List[int] = field(default_factory=lambda: [16])  # Hidden layer sizes for quadratic blocks
    activation: str = "none"  # Activation function (default to none for quadratic blocks)
    use_layer_norm: bool = False  # Whether to use layer normalization
    use_quadratic_norm: bool = True  # Whether to use quadratic normalization
    dropout_rate: float = 0.1  # Dropout rate for regularization
    
    # ResNet wrapper parameters
    num_resnet_blocks: int = 5  # Number of ResNet blocks (default to 5 for quadratic)
    share_parameters: bool = False  # Whether to share parameters across ResNet blocks
    weight_residual: bool = True  # Whether to weight residual connections
    residual_weight: float = 1.0  # Weight for residual connections
    
    # Embedding parameters (for future use)
    embedding_type: Optional[str] = "default"  # Default embedding type for quadratic networks
    
    # Validation parameters
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Validate quadratic-specific parameters
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes list cannot be empty")
        
        if any(h <= 0 for h in self.hidden_sizes):
            raise ValueError("All hidden sizes must be positive")
        
        if self.dropout_rate < 0.0 or self.dropout_rate > 1.0:
            raise ValueError("dropout_rate must be between 0.0 and 1.0")
        
        if self.num_resnet_blocks < 0:
            raise ValueError("num_resnet_blocks must be non-negative")
        
        if self.residual_weight <= 0.0:
            raise ValueError("residual_weight must be positive")
    
    def _validate_model_specific(self) -> None:
        """Validate quadratic-specific parameters."""
        # Quadratic-specific validation is handled in __post_init__
        pass
    
    def get_architecture_summary(self) -> str:
        """Get a summary of the model architecture."""
        hidden_str = " -> ".join(map(str, self.hidden_sizes))
        resnet_info = f" + ResNet({self.num_resnet_blocks} blocks)" if self.num_resnet_blocks > 0 else " + ResNet(0 blocks)"
        
        return f"Quadratic: {self.input_dim} -> {hidden_str} -> {self.output_dim} ({self.activation}){resnet_info}"
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'use_layer_norm': self.use_layer_norm,
            'use_quadratic_norm': self.use_quadratic_norm,
            'dropout_rate': self.dropout_rate,
            'num_resnet_blocks': self.num_resnet_blocks,
            'share_parameters': self.share_parameters,
            'weight_residual': self.weight_residual,
            'residual_weight': self.residual_weight,
            'embedding_type': self.embedding_type
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Quadratic_ET_Config':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Quadratic_ET_Config(type={self.model_type}, arch={self.get_architecture_summary()})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Quadratic_ET_Config("
                f"model_type='{self.model_type}', "
                f"input_dim={self.input_dim}, "
                f"output_dim={self.output_dim}, "
                f"hidden_sizes={self.hidden_sizes}, "
                f"activation='{self.activation}', "
                f"num_resnet_blocks={self.num_resnet_blocks}, "
                f"dropout_rate={self.dropout_rate})")


def create_quadratic_et_config(
    input_dim: int,
    output_dim: int,
    hidden_sizes: List[int] = [16],
    activation: str = "none",
    use_layer_norm: bool = False,
    use_quadratic_norm: bool = True,
    dropout_rate: float = 0.1,
    num_resnet_blocks: int = 5,
    share_parameters: bool = False,
    weight_residual: bool = True,
    residual_weight: float = 1.0,
    embedding_type: Optional[str] = "default",
    **kwargs
) -> Quadratic_ET_Config:
    """
    Create a Quadratic ET configuration with specified parameters.
    
    Args:
        input_dim: Input dimension (natural parameters)
        output_dim: Output dimension (target statistics)
        hidden_sizes: Hidden layer sizes for quadratic blocks
        activation: Activation function (default: "none")
        use_layer_norm: Whether to use layer normalization
        use_quadratic_norm: Whether to use quadratic normalization
        dropout_rate: Dropout rate for regularization
        num_resnet_blocks: Number of ResNet blocks
        share_parameters: Whether to share parameters across ResNet blocks
        weight_residual: Whether to weight residual connections
        residual_weight: Weight for residual connections
        embedding_type: Type of embedding to use
        **kwargs: Additional configuration parameters
        
    Returns:
        Quadratic ET configuration object
    """
    if hidden_sizes is None:
        raise ValueError("hidden_sizes cannot be None")
    
    return Quadratic_ET_Config(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
        use_layer_norm=use_layer_norm,
        use_quadratic_norm=use_quadratic_norm,
        dropout_rate=dropout_rate,
        num_resnet_blocks=num_resnet_blocks,
        share_parameters=share_parameters,
        weight_residual=weight_residual,
        residual_weight=residual_weight,
        embedding_type=embedding_type,
        **kwargs
    )


def create_quadratic_training_config_from_args(args) -> QuadraticTrainingConfig:
    """
    Create a quadratic training configuration from command line arguments.
    
    This method extracts training-related arguments and creates a QuadraticTrainingConfig
    with RMSprop as the default optimizer, optimized for quadratic layers.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        QuadraticTrainingConfig instance with RMSprop as default optimizer
    """
    # Build training config kwargs - only include explicitly provided arguments
    training_kwargs = {}
    
    # Direct mapping (arg_name -> config_name)
    training_attributes = [
        'learning_rate', 'batch_size', 'weight_decay', 'beta1', 'beta2', 'eps',
        'loss_function', 'l1_reg_weight', 'use_mini_batching', 'random_batch_sampling',
        'eval_steps', 'save_steps', 'early_stopping_patience', 'early_stopping_min_delta',
        'log_frequency', 'random_seed', 'optimizer'
    ]
    
    for attribute in training_attributes:
        if hasattr(args, attribute) and getattr(args, attribute) is not None:
            training_kwargs[attribute] = getattr(args, attribute)
    
    # Create quadratic training configuration (defaults to RMSprop)
    return QuadraticTrainingConfig(**training_kwargs)
