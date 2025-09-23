"""
Standard configuration system for all neural network models.

This module provides a unified configuration interface for all model types,
training parameters, and experimental settings.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
import jax.numpy as jnp


@dataclass
class NetworkConfig:
    """Configuration for network architecture."""
    # Basic architecture
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128, 64])
    activation: str = "tanh"  # "tanh", "relu", "swish", "gelu"
    dropout_rate: float = 0.0
    use_batch_norm: bool = False
    use_layer_norm: bool = False
    
    # Input/Output
    input_dim: Optional[int] = None  # Auto-detected if None
    output_dim: int = 9  # Default for 3D tril format
    use_feature_engineering: bool = True
    
    # Advanced options
    residual_connections: bool = False
    skip_connections: bool = False
    weight_init: str = "xavier"  # "xavier", "he", "normal"


@dataclass 
class TrainingConfig:
    """Configuration for training parameters."""
    # Optimization
    learning_rate: float = 1e-3
    optimizer: str = "adam"  # "adam", "adamw", "sgd", "rmsprop"
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    
    # Schedule
    use_lr_schedule: bool = True
    lr_schedule_type: str = "exponential"  # "exponential", "cosine", "polynomial"
    lr_decay_rate: float = 0.95
    lr_decay_steps: int = 1000
    
    # Training loop
    num_epochs: int = 100
    batch_size: int = 64
    patience: int = float('inf')
    min_delta: float = 1e-6
    
    # Validation
    validation_freq: int = 10
    early_stopping: bool = True


@dataclass
class ModelSpecificConfig:
    """Configuration for model-specific parameters."""
    # Flow-based models
    num_flow_layers: int = 20
    flow_hidden_size: int = 64
    
    # Diffusion models
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    noise_schedule: str = "linear"  # "linear", "cosine"
    
    # NoProp-CT models
    num_time_steps: int = 10
    time_horizon: float = 1.0
    ode_solver: str = "euler"  # "euler", "rk4", "dopri5"
    noise_scale: float = 0.1
    
    # Quadratic ResNet
    use_quadratic_terms: bool = True
    quadratic_mixing: str = "adaptive"  # "fixed", "adaptive", "learned"
    
    # Geometric Flow models
    matrix_rank: int = None  # Rank of matrix A in geometric flow (None = full rank)
    n_time_steps: int = 10  # Number of time steps for flow integration
    smoothness_weight: float = 1e-3  # Penalty for large du/dt
    time_embed_dim: int = 16  # Dimension of time embedding
    max_freq: float = 10.0  # Maximum frequency for time embedding
    
    # Transformer models
    num_heads: int = 8
    num_transformer_layers: int = 6
    use_positional_encoding: bool = True
    
    # INN models
    coupling_type: str = "additive"  # "additive", "affine", "spline"
    num_inn_layers: int = 8
    use_actnorm: bool = True


@dataclass
class ExperimentConfig:
    """Configuration for experimental setup."""
    # Data
    dataset_type: str = "3d_gaussian"  # "1d_gaussian", "3d_gaussian"
    data_format: str = "tril"  # "full", "tril"
    train_size: Optional[int] = None  # Use all available if None
    val_split: float = 0.2
    test_split: float = 0.1
    
    # Evaluation
    compute_ground_truth: bool = True
    eval_metrics: List[str] = field(default_factory=lambda: ["mse", "mae", "ground_truth_mse"])
    
    # Logging and saving
    save_model: bool = True
    save_plots: bool = True
    log_frequency: int = 10
    output_dir: str = "artifacts"
    experiment_name: Optional[str] = None  # Auto-generated if None
    
    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True


@dataclass
class FullConfig:
    """Complete configuration combining all aspects."""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model_specific: ModelSpecificConfig = field(default_factory=ModelSpecificConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization."""
        return {
            'network': self.network.__dict__,
            'training': self.training.__dict__,
            'model_specific': self.model_specific.__dict__,
            'experiment': self.experiment.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FullConfig':
        """Create config from dictionary."""
        return cls(
            network=NetworkConfig(**config_dict.get('network', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            model_specific=ModelSpecificConfig(**config_dict.get('model_specific', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {}))
        )


# Predefined configurations for different network types
def get_deep_narrow_config() -> FullConfig:
    """Get configuration for deep narrow networks."""
    config = FullConfig()
    config.network.hidden_sizes = [64] * 12  # 12 layers x 64 units
    config.training.learning_rate = 5e-4
    config.training.batch_size = 32
    config.training.num_epochs = 150
    config.training.patience = float('inf')
    return config


def get_ultra_deep_config() -> FullConfig:
    """Get configuration for ultra-deep networks."""
    config = FullConfig()
    config.network.hidden_sizes = [32] * 20  # 20 layers x 32 units
    config.training.learning_rate = 1e-4
    config.training.batch_size = 16
    config.training.num_epochs = 200
    config.training.patience = float('inf')
    config.training.weight_decay = 1e-5
    config.training.gradient_clip_norm = 0.5
    return config


def get_glow_config() -> FullConfig:
    """Get configuration for GLOW-based models."""
    config = FullConfig()
    config.network.hidden_sizes = [64, 128, 128]  # Base network layers for display
    config.network.output_dim = None  # Auto-detect from data (should match eta_dim for ET models)
    config.network.use_layer_norm = True  # Enable layer normalization for stability
    config.model_specific.num_flow_layers = 20  # Reasonable number of flow layers for Glow
    config.model_specific.flow_hidden_size = 64
    config.training.learning_rate = 1e-3
    config.training.num_epochs = 100
    config.training.batch_size = 64
    config.training.gradient_clip_norm = 1.0  # Add gradient clipping for stability
    return config


def get_diffusion_config() -> FullConfig:
    """Get configuration for diffusion models."""
    config = FullConfig()
    config.network.hidden_sizes = [64] * 10
    config.model_specific.num_timesteps = 100
    config.model_specific.beta_start = 1e-4
    config.model_specific.beta_end = 2e-2
    config.training.learning_rate = 1e-3
    return config


def get_noprop_config() -> FullConfig:
    """Get configuration for NoProp-CT models."""
    config = FullConfig()
    config.network.hidden_sizes = [64] * 8
    config.model_specific.num_time_steps = 10
    config.model_specific.time_horizon = 1.0
    config.training.learning_rate = 1e-3
    return config


def get_quadratic_config() -> FullConfig:
    """Get configuration for quadratic ResNet models."""
    config = FullConfig()
    config.network.hidden_sizes = [128] * 8
    config.network.residual_connections = True
    config.model_specific.use_quadratic_terms = True
    config.model_specific.quadratic_mixing = "adaptive"
    config.training.learning_rate = 1e-3
    return config


def get_transformer_config() -> FullConfig:
    """Get configuration for transformer models."""
    config = FullConfig()
    config.network.hidden_sizes = [256, 256]  # Feedforward layers
    config.model_specific.num_heads = 8
    config.model_specific.num_transformer_layers = 6
    config.training.learning_rate = 1e-4
    return config


# Registry of all available configurations
CONFIG_REGISTRY = {
    'deep_narrow': get_deep_narrow_config,
    'ultra_deep': get_ultra_deep_config,
    'glow': get_glow_config,
    'diffusion': get_diffusion_config,
    'noprop': get_noprop_config,
    'quadratic': get_quadratic_config,
    'transformer': get_transformer_config,
}


def get_config(config_name: str) -> FullConfig:
    """Get a predefined configuration by name."""
    if config_name not in CONFIG_REGISTRY:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIG_REGISTRY.keys())}")
    return CONFIG_REGISTRY[config_name]()


def list_available_configs() -> List[str]:
    """List all available configuration names."""
    return list(CONFIG_REGISTRY.keys())