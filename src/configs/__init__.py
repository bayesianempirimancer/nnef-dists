"""
Configuration classes for all models.

This module provides configuration classes for all neural network models
in the nnef-dists package, following Hugging Face conventions.
"""

# Import all configuration classes
from .mlp_et_config import MLP_ET_Config
# from .base_config import BaseConfig  # File doesn't exist
from .base_training_config import BaseTrainingConfig
from .base_model_config import BaseModelConfig

# Public API
__all__ = [
    "MLP_ET_Config",
    # "BaseConfig",  # File doesn't exist
    "BaseTrainingConfig",
    "BaseModelConfig",
]
