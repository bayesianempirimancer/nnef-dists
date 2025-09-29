"""
LogZ Models - Hugging Face Compatible Neural Networks for Exponential Family Distributions

This package provides neural network models for learning the log normalizer
of exponential family distributions, compatible with Hugging Face standards.

Available Models:
- MLP_LogZ_Network: Multi-layer perceptron for log normalizer approximation
- Convex_LogZ_Network: Convex neural network maintaining convexity constraints
- Quadratic_LogZ_Network: Quadratic transformation networks
- GLU_LogZ_Network: Gated Linear Unit networks

Usage:
    from src.models import MLP_LogZ_Network, MLP_LogZ_Config
    
    config = MLP_LogZ_Config.from_pretrained("path/to/model")
    model = MLP_LogZ_Network.from_pretrained("path/to/model", config=config)
"""

# Import configuration classes
from ..configs.mlp_et_config import MLP_ET_Config

# Import model classes
from .mlp_et_net import MLP_ET_Network

# Model registry for Hugging Face compatibility
CONFIG_MAPPING_REGISTRY = {
    "mlp_et": MLP_ET_Config,
}

MODEL_MAPPING_REGISTRY = {
    "mlp_et": MLP_ET_Network,
}

# Auto-configuration mapping for Hugging Face
AUTO_CONFIG_MAPPING = {
    "mlp_et": MLP_ET_Config,
}

# Auto-model mapping
AUTO_MODEL_MAPPING = {
    "mlp_et": MLP_ET_Network,
}

# Public API
__all__ = [
    # Configuration classes
    "MLP_ET_Config",
    
    # Model classes
    "MLP_ET_Network",
    
    # Utilities
    "CONFIG_MAPPING_REGISTRY",
    "MODEL_MAPPING_REGISTRY",
    "AUTO_CONFIG_MAPPING",
    "AUTO_MODEL_MAPPING",
]

# Version information
__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__description__ = "Neural Networks for Exponential Family Log Normalizers"
