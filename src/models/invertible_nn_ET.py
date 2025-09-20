"""
Invertible Neural Network ET implementation using the unified ETNetwork.

This module provides Invertible NN-based ET models that inherit from the unified ETNetwork class.
"""

from .ET_Net import ETNetwork, ETTrainer, create_invertible_et
from ..config import FullConfig


class InvertibleNetwork(ETNetwork):
    """Invertible NN-based ET Network."""
    
    def __init__(self, config):
        super().__init__(config=config, architecture="invertible")


class InvertibleTrainer(ETTrainer):
    """Trainer for Invertible NN ET Network."""
    
    def __init__(self, config: FullConfig):
        super().__init__(config, architecture="invertible")


def create_model_and_trainer(config: FullConfig) -> InvertibleTrainer:
    """Factory function to create Invertible NN model and trainer."""
    return create_invertible_et(config)