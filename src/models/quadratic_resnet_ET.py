"""
Quadratic ResNet ET implementation using the unified ETNetwork.

This module provides Quadratic ResNet-based ET models that inherit from the unified ETNetwork class.
"""

from .ET_Net import ETNetwork, ETTrainer, create_quadratic_et
from ..config import FullConfig


class QuadraticResNetNetwork(ETNetwork):
    """Quadratic ResNet-based ET Network."""
    
    architecture: str = "quadratic"
    
    def __init__(self, config):
        super().__init__(config=config)


class QuadraticResNetTrainer(ETTrainer):
    """Trainer for Quadratic ResNet ET Network."""
    
    def __init__(self, config: FullConfig):
        super().__init__(config, architecture="quadratic")


def create_model_and_trainer(config: FullConfig) -> QuadraticResNetTrainer:
    """Factory function to create Quadratic ResNet model and trainer."""
    return create_quadratic_et(config)