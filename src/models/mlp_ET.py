"""
MLP ET implementation using the unified ETNetwork.

This module provides MLP-based ET models that inherit from the unified ETNetwork class.
"""

from .ET_Net import ETNetwork, ETTrainer, create_mlp_et
from ..config import FullConfig


class MLPNetwork(ETNetwork):
    """MLP-based ET Network."""
    
    def __init__(self, config):
        super().__init__(config=config, architecture="mlp")


class MLPTrainer(ETTrainer):
    """Trainer for MLP ET Network."""
    
    def __init__(self, config: FullConfig):
        super().__init__(config, architecture="mlp")


def create_model_and_trainer(config: FullConfig) -> MLPTrainer:
    """Factory function to create MLP model and trainer."""
    return create_mlp_et(config)
