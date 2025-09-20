"""
Deep Flow ET implementation using the unified ETNetwork.

This module provides Deep Flow-based ET models that inherit from the unified ETNetwork class.
"""

from .ET_Net import ETNetwork, ETTrainer, create_deepflow_et
from ..config import FullConfig


class DeepFlowNetwork(ETNetwork):
    """Deep Flow-based ET Network."""
    
    def __init__(self, config):
        super().__init__(config=config, architecture="deepflow")


class DeepFlowTrainer(ETTrainer):
    """Trainer for Deep Flow ET Network."""
    
    def __init__(self, config: FullConfig):
        super().__init__(config, architecture="deepflow")


def create_model_and_trainer(config: FullConfig) -> DeepFlowTrainer:
    """Factory function to create Deep Flow model and trainer."""
    return create_deepflow_et(config)