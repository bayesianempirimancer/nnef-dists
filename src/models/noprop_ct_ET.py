"""
NoProp-CT ET implementation using the unified ETNetwork.

This module provides NoProp-CT-based ET models that inherit from the unified ETNetwork class.
"""

from .ET_Net import ETNetwork, ETTrainer, create_nopropct_et
from ..config import FullConfig


class NoPropCTNetwork(ETNetwork):
    """NoProp-CT-based ET Network."""
    
    architecture: str = "nopropct"
    
    def __init__(self, config):
        super().__init__(config=config)


class NoPropCTTrainer(ETTrainer):
    """Trainer for NoProp-CT ET Network."""
    
    def __init__(self, config: FullConfig):
        super().__init__(config, architecture="nopropct")


def create_model_and_trainer(config: FullConfig) -> NoPropCTTrainer:
    """Factory function to create NoProp-CT model and trainer."""
    return create_nopropct_et(config)