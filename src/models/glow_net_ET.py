"""
Glow Network ET implementation using the unified ETNetwork.

This module provides Glow-based ET models (normalizing flows with affine coupling)
that inherit from the unified ETNetwork class.
"""

from .ET_Net import ETNetwork, ETTrainer, create_deepflow_et
from ..config import FullConfig


class GlowNetworkET(ETNetwork):
    """Glow Network-based ET Network using affine coupling layers."""
    
    def __init__(self, config):
        super().__init__(config=config, architecture="deepflow")


class GlowTrainerET(ETTrainer):
    """Trainer for Glow Network ET."""
    
    def __init__(self, config: FullConfig):
        super().__init__(config, architecture="deepflow")


def create_glow_et_model_and_trainer(config: FullConfig) -> GlowTrainerET:
    """Factory function to create Glow ET model and trainer."""
    return create_deepflow_et(config)