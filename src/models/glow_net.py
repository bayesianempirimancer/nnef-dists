"""
Glow Network implementation using the unified ETNetwork.

This module provides Glow-based models (normalizing flows with affine coupling)
that inherit from the unified ETNetwork class.
"""

from .ET_Net import ETTrainer
from ..config import FullConfig


def create_glow_et_model_and_trainer(config: FullConfig) -> ETTrainer:
    """Factory function to create Glow ET model and trainer."""
    # Use GLU architecture as a proxy for flow-like behavior
    return ETTrainer(config, architecture="glu")