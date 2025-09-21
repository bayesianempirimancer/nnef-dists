"""
GLU ET implementation using the unified ETNetwork.

This module provides GLU-based ET models that inherit from the unified ETNetwork class.
"""

from .ET_Net import ETNetwork, ETTrainer, create_glu_et
from ..config import FullConfig


class GLUNetwork(ETNetwork):
    """GLU-based ET Network."""
    
    architecture: str = "glu"

class GLUTrainer(ETTrainer):
    """Trainer for GLU ET Network."""
    
    def __init__(self, config: FullConfig):
        super().__init__(config, architecture="glu")


def create_model_and_trainer(config: FullConfig) -> GLUTrainer:
    """Factory function to create GLU model and trainer."""
    return create_glu_et(config)