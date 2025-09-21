"""
GLU LogZ implementation using the unified LogZNetwork.

This module provides GLU-based LogZ models that inherit from the unified LogZNetwork class.
"""

from .logZ_Net import LogZNetwork, LogZTrainer
from ..config import FullConfig


class GLULogNormalizerNetwork(LogZNetwork):
    """GLU-based LogZ Network."""
    
    architecture: str = "glu"


class GLULogNormalizerTrainer(LogZTrainer):
    """Trainer for GLU LogZ Network."""
    
    def __init__(self, config: FullConfig, hessian_method='diagonal', adaptive_weights=True):
        super().__init__(config, hessian_method=hessian_method, adaptive_weights=adaptive_weights)


def create_model_and_trainer(config: FullConfig, hessian_method='diagonal', adaptive_weights=True):
    """Factory function to create GLU LogZ model and trainer."""
    return GLULogNormalizerTrainer(config, hessian_method=hessian_method, adaptive_weights=adaptive_weights)