"""
MLP LogZ implementation using the unified LogZNetwork.

This module provides MLP-based LogZ models that inherit from the unified LogZNetwork class.
"""

from .logZ_Net import LogZNetwork, LogZTrainer
from ..config import FullConfig


class MLPLogNormalizerNetwork(LogZNetwork):
    """MLP-based LogZ Network."""
    
    architecture: str = "mlp"

class MLPLogNormalizerTrainer(LogZTrainer):
    """Trainer for MLP LogZ Network."""
    
    def __init__(self, config: FullConfig, hessian_method='diagonal', adaptive_weights=True):
        super().__init__(config, hessian_method=hessian_method, adaptive_weights=adaptive_weights)


def create_model_and_trainer(config: FullConfig, hessian_method='diagonal', adaptive_weights=True):
    """Factory function to create MLP LogZ model and trainer."""
    return MLPLogNormalizerTrainer(config, hessian_method=hessian_method, adaptive_weights=adaptive_weights)