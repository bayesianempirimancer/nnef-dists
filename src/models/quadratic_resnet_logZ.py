"""
Quadratic ResNet LogZ implementation using the unified LogZNetwork.

This module provides Quadratic ResNet-based LogZ models that inherit from the unified LogZNetwork class.
"""

from .logZ_Net import LogZNetwork, LogZTrainer
from ..config import FullConfig


class QuadraticResNetLogNormalizer(LogZNetwork):
    """Quadratic ResNet-based LogZ Network."""
    
    architecture: str = "quadratic"


class QuadraticResNetLogNormalizerTrainer(LogZTrainer):
    """Trainer for Quadratic ResNet LogZ Network."""
    
    def __init__(self, config: FullConfig, hessian_method='diagonal', adaptive_weights=True):
        super().__init__(config, hessian_method=hessian_method, adaptive_weights=adaptive_weights)


def create_model_and_trainer(config: FullConfig, hessian_method='diagonal', adaptive_weights=True):
    """Factory function to create Quadratic ResNet LogZ model and trainer."""
    return QuadraticResNetLogNormalizerTrainer(config, hessian_method=hessian_method, adaptive_weights=adaptive_weights)