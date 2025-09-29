"""
Training implementations for all models.

This module provides training classes and utilities for all neural network models
in the nnef-dists package, following Hugging Face conventions.
"""

# Import base training classes
from .base_et_trainer import BaseETTrainer

# Import training factory
from .trainer_factory import (
    create_mlp_et_trainer,
)

# Public API
__all__ = [
    # Base classes
    "BaseETTrainer",
    
    # Factory functions
    "create_mlp_et_trainer",
]
