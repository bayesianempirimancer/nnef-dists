"""
Training implementation for Quadratic ET models.

This module implements a Hugging Face compatible training protocol for
Quadratic-based ET networks that directly predict expected sufficient statistics
from natural parameters.
"""

from typing import Dict, Any, Optional, Tuple, List
import jax
import jax.numpy as jnp

from .base_et_trainer import BaseETTrainer
from ..models.quadratic_et_net import Quadratic_ET_Network
from ..configs.quadratic_et_config import Quadratic_ET_Config


class QuadraticETTrainer(BaseETTrainer):
    """
    Trainer for Quadratic ET models using standard backpropagation.
    
    This implements a standard training loop with:
    - Standard backpropagation
    - MSE loss for ET prediction
    - Optional regularization (weight decay, dropout)
    - Standard optimization (Adam, etc.)
    """
    
