"""
NoProp-FM: Flow Matching NoProp implementation.

This module implements the flow matching variant of NoProp.
The key idea is to model the denoising process as a flow that transforms
a base distribution to the target distribution.
"""

from typing import Any, Dict, Optional, Tuple, Callable
from functools import partial
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
import optax

from ..base_model import BaseModel
from ..base_config import BaseConfig
from ...utils.ode_integration import integrate_ode
from ...utils.jacobian_utils import trace_jacobian


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

@dataclass
class Config(BaseConfig):
    """Configuration for NoProp FM Network."""
    
    # === MODEL IDENTIFICATION ===
    model_name: str = "noprop_fm_net"
    
    # === NOPROP FM SPECIFIC PARAMETERS ===
    num_timesteps: int = 20
    integration_method: str = "euler"  # "euler" or "heun"
    reg_weight: float = 0.0  # Hyperparameter from the paper
    sigma_t: float = 0.1  # Standard deviation of noise added to z_t


# ============================================================================
# MODEL IMPLEMENTATION
# ============================================================================

class NoPropFM(BaseModel[Config]):
    """Flow Matching NoProp implementation.
    
    This class implements the flow matching variant where the denoising
    process is modeled as a flow that transforms a base distribution
    to the target distribution over continuous time.
    """
    
    config: Config

    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True, rngs: dict = None, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Main forward pass through the NoProp FM network.
        
        Args:
            eta: Natural parameters of shape (batch_size, eta_dim)
            training: Whether in training mode (affects dropout)
            rngs: Random number generator keys
            **kwargs: Additional arguments (for HF compatibility)
            
        Returns:
            Tuple of (predictions, internal_loss) where:
            - predictions: Expected sufficient statistics E[T(X)|η] of shape (batch_size, output_dim)
            - internal_loss: Internal regularization loss (scalar, usually 0.0)
        """
        # For now, implement a simple forward pass that returns the input
        # This is a placeholder implementation
        predictions = eta  # Simple identity mapping for now
        internal_loss = jnp.array(0.0)
        
        return predictions, internal_loss
