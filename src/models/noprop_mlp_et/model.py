"""
NoProp MLP Network

A simplified version of the NoProp Geometric Flow ET Network that uses a regular 3-layer MLP
instead of the Fisher flow field. This helps isolate the contribution of the Fisher flow
vs. a standard MLP approach while maintaining the same time embedding and NoProp training protocol.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field

from ..base_model import BaseModel
from ..base_config import BaseConfig
from ...utils.activation_utils import get_activation_function
from ...layers.mlp import MLPBlock
from ...layers.concatsquash import ConcatSquash
from ...embeddings.time_embeddings import LogFreqTimeEmbedding, ConstantTimeEmbedding
from ...embeddings.eta_embedding import EtaEmbedding


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

@dataclass
class Config(BaseConfig):
    """Configuration for NoProp MLP Network."""
    
    # === MODEL IDENTIFICATION ===
    model_name: str = "noprop_mlp_et_net"
    
    # === NOPROP MLP SPECIFIC PARAMETERS ===
    layer_norm_type: str = "weak_layer_norm"
    eta_embed_dim: Optional[int] = 8  # Default eta embedding dimension
    eta_embedding_type: str = "default"  # Type of eta embedding to use
    loss_type: str = "noprop"  # "noprop" or "flow_matching"
    n_time_steps: int = 20  # Number of time steps for integration during prediction
    
    


@dataclass
class FlowMatchingConfig(BaseConfig):
    """Configuration for NoProp MLP Network with Flow Matching."""
    
    # === MODEL IDENTIFICATION ===
    model_name: str = "noprop_mlp_et_flow_matching_net"
    
    # === FLOW MATCHING SPECIFIC PARAMETERS ===
    layer_norm_type: str = "weak_layer_norm"
    eta_embed_dim: Optional[int] = 8  # Default eta embedding dimension
    eta_embedding_type: str = "default"  # Type of eta embedding to use
    loss_type: str = "flow_matching"  # Fixed to "flow_matching"
    n_time_steps: int = 20  # Number of time steps for integration during prediction
    
    


# ============================================================================
# MODEL IMPLEMENTATION
# ============================================================================


class NoProp_MLP_ET_Network(BaseModel[Config]):
    """
    NoProp MLP Network that learns flow dynamics using a simple 3-layer MLP.
    
    This is a simplified version of the NoProp Geometric Flow ET Network that replaces
    the uses standard MLP to compute the flow field in the case of flow matching or the 
    estimated output in the noprop_ct case.
    
    In the original noprop paper, the authors parameterized SNR directly via a real valued gamma(t) so that
    SNR(t) = exp(-gamma(t)), where gamma(t) is a real value decreasing function.  Here we choose a different
    parameterization that seem more consistent with the continuous time formulation.  Specifically, we define
    delta(t)dt to be the variance of the weiner process noise added during the backward proccess.  
    In terms of the original no-prop paper this is equivalent to defining 1-alpha(t) = dt*delta(t) and 
    1-alpha_bar(t) = Delta(t) = Delta(T=1) + int_t^T delta(t)dt which imlies alpha_bar_prime(t) = delta(t).  
    This parameterization is easier to constrain in a manner that is consistent with the continuous time formulation.
    For example, the cumulative noise added to the backward process from t = 1 to t = 0, should be Delta(0) = 1.0.  
    This suggests the parameterization Delta(t) = exp(-gamma(t)) where gamma(0) = 0 and gamma(t) is an increasing function, 
    terminating at gamma(1) = -log(Delta(1)) > 0.  
    This lead to the following identities

        Delta(t) = exp(-gamma(t))

        1-alpha_bar(t) = Delta(t) 
        1-alpha(t) = -Delta'(t)dt
        SNR(t) = alpha_bar(t)/(1-alpha_bar(t)) = (1-Delta(t))/Delta(t) = 1/Delta(t) - 1
        SNR'(t) = Delta'(t)/Delta(t)**2

    With forward prcess identities a,b,c for mu_t = a(t) u_net(z(t-dt), x , t) + b(t) z(t-dt) + c(t) noise, given by 
       a(t) = -sqrt(1-Delta(t))/Delta(t)*Delta'(t) * dt  
       b(t) = 1 + ( 1/2 + 1/Delta(t) ) * Delta'(t) * dt
       c(t) =     stuff we dont need to know about

    """
    config: 'Config'
    
    def gamma_prime(self, gamma_rate: jnp.ndarray) -> jnp.ndarray:
        """Positive valued function so that gamma is increasing."""
        return nn.softplus(gamma_rate)
    
    def gamma(self, t: jnp.ndarray, gamma_rate: jnp.ndarray) -> jnp.ndarray:
        """Gamma function for noise schedule."""
        return self.gamma_prime(gamma_rate) * t
    
    def Delta(self, t: jnp.ndarray, gamma_rate: jnp.ndarray) -> jnp.ndarray:
        """Delta function for noise schedule."""
        return jnp.exp(-self.gamma(t, gamma_rate))

    def Delta_prime(self, t: jnp.ndarray, gamma_rate: jnp.ndarray) -> jnp.ndarray:
        """Derivative of Delta function."""
        return -self.gamma_prime(gamma_rate) * jnp.exp(-self.gamma(t, gamma_rate))

    def alpha_bar(self, t: jnp.ndarray, gamma_rate: jnp.ndarray) -> jnp.ndarray:
        # this is called alpha_bar in the paper: recall that noise added from time t to T is 1 - alpha_bar(t)
        return 1.0 - self.Delta(t, gamma_rate)

    def SNR(self, t: jnp.ndarray, gamma_rate: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(self.gamma(t, gamma_rate))-1.0
    
    def SNR_prime(self, t: jnp.ndarray, gamma_rate: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(self.gamma(t, gamma_rate)) * self.gamma_prime(gamma_rate)


    def one_minus_alpha_over_dt(self, t: jnp.ndarray, gamma_rate: jnp.ndarray) -> jnp.ndarray:
        """One minus alpha over dt."""
        return -self.Delta_prime(t, gamma_rate)
    
    def get_a_b_minus_1(self, t: jnp.ndarray, gamma_rate: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        Delta = self.Delta(t, gamma_rate)
        Delta_prime = self.Delta_prime(t, gamma_rate)
#        gamma_prime = self.gamma_prime(gamma_rate)

        a = -jnp.sqrt(1-Delta)*(Delta_prime/Delta)
        b_minus_1 = (Delta/2.0 + 1)*(Delta_prime/Delta)
        return a, b_minus_1
    
    @nn.compact
    def __call__(self, z: jnp.ndarray, eta: jnp.ndarray, t: jnp.ndarray, training: bool = True, rngs: dict = None) -> jnp.ndarray:
        """
        Forward pass of the NoProp MLP network.
        
        Args:
            z: Current state [batch_shape, output_dim]
            eta: Target natural parameters [batch_shape, input_dim]
            t: Time [batch_shape] or scalar
            training: Whether in training mode
            rngs: Random number generators
            
        Returns:
            du/dt: Flow field [batch_shape, output_dim]
        """
        # NoProp-specific parameters - initialize with a more reasonable value
        gamma_rate = self.param('gamma_rate', nn.initializers.constant(4.0), ())
        
        # Handle time embedding
        if t.ndim == 0:  # Scalar time
            t = jnp.broadcast_to(t, z.shape[:-1])
        
        # Create time embedding using the configured method
        from ...embeddings.time_embeddings import create_time_embedding
        t_embedding = create_time_embedding(
            embed_dim=self.config.time_embed_dim,
            method=self.config.time_embed_method,
            min_freq=self.config.time_embed_min_freq,
            max_freq=self.config.time_embed_max_freq,
            T_max=self.config.time_embed_T_max
        )
        t_embed = t_embedding(t)
        
        # Handle eta embedding
        if self.config.eta_embed_dim is not None and self.config.eta_embed_dim > 0:
            # Use the proper EtaEmbedding class
            eta_embedding_fn = EtaEmbedding(embedding_type=self.config.eta_embedding_type)
            eta_embedded = eta_embedding_fn(eta)
        else:
            eta_embedded = eta
        
        # Concatenate inputs: [z, eta_embedded, t_embed]
#        x = jnp.concatenate([z, eta_embedded, t_embed], axis=-1)
        x = ConcatSquash(self.config.hidden_sizes[0])(z, eta_embedded, t_embed)
        x = get_activation_function(self.config.activation)(x)
        
        # Apply MLP block
        x = MLPBlock(
            features=tuple(self.config.hidden_sizes[1:]),
            use_bias=True,
            activation=get_activation_function(self.config.activation),
            use_layer_norm=self.config.use_layer_norm,
            dropout_rate=self.config.dropout_rate,
            name='mlp_block'
        )(x, training=training, rngs=rngs)
        
        # Output layer with residual connection
        u_t = nn.Dense(self.config.output_dim, name='output_layer')(x)
        
        return u_t
    
    def _compute_noprop_loss_impl(self, params: dict, eta: jnp.ndarray, mu_T: jnp.ndarray, training: bool = True, rngs: dict = None) -> jnp.ndarray:
        """Implementation of NoProp loss (no JIT compilation for debugging)."""
        # Generate random times t ~ Uniform(0, 1) for each sample in the batch
        if rngs is not None and 'noise' in rngs:
            noise_rng = rngs['noise']
        else:
            noise_rng = jax.random.PRNGKey(0)
        t_rng, noise_rng = jax.random.split(noise_rng, 2)
        
        # Sample time for each sample in the batch (shape: eta.shape[:-1])
        batch_shape = eta.shape[:-1]
        t = jax.random.uniform(t_rng, batch_shape)  # Batch of time values
        
        # Use the flow field computation
        gamma_rate = params['params']['gamma_rate']
        Delta_t = self.Delta(t, gamma_rate)  # Now Delta_t has batch_shape
        
        # Compute noisy state
        z_t = mu_T*jnp.sqrt(1-Delta_t[..., None]) + jax.random.normal(noise_rng, mu_T.shape)*jnp.sqrt(Delta_t[..., None])
        u_t = self.apply(params, z_t, eta, t, training=training, rngs=rngs)
        
        # NoProp loss (penalize large derivatives)
        # SNR_prime now has batch_shape, so we need to broadcast it properly
        SNR_prime_t = self.SNR_prime(t, gamma_rate)  # Shape: batch_shape
        
        # Compute loss with better numerical stability
        loss = jnp.mean(SNR_prime_t[..., None] * (u_t - mu_T) ** 2)
        
        # Add numerical stability - check for NaN/Inf
        loss = jnp.where(jnp.isfinite(loss), loss, 1.0)
        
        return loss

    def _compute_noprop_loss(self, params: dict, eta: jnp.ndarray, mu_T: jnp.ndarray, training: bool = True, rngs: dict = None) -> jnp.ndarray:
        """Compute NoProp loss (no JIT compilation for debugging)."""
        return self._compute_noprop_loss_impl(params, eta, mu_T, training, rngs)

    def _noprop_predict(self, params: dict, eta: jnp.ndarray, n_time_steps: int) -> jnp.ndarray:
        """
        Internal NoProp prediction method.
        
        Args:
            params: Model parameters
            eta: Target natural parameters [batch_shape, eta_dim]
            n_time_steps: Number of integration steps
            
        Returns:
            Predicted expected sufficient statistics [batch_shape, output_dim]
        """
        # Simple forward Euler integration starting at z = 0
        dt = jnp.array(1.0 / n_time_steps)
        z_current = jnp.zeros_like(eta)
        
        # Get gamma_rate parameter - try different possible structures
        try:
            gamma_rate = params['params']['gamma_rate']
        except (KeyError, TypeError):
            try:
                gamma_rate = params['gamma_rate']
            except (KeyError, TypeError):
                # Fallback to default value
                gamma_rate = jnp.array(4.0)
        
        for i in range(n_time_steps):
            t_current = i * dt
            u_t = self.apply(params, z_current, eta, t_current, training=False)
            a, b_minus_1 = self.get_a_b_minus_1(t_current, gamma_rate)
            
            # Add numerical stability check
            u_t = jnp.where(jnp.isfinite(u_t), u_t, jnp.zeros_like(u_t))
            z_current = z_current + dt * (a*u_t + (b_minus_1)*z_current)
#            z_current = z_current + dt*(u_t-z_current)/self.Delta(t_current, gamma_rate)
            
            # Ensure z_current remains finite
            z_current = jnp.where(jnp.isfinite(z_current), z_current, jnp.zeros_like(z_current))
        
        return z_current
    
    def _compute_flow_matching_loss_impl(self, params: dict, eta: jnp.ndarray, mu_T: jnp.ndarray, training: bool = True, rngs: dict = None) -> jnp.ndarray:
        """Implementation of flow matching loss (no JIT compilation for debugging)."""
        # Use the provided rngs or create a default one
        if rngs is not None and 'noise' in rngs:
            noise_rng = rngs['noise']
        else:
            noise_rng = jax.random.PRNGKey(0)
        t_rng, noise_rng = jax.random.split(noise_rng, 2)
        
        # Sample time for each sample in the batch
        batch_shape = eta.shape[:-1]
        t = jax.random.uniform(t_rng, batch_shape)
        
        # Sample noise
        z_0 = jax.random.normal(noise_rng, mu_T.shape)
        
        # Interpolate between z_0 and mu_T
        z_t = t[..., None] * mu_T + (1 - t[..., None]) * z_0
        
        # Compute predicted flow field
        du_dt_predicted = self.apply(params, z_t, eta, t, training=training, rngs=rngs)
        
        # Compute target flow field (simplified for now)
        loss = jnp.mean((du_dt_predicted - (mu_T - z_0)) ** 2)
        return loss

    def _compute_flow_matching_loss(self, params: dict, eta: jnp.ndarray, mu_T: jnp.ndarray, training: bool = True, rngs: dict = None) -> jnp.ndarray:
        """Compute flow matching loss (no JIT compilation for debugging)."""
        return self._compute_flow_matching_loss_impl(params, eta, mu_T, training, rngs)
    
    def _flow_matching_predict(self, params: dict, eta: jnp.ndarray, n_time_steps: int) -> jnp.ndarray:
        """Compute flow matching prediction."""
        # Simple forward Euler integration starting at z = 0
        dt = jnp.array(1.0 / n_time_steps)
        z_current = jnp.zeros_like(eta)
        
        for i in range(n_time_steps):
            t = i * dt
            du_dt = self.apply(params, z_current, eta, t, training=False)
            z_current = z_current + dt * du_dt
        
        return z_current
    
    def loss(self, params: dict, eta: jnp.ndarray, mu_T: jnp.ndarray, training: bool = True, rngs: dict = None) -> jnp.ndarray:
        """Compute loss based on the configured loss type."""
        if self.config.loss_type == "noprop":
            return self._compute_noprop_loss(params, eta, mu_T, training, rngs)
        elif self.config.loss_type == "flow_matching":
            return self._compute_flow_matching_loss(params, eta, mu_T, training, rngs)
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
    
    def predict(self, params: dict, eta: jnp.ndarray, n_time_steps: int = None) -> jnp.ndarray:
        """Make predictions using the configured prediction method."""
        if n_time_steps is None:
            n_time_steps = self.config.n_time_steps
        
        if self.config.loss_type == "noprop":
            return self._noprop_predict(params, eta, n_time_steps)
        elif self.config.loss_type == "flow_matching":
            return self._flow_matching_predict(params, eta, n_time_steps)
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")


