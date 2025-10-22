"""
NoProp-FM: Flow Matching NoProp implementation.

This module implements the flow matching variant of NoProp.
The key idea is to model the denoising process as a flow that transforms
a base distribution to the target distribution.
"""

from typing import Any, Dict, Tuple, Optional
from functools import partial, cached_property
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
from .crn import ConditionalResnet_MLP as ConditionalResnet


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

@dataclass(frozen=True)
class Config(BaseConfig):
    """Configuration for NoProp FM Network."""
    
    # === MODEL IDENTIFICATION ===
    model_name: str = "noprop_fm_net"
    
    # === NOPROP FM SPECIFIC PARAMETERS ===
    z_shape: Tuple[int, ...] = (9,)  # Shape of target z (excluding batch dimensions)
    x_shape: Tuple[int, ...] = (9,)  # Shape of input x (excluding batch dimensions)
    num_timesteps: int = 20
    integration_method: str = "euler"  # "euler" or "heun"
    reg_weight: float = 0.0  # Hyperparameter from the paper
    sigma_t: float = 0.1  # Standard deviation of noise added to z_t

    
    # === MODEL CONFIGURATION ===
    model_type: str = "conditional_resnet"
    hidden_sizes: Tuple[int, ...] = (128, 128, 128)
    model_dropout_rate: float = 0.1
    model_activation: str = "swish"
    time_embed_dim: int = 64
    time_embed_method: str = "sinusoidal"
    time_embed_min_freq: float = 1.0
    time_embed_max_freq: float = 1000.0
    eta_embed_type: str = "default"
    eta_embed_dim: Optional[int] = None
    activation: str = "swish"
    use_batch_norm: bool = False
    dropout_rate: float = 0.1
    
    # Additional attributes needed for BaseConfig compatibility
    use_resnet: bool = False
    num_resnet_blocks: int = 0
    use_layer_norm: bool = False
    loss_type: str = "flow_matching"



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
    
    @property
    def z_shape(self):
        return self.config.z_shape
    
    @property
    def x_shape(self):
        return self.config.x_shape
    
    @cached_property
    def _get_z_dim(self) -> int:
        """Calculate the flattened dimension from z_shape."""
        z_dim = 1
        for dim in self.config.z_shape:
            z_dim *= dim
        return z_dim

    def _flatten_z(self, z: jnp.ndarray) -> jnp.ndarray:
        """Flatten the z tensor."""
        if len(self.config.z_shape) > 1:
            return z.reshape(z.shape[:-1] + (self._get_z_dim(),))
        else:
            return z

    def _unflatten_z(self, z: jnp.ndarray) -> jnp.ndarray:
        """Unflatten the z tensor."""
        if len(self.config.z_shape) > 1:
            return z.reshape(z.shape[:-1] + self.config.z_shape)
        else:
            return z

    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray, training: bool = True, rngs: dict = None, **kwargs) -> jnp.ndarray:
        """
        Main forward pass to compute model output.
        
        Args:
            z: Current state/trajectory of shape (batch_size, z_dim)
            x: Input data (natural parameters eta) of shape (batch_size, x_dim)
            t: Time step of shape (batch_size,)
            training: Whether in training mode (affects dropout)
            rngs: Random number generator keys
            **kwargs: Additional arguments (for HF compatibility)
            
        Returns:
            Model output [batch_shape + z_shape]
        """
        z = self._unflatten_z(z)
        model_output = self._get_model_output(z, x, t, training=training)
        output = self._flatten_z(model_output)
        return output

    @nn.compact
    def _get_model_output(
        self, 
        z: jnp.ndarray, 
        x: jnp.ndarray, 
        t: jnp.ndarray,
        training: bool = True
    ) -> jnp.ndarray:
        """Get model output - @nn.compact method for parameter initialization."""
        # Create the conditional ResNet model using config parameters
        from .crn import get_act
        model = ConditionalResnet(
            hidden_dims=self.config.hidden_sizes,
            time_embed_dim=self.config.time_embed_dim,
            time_embed_method=self.config.time_embed_method,
            eta_embed_type=self.config.eta_embed_type,
            eta_embed_dim=self.config.eta_embed_dim,
            activation_fn=get_act(self.config.model_activation),
            dropout_rate=self.config.model_dropout_rate
        )
        return model(z, x, t, training=training)

    def dz_dt(
        self,
        params: Dict[str, Any],
        z: jnp.ndarray,
        x: jnp.ndarray,
        t: jnp.ndarray
    ) -> jnp.ndarray:
        """Wrapper for ode solver

        Args:
            params: Model parameters
            z: Current state [batch_size + z_shape]
            x: Input data [batch_size, ...]
            t: Current time [batch_size]
            
        Returns:
            Vector field dz/dt [batch_size + z_shape]
        """
        return self.apply(params, z, x, t, training=False)
    
    @partial(jax.jit, static_argnums=(0,))  # self is static
    def compute_loss(
        self,
        params: Dict[str, Any],
        x: jnp.ndarray,
        target: jnp.ndarray,
        key: jr.PRNGKey
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute the NoProp-FM loss.
        
        For flow matching, the loss is simply the MSE between model output and target:
        L_FM = E[||model_output - (z_target - z_0)||²]  where z_0 is a random initial condition
        
        Args:
            params: Model parameters
            x: Input data [batch_shape, ...]
            target: Clean target [batch_shape + z_shape]
            key: Random key for sampling t and z_t
            
        Returns:
            Tuple of (loss, metrics)
        """

        target = self._flatten_z(target)
        batch_shape = target.shape[:-1]
        # Split keys for all random operations
        key, t_key, z0_key, z_t_noise_key = jr.split(key, 4)
        
        t = jr.uniform(t_key, batch_shape, minval=0.0, maxval=1.0)
        z_0 = jr.normal(z0_key, target.shape)

        z_t = z_0 + t[...,None] * (target - z_0)
        z_t = z_t + self.config.sigma_t * jr.normal(z_t_noise_key, z_t.shape)

        # Get model output
        dz_dt = self.apply(params, z_t, x, t)
        # Construct estimated target 
        z_1_est = z_t + dz_dt*(1.0-t[:,None])

        # Compute MSE loss
        squared_error = (z_1_est - target) ** 2
        mse = jnp.mean(squared_error)
        
        # Regularization loss
        reg_loss = jnp.mean(dz_dt ** 2)
        fm_loss = jnp.mean((dz_dt - (target - z_0)) ** 2)
#        no_prop_fm_loss = jnp.mean((dz_dt - (target - z_t)/(1.0-t[:,None])) ** 2)

        total_loss = fm_loss + self.config.reg_weight * reg_loss

        # Compute additional metrics
        metrics = {
            "fm_loss": fm_loss,
            "mse": mse,
            "reg_loss": reg_loss,
            "total_loss": total_loss,
        }
        
        return total_loss, metrics
    
    @partial(jax.jit, static_argnums=(0, 3, 4, 5, 6))  # self, num_steps, integration_method, output_type, with_logp are static arguments    
    def predict(
        self,
        params: Dict[str, Any],
        x: jnp.ndarray,
        num_steps: int,
        integration_method: str = "euler",
        output_type: str = "end_point",
        with_logp: bool = False,
        key: jr.PRNGKey = None
    ) -> jnp.ndarray:
        """
        Generate predictions using the trained NoProp-FM neural ODE.
        
        This integrates the learned vector field from zeros (t=0)
        to the final prediction (t=1), following the paper's approach.
        Uses scan-based integration for better performance.
        
        Args:
            params: Model parameters
            x: Input data [batch_shape, ...]
            num_steps: Number of integration steps
            integration_method: Integration method to use ("euler", "heun", "rk4", "adaptive")
            output_type: Type of output ("end_point" or "trajectory")
            key: Random key
            
        Returns:
            If output_type="end_point": Final prediction [batch_shape + z_shape]
            If output_type="trajectory": Full trajectory [num_steps+1, batch_shape + z_shape]
            Note that in fully generative model you need to provide a key, and if x=None, then you need
                to provide something that returns x.shape[:-len(self.x_shape)] = (number_of_samples,)
        """
        # Disable gradient tracking through parameters for inference
        params_no_grad = jax.lax.stop_gradient(params)
        
        # Infer batch shape from input tensor
        batch_shape = x.shape[:-len(self.x_shape)]

        if key is None:
            z0 = jnp.zeros(batch_shape + self.z_shape)
        else:
            z0 = jr.normal(key, batch_shape + self.z_shape)  # check for sensitivity to initial conditions
        
        if with_logp:
            # Combined vector field for z and logp integration
            def vector_field(params, state, x, t):
                z = state[..., :-1]  # All but last dimension
                logp = state[..., -1:]  # Last dimension
                dz_dt = self.dz_dt(params, z, x, t)
                # Compute Jacobian trace for logp evolution
                # Broadcast t to match batch size
                t_broadcast = jnp.broadcast_to(t, z.shape[:-1])
                trace_jac = trace_jacobian(self.dz_dt, params, z, x, t_broadcast)
                dlogp_dt = -trace_jac
                return jnp.concatenate([dz_dt, dlogp_dt[..., None]], axis=-1)
            
            # Initialize combined state
            logp0 = jnp.zeros(batch_shape)
            state0 = jnp.concatenate([z0, logp0[..., None]], axis=-1)
            
            return integrate_ode(
                vector_field=vector_field,
                params=params_no_grad,
                z0=state0,
                x=x,
                time_span=(0.0, 1.0),
                num_steps=num_steps,
                method=integration_method,
                output_type=output_type
            )
        else:
            # Standard vector field for z only
            def vector_field(params, z, x, t):
                return self.dz_dt(params, z, x, t)

            return integrate_ode(
                vector_field=vector_field,
                params=params_no_grad,
                z0=z0,
                x=x,
                time_span=(0.0, 1.0),
                num_steps=num_steps,
                method=integration_method,
                output_type=output_type
            )

    @partial(jax.jit, static_argnums=(0, 5))  # self and optimizer are static arguments
    def train_step(
        self,
        params: Dict[str, Any],
        x: jnp.ndarray,
        target: jnp.ndarray,
        opt_state: optax.OptState,
        optimizer: optax.GradientTransformation,
        key: jr.PRNGKey,
    ) -> Tuple[Dict[str, Any], optax.OptState, jnp.ndarray, Dict[str, jnp.ndarray]]:

        """Single training step for NoProp-FM.
        
        Args:
            params: Model parameters
            opt_state: Optimizer state
            x: Input data [batch_size, height, width, channels]
            target: Clean target [batch_size, num_classes]
            key: Random key
            optimizer: Optax optimizer
            
        Returns:
            Tuple of (updated_params, updated_opt_state, loss, metrics)
        """
        # Compute loss and gradients first
        (loss, metrics), grads = jax.value_and_grad(
            self.compute_loss, has_aux=True)(params, x, target, key)
        
        # Update parameters using optimizer (outside JIT)
        updates, updated_opt_state = optimizer.update(grads, opt_state, params)
        updated_params = optax.apply_updates(params, updates)
        
        return updated_params, updated_opt_state, loss, metrics