from __future__ import annotations

from typing import Sequence, Callable, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from src.ef import ExponentialFamily, MultivariateNormal

Array = jax.Array


def get_activation(name: str) -> Callable[[Array], Array]:
    if name == "relu":
        return nn.relu
    if name == "gelu":
        return nn.gelu
    if name == "tanh":
        return nn.tanh
    if name == "swish":
        return nn.swish
    raise ValueError(f"Unknown activation: {name}")


class nat2statMLP(nn.Module):
    """
    Natural parameters to sufficient statistics MLP.
    Maps natural parameters (eta) to sufficient statistics (moments).
    """
    dist: ExponentialFamily = MultivariateNormal(x_shape=(2,))
    hidden_sizes: Sequence[int] = (8*dist.eta_dim, 4*dist.eta_dim, 2*dist.eta_dim, dist.eta_dim)
    activation: str = "swish"
    output_dim: int = 2
    dropout_rate: float = 0.0
    use_batch_norm: bool = False
    use_feature_engineering: bool = False

    @nn.compact
    def __call__(self, eta: Array, training: bool = True) -> Array:
        # Apply feature engineering
        if self.use_feature_engineering:
            x = self.nat_features(eta)
        else:
            x = eta
        
        act = get_activation(self.activation)
        
        # Deep network with optional batch normalization and dropout
        for i, size in enumerate(self.hidden_sizes):
            dn = nn.Dense(size, kernel_init=nn.initializers.lecun_normal())(x)
            x = nn.Dense(size, kernel_init=nn.initializers.lecun_normal())(x) - dn*nn.Dense(size, kernel_init=nn.initializers.lecun_normal())(x)
            
            if self.use_batch_norm:
                x = nn.BatchNorm(use_running_average=not training)(x)
            
            x = act(x)
            
            if self.dropout_rate > 0 and training:
                x = nn.Dropout(rate=self.dropout_rate)(x, rng=self.make_rng('dropout'))
        
        # Output layer
        x = nn.Dense(self.output_dim)(x)
        return x
    
    def loss(self, params: dict, eta: Array, y: Array) -> Array:
        """Compute MSE loss for the model."""
        preds = self.apply(params, eta, training=False)
        return jnp.mean(jnp.square(preds - y))

    def KL_loss(self, params: dict, eta: Array, y: Array) -> Array:
        raise NotImplementedError

    def nat_features(self, eta: Array) -> Array:
        """
        Natural parameter feature engineering.
        Creates non-linear features from the input eta parameters with numerical stability.
        """
        # Original eta parameters
        features = [eta]
        
        # # 1. clip(1/eta) - inverse with aggressive clipping for numerical stability
        # eta_abs = jnp.abs(eta)
        # eta_safe = jnp.where(eta_abs < 1e-6, jnp.sign(eta) * 1e-6, eta)
        # inv_eta = jnp.clip(1.0 / eta_safe, -100.0, 100.0)
        # features.append(inv_eta)
        
        # # 2. log(abs(eta)) - logarithmic features with proper clipping
        # eta_abs_clipped = jnp.clip(eta_abs, 1e-8, 1e8)
        # log_eta = jnp.log(eta_abs_clipped)
        # features.append(log_eta)
        
        # 3. eta/norm(eta) - normalized eta (unit vector)
        eta_inv = jnp.clip(1.0/eta,-1000.0,1000.0)
        features.append(eta_inv)
        eta_norm = jnp.linalg.norm(eta, axis=-1, keepdims=True)
        features.append(eta/eta_norm)
        features.append(eta_norm*eta_inv)
        features.append(eta_norm)
        features.append(jnp.log(eta_norm))


        # Concatenate all features
        result = jnp.concatenate(features, axis=-1)
        result = jnp.concatenate([result, jnp.abs(result)], axis=-1)

        # Final safety check - replace any remaining NaN or Inf values
        result = jnp.where(jnp.isfinite(result), result, 0.0)
        
        # Additional safety: ensure all values are finite and bounded
        result = jnp.clip(result, -1e6, 1e6)
        return result

# def newtons_method(fun: Callable[[Array], Array], x0: Array, max_iters: int = 10, tol: float = 1e-6) -> Array:
#     """
#     Newton's method for finding the root of a function.
#     """
#     for i in range(max_iters):
#         x0 = x0 - (1.0-tol)*jax.grad(fun)(x0) / jax.hessian(fun)(x0)
#         if jnp.linalg.norm(jax.grad(fun)(x0)) < tol:
#             break

#     return x0, i