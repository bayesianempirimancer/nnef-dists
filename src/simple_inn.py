"""
Simplified Invertible Neural Network for Exponential Family Moment Mapping

A simpler implementation focusing on the core invertible transformations
without the complex mask handling that was causing JIT issues.
"""

from __future__ import annotations

from typing import Dict, Tuple, Any
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random
import optax
from flax.training import train_state

from src.ef import ExponentialFamily

Array = jax.Array


@dataclass
class SimpleINNConfig:
    """Configuration for Simple INN."""
    num_layers: int = 4
    hidden_size: int = 64
    activation: str = "tanh"
    learning_rate: float = 1e-3
    clamp_alpha: float = 2.0


class SimpleCouplingLayer(nn.Module):
    """
    Simplified affine coupling layer.
    For 2D case, first dimension predicts transformation for second dimension.
    """
    hidden_size: int = 64
    activation: str = "tanh"
    clamp_alpha: float = 2.0
    
    @nn.compact
    def __call__(self, x: Array, reverse: bool = False) -> Tuple[Array, Array]:
        """
        Forward/reverse pass through coupling layer.
        
        For 2D input [x1, x2]:
        - Forward: x1 unchanged, x2 = x2 * exp(s(x1)) + t(x1)
        - Reverse: x1 unchanged, x2 = (x2 - t(x1)) / exp(s(x1))
        """
        if x.shape[-1] != 2:
            raise ValueError("SimpleCouplingLayer only supports 2D inputs")
        
        x1, x2 = x[..., 0:1], x[..., 1:2]  # Split into two parts
        
        # Neural network to predict scale and translation from x1
        if self.activation == "tanh":
            act = nn.tanh
        elif self.activation == "relu":
            act = nn.relu
        else:
            act = nn.swish
        
        h = x1
        h = nn.Dense(self.hidden_size)(h)
        h = act(h)
        h = nn.Dense(self.hidden_size)(h)
        h = act(h)
        
        # Output scale and translation
        st = nn.Dense(2)(h)  # [log_s, t]
        log_s, t = st[..., 0:1], st[..., 1:2]
        
        # Clamp for stability
        log_s = jnp.tanh(log_s / self.clamp_alpha) * self.clamp_alpha
        
        if not reverse:
            # Forward transformation
            s = jnp.exp(log_s)
            x2_new = x2 * s + t
            y = jnp.concatenate([x1, x2_new], axis=-1)
            log_det_J = jnp.sum(log_s, axis=-1)  # Log determinant
        else:
            # Reverse transformation
            s = jnp.exp(log_s)
            x2_new = (x2 - t) / s
            y = jnp.concatenate([x1, x2_new], axis=-1)
            log_det_J = -jnp.sum(log_s, axis=-1)  # Negative for inverse
        
        return y, log_det_J


class SimplePermutation(nn.Module):
    """Simple permutation layer - swaps the two dimensions."""
    
    def __call__(self, x: Array, reverse: bool = False) -> Tuple[Array, Array]:
        """Permute dimensions."""
        if x.shape[-1] != 2:
            raise ValueError("SimplePermutation only supports 2D inputs")
        
        # Swap dimensions: [x1, x2] -> [x2, x1]
        y = jnp.flip(x, axis=-1)
        log_det_J = jnp.zeros(x.shape[0])  # Permutation has det = ±1, log det = 0
        
        return y, log_det_J


class SimpleInvertibleNet(nn.Module):
    """
    Simple invertible network for 2D moment mapping.
    Alternates between coupling layers and permutations.
    """
    ef: ExponentialFamily
    config: SimpleINNConfig
    
    @nn.compact
    def __call__(self, eta: Array, reverse: bool = False) -> Tuple[Array, Array]:
        """
        Forward/reverse pass through the network.
        
        Args:
            eta: Input [batch_size, 2]
            reverse: If True, perform inverse transformation
            
        Returns:
            (output, log_det_jacobian)
        """
        if eta.shape[-1] != 2:
            raise ValueError("SimpleInvertibleNet only supports 2D inputs")
        
        x = eta
        log_det_total = jnp.zeros(x.shape[0])
        
        if not reverse:
            # Forward pass
            for i in range(self.config.num_layers):
                # Coupling layer
                x, log_det = SimpleCouplingLayer(
                    hidden_size=self.config.hidden_size,
                    activation=self.config.activation,
                    clamp_alpha=self.config.clamp_alpha
                )(x, reverse=False)
                log_det_total += log_det
                
                # Permutation (except last layer)
                if i < self.config.num_layers - 1:
                    x, log_det = SimplePermutation()(x, reverse=False)
                    log_det_total += log_det
        else:
            # Reverse pass (apply in reverse order)
            for i in reversed(range(self.config.num_layers)):
                # Permutation (except first layer in reverse)
                if i < self.config.num_layers - 1:
                    x, log_det = SimplePermutation()(x, reverse=True)
                    log_det_total += log_det
                
                # Coupling layer
                x, log_det = SimpleCouplingLayer(
                    hidden_size=self.config.hidden_size,
                    activation=self.config.activation,
                    clamp_alpha=self.config.clamp_alpha
                )(x, reverse=True)
                log_det_total += log_det
        
        return x, log_det_total
    
    def compute_loss(self, eta: Array, moments: Array) -> Dict[str, Array]:
        """Compute training loss."""
        # Forward pass: η → predicted moments
        pred_moments, log_det_J = self(eta, reverse=False)
        
        # Reconstruction loss
        mse_loss = jnp.mean(jnp.square(pred_moments - moments))
        
        # Regularization: encourage reasonable log determinants
        log_det_reg = jnp.mean(jnp.square(log_det_J))
        
        total_loss = mse_loss + 1e-3 * log_det_reg
        
        return {
            "total_loss": total_loss,
            "mse_loss": mse_loss,
            "log_det_reg": log_det_reg,
            "log_det_mean": jnp.mean(log_det_J),
            "mse": mse_loss,  # For compatibility
        }


def train_simple_inn(
    ef: ExponentialFamily,
    train_data: Dict[str, Array],
    val_data: Dict[str, Array],
    config: SimpleINNConfig,
    num_epochs: int,
    batch_size: int,
    seed: int,
) -> Tuple[train_state.TrainState, Dict[str, list]]:
    """Train the simple invertible network."""
    
    rng = random.PRNGKey(seed)
    
    # Create model
    model = SimpleInvertibleNet(ef=ef, config=config)
    
    # Initialize
    dummy_eta = jnp.zeros((1, ef.eta_dim))
    params = model.init(rng, dummy_eta, reverse=False)
    
    # Create train state
    tx = optax.adam(config.learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    
    num_train = train_data["eta"].shape[0]
    steps_per_epoch = max(1, (num_train + batch_size - 1) // batch_size)
    
    @jax.jit
    def train_step(state, batch_eta, batch_y):
        def loss_fn(params):
            loss_dict = model.apply(params, method=model.compute_loss, eta=batch_eta, moments=batch_y)
            return loss_dict["total_loss"], loss_dict
        
        (loss, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss_dict
    
    # Training history
    history = {
        "train_loss": [], "train_mse": [], "train_log_det": [],
        "val_loss": [], "val_mse": [], "val_log_det": []
    }
    
    indices = jnp.arange(num_train)
    
    for epoch in range(num_epochs):
        # Shuffle
        perm_key = random.fold_in(rng, epoch)
        perm = random.permutation(perm_key, indices)
        eta_shuf = train_data["eta"][perm]
        y_shuf = train_data["y"][perm]
        
        # Mini-batch training
        epoch_metrics = {"total_loss": [], "mse_loss": [], "log_det_reg": []}
        
        for step in range(steps_per_epoch):
            start = step * batch_size
            end = min(start + batch_size, num_train)
            batch_eta = eta_shuf[start:end]
            batch_y = y_shuf[start:end]
            
            state, step_metrics = train_step(state, batch_eta, batch_y)
            
            for key in epoch_metrics:
                epoch_metrics[key].append(float(step_metrics[key]))
        
        # Epoch averages
        train_loss = jnp.mean(jnp.array(epoch_metrics["total_loss"]))
        train_mse = jnp.mean(jnp.array(epoch_metrics["mse_loss"]))
        train_log_det = jnp.mean(jnp.array(epoch_metrics["log_det_reg"]))
        
        # Validation
        val_metrics = model.apply(state.params, method=model.compute_loss,
                                 eta=val_data["eta"], moments=val_data["y"])
        
        # Store history
        history["train_loss"].append(float(train_loss))
        history["train_mse"].append(float(train_mse))
        history["train_log_det"].append(float(train_log_det))
        history["val_loss"].append(float(val_metrics["total_loss"]))
        history["val_mse"].append(float(val_metrics["mse_loss"]))
        history["val_log_det"].append(float(val_metrics["log_det_reg"]))
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_metrics['total_loss']:.6f}")
    
    return state, history
