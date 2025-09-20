"""
Invertible Neural Network (INN) for Exponential Family Moment Mapping

This module implements a GLOW-inspired invertible neural network for learning
the bijective mapping between natural parameters (eta) and expected sufficient statistics (moments).

Key components:
- Affine coupling layers for invertible transformations
- Invertible 1x1 convolutions (adapted for tabular data)
- Multi-scale architecture with squeeze/split operations
- Maximum likelihood training with change of variables
"""

from __future__ import annotations

import math
from typing import Dict, Tuple, Any, Optional, List
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random
import optax
from flax.training import train_state
from flax.core import FrozenDict

from src.ef import ExponentialFamily

Array = jax.Array


@dataclass
class INNConfig:
    """Configuration for Invertible Neural Network."""
    # Architecture
    num_flow_layers: int = 8
    hidden_sizes: Tuple[int, ...] = (64, 64, 32)
    activation: str = "relu"
    use_batch_norm: bool = False
    
    # Coupling layers
    coupling_type: str = "affine"  # "affine", "additive"
    clamp_alpha: float = 2.0  # Clamping for numerical stability
    
    # Invertible convolutions
    use_invertible_conv: bool = True
    conv_lu_decomposition: bool = True
    
    # Multi-scale
    num_scales: int = 1  # For tabular data, typically 1
    squeeze_factor: int = 2
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    
    # Loss components
    reconstruction_weight: float = 1.0
    regularization_weight: float = 1e-3


class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer as in GLOW.
    
    Splits input into two parts, uses one part to predict affine transformation
    parameters for the other part.
    """
    hidden_sizes: Tuple[int, ...] = (64, 64, 32)
    activation: str = "relu"
    clamp_alpha: float = 2.0
    mask: Array = None  # Binary mask to split channels
    
    def setup(self):
        if self.activation == "relu":
            self.act = nn.relu
        elif self.activation == "gelu":
            self.act = nn.gelu
        elif self.activation == "swish":
            self.act = nn.swish
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    @nn.compact
    def __call__(self, x: Array, reverse: bool = False) -> Tuple[Array, Array]:
        """
        Forward/reverse pass through affine coupling layer.
        
        Args:
            x: Input tensor [batch_size, dim]
            reverse: If True, perform inverse transformation
            
        Returns:
            (output, log_det_jacobian)
        """
        batch_size, dim = x.shape
        
        if self.mask is None:
            # Create alternating mask if not provided
            mask = jnp.arange(dim) % 2
            mask = mask.astype(jnp.float32)
        else:
            mask = self.mask
        
        # Split input
        x_masked = x * mask
        x_unmasked = x * (1 - mask)
        
        # Neural network to predict scale and translation
        h = x_masked
        for i, hidden_size in enumerate(self.hidden_sizes):
            h = nn.Dense(hidden_size, name=f"dense_{i}")(h)
            h = self.act(h)
        
        # Output layer: predict both scale (log_s) and translation (t)
        # Output dimension is 2 * number of unmasked channels
        n_unmasked = jnp.sum(1 - mask).astype(jnp.int32)
        st = nn.Dense(2 * n_unmasked, name="output")(h)
        
        # Extract scale and translation parameters
        log_s = st[..., :n_unmasked]
        t = st[..., n_unmasked:]
        
        # Clamp log_s for numerical stability
        log_s = jnp.tanh(log_s / self.clamp_alpha) * self.clamp_alpha
        
        # Apply affine transformation
        if not reverse:
            # Forward: y = x * exp(log_s) + t
            s = jnp.exp(log_s)
            y_unmasked = x_unmasked * s + t
            y = x_masked + y_unmasked * (1 - mask)
            
            # Log determinant of Jacobian
            log_det_J = jnp.sum(log_s * (1 - mask), axis=-1)
            
        else:
            # Reverse: x = (y - t) / exp(log_s)
            s = jnp.exp(log_s)
            x_unmasked_new = (x_unmasked - t) / s
            y = x_masked + x_unmasked_new * (1 - mask)
            
            # Log determinant of Jacobian (negative for inverse)
            log_det_J = -jnp.sum(log_s * (1 - mask), axis=-1)
        
        return y, log_det_J


class InvertibleLinear(nn.Module):
    """
    Invertible linear transformation using LU decomposition.
    
    Adapted from GLOW's invertible 1x1 convolutions for tabular data.
    """
    use_lu_decomposition: bool = True
    
    def setup(self):
        pass
    
    @nn.compact
    def __call__(self, x: Array, reverse: bool = False) -> Tuple[Array, Array]:
        """
        Forward/reverse pass through invertible linear layer.
        
        Args:
            x: Input tensor [batch_size, dim]
            reverse: If True, perform inverse transformation
            
        Returns:
            (output, log_det_jacobian)
        """
        batch_size, dim = x.shape
        
        if self.use_lu_decomposition:
            # LU decomposition parameterization for numerical stability
            
            # Lower triangular matrix (with 1s on diagonal)
            L_mask = jnp.tril(jnp.ones((dim, dim)), k=-1)
            L_params = self.param('L_params', nn.initializers.zeros, (dim, dim))
            L = L_mask * L_params + jnp.eye(dim)
            
            # Upper triangular matrix  
            U_mask = jnp.triu(jnp.ones((dim, dim)), k=1)
            U_params = self.param('U_params', nn.initializers.normal(stddev=0.1), (dim, dim))
            U_diag = self.param('U_diag', nn.initializers.normal(stddev=0.1), (dim,))
            U = U_mask * U_params + jnp.diag(U_diag)
            
            # Permutation matrix (fixed, learnable would be discrete)
            P = jnp.eye(dim)  # Identity for simplicity, could randomize
            
            # Full matrix: W = P @ L @ U
            W = P @ L @ U
            
            # Log determinant is sum of log diagonal elements of U
            log_det_W = jnp.sum(jnp.log(jnp.abs(U_diag)))
            
        else:
            # Direct parameterization (less stable but simpler)
            W = self.param('W', nn.initializers.orthogonal(), (dim, dim))
            log_det_W = jnp.linalg.slogdet(W)[1]
        
        if not reverse:
            # Forward: y = W @ x
            y = x @ W.T  # Transpose for batch-first format
            log_det_J = jnp.full((batch_size,), log_det_W)
        else:
            # Reverse: x = W^(-1) @ y
            W_inv = jnp.linalg.inv(W)
            y = x @ W_inv.T
            log_det_J = jnp.full((batch_size,), -log_det_W)
        
        return y, log_det_J


class ActNorm(nn.Module):
    """
    Activation normalization layer from GLOW.
    
    Performs data-dependent initialization to have zero mean and unit variance.
    """
    
    @nn.compact
    def __call__(self, x: Array, reverse: bool = False) -> Tuple[Array, Array]:
        """
        Forward/reverse pass through ActNorm layer.
        
        Args:
            x: Input tensor [batch_size, dim]
            reverse: If True, perform inverse transformation
            
        Returns:
            (output, log_det_jacobian)
        """
        batch_size, dim = x.shape
        
        # Learnable scale and bias parameters
        log_s = self.param('log_s', nn.initializers.zeros, (dim,))
        b = self.param('b', nn.initializers.zeros, (dim,))
        
        # Data-dependent initialization (would be done in practice)
        # For now, we'll rely on the initialization
        
        if not reverse:
            # Forward: y = s * x + b
            s = jnp.exp(log_s)
            y = s * x + b
            log_det_J = jnp.sum(log_s) * jnp.ones(batch_size)
        else:
            # Reverse: x = (y - b) / s
            s = jnp.exp(log_s)
            y = (x - b) / s
            log_det_J = -jnp.sum(log_s) * jnp.ones(batch_size)
        
        return y, log_det_J


class FlowBlock(nn.Module):
    """
    Single flow block consisting of:
    1. ActNorm
    2. Invertible linear transformation
    3. Affine coupling layer
    """
    config: INNConfig
    mask: Optional[Array] = None
    
    @nn.compact
    def __call__(self, x: Array, reverse: bool = False) -> Tuple[Array, Array]:
        """
        Forward/reverse pass through flow block.
        
        Args:
            x: Input tensor [batch_size, dim]
            reverse: If True, perform inverse transformation
            
        Returns:
            (output, log_det_jacobian)
        """
        log_det_total = jnp.zeros(x.shape[0])
        
        if not reverse:
            # Forward pass: ActNorm -> InvLinear -> Coupling
            
            # 1. ActNorm
            x, log_det = ActNorm()(x, reverse=False)
            log_det_total += log_det
            
            # 2. Invertible linear (if enabled)
            if self.config.use_invertible_conv:
                x, log_det = InvertibleLinear(
                    use_lu_decomposition=self.config.conv_lu_decomposition
                )(x, reverse=False)
                log_det_total += log_det
            
            # 3. Affine coupling
            x, log_det = AffineCouplingLayer(
                hidden_sizes=self.config.hidden_sizes,
                activation=self.config.activation,
                clamp_alpha=self.config.clamp_alpha,
                mask=self.mask
            )(x, reverse=False)
            log_det_total += log_det
            
        else:
            # Reverse pass: Coupling -> InvLinear -> ActNorm
            
            # 3. Affine coupling (reverse)
            x, log_det = AffineCouplingLayer(
                hidden_sizes=self.config.hidden_sizes,
                activation=self.config.activation,
                clamp_alpha=self.config.clamp_alpha,
                mask=self.mask
            )(x, reverse=True)
            log_det_total += log_det
            
            # 2. Invertible linear (reverse, if enabled)
            if self.config.use_invertible_conv:
                x, log_det = InvertibleLinear(
                    use_lu_decomposition=self.config.conv_lu_decomposition
                )(x, reverse=True)
                log_det_total += log_det
            
            # 1. ActNorm (reverse)
            x, log_det = ActNorm()(x, reverse=True)
            log_det_total += log_det
        
        return x, log_det_total


class InvertibleMomentNet(nn.Module):
    """
    Main invertible neural network for moment mapping.
    
    Learns bijective mapping between natural parameters η and moments μ.
    """
    ef: ExponentialFamily
    config: INNConfig
    
    def get_masks(self) -> List[Array]:
        """Get static masks for coupling layers."""
        dim = self.ef.eta_dim
        masks = []
        for i in range(self.config.num_flow_layers):
            # Alternate between different masking patterns
            if i % 2 == 0:
                mask = jnp.arange(dim) % 2
            else:
                mask = (jnp.arange(dim) + 1) % 2
            masks.append(mask.astype(jnp.float32))
        return masks
    
    @nn.compact
    def __call__(self, eta: Array, reverse: bool = False) -> Tuple[Array, Array]:
        """
        Forward/reverse pass through the invertible network.
        
        Args:
            eta: Natural parameters [batch_size, eta_dim] (forward) or
                 Moments [batch_size, moment_dim] (reverse)
            reverse: If True, map from moments to natural parameters
            
        Returns:
            (output, log_det_jacobian)
        """
        x = eta
        log_det_total = jnp.zeros(x.shape[0])
        
        # Get static masks
        masks = self.get_masks()
        
        if not reverse:
            # Forward: η → μ
            for i in range(self.config.num_flow_layers):
                x, log_det = FlowBlock(
                    config=self.config,
                    mask=masks[i]
                )(x, reverse=False)
                log_det_total += log_det
        else:
            # Reverse: μ → η (apply layers in reverse order)
            for i in reversed(range(self.config.num_flow_layers)):
                x, log_det = FlowBlock(
                    config=self.config,
                    mask=masks[i]
                )(x, reverse=True)
                log_det_total += log_det
        
        return x, log_det_total
    
    def log_likelihood(self, eta: Array, moments: Array) -> Dict[str, Array]:
        """
        Compute log-likelihood for training.
        
        Args:
            eta: Natural parameters [batch_size, eta_dim]
            moments: True moments [batch_size, moment_dim]
            
        Returns:
            Dictionary with loss components
        """
        # Forward pass: η → predicted moments
        pred_moments, log_det_J = self(eta, reverse=False)
        
        # Reconstruction loss (negative log-likelihood)
        reconstruction_loss = jnp.mean(jnp.square(pred_moments - moments))
        
        # Change of variables term
        # For maximum likelihood: log p(moments|eta) = log p(pred_moments) + log|det(J)|
        # We want to maximize this, so minimize the negative
        log_det_loss = -jnp.mean(log_det_J)
        
        # Total loss
        total_loss = (self.config.reconstruction_weight * reconstruction_loss + 
                     self.config.regularization_weight * log_det_loss)
        
        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "log_det_loss": log_det_loss,
            "log_det_mean": jnp.mean(log_det_J),
            "mse": reconstruction_loss,
        }
    
    def sample_moments(self, eta: Array) -> Array:
        """Sample moments given natural parameters."""
        moments, _ = self(eta, reverse=False)
        return moments
    
    def infer_eta(self, moments: Array) -> Array:
        """Infer natural parameters given moments."""
        eta, _ = self(moments, reverse=True)
        return eta


class INNTrainState(train_state.TrainState):
    """Extended train state for INN."""
    pass


def create_inn_train_state(rng: Array, model: InvertibleMomentNet, 
                          config: INNConfig) -> INNTrainState:
    """Create training state for invertible network."""
    init_rng, train_rng = random.split(rng)
    
    # Initialize model parameters
    dummy_eta = jnp.zeros((1, model.ef.eta_dim))
    params = model.init(init_rng, dummy_eta, reverse=False)
    
    # Create optimizer with gradient clipping
    tx = optax.chain(
        optax.clip_by_global_norm(config.gradient_clip_norm),
        optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
    )
    
    return INNTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


def train_inn_moment_net(
    ef: ExponentialFamily,
    train_data: Dict[str, Array],
    val_data: Dict[str, Array],
    config: INNConfig,
    num_epochs: int,
    batch_size: int,
    seed: int,
) -> Tuple[INNTrainState, Dict[str, list]]:
    """
    Train the invertible moment network.
    
    Args:
        ef: Exponential family distribution
        train_data: Training data dict with 'eta' and 'y' keys
        val_data: Validation data dict with 'eta' and 'y' keys
        config: INN configuration
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        seed: Random seed
        
    Returns:
        Trained model state and training history
    """
    rng = random.PRNGKey(seed)
    
    # Create model and training state
    model = InvertibleMomentNet(ef=ef, config=config)
    state = create_inn_train_state(rng, model, config)
    
    num_train = train_data["eta"].shape[0]
    steps_per_epoch = max(1, (num_train + batch_size - 1) // batch_size)
    
    @jax.jit
    def train_step(state: INNTrainState, batch_eta: Array, batch_y: Array) -> Tuple[INNTrainState, Dict[str, Array]]:
        # Compute loss and gradients
        grad_fn = jax.value_and_grad(
            lambda p: model.apply(p, method=model.log_likelihood, eta=batch_eta, moments=batch_y),
            has_aux=True
        )
        (loss_dict, grads) = grad_fn(state.params)
        
        # Update parameters
        new_state = state.apply_gradients(grads=grads)
        
        return new_state, loss_dict
    
    # Training history
    history = {
        "train_loss": [], "train_reconstruction": [], "train_log_det": [],
        "val_loss": [], "val_reconstruction": [], "val_log_det": []
    }
    
    indices = jnp.arange(num_train)
    
    for epoch in range(num_epochs):
        # Shuffle training data
        perm_key = random.fold_in(rng, epoch)
        perm = random.permutation(perm_key, indices)
        eta_shuf = train_data["eta"][perm]
        y_shuf = train_data["y"][perm]
        
        # Mini-batch training
        epoch_metrics = {"total_loss": [], "reconstruction_loss": [], "log_det_loss": []}
        
        for step in range(steps_per_epoch):
            start = step * batch_size
            end = min(start + batch_size, num_train)
            batch_eta = eta_shuf[start:end]
            batch_y = y_shuf[start:end]
            
            state, step_metrics = train_step(state, batch_eta, batch_y)
            
            # Accumulate metrics
            for key in epoch_metrics:
                epoch_metrics[key].append(float(step_metrics[key]))
        
        # Compute epoch averages
        train_loss = jnp.mean(jnp.array(epoch_metrics["total_loss"]))
        train_reconstruction = jnp.mean(jnp.array(epoch_metrics["reconstruction_loss"]))
        train_log_det = jnp.mean(jnp.array(epoch_metrics["log_det_loss"]))
        
        # Validation metrics
        val_metrics = model.apply(
            state.params, method=model.log_likelihood,
            eta=val_data["eta"], moments=val_data["y"]
        )
        
        # Store history
        history["train_loss"].append(float(train_loss))
        history["train_reconstruction"].append(float(train_reconstruction))
        history["train_log_det"].append(float(train_log_det))
        history["val_loss"].append(float(val_metrics["total_loss"]))
        history["val_reconstruction"].append(float(val_metrics["reconstruction_loss"]))
        history["val_log_det"].append(float(val_metrics["log_det_loss"]))
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_metrics['total_loss']:.6f}")
            print(f"  Reconstruction: {train_reconstruction:.6f}, Log Det: {train_log_det:.6f}")
    
    return state, history
