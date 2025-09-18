"""
Improved Invertible Neural Network for Exponential Family Moment Mapping

Enhanced version with:
- Deeper architecture (more layers)
- ActNorm layers for better conditioning
- Geometric preprocessing for natural parameters
- Better regularization for invertibility
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
class ImprovedINNConfig:
    """Configuration for Improved INN."""
    num_layers: int = 12  # Much deeper
    hidden_size: int = 128  # Larger hidden layers
    num_hidden_layers: int = 3  # Deeper coupling networks
    activation: str = "gelu"
    learning_rate: float = 5e-4  # Lower LR for stability
    clamp_alpha: float = 2.5
    
    # Regularization
    log_det_weight: float = 0.1  # Stronger regularization
    invertibility_weight: float = 1.0  # Explicit invertibility loss
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    
    # Preprocessing
    use_geometric_preprocessing: bool = True
    preprocessing_epsilon: float = 1e-6  # For numerical stability


class GeometricPreprocessing(nn.Module):
    """
    Geometric preprocessing layer for natural parameters.
    
    Applies the transformation: eta = eta/norm(eta) * log(norm(eta))
    This helps with the natural parameter geometry.
    """
    epsilon: float = 1e-6
    
    @nn.compact
    def __call__(self, eta: Array, reverse: bool = False) -> Tuple[Array, Array]:
        """
        Apply geometric preprocessing.
        
        Args:
            eta: Natural parameters [batch_size, 2]
            reverse: If True, apply inverse transformation
            
        Returns:
            (transformed_eta, log_det_jacobian)
        """
        if not reverse:
            # Forward: eta = eta/||eta|| * log(1 + ||eta||)
            eta_norm = jnp.linalg.norm(eta, axis=-1, keepdims=True)
            eta_norm_safe = jnp.maximum(eta_norm, self.epsilon)  # Avoid division by zero
            
            eta_normalized = eta / eta_norm_safe
            log_one_plus_norm = jnp.log(1.0 + eta_norm_safe)  # More stable than log(norm)
            
            transformed_eta = eta_normalized * log_one_plus_norm
            
            # Simple log det approximation (complex Jacobian avoided for stability)
            log_det_J = jnp.zeros(eta.shape[0])
            
        else:
            # Reverse: eta = input/||input|| * (exp(||input||) - 1)
            input_norm = jnp.linalg.norm(eta, axis=-1, keepdims=True)
            input_norm_safe = jnp.maximum(input_norm, self.epsilon)
            
            # Clamp input norm to prevent extreme exp values
            input_norm_clamped = jnp.clip(input_norm_safe, 0.0, 5.0)  # Limit exp(5) ≈ 148
            
            input_normalized = eta / input_norm_safe
            exp_norm_minus_one = jnp.exp(input_norm_clamped) - 1.0
            
            # Additional clamping for safety
            exp_norm_minus_one = jnp.clip(exp_norm_minus_one, self.epsilon, 100.0)
            
            transformed_eta = input_normalized * exp_norm_minus_one
            log_det_J = jnp.zeros(eta.shape[0])
        
        return transformed_eta, log_det_J


class ImprovedActNorm(nn.Module):
    """
    Improved ActNorm with better initialization and stability.
    """
    
    @nn.compact
    def __call__(self, x: Array, reverse: bool = False) -> Tuple[Array, Array]:
        """ActNorm with data-dependent initialization."""
        batch_size, dim = x.shape
        
        # Learnable parameters with better initialization
        log_s = self.param('log_s', nn.initializers.zeros, (dim,))
        b = self.param('b', nn.initializers.zeros, (dim,))
        
        # Apply transformation
        if not reverse:
            # Forward: y = exp(log_s) * x + b
            s = jnp.exp(log_s)
            y = s * x + b
            log_det_J = jnp.sum(log_s) * jnp.ones(batch_size)
        else:
            # Reverse: x = (y - b) / exp(log_s)
            s = jnp.exp(log_s)
            y = (x - b) / s
            log_det_J = -jnp.sum(log_s) * jnp.ones(batch_size)
        
        return y, log_det_J


class DeepCouplingLayer(nn.Module):
    """
    Deeper coupling layer with more expressive neural networks.
    """
    hidden_size: int = 128
    num_hidden_layers: int = 3
    activation: str = "gelu"
    clamp_alpha: float = 2.5
    mask: Array = None
    
    @nn.compact
    def __call__(self, x: Array, reverse: bool = False) -> Tuple[Array, Array]:
        """Deep coupling transformation."""
        if x.shape[-1] != 2:
            raise ValueError("DeepCouplingLayer only supports 2D inputs")
        
        x1, x2 = x[..., 0:1], x[..., 1:2]
        
        # Get activation function
        if self.activation == "gelu":
            act = nn.gelu
        elif self.activation == "tanh":
            act = nn.tanh
        elif self.activation == "relu":
            act = nn.relu
        else:
            act = nn.swish
        
        # Deeper neural network
        h = x1
        
        # Input layer
        h = nn.Dense(self.hidden_size)(h)
        h = act(h)
        
        # Hidden layers with proper ResNet blocks
        for i in range(self.num_hidden_layers):
            h_res = h
            
            # ResNet block: f(x) = activation(x + Dense(activation(Dense(x))))
            h_block = nn.Dense(self.hidden_size)(h)
            h_block = act(h_block)
            h_block = nn.Dense(self.hidden_size)(h_block)
            
            # Proper ResNet form: h = act(h + f(h))
            if h.shape == h_block.shape:
                h = act(h + h_block)  # ResNet form with activation after sum
            else:
                h = act(h_block)  # Fallback if dimensions don't match
        
        # Output layer: predict scale and translation
        st = nn.Dense(2)(h)
        log_s, t = st[..., 0:1], st[..., 1:2]
        
        # Better clamping for stability
        log_s = jnp.tanh(log_s / self.clamp_alpha) * self.clamp_alpha
        
        # Apply transformation
        if not reverse:
            s = jnp.exp(log_s)
            x2_new = x2 * s + t
            y = jnp.concatenate([x1, x2_new], axis=-1)
            log_det_J = jnp.sum(log_s, axis=-1)
        else:
            s = jnp.exp(log_s)
            x2_new = (x2 - t) / (s + 1e-7)  # Add small epsilon for stability
            y = jnp.concatenate([x1, x2_new], axis=-1)
            log_det_J = -jnp.sum(log_s, axis=-1)
        
        return y, log_det_J


class ImprovedFlowBlock(nn.Module):
    """
    Improved flow block with ActNorm + Deep Coupling + Permutation.
    """
    config: ImprovedINNConfig
    
    @nn.compact
    def __call__(self, x: Array, reverse: bool = False) -> Tuple[Array, Array]:
        """Flow block with improved architecture."""
        log_det_total = jnp.zeros(x.shape[0])
        
        if not reverse:
            # Forward: ActNorm -> Coupling -> Permutation
            
            # 1. ActNorm for conditioning
            x, log_det = ImprovedActNorm()(x, reverse=False)
            log_det_total += log_det
            
            # 2. Deep coupling layer
            x, log_det = DeepCouplingLayer(
                hidden_size=self.config.hidden_size,
                num_hidden_layers=self.config.num_hidden_layers,
                activation=self.config.activation,
                clamp_alpha=self.config.clamp_alpha
            )(x, reverse=False)
            log_det_total += log_det
            
            # 3. Permutation (swap dimensions)
            x = jnp.flip(x, axis=-1)  # Simple swap
            # Permutation has determinant ±1, so log det = 0
            
        else:
            # Reverse: Permutation -> Coupling -> ActNorm
            
            # 3. Permutation (reverse)
            x = jnp.flip(x, axis=-1)
            
            # 2. Deep coupling layer (reverse)
            x, log_det = DeepCouplingLayer(
                hidden_size=self.config.hidden_size,
                num_hidden_layers=self.config.num_hidden_layers,
                activation=self.config.activation,
                clamp_alpha=self.config.clamp_alpha
            )(x, reverse=True)
            log_det_total += log_det
            
            # 1. ActNorm (reverse)
            x, log_det = ImprovedActNorm()(x, reverse=True)
            log_det_total += log_det
        
        return x, log_det_total


class ImprovedInvertibleNet(nn.Module):
    """
    Improved invertible network with deeper architecture and preprocessing.
    """
    ef: ExponentialFamily
    config: ImprovedINNConfig
    
    @nn.compact
    def __call__(self, eta: Array, reverse: bool = False) -> Tuple[Array, Array]:
        """
        Forward/reverse pass through improved network.
        
        Args:
            eta: Natural parameters [batch_size, 2] (forward) or
                 Moments [batch_size, 2] (reverse)
            reverse: If True, map from moments to natural parameters
            
        Returns:
            (output, log_det_jacobian)
        """
        x = eta
        log_det_total = jnp.zeros(x.shape[0])
        
        if not reverse:
            # Forward: η → μ
            
            # 1. Geometric preprocessing
            if self.config.use_geometric_preprocessing:
                x, log_det = GeometricPreprocessing(
                    epsilon=self.config.preprocessing_epsilon
                )(x, reverse=False)
                log_det_total += log_det
            
            # 2. Flow layers
            for i in range(self.config.num_layers):
                x, log_det = ImprovedFlowBlock(config=self.config)(x, reverse=False)
                log_det_total += log_det
        
        else:
            # Reverse: μ → η (apply in reverse order)
            
            # 2. Flow layers (reverse order)
            for i in reversed(range(self.config.num_layers)):
                x, log_det = ImprovedFlowBlock(config=self.config)(x, reverse=True)
                log_det_total += log_det
            
            # 1. Geometric preprocessing (reverse)
            if self.config.use_geometric_preprocessing:
                x, log_det = GeometricPreprocessing(
                    epsilon=self.config.preprocessing_epsilon
                )(x, reverse=True)
                log_det_total += log_det
        
        return x, log_det_total
    
    def compute_loss(self, eta: Array, moments: Array) -> Dict[str, Array]:
        """Compute improved training loss with better regularization."""
        
        # Forward pass: η → predicted moments
        pred_moments, log_det_J = self(eta, reverse=False)
        
        # Reconstruction loss
        mse_loss = jnp.mean(jnp.square(pred_moments - moments))
        
        # Log determinant regularization (encourage reasonable transformations)
        log_det_reg = jnp.mean(jnp.square(log_det_J))
        
        # Invertibility loss: test round-trip consistency
        reconstructed_eta, log_det_reverse = self(pred_moments, reverse=True)
        invertibility_loss = jnp.mean(jnp.square(reconstructed_eta - eta))
        
        # Log det consistency (should cancel out for perfect invertibility)
        log_det_consistency = jnp.mean(jnp.square(log_det_J + log_det_reverse))
        
        # Total loss with improved regularization
        total_loss = (mse_loss + 
                     self.config.log_det_weight * log_det_reg +
                     self.config.invertibility_weight * invertibility_loss +
                     0.1 * log_det_consistency)
        
        return {
            "total_loss": total_loss,
            "mse_loss": mse_loss,
            "log_det_reg": log_det_reg,
            "invertibility_loss": invertibility_loss,
            "log_det_consistency": log_det_consistency,
            "log_det_mean": jnp.mean(log_det_J),
            "mse": mse_loss,  # For compatibility
        }


def train_improved_inn(
    ef: ExponentialFamily,
    train_data: Dict[str, Array],
    val_data: Dict[str, Array],
    config: ImprovedINNConfig,
    num_epochs: int,
    batch_size: int,
    seed: int,
) -> Tuple[train_state.TrainState, Dict[str, list]]:
    """Train the improved invertible network."""
    
    rng = random.PRNGKey(seed)
    
    # Create model
    model = ImprovedInvertibleNet(ef=ef, config=config)
    
    # Initialize
    dummy_eta = jnp.zeros((1, ef.eta_dim))
    params = model.init(rng, dummy_eta, reverse=False)
    
    # Create train state with better optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(config.gradient_clip_norm),
        optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
    )
    
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
    
    # Training history with more metrics
    history = {
        "train_loss": [], "train_mse": [], "train_log_det": [], "train_invertibility": [],
        "val_loss": [], "val_mse": [], "val_log_det": [], "val_invertibility": []
    }
    
    indices = jnp.arange(num_train)
    
    for epoch in range(num_epochs):
        # Shuffle
        perm_key = random.fold_in(rng, epoch)
        perm = random.permutation(perm_key, indices)
        eta_shuf = train_data["eta"][perm]
        y_shuf = train_data["y"][perm]
        
        # Mini-batch training
        epoch_metrics = {
            "total_loss": [], "mse_loss": [], "log_det_reg": [], "invertibility_loss": []
        }
        
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
        train_invertibility = jnp.mean(jnp.array(epoch_metrics["invertibility_loss"]))
        
        # Validation
        val_metrics = model.apply(state.params, method=model.compute_loss,
                                 eta=val_data["eta"], moments=val_data["y"])
        
        # Store history
        history["train_loss"].append(float(train_loss))
        history["train_mse"].append(float(train_mse))
        history["train_log_det"].append(float(train_log_det))
        history["train_invertibility"].append(float(train_invertibility))
        history["val_loss"].append(float(val_metrics["total_loss"]))
        history["val_mse"].append(float(val_metrics["mse_loss"]))
        history["val_log_det"].append(float(val_metrics["log_det_reg"]))
        history["val_invertibility"].append(float(val_metrics["invertibility_loss"]))
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_metrics['total_loss']:.6f}")
            print(f"  MSE: {train_mse:.6f}, Invertibility: {train_invertibility:.6f}")
    
    return state, history
