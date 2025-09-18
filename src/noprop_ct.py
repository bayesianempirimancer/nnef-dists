"""
Continuous-Time NoProp Implementation for Exponential Family Moment Mapping

This module implements a Neural ODE-based approach inspired by NoProp-CT for learning
the mapping from natural parameters (eta) to expected sufficient statistics.

The key idea is to model the network as a continuous-time dynamical system that
evolves from noisy inputs toward the correct moment predictions through a learned
vector field.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple, Any
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
class NoPropCTConfig:
    """Configuration for NoProp-CT training."""
    hidden_sizes: Tuple[int, ...] = (64, 64, 32)
    activation: str = "swish"
    noise_scale: float = 0.1
    time_horizon: float = 1.0
    num_time_steps: int = 10
    ode_solver: str = "euler"  # "euler", "rk4", "dopri5"
    learning_rate: float = 1e-3
    denoising_weight: float = 1.0
    consistency_weight: float = 0.1


class VectorField(nn.Module):
    """Neural network that defines the vector field for the ODE."""
    
    hidden_sizes: Tuple[int, ...] = (64, 64, 32)
    activation: str = "swish"
    output_dim: int = 1
    
    @nn.compact
    def __call__(self, state: Array, eta: Array, time: Array) -> Array:
        """
        Compute the vector field: dx/dt = f(x, eta, t)
        
        Args:
            state: Current network state [batch_size, output_dim]
            eta: Natural parameters [batch_size, eta_dim] 
            time: Current time [batch_size, 1] or scalar
            
        Returns:
            Time derivative of state [batch_size, output_dim]
        """
        # Get activation function
        if self.activation == "swish":
            act = nn.swish
        elif self.activation == "relu":
            act = nn.relu
        elif self.activation == "tanh":
            act = nn.tanh
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        
        # Ensure time has proper shape
        time = jnp.asarray(time)  # Convert to JAX array
        if time.ndim == 0:
            time = jnp.broadcast_to(time, (state.shape[0], 1))
        elif time.ndim == 1:
            time = time[:, None]
            
        # Concatenate state, eta, and time as input to vector field
        x = jnp.concatenate([state, eta, time], axis=-1)
        
        # Forward pass through the network
        for size in self.hidden_sizes:
            x = nn.Dense(size)(x)
            x = act(x)
            
        # Output the vector field
        dx_dt = nn.Dense(self.output_dim)(x)
        return dx_dt


class NeuralODESolver:
    """Simple ODE solvers for the neural ODE."""
    
    @staticmethod
    def euler_step(vector_field: Callable, state: Array, eta: Array, t: float, dt: float) -> Array:
        """Single Euler integration step."""
        dx_dt = vector_field(state, eta, t)
        return state + dt * dx_dt
    
    @staticmethod
    def rk4_step(vector_field: Callable, state: Array, eta: Array, t: float, dt: float) -> Array:
        """Single RK4 integration step."""
        k1 = vector_field(state, eta, t)
        k2 = vector_field(state + dt/2 * k1, eta, t + dt/2)
        k3 = vector_field(state + dt/2 * k2, eta, t + dt/2)
        k4 = vector_field(state + dt * k3, eta, t + dt)
        return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    @classmethod
    def integrate(cls, vector_field: Callable, initial_state: Array, eta: Array, 
                  time_span: Tuple[float, float], num_steps: int, method: str = "euler") -> Array:
        """
        Integrate the ODE from t0 to t1.
        
        Returns:
            Final state after integration
        """
        t0, t1 = time_span
        dt = (t1 - t0) / num_steps
        
        state = initial_state
        t = t0
        
        if method == "euler":
            step_fn = cls.euler_step
        elif method == "rk4":
            step_fn = cls.rk4_step
        else:
            raise ValueError(f"Unknown solver method: {method}")
        
        for _ in range(num_steps):
            state = step_fn(vector_field, state, eta, t, dt)
            t += dt
            
        return state


class NoPropCTMomentNet(nn.Module):
    """
    Continuous-Time NoProp neural network for moment mapping.
    
    This network learns to map natural parameters to expected sufficient statistics
    by modeling the process as a continuous-time denoising problem.
    """
    
    ef: ExponentialFamily
    config: NoPropCTConfig
    
    def setup(self):
        self.vector_field = VectorField(
            hidden_sizes=self.config.hidden_sizes,
            activation=self.config.activation,
            output_dim=self.ef.eta_dim  # Output dimension matches sufficient stats
        )
        
    def __call__(self, eta: Array, training: bool = True, 
                 rng: Array = None) -> Array:
        """
        Forward pass: evolve from noisy initial state to moment prediction.
        
        Args:
            eta: Natural parameters [batch_size, eta_dim]
            training: Whether in training mode
            rng: Random key for noise sampling
            
        Returns:
            Predicted moments [batch_size, eta_dim]
        """
        batch_size = eta.shape[0]
        
        if training and rng is not None:
            # Start from noisy initial condition
            noise = random.normal(rng, (batch_size, self.ef.eta_dim)) * self.config.noise_scale
            initial_state = eta + noise
        else:
            # For inference, start from the input directly or small noise
            initial_state = eta
        
        # Define the vector field function for this batch
        def vector_field_fn(state, eta_batch, t):
            return self.vector_field(state, eta_batch, t)
        
        # Integrate the ODE
        final_state = NeuralODESolver.integrate(
            vector_field_fn, 
            initial_state, 
            eta,
            time_span=(0.0, self.config.time_horizon),
            num_steps=self.config.num_time_steps,
            method=self.config.ode_solver
        )
        
        return final_state
    
    def compute_loss(self, params: Dict, eta: Array, target_moments: Array, 
                     rng: Array) -> Tuple[Array, Dict[str, Array]]:
        """
        Compute NoProp-CT training loss.
        
        The loss consists of:
        1. Denoising loss: prediction should match target moments
        2. Consistency loss: similar inputs should have similar trajectories
        """
        # Forward pass
        predicted_moments = self.apply(params, eta, training=True, rng=rng)
        
        # Denoising loss: final prediction should match target
        denoising_loss = jnp.mean(jnp.square(predicted_moments - target_moments))
        
        # Consistency loss: perturb inputs slightly and ensure similar outputs
        rng_pert, rng_main = random.split(rng)
        eta_perturbed = eta + random.normal(rng_pert, eta.shape) * 0.01
        predicted_perturbed = self.apply(params, eta_perturbed, training=True, rng=rng_main)
        consistency_loss = jnp.mean(jnp.square(predicted_moments - predicted_perturbed))
        
        # Total loss
        total_loss = (self.config.denoising_weight * denoising_loss + 
                     self.config.consistency_weight * consistency_loss)
        
        metrics = {
            "total_loss": total_loss,
            "denoising_loss": denoising_loss,
            "consistency_loss": consistency_loss,
            "mse": denoising_loss  # For compatibility with existing code
        }
        
        return total_loss, metrics


class NoPropCTTrainState(train_state.TrainState):
    """Extended train state for NoProp-CT."""
    rng: Array


def create_noprop_ct_train_state(rng: Array, model: NoPropCTMomentNet, 
                                 config: NoPropCTConfig) -> NoPropCTTrainState:
    """Create training state for NoProp-CT model."""
    init_rng, train_rng = random.split(rng)
    
    # Initialize model parameters
    dummy_eta = jnp.zeros((1, model.ef.eta_dim))
    params = model.init(init_rng, dummy_eta, training=False)
    
    # Create optimizer
    tx = optax.adam(config.learning_rate)
    
    return NoPropCTTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        rng=train_rng
    )


def train_noprop_ct_moment_net(
    ef: ExponentialFamily,
    train_data: Dict[str, Array],
    val_data: Dict[str, Array], 
    config: NoPropCTConfig,
    num_epochs: int,
    batch_size: int,
    seed: int,
) -> Tuple[NoPropCTTrainState, Dict[str, list]]:
    """
    Train the NoProp-CT moment network.
    
    Args:
        ef: Exponential family distribution
        train_data: Training data dict with 'eta' and 'y' keys
        val_data: Validation data dict with 'eta' and 'y' keys
        config: NoProp-CT configuration
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        seed: Random seed
        
    Returns:
        Trained model state and training history
    """
    rng = random.PRNGKey(seed)
    
    # Create model and training state
    model = NoPropCTMomentNet(ef=ef, config=config)
    state = create_noprop_ct_train_state(rng, model, config)
    
    num_train = train_data["eta"].shape[0]
    steps_per_epoch = max(1, (num_train + batch_size - 1) // batch_size)
    
    @jax.jit
    def train_step(state: NoPropCTTrainState, batch_eta: Array, batch_y: Array) -> Tuple[NoPropCTTrainState, Dict[str, Array]]:
        # Get fresh random key for this step
        step_rng, new_rng = random.split(state.rng)
        
        # Compute gradients
        grad_fn = jax.value_and_grad(
            lambda p: model.compute_loss(p, batch_eta, batch_y, step_rng), 
            has_aux=True
        )
        (loss, metrics), grads = grad_fn(state.params)
        
        # Update parameters
        new_state = state.apply_gradients(grads=grads)
        new_state = new_state.replace(rng=new_rng)
        
        return new_state, metrics
    
    # Training history
    history = {
        "train_loss": [], "train_denoising": [], "train_consistency": [],
        "val_loss": [], "val_denoising": [], "val_consistency": []
    }
    
    indices = jnp.arange(num_train)
    
    for epoch in range(num_epochs):
        # Shuffle training data
        perm_key = random.fold_in(state.rng, epoch)
        perm = random.permutation(perm_key, indices)
        eta_shuf = train_data["eta"][perm]
        y_shuf = train_data["y"][perm]
        
        # Mini-batch training
        epoch_metrics = {"total_loss": [], "denoising_loss": [], "consistency_loss": []}
        
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
        train_denoising = jnp.mean(jnp.array(epoch_metrics["denoising_loss"]))
        train_consistency = jnp.mean(jnp.array(epoch_metrics["consistency_loss"]))
        
        # Validation metrics
        val_rng = random.fold_in(state.rng, epoch + 1000)
        _, val_metrics = model.compute_loss(state.params, val_data["eta"], val_data["y"], val_rng)
        
        # Store history
        history["train_loss"].append(float(train_loss))
        history["train_denoising"].append(float(train_denoising))
        history["train_consistency"].append(float(train_consistency))
        history["val_loss"].append(float(val_metrics["total_loss"]))
        history["val_denoising"].append(float(val_metrics["denoising_loss"]))
        history["val_consistency"].append(float(val_metrics["consistency_loss"]))
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_metrics['total_loss']:.6f}")
    
    return state, history
