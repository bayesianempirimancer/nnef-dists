"""
Deep Flow Network for Natural Parameter to Statistics Mapping

This implements a discrete flow network with N=20 layers that combines:
1. Flow structure similar to NoProp-CT but with discrete layers
2. Diffusion-based training protocol 
3. Adaptive quadratic ResNet-style error prediction layers
4. ResNet form: x = x + layer(x)

The network learns to predict statistics by flowing through discrete time steps,
with each layer predicting the "error" or residual to add to the current state.
"""

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
import optax
from typing import Tuple, Dict, Any, Optional
import numpy as np


class AdaptiveQuadraticErrorLayer(nn.Module):
    """
    Adaptive quadratic ResNet layer for error prediction in flow network.
    
    Implements: error = α*(Wx) + β*((B*x)*x)
    where α and β are learnable mixing coefficients.
    """
    
    hidden_size: int
    activation: str = "tanh"
    
    def setup(self):
        if self.activation == "relu":
            self.act = nn.relu
        elif self.activation == "tanh":
            self.act = nn.tanh
        elif self.activation == "gelu":
            self.act = nn.gelu
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Linear term: Wx
        linear_term = nn.Dense(self.hidden_size, name='linear')(x)
        
        # Quadratic term: (B*x)*x
        quad_weights = nn.Dense(self.hidden_size, use_bias=False, name='quad')(x)
        quad_term = quad_weights * x
        
        # Learnable mixing coefficients
        alpha = self.param('alpha', nn.initializers.ones, (self.hidden_size,))
        beta = self.param('beta', nn.initializers.zeros, (self.hidden_size,))
        
        # Adaptive combination
        error = alpha * linear_term + beta * quad_term
        error = self.act(error)
        
        return error


class DeepFlowLayer(nn.Module):
    """
    Single layer of the deep flow network.
    
    Each layer predicts an error/residual to add to the current state.
    Uses adaptive quadratic ResNet for error prediction.
    """
    
    hidden_size: int = 256
    output_dim: int = 12
    activation: str = "tanh"
    dropout_rate: float = 0.1
    
    def setup(self):
        # Error prediction network
        self.error_net = AdaptiveQuadraticErrorLayer(
            hidden_size=self.hidden_size,
            activation=self.activation
        )
        
        # Output projection
        self.output_proj = nn.Dense(self.output_dim)
        
        # Optional dropout for regularization
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(rate=self.dropout_rate)
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Predict error/residual
        error = self.error_net(x)
        
        # Apply dropout if enabled (disabled for now to avoid RNG issues)
        # if self.dropout_rate > 0 and training:
        #     error = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(error, rng=self.make_rng('dropout'))
        
        # Project to output dimension
        error = self.output_proj(error)
        
        return error


class DeepFlowNetwork(nn.Module):
    """
    Deep Flow Network with N=20 discrete layers.
    
    Each layer predicts an error to add to the current state, following
    the flow: x_{t+1} = x_t + error_t(x_t)
    
    Uses diffusion-based training where we add noise at different time steps
    and train the network to predict the noise/error.
    """
    
    num_layers: int = 20
    hidden_size: int = 256
    output_dim: int = 12
    activation: str = "tanh"
    dropout_rate: float = 0.1
    use_feature_engineering: bool = True
    
    def setup(self):
        # Input feature engineering (same as standard MLP)
        if self.use_feature_engineering:
            self.input_proj = nn.Dense(self.hidden_size)
        
        # Flow layers
        self.flow_layers = [DeepFlowLayer(
            hidden_size=self.hidden_size,
            output_dim=self.output_dim,
            activation=self.activation,
            dropout_rate=self.dropout_rate
        ) for _ in range(self.num_layers)]
        
        # Optional output normalization
        self.output_norm = nn.LayerNorm()
    
    def nat_features(self, eta: jnp.ndarray) -> jnp.ndarray:
        """
        Natural parameter feature engineering (same as standard MLP).
        """
        # Original eta parameters
        features = [eta]
        
        # 1. clip(1/eta) - inverse with aggressive clipping
        eta_inv = jnp.clip(1.0/eta, -1000.0, 1000.0)
        features.append(eta_inv)
        
        # 2. eta/norm(eta) - normalized eta (unit vector)
        eta_norm = jnp.linalg.norm(eta, axis=-1, keepdims=True)
        features.append(eta/eta_norm)
        
        # 3. norm(eta) * (1/eta)
        features.append(eta_norm * eta_inv)
        
        # 4. norm(eta)
        features.append(eta_norm)
        
        # 5. log(norm(eta))
        features.append(jnp.log(eta_norm))
        
        # Concatenate all features
        result = jnp.concatenate(features, axis=-1)
        
        # Add absolute values
        result = jnp.concatenate([result, jnp.abs(result)], axis=-1)
        
        # Safety checks
        result = jnp.where(jnp.isfinite(result), result, 0.0)
        result = jnp.clip(result, -1e6, 1e6)
        
        return result
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through the flow network.
        
        Args:
            eta: Natural parameters (batch_size, input_dim)
            training: Whether in training mode
            
        Returns:
            Predicted statistics (batch_size, output_dim)
        """
        # Apply feature engineering if enabled
        if self.use_feature_engineering:
            x = self.nat_features(eta)
            x = self.input_proj(x)
        else:
            x = eta
        
        # Initialize state (could be zero or learnable)
        state = jnp.zeros((x.shape[0], self.output_dim))
        
        # Flow through discrete layers
        for i, layer in enumerate(self.flow_layers):
            # Predict error/residual for this layer
            error = layer(x, training=training)
            
            # ResNet update: state = state + error
            state = state + error
        
        # Optional output normalization
        if self.output_norm is not None:
            state = self.output_norm(state)
        
        return state
    
    def forward_with_intermediates(self, eta: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Forward pass that returns intermediate states for analysis.
        
        Returns:
            Final output and dictionary of intermediate states
        """
        # Apply feature engineering if enabled
        if self.use_feature_engineering:
            x = self.nat_features(eta)
            x = self.input_proj(x)
        else:
            x = eta
        
        # Initialize state
        state = jnp.zeros((x.shape[0], self.output_dim))
        intermediates = {"initial": state}
        
        # Flow through discrete layers
        for i, layer in enumerate(self.flow_layers):
            # Predict error/residual for this layer
            error = layer(x, training=training)
            
            # ResNet update: state = state + error
            state = state + error
            intermediates[f"layer_{i}"] = state
            intermediates[f"error_{i}"] = error
        
        # Optional output normalization
        if self.output_norm is not None:
            state = self.output_norm(state)
        
        return state, intermediates


class DiffusionNoiseScheduler:
    """
    Noise scheduler for diffusion-based training of the flow network.
    
    Adds noise at different time steps and trains the network to predict
    the noise/error that would recover the clean statistics.
    """
    
    def __init__(self, num_timesteps: int = 100, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Linear beta schedule
        self.betas = jnp.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
    
    def add_noise(self, clean_stats: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray) -> jnp.ndarray:
        """
        Add noise to clean statistics at given timesteps.
        
        Args:
            clean_stats: Clean target statistics
            noise: Random noise to add
            timesteps: Timestep indices (batch_size,)
            
        Returns:
            Noisy statistics
        """
        # Get alpha values for the timesteps
        alphas_cumprod_t = self.alphas_cumprod[timesteps]
        
        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = jnp.sqrt(alphas_cumprod_t)[:, None]
        sqrt_one_minus_alphas_cumprod_t = jnp.sqrt(1.0 - alphas_cumprod_t)[:, None]
        
        # Add noise: sqrt(alpha_cumprod) * clean + sqrt(1 - alpha_cumprod) * noise
        noisy_stats = sqrt_alphas_cumprod_t * clean_stats + sqrt_one_minus_alphas_cumprod_t * noise
        
        return noisy_stats
    
    def get_target(self, clean_stats: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray) -> jnp.ndarray:
        """
        Get the target for the network to predict (the noise).
        
        In standard diffusion, we predict the noise. In our flow network,
        we predict the error/residual that would recover the clean statistics.
        """
        # For flow networks, we want to predict the error that would
        # transform the noisy statistics back to clean statistics
        noisy_stats = self.add_noise(clean_stats, noise, timesteps)
        
        # The target is the difference (error) between clean and noisy
        target = clean_stats - noisy_stats
        
        return target, noisy_stats


def flow_loss_fn(model: DeepFlowNetwork, params: dict, eta: jnp.ndarray, 
                 clean_stats: jnp.ndarray, noise_scheduler: DiffusionNoiseScheduler,
                 rng: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute loss for the flow network using diffusion-based training.
    
    Args:
        model: DeepFlowNetwork model
        params: Model parameters
        eta: Natural parameters
        clean_stats: Clean target statistics
        noise_scheduler: Noise scheduler
        rng: Random key
        
    Returns:
        Loss and auxiliary information
    """
    batch_size = eta.shape[0]
    
    # Sample random timesteps
    rng, timestep_rng = jax.random.split(rng)
    timesteps = jax.random.randint(
        timestep_rng, (batch_size,), 0, noise_scheduler.num_timesteps
    )
    
    # Generate random noise
    rng, noise_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, clean_stats.shape)
    
    # Get target (error to predict) and noisy input
    target, noisy_stats = noise_scheduler.get_target(clean_stats, noise, timesteps)
    
    # Forward pass through the network
    # We pass the noisy statistics as the "current state" that the network should correct
    predicted_error = model.apply(params, eta, training=True)
    
    # Compute MSE loss between predicted and target error
    mse_loss = jnp.mean(jnp.square(predicted_error - target))
    
    # Optional: Add consistency loss (encourage smooth flow)
    # This helps ensure the flow is well-behaved
    consistency_loss = jnp.mean(jnp.square(predicted_error))
    
    # Total loss
    total_loss = mse_loss + 0.01 * consistency_loss
    
    aux_info = {
        "mse_loss": mse_loss,
        "consistency_loss": consistency_loss,
        "target_norm": jnp.mean(jnp.linalg.norm(target, axis=-1)),
        "pred_norm": jnp.mean(jnp.linalg.norm(predicted_error, axis=-1)),
        "timesteps_mean": jnp.mean(timesteps)
    }
    
    return total_loss, aux_info


def create_flow_train_state(model: DeepFlowNetwork, rng: jnp.ndarray, 
                           eta_sample: jnp.ndarray, learning_rate: float = 1e-3) -> Tuple[dict, optax.OptState]:
    """
    Create training state for the flow network.
    
    Args:
        model: DeepFlowNetwork model
        rng: Random key for initialization
        eta_sample: Sample input for shape inference
        learning_rate: Learning rate for optimizer
        
    Returns:
        Initial parameters and optimizer state
    """
    # Initialize model parameters
    params = model.init(rng, eta_sample)
    
    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate)  # Use regular adam instead of adamw
    )
    
    opt_state = optimizer.init(params)
    
    return params, optimizer, opt_state


def train_flow_network(model: DeepFlowNetwork, params: dict, optimizer: optax.GradientTransformation,
                      opt_state: optax.OptState, train_data: Dict[str, jnp.ndarray],
                      val_data: Dict[str, jnp.ndarray], config: Dict[str, Any]) -> Tuple[dict, Dict[str, list]]:
    """
    Train the deep flow network.
    
    Args:
        model: DeepFlowNetwork model
        params: Initial parameters
        optimizer: Optimizer
        opt_state: Initial optimizer state
        train_data: Training data
        val_data: Validation data
        config: Training configuration
        
    Returns:
        Trained parameters and training history
    """
    num_epochs = config.get('num_epochs', 100)
    batch_size = config.get('batch_size', 32)
    noise_scheduler = DiffusionNoiseScheduler(
        num_timesteps=config.get('num_timesteps', 100),
        beta_start=config.get('beta_start', 0.0001),
        beta_end=config.get('beta_end', 0.02)
    )
    
    train_losses = []
    val_losses = []
    best_params = params
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.get('patience', 20)
    
    @jax.jit
    def train_step(params, opt_state, eta_batch, y_batch, rng):
        loss, aux = flow_loss_fn(model, params, eta_batch, y_batch, noise_scheduler, rng)
        grads = jax.grad(lambda p: flow_loss_fn(model, p, eta_batch, y_batch, noise_scheduler, rng)[0])(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux
    
    rng = random.PRNGKey(42)
    
    for epoch in range(num_epochs):
        # Training with mini-batches
        n_train = train_data['eta'].shape[0]
        indices = jnp.arange(n_train)
        rng, epoch_rng = jax.random.split(rng)
        indices = jax.random.permutation(epoch_rng, indices)
        
        epoch_train_loss = 0.0
        num_batches = 0
        
        for i in range(0, n_train, batch_size):
            batch_indices = indices[i:i+batch_size]
            eta_batch = train_data['eta'][batch_indices]
            y_batch = train_data['y'][batch_indices]
            
            rng, batch_rng = jax.random.split(rng)
            params, opt_state, loss, aux = train_step(params, opt_state, eta_batch, y_batch, batch_rng)
            
            if not jnp.isfinite(loss):
                print(f"    Non-finite loss at epoch {epoch}, batch {i//batch_size}")
                break
                
            epoch_train_loss += float(loss)
            num_batches += 1
        
        if num_batches == 0:
            break
            
        epoch_train_loss /= num_batches
        
        # Validation
        val_pred = model.apply(params, val_data['eta'], training=False)
        val_loss = float(jnp.mean(jnp.square(val_pred - val_data['y'])))
        
        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        
        # Best model tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d}: Train={epoch_train_loss:.2f}, Val={val_loss:.2f}, Best={best_val_loss:.2f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break
    
    return best_params, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }
