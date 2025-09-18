"""
Diffusion Model for Exponential Family Moment Mapping

This module implements a DDPM-style diffusion model for learning the mapping 
from natural parameters (eta) to expected sufficient statistics (moments).

The key idea is to treat moment prediction as a conditional denoising problem:
- Forward process: q(x_t | x_0) adds Gaussian noise to true moments
- Reverse process: p_θ(x_{t-1} | x_t, η) learns to denoise conditioned on natural parameters
"""

from __future__ import annotations

import math
from typing import Dict, Tuple, Any, Optional
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
class DiffusionConfig:
    """Configuration for diffusion model training."""
    # Model architecture
    hidden_sizes: Tuple[int, ...] = (128, 128, 64)
    activation: str = "swish"
    use_time_embedding: bool = True
    time_embed_dim: int = 64
    
    # Diffusion schedule
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule_type: str = "linear"  # "linear", "cosine"
    
    # Training
    learning_rate: float = 1e-3
    prediction_type: str = "epsilon"  # "epsilon", "x0", "v"
    loss_type: str = "mse"  # "mse", "l1"
    
    # Sampling
    num_inference_steps: int = 50
    eta_guidance_scale: float = 1.0


def get_beta_schedule(schedule_type: str, num_timesteps: int, 
                     beta_start: float, beta_end: float) -> Array:
    """Create noise schedule for diffusion process."""
    if schedule_type == "linear":
        return jnp.linspace(beta_start, beta_end, num_timesteps)
    elif schedule_type == "cosine":
        # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
        s = 0.008
        steps = jnp.arange(num_timesteps + 1)
        alphas_cumprod = jnp.cos(((steps / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return jnp.clip(betas, 0.0001, 0.9999)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps."""
    embed_dim: int = 64
    max_period: int = 10000
    
    @nn.compact
    def __call__(self, timesteps: Array) -> Array:
        """
        Args:
            timesteps: [batch_size] or [batch_size, 1]
        Returns:
            [batch_size, embed_dim] time embeddings
        """
        if timesteps.ndim > 1:
            timesteps = timesteps.squeeze(-1)
            
        half_dim = self.embed_dim // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        
        if self.embed_dim % 2 == 1:  # Handle odd embed_dim
            emb = jnp.concatenate([emb, jnp.zeros((emb.shape[0], 1))], axis=-1)
            
        return emb


class DiffusionUNet(nn.Module):
    """
    U-Net style architecture for conditional diffusion model.
    
    Takes noisy moments x_t, timestep t, and natural parameters η,
    and predicts the noise ε or denoised moments x_0.
    """
    config: DiffusionConfig
    output_dim: int
    
    def setup(self):
        self.time_embedding = TimeEmbedding(embed_dim=self.config.time_embed_dim)
        
        # Get activation function
        if self.config.activation == "swish":
            self.act = nn.swish
        elif self.config.activation == "relu":
            self.act = nn.relu
        elif self.config.activation == "gelu":
            self.act = nn.gelu
        else:
            raise ValueError(f"Unknown activation: {self.config.activation}")
    
    @nn.compact
    def __call__(self, x_t: Array, timesteps: Array, eta: Array, training: bool = True) -> Array:
        """
        Forward pass of diffusion U-Net.
        
        Args:
            x_t: Noisy moments [batch_size, moment_dim]
            timesteps: Diffusion timesteps [batch_size] or [batch_size, 1]
            eta: Natural parameters [batch_size, eta_dim]
            training: Whether in training mode
            
        Returns:
            Predicted noise or denoised moments [batch_size, moment_dim]
        """
        batch_size = x_t.shape[0]
        
        # Time embedding
        if self.config.use_time_embedding:
            time_emb = self.time_embedding(timesteps)  # [batch_size, time_embed_dim]
        else:
            if timesteps.ndim == 1:
                timesteps = timesteps[:, None]
            time_emb = timesteps.astype(jnp.float32)  # [batch_size, 1]
        
        # Concatenate inputs: [x_t, eta, time_emb]
        h = jnp.concatenate([x_t, eta, time_emb], axis=-1)
        
        # Encoder (downsampling path)
        encoder_features = []
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            h = nn.Dense(hidden_size, name=f"encoder_{i}")(h)
            h = nn.LayerNorm(name=f"encoder_ln_{i}")(h)
            h = self.act(h)
            
            # Skip dropout for simplicity
            # if training and i < len(self.config.hidden_sizes) - 1:
            #     h = nn.Dropout(rate=0.1, name=f"encoder_dropout_{i}")(h, deterministic=not training)
            
            encoder_features.append(h)
        
        # Bottleneck
        h = nn.Dense(self.config.hidden_sizes[-1] // 2, name="bottleneck")(h)
        h = self.act(h)
        
        # Decoder (upsampling path with skip connections)
        for i, hidden_size in enumerate(reversed(self.config.hidden_sizes[:-1])):
            # Skip connection from encoder
            skip = encoder_features[-(i+2)]  # Corresponding encoder layer
            h = jnp.concatenate([h, skip], axis=-1)
            
            h = nn.Dense(hidden_size, name=f"decoder_{i}")(h)
            h = nn.LayerNorm(name=f"decoder_ln_{i}")(h)
            h = self.act(h)
            
            # Skip dropout for simplicity
            # if training:
            #     h = nn.Dropout(rate=0.1, name=f"decoder_dropout_{i}")(h, deterministic=not training)
        
        # Output layer
        output = nn.Dense(self.output_dim, name="output")(h)
        
        return output


class DiffusionMomentNet(nn.Module):
    """
    Main diffusion model for moment mapping.
    """
    ef: ExponentialFamily
    config: DiffusionConfig
    
    def setup(self):
        self.unet = DiffusionUNet(
            config=self.config, 
            output_dim=self.ef.eta_dim  # Output same dimension as moments
        )
        
        # Precompute diffusion schedule
        betas = get_beta_schedule(
            self.config.schedule_type,
            self.config.num_timesteps,
            self.config.beta_start,
            self.config.beta_end
        )
        
        alphas = 1.0 - betas
        alphas_cumprod = jnp.cumprod(alphas)
        
        # Store as non-trainable parameters
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - alphas_cumprod)
    
    def init_for_training(self, x_t: Array, timesteps: Array, eta: Array, training: bool = True) -> Array:
        """Initialization method for training setup."""
        return self.unet(x_t, timesteps, eta, training)
    
    def q_sample(self, x_0: Array, t: Array, noise: Array) -> Array:
        """
        Forward diffusion process: q(x_t | x_0)
        
        Args:
            x_0: Clean moments [batch_size, moment_dim]
            t: Timesteps [batch_size]
            noise: Gaussian noise [batch_size, moment_dim]
            
        Returns:
            Noisy moments x_t
        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Broadcast to match x_0 shape
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None]
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_0: Array, eta: Array, t: Array, noise: Array) -> Dict[str, Array]:
        """
        Compute training losses for reverse process.
        
        Args:
            x_0: Clean moments [batch_size, moment_dim]
            eta: Natural parameters [batch_size, eta_dim]
            t: Timesteps [batch_size]
            noise: Gaussian noise [batch_size, moment_dim]
            
        Returns:
            Dictionary of losses
        """
        # Forward process: add noise
        x_t = self.q_sample(x_0, t, noise)
        
        # Reverse process: predict noise/clean image
        predicted = self.unet(x_t, t, eta, training=True)
        
        if self.config.prediction_type == "epsilon":
            target = noise
        elif self.config.prediction_type == "x0":
            target = x_0
        elif self.config.prediction_type == "v":
            # v-parameterization: v = α_t * ε - σ_t * x_0
            sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None]
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
            target = sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * x_0
        else:
            raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")
        
        if self.config.loss_type == "mse":
            loss = jnp.mean(jnp.square(predicted - target))
        elif self.config.loss_type == "l1":
            loss = jnp.mean(jnp.abs(predicted - target))
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
        return {
            "loss": loss,
            "mse": jnp.mean(jnp.square(predicted - target)),
            "mae": jnp.mean(jnp.abs(predicted - target)),
        }
    
    def __call__(self, eta: Array, num_inference_steps: Optional[int] = None, 
                 rng: Optional[Array] = None) -> Array:
        """
        Sample moments from natural parameters using DDPM sampling.
        
        Args:
            eta: Natural parameters [batch_size, eta_dim]
            num_inference_steps: Number of denoising steps
            rng: Random key for sampling
            
        Returns:
            Predicted moments [batch_size, moment_dim]
        """
        if num_inference_steps is None:
            num_inference_steps = self.config.num_inference_steps
            
        if rng is None:
            rng = random.PRNGKey(0)
        
        batch_size = eta.shape[0]
        moment_dim = self.ef.eta_dim
        
        # Start from pure noise
        x_t = random.normal(rng, (batch_size, moment_dim))
        
        # Create sampling schedule
        timesteps = jnp.linspace(self.config.num_timesteps - 1, 0, num_inference_steps).astype(jnp.int32)
        
        for i, t in enumerate(timesteps):
            t_batch = jnp.full((batch_size,), t)
            
            # Predict noise
            predicted_noise = self.unet(x_t, t_batch, eta, training=False)
            
            # DDPM sampling step
            if i < len(timesteps) - 1:  # Not the last step
                # Add noise for non-deterministic sampling
                rng, noise_rng = random.split(rng)
                noise = random.normal(noise_rng, x_t.shape)
                
                # Compute denoising step
                alpha_t = self.alphas[t]
                alpha_cumprod_t = self.alphas_cumprod[t]
                alpha_cumprod_prev = self.alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else 1.0
                
                # DDPM formula
                pred_x0 = (x_t - jnp.sqrt(1 - alpha_cumprod_t) * predicted_noise) / jnp.sqrt(alpha_cumprod_t)
                
                # Compute x_{t-1}
                pred_x0 = jnp.clip(pred_x0, -5.0, 5.0)  # Clip for stability
                
                direction_pointing_to_xt = jnp.sqrt(1 - alpha_cumprod_prev) * predicted_noise
                x_t = jnp.sqrt(alpha_cumprod_prev) * pred_x0 + direction_pointing_to_xt
                
                # Add noise
                if t > 0:
                    beta_t = self.betas[t]
                    sigma_t = jnp.sqrt(beta_t)
                    x_t = x_t + sigma_t * noise
            else:
                # Final step: deterministic
                alpha_cumprod_t = self.alphas_cumprod[t]
                x_t = (x_t - jnp.sqrt(1 - alpha_cumprod_t) * predicted_noise) / jnp.sqrt(alpha_cumprod_t)
        
        return x_t


class DiffusionTrainState(train_state.TrainState):
    """Extended train state for diffusion model."""
    rng: Array


def create_diffusion_train_state(rng: Array, model: DiffusionMomentNet, 
                                config: DiffusionConfig) -> DiffusionTrainState:
    """Create training state for diffusion model."""
    init_rng, train_rng = random.split(rng)
    
    # Initialize model parameters using the UNet directly
    dummy_eta = jnp.zeros((1, model.ef.eta_dim))
    dummy_moments = jnp.zeros((1, model.ef.eta_dim))
    dummy_t = jnp.zeros((1,), dtype=jnp.int32)
    
    # Initialize using the init_for_training method
    params = model.init(init_rng, method=model.init_for_training, x_t=dummy_moments, timesteps=dummy_t, eta=dummy_eta, training=False)
    
    # Create optimizer
    tx = optax.adam(config.learning_rate)
    
    return DiffusionTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        rng=train_rng
    )


def train_diffusion_moment_net(
    ef: ExponentialFamily,
    train_data: Dict[str, Array],
    val_data: Dict[str, Array],
    config: DiffusionConfig,
    num_epochs: int,
    batch_size: int,
    seed: int,
) -> Tuple[DiffusionTrainState, Dict[str, list]]:
    """
    Train the diffusion moment network.
    
    Args:
        ef: Exponential family distribution
        train_data: Training data dict with 'eta' and 'y' keys
        val_data: Validation data dict with 'eta' and 'y' keys
        config: Diffusion configuration
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        seed: Random seed
        
    Returns:
        Trained model state and training history
    """
    rng = random.PRNGKey(seed)
    
    # Create model and training state
    model = DiffusionMomentNet(ef=ef, config=config)
    state = create_diffusion_train_state(rng, model, config)
    
    num_train = train_data["eta"].shape[0]
    steps_per_epoch = max(1, (num_train + batch_size - 1) // batch_size)
    
    @jax.jit
    def train_step(state: DiffusionTrainState, batch_eta: Array, batch_y: Array) -> Tuple[DiffusionTrainState, Dict[str, Array]]:
        # Get fresh random keys
        step_rng, noise_rng, time_rng, new_rng = random.split(state.rng, 4)
        
        # Sample random timesteps
        batch_size = batch_eta.shape[0]
        t = random.randint(time_rng, (batch_size,), 0, config.num_timesteps)
        
        # Sample noise
        noise = random.normal(noise_rng, batch_y.shape)
        
        # Compute loss
        grad_fn = jax.value_and_grad(
            lambda p: model.apply(p, method=model.p_losses, x_0=batch_y, eta=batch_eta, t=t, noise=noise),
            has_aux=True
        )
        (loss_dict, grads) = grad_fn(state.params)
        
        # Update parameters
        new_state = state.apply_gradients(grads=grads)
        new_state = new_state.replace(rng=new_rng)
        
        return new_state, loss_dict
    
    # Training history
    history = {
        "train_loss": [], "train_mse": [], "train_mae": [],
        "val_loss": [], "val_mse": [], "val_mae": []
    }
    
    indices = jnp.arange(num_train)
    
    for epoch in range(num_epochs):
        # Shuffle training data
        perm_key = random.fold_in(state.rng, epoch)
        perm = random.permutation(perm_key, indices)
        eta_shuf = train_data["eta"][perm]
        y_shuf = train_data["y"][perm]
        
        # Mini-batch training
        epoch_metrics = {"loss": [], "mse": [], "mae": []}
        
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
        train_loss = jnp.mean(jnp.array(epoch_metrics["loss"]))
        train_mse = jnp.mean(jnp.array(epoch_metrics["mse"]))
        train_mae = jnp.mean(jnp.array(epoch_metrics["mae"]))
        
        # Validation metrics (simplified - just use a few timesteps)
        val_rng = random.fold_in(state.rng, epoch + 1000)
        val_noise_rng, val_time_rng = random.split(val_rng)
        
        val_t = random.randint(val_time_rng, (val_data["eta"].shape[0],), 0, config.num_timesteps)
        val_noise = random.normal(val_noise_rng, val_data["y"].shape)
        
        val_metrics = model.apply(
            state.params, method=model.p_losses,
            x_0=val_data["y"], eta=val_data["eta"], t=val_t, noise=val_noise
        )
        
        # Store history
        history["train_loss"].append(float(train_loss))
        history["train_mse"].append(float(train_mse))
        history["train_mae"].append(float(train_mae))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_mse"].append(float(val_metrics["mse"]))
        history["val_mae"].append(float(val_metrics["mae"]))
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_metrics['loss']:.6f}")
    
    return state, history
