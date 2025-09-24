"""
True NoProp-CT ET implementation with diffusion-based training protocol.

This module implements a proper NoProp-CT (Non-propagating Continuous-Time) model
that follows the true noprop algorithm principles:
- Layer-wise training without backpropagation
- Diffusion-based training protocol
- Time-step specific denoising objectives
- No gradient flow between layers during training
"""

import jax
import jax.numpy as jnp
import jax.nn
import flax.linen as nn
from typing import Dict, Any, Tuple, Optional, Union, List
import optax
from jax import random
from tqdm import tqdm

from ..base_model import BaseNeuralNetwork
from ..config import FullConfig


class NoProp_CT_Block(nn.Module):
    """
    Individual block for NoProp-CT that learns to denoise noisy targets.
    
    Following the paper's approach, each block is trained independently to predict
    the clean target from a noisy target and the input. This implements the core
    NoProp algorithm where each block learns a local denoising process.
    
    Supports both standard MLP and ResNet-style architectures.
    """
    
    hidden_sizes: List[int]  # Multiple hidden layer sizes for expressive power
    output_dim: int
    noise_schedule: str = "linear"  # "linear", "cosine"
    use_resnet: bool = True  # Whether to use ResNet-style connections (default: True)
    resnet_skip_every: int = 2  # Skip connection every N layers
    use_feature_engineering: bool = True  # Whether to use eta feature engineering
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, time_embedding: jnp.ndarray, 
                 z: jnp.ndarray = None, previous_output: jnp.ndarray = None, 
                 training: bool = True) -> jnp.ndarray:
        """
        Forward pass through the NoProp-CT MLP.
        
        Args:
            eta: Natural parameters [batch_size, eta_dim]
            time_embedding: Time embedding [batch_size, time_embed_dim]
            z: Noisy target from diffusion schedule [batch_size, output_dim] (for training)
            previous_output: Previous time step output [batch_size, output_dim] (for inference)
            training: Whether in training mode
            
        Returns:
            Denoised output [batch_size, output_dim]
        """
        # Apply feature engineering to eta if enabled
        if self.use_feature_engineering:
            from ..eta_features import compute_eta_features
            eta_features = compute_eta_features(eta, method='noprop')
        else:
            eta_features = eta
        
        # During inference: concatenate [eta_features, previous_output, time_embedding]
        # During training: concatenate [eta_features, z, time_embedding] where z is noisy target
        if training and z is not None:
            # Training mode: use eta features, noisy target z, and time embedding
            x = jnp.concatenate([eta_features, z, time_embedding], axis=-1)
        elif previous_output is not None:
            # Inference mode: use previous output
            x = jnp.concatenate([eta_features, previous_output, time_embedding], axis=-1)
        else:
            # Fallback: use eta features and time embedding only
            x = jnp.concatenate([eta_features, time_embedding], axis=-1)
        
        # Denoising network - full MLP with multiple hidden layers
        net_out = x
        
        # Apply each hidden layer with optional ResNet connections
        for i, hidden_size in enumerate(self.hidden_sizes):
            # Store input for potential skip connection
            if self.use_resnet and i > 0 and i % self.resnet_skip_every == 0:
                skip_input = net_out
            
            # Apply dense layer
            net_out = nn.Dense(hidden_size, name=f'denoise_hidden_{i}')(net_out)
            net_out = nn.swish(net_out)
            
            # Apply ResNet skip connection if enabled
            if self.use_resnet and i > 0 and i % self.resnet_skip_every == 0:
                # Ensure dimensions match for skip connection
                if skip_input.shape[-1] == net_out.shape[-1]:
                    net_out = net_out + skip_input
                else:
                    # Project skip input to match current dimension
                    skip_proj = nn.Dense(hidden_size, name=f'skip_proj_{i}')(skip_input)
                    net_out = net_out + skip_proj
        
        # Final output layer
        net_out = nn.Dense(self.output_dim, name='denoise_output')(net_out)
        
        return net_out
    
    def _time_embedding(self, t: float, dim: int) -> jnp.ndarray:
        """Create time embedding for continuous time t ∈ [0, 1]."""
        # Simple sinusoidal time embedding for continuous time
        embed_dim = min(dim, 16)  # Limit embedding dimension
        freqs = jnp.linspace(0, 1, embed_dim // 2)
        time_embed = jnp.concatenate([
            jnp.sin(2 * jnp.pi * freqs * t),
            jnp.cos(2 * jnp.pi * freqs * t)
        ])
        return time_embed


class NoProp_CT_ET_Network(BaseNeuralNetwork):
    """
    True NoProp-CT-based ET Network with diffusion-based training.
    
    This network implements the proper noprop algorithm:
    - Single MLP that takes time embedding as input
    - Multiple hidden layers for sufficient expressive power
    - Uses diffusion-based denoising objectives
    - No backpropagation (each time step trained independently)
    """
    
    # Flax module fields
    num_time_steps: int = 10
    noise_schedule: str = "linear"  # "linear", "cosine", or "flow_matching"
    max_noise: float = 1.0
    time_embed_dim: int = 16
    flow_matching_sigma: float = 0.1  # Fixed noise for flow matching
    mlp_config: Dict = None
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through the NoProp-CT network.
        
        During inference, this applies the MLP sequentially across time steps.
        During training, the MLP is trained on individual time steps.
        
        Args:
            eta: Natural parameters [batch_size, input_dim]
            training: Whether in training mode
            
        Returns:
            Predicted expected statistics [batch_size, output_dim]
        """
        # Create the single MLP
        mlp = NoProp_CT_Block(**self.mlp_config)
        
        if training:
            # During training, we don't use this forward pass
            # The MLP is trained on individual time steps in the trainer
            return jnp.zeros((eta.shape[0], self.mlp_config['output_dim']))
        else:
            # Inference: deterministic sequential application
            # Start with eta as initial input
            z = eta  # Initialize with natural parameters
            
            # Apply MLP sequentially across time steps
            for i in range(self.num_time_steps):
                # Convert discrete step to continuous time t ∈ [0, 1]
                t = i / (self.num_time_steps - 1) if self.num_time_steps > 1 else 0.0
                
                # Create time embedding for continuous time t
                time_embed = self._time_embedding(t, self.time_embed_dim)
                time_embed_batch = jnp.broadcast_to(time_embed, (eta.shape[0], time_embed.shape[0]))
                
                # Apply MLP with current z as input
                # MLP input: [eta, z, time_embedding]
                z = mlp(eta, time_embed_batch, z=z, previous_output=None, training=False)
            
            return z
    
    def _time_embedding(self, t: float, dim: int) -> jnp.ndarray:
        """Create time embedding for continuous time t ∈ [0, 1]."""
        # Simple sinusoidal time embedding for continuous time
        embed_dim = min(dim, 16)  # Limit embedding dimension
        freqs = jnp.linspace(0, 1, embed_dim // 2)
        time_embed = jnp.concatenate([
            jnp.sin(2 * jnp.pi * freqs * t),
            jnp.cos(2 * jnp.pi * freqs * t)
        ])
        return time_embed
    
    def get_noise_schedule(self) -> jnp.ndarray:
        """Get noise schedule for diffusion process as continuous time function."""
        if self.noise_schedule == "noprop_ct":
            return jnp.linspace(0, self.max_noise, self.num_time_steps)
        elif self.noise_schedule == "flow_matching":
            # Flow matching uses fixed noise, return constant schedule
            return jnp.full(self.num_time_steps, self.flow_matching_sigma)
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}. Available: 'noprop_ct', 'flow_matching'")
    
    def get_gamma_at_time(self, t: float) -> float:
        """Get γ(t) at continuous time t ∈ [0, 1]."""
        if self.noise_schedule == "noprop_ct":
            # For NoProp-CT noise schedule, use linear γ(t) = t
            return t
        elif self.noise_schedule == "flow_matching":
            # For flow matching, γ(t) is not directly used in the same way
            # We'll return a constant value for consistency
            return 0.0
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}. Available: 'noprop_ct', 'flow_matching'")
    
    def get_noise_at_time(self, t: float) -> float:
        """Get noise level at continuous time t ∈ [0, 1] using ᾱ_t = σ(-γ(t))."""
        gamma_t = self.get_gamma_at_time(t)
        # Following the paper: ᾱ_t = σ(-γ(t))
        alpha_bar_t = jax.nn.sigmoid(-gamma_t)
        return alpha_bar_t
    
    def get_snr_at_time(self, t: float) -> float:
        """Get signal-to-noise ratio at continuous time t ∈ [0, 1]."""
        # Following the paper: SNR(t) = ᾱ_t/(1-ᾱ_t)
        alpha_bar_t = self.get_noise_at_time(t)
        return alpha_bar_t / (1 - alpha_bar_t)
    
    def get_snr_derivative_at_time(self, t: float) -> float:
        """Get derivative of SNR at continuous time t ∈ [0, 1]."""
        # d/dt SNR(t) = d/dt [ᾱ_t/(1-ᾱ_t)]
        # Using quotient rule: d/dt [f/g] = (f'g - fg')/g²
        # where f = ᾱ_t, g = 1-ᾱ_t
        # f' = d/dt ᾱ_t = d/dt σ(-γ(t)) = -γ'(t) * σ(-γ(t)) * (1-σ(-γ(t)))
        # g' = d/dt (1-ᾱ_t) = -f'
        
        gamma_t = self.get_gamma_at_time(t)
        gamma_prime_t = 1.0  # For linear γ(t) = t, we have γ'(t) = 1
        
        alpha_bar_t = self.get_noise_at_time(t)
        
        # d/dt ᾱ_t = -γ'(t) * ᾱ_t * (1-ᾱ_t)
        alpha_bar_prime = -gamma_prime_t * alpha_bar_t * (1 - alpha_bar_t)
        
        # d/dt SNR(t) = d/dt [ᾱ_t/(1-ᾱ_t)]
        # = [ᾱ_t' * (1-ᾱ_t) - ᾱ_t * (-ᾱ_t')] / (1-ᾱ_t)²
        # = [ᾱ_t' * (1-ᾱ_t) + ᾱ_t * ᾱ_t'] / (1-ᾱ_t)²
        # = ᾱ_t' / (1-ᾱ_t)²
        snr_derivative = alpha_bar_prime / ((1 - alpha_bar_t) ** 2)
        
        return snr_derivative
    
    def generate_flow_matching_sample(self, z_0: jnp.ndarray, z_1: jnp.ndarray, t: float) -> jnp.ndarray:
        """
        Generate sample z_t for flow matching: p(z(t)|z_0,z_1,x) = Normal(t*z_1 + (1-t)*z_0, sigma^2)
        
        Args:
            z_0: Initial state (noisy target)
            z_1: Final state (clean target)
            t: Time parameter ∈ [0, 1]
            
        Returns:
            z_t: Sample at time t
        """
        # Mean: t*z_1 + (1-t)*z_0
        mean = t * z_1 + (1 - t) * z_0
        
        # Fixed noise sigma^2
        noise = self.flow_matching_sigma * jnp.ones_like(mean)
        
        # Generate sample (in practice, this would be done during training)
        # For now, just return the mean (deterministic)
        return mean
    
    def add_noise(self, x: jnp.ndarray, noise: jnp.ndarray, t: float) -> jnp.ndarray:
        """Add noise to input at continuous time t ∈ [0, 1] using variance-preserving OU process."""
        alpha_bar_t = self.get_noise_at_time(t)
        
        # Variance-preserving OU process: z_t ~ N(√(ᾱ_t)z_T, 1-ᾱ_t)
        # where z_T is the clean target (x) and z_t is the noisy version
        sqrt_alpha_bar = jnp.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = jnp.sqrt(1 - alpha_bar_t)
        
        # z_t = √(ᾱ_t) * z_T + √(1-ᾱ_t) * noise
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
    
    def compute_internal_loss(self, params: Dict, eta: jnp.ndarray, 
                            predicted_stats: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """No internal loss for noprop-CT (losses are computed per layer)."""
        return 0.0


class NoProp_CT_ET_Trainer:
    """
    Trainer for NoProp-CT ET Network with proper noprop algorithm.
    
    This trainer implements the true noprop training protocol:
    - Single MLP trained on individual time steps
    - Diffusion-based denoising objectives
    - No backpropagation between time steps
    """
    
    def __init__(self, config: FullConfig, loss_type: str = "simple_target"):
        self.config = config
        self.loss_type = loss_type
        
        # Create MLP configuration
        mlp_config = {
            'hidden_sizes': config.network.hidden_sizes if config.network.hidden_sizes else [64, 64, 64],
            'output_dim': config.network.output_dim,
            'noise_schedule': getattr(config.network, 'noise_schedule', 'linear'),
            'use_resnet': getattr(config.network, 'use_resnet', True),  # Default to ResNet
            'resnet_skip_every': getattr(config.network, 'resnet_skip_every', 2)
        }
        
        # Initialize model with proper Flax module structure
        self.model = NoProp_CT_ET_Network(
            config=config,
            num_time_steps=getattr(config.network, 'num_time_steps', 10),
            noise_schedule=getattr(config.network, 'noise_schedule', 'linear'),
            max_noise=getattr(config.network, 'max_noise', 1.0),
            time_embed_dim=getattr(config.network, 'time_embed_dim', 16),
            mlp_config=mlp_config
        )
        self.rng = random.PRNGKey(config.experiment.random_seed)
        
        # Training configuration
        self.num_time_steps = getattr(config.network, 'num_time_steps', 10)
        self.learning_rate = config.training.learning_rate
        self.batch_size = config.training.batch_size
        self.time_embed_dim = getattr(config.network, 'time_embed_dim', 16)
        
        # Single MLP parameters and optimizer
        self.mlp_params = None
        self.mlp_opt_state = None
        self.mlp_optimizer = None
        
        # Don't create MLP instance here - create it fresh each time
        
    def initialize_mlp(self, sample_input: jnp.ndarray) -> Tuple[Dict, optax.OptState]:
        """Initialize the single MLP's parameters and optimizer."""
        self.rng, init_rng = random.split(self.rng)
        
        # Create sample time embedding
        sample_time_embed = jnp.ones((sample_input.shape[0], self.time_embed_dim))
        
        # Create dummy noisy target for initialization (same shape as target)
        sample_z = jnp.ones((sample_input.shape[0], self.model.mlp_config['output_dim']))
        
        # Apply feature engineering to get the correct input dimension
        use_feature_engineering = getattr(self.model.mlp_config, 'use_feature_engineering', True)
        if use_feature_engineering:
            from ..eta_features import compute_eta_features
            sample_eta_features = compute_eta_features(sample_input, method='default')
        else:
            sample_eta_features = sample_input
        
        # Create MLP with feature engineering setting
        mlp_config = self.model.mlp_config.copy()
        mlp_config['use_feature_engineering'] = use_feature_engineering
        mlp = NoProp_CT_Block(**mlp_config)
        params = mlp.init(init_rng, sample_input, sample_time_embed, z=sample_z)
        
        # Create optimizer for the MLP
        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(params)
        
        return params, opt_state
    
    def continuous_time_loss_fn(self, t: float, params: Dict, eta: jnp.ndarray, 
                               target_mu_T: jnp.ndarray, loss_strategy: str = "simple_target",
                               input_output_penalty_weight: float = 0.0) -> jnp.ndarray:
        """
        Compute loss for a specific continuous time t using different training strategies.
        
        Args:
            t: Continuous time t ∈ [0, 1] to train on
            params: Parameters for the MLP
            eta: Natural parameters
            target_mu_T: Target expected statistics (clean target)
            loss_strategy: Training strategy ("simple_target", "noprop", "flow_matching")
            input_output_penalty_weight: Weight for input-output difference penalty
            
        Returns:
            Loss value for this time point
        """
        
        # Create time embedding for continuous time t
        time_embed = self.model._time_embedding(t, self.time_embed_dim)
        time_embed_batch = jnp.broadcast_to(time_embed, (eta.shape[0], time_embed.shape[0]))
        
        # Generate noisy target z using continuous time noise schedule
        # z = sqrt(1 - noise_level) * target + sqrt(noise_level) * noise
        noise_level = self.model.get_noise_at_time(t)
        
        # Generate random noise with same shape as target_mu_T
        self.rng, noise_rng = random.split(self.rng)
        noise = random.normal(noise_rng, target_mu_T.shape)
        
        # Create noisy target z with same shape as target_mu_T
        sqrt_one_minus_noise = jnp.sqrt(1 - noise_level)
        sqrt_noise = jnp.sqrt(noise_level)
        z = sqrt_one_minus_noise * target_mu_T + sqrt_noise * noise
        
        # Get MLP prediction with noisy target z as input
        mlp_config = self.model.mlp_config.copy()
        mlp_config['use_feature_engineering'] = getattr(self.model.mlp_config, 'use_feature_engineering', True)
        mlp = NoProp_CT_Block(**mlp_config)
        predicted_output = mlp.apply(params, eta, time_embed_batch, z=z, 
                                   previous_output=None, training=True)
        
        # Compute base loss
        if loss_strategy == "noprop":
            base_loss = self._noprop_loss(t, predicted_output, target_mu_T)
        elif loss_strategy == "simple_target":
            base_loss = self._simple_target_loss(predicted_output, target_mu_T)
        elif loss_strategy == "flow_matching":
            # For flow matching, we need to sample z_0 and z_t properly
            self.rng, flow_rng = random.split(self.rng)
            z_0, z_t = self.sample_flow_matching_data(target_mu_T, t, flow_rng)
            base_loss = self._flow_matching_loss(t, predicted_output, z_0, target_mu_T, flow_rng)
        else:
            raise ValueError(f"Unknown loss strategy: {loss_strategy}. Available: 'noprop', 'simple_target', 'flow_matching'")
        
        # Add input-output difference penalty if specified
        if input_output_penalty_weight > 0:
            # For ResNet-style training, penalize large differences between input and output
            # This encourages smooth transitions between time steps
            input_output_diff = jnp.mean((predicted_output - eta) ** 2)
            base_loss += input_output_penalty_weight * input_output_diff
        
        return base_loss
    
    def _noprop_loss(self, t: float, predicted_output: jnp.ndarray, 
                    target_mu_T: jnp.ndarray) -> jnp.ndarray:
        """
        Standard NoProp loss: weighted by derivative of signal-to-noise ratio.
        
        Following the NoProp paper, this weights the loss by the derivative
        of the SNR at time t: weight = d/dt SNR(t)
        """
        # Get derivative of SNR at time t
        snr_derivative = self.model.get_snr_derivative_at_time(t)
        
        # Use absolute value of derivative as weight (since derivative can be negative)
        variance_weight = jnp.abs(snr_derivative)
        
        # Base MSE loss
        mse_loss = jnp.mean((predicted_output - target_mu_T) ** 2)
        
        # Apply variance weighting
        weighted_loss = variance_weight * mse_loss
        
        # Add regularization
        reg_loss = 1e-4 * jnp.mean(predicted_output ** 2)
        
        return weighted_loss + reg_loss
    
    def _flow_matching_loss(self, t: float, predicted_output: jnp.ndarray, 
                           z_0: jnp.ndarray, z_1: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        """
        Flow matching loss: |NN(z_t, x, t) - (z_1 - z_0)|^2
        
        Args:
            t: Time parameter ∈ [0, 1]
            predicted_output: Network output
            z_0: Initial state sampled from N(0, I)
            z_1: Final state (clean target)
            rng: Random key for sampling
            
        Returns:
            Flow matching loss
        """
        # Target is the difference z_1 - z_0
        target = z_1 - z_0
        
        # MSE loss between network output and target
        mse_loss = jnp.mean((predicted_output - target) ** 2)
        
        # Add regularization
        reg_loss = 1e-4 * jnp.mean(predicted_output ** 2)
        
        return mse_loss + reg_loss
    
    def sample_flow_matching_data(self, z_1: jnp.ndarray, t: float, rng: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Sample flow matching data: z_0 ~ N(0, I), z_t ~ N(t*z_1 + (1-t)*z_0, σ²)
        
        Args:
            z_1: Final state (clean target)
            t: Time parameter ∈ [0, 1]
            rng: Random key for sampling
            
        Returns:
            z_0: Initial state sampled from N(0, I)
            z_t: Intermediate state sampled from N(t*z_1 + (1-t)*z_0, σ²)
        """
        # Sample z_0 from N(0, I)
        rng, z_0_rng = jax.random.split(rng)
        z_0 = jax.random.normal(z_0_rng, z_1.shape)
        
        # Sample z_t from N(t*z_1 + (1-t)*z_0, σ²)
        mean = t * z_1 + (1 - t) * z_0
        rng, z_t_rng = jax.random.split(rng)
        noise = jax.random.normal(z_t_rng, z_1.shape)
        z_t = mean + self.model.flow_matching_sigma * noise
        
        return z_0, z_t
    
    def _simple_target_loss(self, predicted_output: jnp.ndarray, target_mu_T: jnp.ndarray) -> jnp.ndarray:
        """
        Simple target loss: direct MSE to final target.
        
        This is the simplest approach where each time step is trained
        to predict the final clean target directly.
        """
        # Direct MSE loss to final target
        mse_loss = jnp.mean((predicted_output - target_mu_T) ** 2)
        
        # Add regularization
        reg_loss = 1e-4 * jnp.mean(predicted_output ** 2)
        
        return mse_loss + reg_loss
    
    
    def train_continuous_time(self, t: float, batch: Dict[str, jnp.ndarray], 
                             loss_strategy: str = None,
                             input_output_penalty_weight: float = 0.0) -> Tuple[Dict, Any, float]:
        """
        Single training step for a specific continuous time t.
        
        This is the core of the noprop algorithm - the MLP is trained
        on individual time points without gradients flowing between time points.
        
        Args:
            t: Continuous time t ∈ [0, 1] to train on
            batch: Training batch data
            loss_strategy: Loss strategy to use ("noprop", "simple_target", "flow_matching")
            input_output_penalty_weight: Weight for input-output difference penalty
        """
        # Use loss_type if loss_strategy is not specified
        if loss_strategy is None:
            loss_strategy = self.loss_type
        
        # Compute loss and gradients for this time point only
        loss_value, grads = jax.value_and_grad(self.continuous_time_loss_fn, argnums=1)(
            t, self.mlp_params, batch['eta'], batch['mu_T'], 
            loss_strategy, input_output_penalty_weight
        )
        
        # Clip gradients to prevent instability
        grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        
        # Update MLP parameters
        updates, new_opt_state = self.mlp_optimizer.update(grads, self.mlp_opt_state, self.mlp_params)
        new_params = optax.apply_updates(self.mlp_params, updates)
        
        # Store updated parameters and optimizer state
        self.mlp_params = new_params
        self.mlp_opt_state = new_opt_state
        
        return new_params, new_opt_state, float(loss_value)
    
    def train_epoch(self, train_data: Dict[str, jnp.ndarray], 
                   loss_strategy: str = "noprop",
                   input_output_penalty_weight: float = 0.0) -> List[float]:
        """
        Train the MLP for one epoch using noprop protocol.
        
        The MLP is trained on individual time steps independently.
        For each batch, we randomly sample a single time step to train on.
        
        Args:
            train_data: Training data
            loss_strategy: Loss strategy to use ("noprop", "simple_target", "flow_matching")
            input_output_penalty_weight: Weight for input-output difference penalty
        """
        n_train = train_data['eta'].shape[0]
        batch_size = self.batch_size
        
        # Shuffle data
        self.rng, shuffle_rng = random.split(self.rng)
        perm = random.permutation(shuffle_rng, n_train)
        train_data_shuffled = {k: v[perm] for k, v in train_data.items()}
        
        epoch_losses = []
        
        # Train each batch
        for i in range(0, n_train, batch_size):
            batch_data = {k: v[i:i+batch_size] for k, v in train_data_shuffled.items()}
            
            # Randomly sample continuous time t ~ Uniform(0,1) for this batch
            self.rng, time_rng = random.split(self.rng)
            t = random.uniform(time_rng, (), minval=0.0, maxval=1.0)
            
            # Train on the sampled continuous time
            _, _, batch_loss = self.train_continuous_time(
                t, batch_data, loss_strategy, input_output_penalty_weight
            )
            epoch_losses.append(batch_loss)
        
        return epoch_losses
    
    def train(self, train_data: Dict[str, jnp.ndarray], 
              val_data: Optional[Dict[str, jnp.ndarray]] = None,
              epochs: int = 300, loss_strategy: str = "noprop",
              input_output_penalty_weight: float = 0.0) -> Tuple[Dict, Dict]:
        """
        Full training loop using noprop protocol.
        
        The MLP is trained on individual time steps without backpropagation.
        
        Args:
            train_data: Training data
            val_data: Validation data
            epochs: Number of training epochs
            loss_strategy: Loss strategy ("noprop", "simple_target", "flow_matching")
            input_output_penalty_weight: Weight for input-output difference penalty
        """
        # Initialize the MLP
        sample_eta = train_data['eta'][:1]
        self.mlp_params, self.mlp_opt_state = self.initialize_mlp(sample_eta)
        self.mlp_optimizer = optax.adam(self.learning_rate)
        
        print(f"Training NoProp-CT ET with {self.num_time_steps} time steps for inference")
        print(f"Single MLP trained on continuous time t ~ Uniform(0,1) (no backpropagation)")
        print(f"MLP Architecture: {self.config.network.hidden_sizes}")
        print(f"Use ResNet: {self.model.mlp_config['use_resnet']}")
        print(f"Loss Strategy: {loss_strategy}")
        if input_output_penalty_weight > 0:
            print(f"Input-Output Penalty Weight: {input_output_penalty_weight}")
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        best_params = self.mlp_params
        
        pbar = tqdm(range(epochs), desc="Training NoProp ET")
        for epoch in pbar:
            # Train the MLP for one epoch
            epoch_losses = self.train_epoch(train_data, loss_strategy, input_output_penalty_weight)
            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            history['train_loss'].append(avg_train_loss)
            
            # Validation loss (using full forward pass)
            if val_data is not None:
                val_loss = self.evaluate(val_data)
                history['val_loss'].append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = self.mlp_params
                
                # Update progress bar with loss information
                pbar.set_postfix({
                    'train_loss': f'{avg_train_loss:.6f}',
                    'val_loss': f'{val_loss:.6f}'
                })
            else:
                history['val_loss'].append(avg_train_loss)
                
                # Update progress bar with training loss only
                pbar.set_postfix({
                    'train_loss': f'{avg_train_loss:.6f}'
                })
        
        # Return best parameters
        self.mlp_params = best_params
        return self.mlp_params, history
    
    def evaluate(self, val_data: Dict[str, jnp.ndarray]) -> float:
        """Evaluate the model using full forward pass."""
        # Use the MLP directly for inference
        eta = val_data['eta']
        z = jnp.zeros((eta.shape[0], self.model.mlp_config['output_dim']))  # Start with zeros
        
        # Apply MLP sequentially across time steps
        for t in range(self.num_time_steps):
            # Create time embedding for this time step
            time_embed = self.model._time_embedding(t, self.time_embed_dim)
            time_embed_batch = jnp.broadcast_to(time_embed, (eta.shape[0], time_embed.shape[0]))
            
            # Apply MLP with current z as input
            mlp_config = self.model.mlp_config.copy()
            mlp_config['use_feature_engineering'] = getattr(self.model.mlp_config, 'use_feature_engineering', True)
            mlp = NoProp_CT_Block(**mlp_config)
            z = mlp.apply(self.mlp_params, eta, time_embed_batch, z=z, 
                         previous_output=None, training=True)
        
        # Compute MSE loss
        mse_loss = jnp.mean((z - val_data['mu_T']) ** 2)
        return float(mse_loss)
    
    def predict(self, eta: jnp.ndarray) -> jnp.ndarray:
        """Make predictions using the trained model."""
        # Use the MLP directly for inference
        z = jnp.zeros((eta.shape[0], self.model.mlp_config['output_dim']))  # Start with zeros
        
        # Apply MLP sequentially across time steps
        for t in range(self.num_time_steps):
            # Create time embedding for this time step
            time_embed = self.model._time_embedding(t, self.time_embed_dim)
            time_embed_batch = jnp.broadcast_to(time_embed, (eta.shape[0], time_embed.shape[0]))
            
            # Apply MLP with current z as input
            mlp_config = self.model.mlp_config.copy()
            mlp_config['use_feature_engineering'] = getattr(self.model.mlp_config, 'use_feature_engineering', True)
            mlp = NoProp_CT_Block(**mlp_config)
            z = mlp.apply(self.mlp_params, eta, time_embed_batch, z=z, 
                         previous_output=None, training=True)
        
        return z


def create_model_and_trainer(config: FullConfig, loss_type: str = "simple_target") -> NoProp_CT_ET_Trainer:
    """Factory function to create NoProp-CT ET model and trainer."""
    return NoProp_CT_ET_Trainer(config, loss_type)