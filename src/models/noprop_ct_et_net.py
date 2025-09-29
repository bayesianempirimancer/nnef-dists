"""
NoProp CT ET Network implementation compatible with Hugging Face.

This module provides a Hugging Face compatible NoProp-CT (Non-propagating Continuous-Time) 
ET model that uses diffusion-based training protocols.
"""

from typing import Optional, List
import jax
import jax.numpy as jnp
import flax.linen as nn
from ..configs.noprop_ct_et_config import NoProp_CT_ET_Config
from ..embeddings.eta_embedding import EtaEmbedding
from ..utils.activation_utils import get_activation_function


class NoProp_CT_Block(nn.Module):
    """
    Individual block for NoProp-CT that learns to denoise noisy targets.
    
    Following the paper's approach, each block is trained independently to predict
    the clean target from a noisy target and the input. This implements the core
    NoProp algorithm where each block learns a local denoising process.
    """
    
    hidden_sizes: List[int]  # Multiple hidden layer sizes for expressive power
    output_dim: int
    noise_schedule: str = "linear"  # "linear", "cosine"
    use_resnet: bool = True  # Whether to use ResNet-style connections (default: True)
    resnet_skip_every: int = 1  # Skip connection every N layers
    use_feature_engineering: bool = True  # Whether to use eta feature engineering
    embedding_type: Optional[str] = None  # Eta embedding type
    
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
        # Apply eta embedding if specified
        if hasattr(self, 'embedding_type') and self.embedding_type is not None:
            eta_embedding = EtaEmbedding(
                embedding_type=self.embedding_type,
                eta_dim=eta.shape[-1]
            )
            eta_features = eta_embedding(eta)
        elif self.use_feature_engineering:
            # Fallback to old feature engineering method
            from embeddings.eta_features import compute_eta_features
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
            net_out = get_activation_function(self.config.activation)(net_out)
            
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


class NoProp_CT_ET_Network(nn.Module):
    """
    True NoProp-CT-based ET Network with diffusion-based training.
    
    This network implements the proper noprop algorithm:
    - Single MLP that takes time embedding as input
    - Multiple hidden layers for sufficient expressive power
    - Uses diffusion-based denoising objectives
    - No backpropagation (each time step trained independently)
    """
    config: NoProp_CT_ET_Config

    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True, **kwargs) -> jnp.ndarray:
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
        # Create MLP configuration
        mlp_config = {
            'hidden_sizes': self.config.hidden_sizes if self.config.hidden_sizes else [64, 64, 64],
            'output_dim': self.config.output_dim,
            'noise_schedule': self.config.noise_schedule,
            'use_resnet': self.config.use_resnet,
            'resnet_skip_every': self.config.resnet_skip_every,
            'use_feature_engineering': self.config.use_feature_engineering,
            'embedding_type': getattr(self.config, 'embedding_type', None)
        }
        
        # Create the single MLP
        mlp = NoProp_CT_Block(**mlp_config)
        
        if training:
            # During training, we don't use this forward pass
            # The MLP is trained on individual time steps in the trainer
            return jnp.zeros((eta.shape[0], self.config.output_dim))
        else:
            # Inference: deterministic sequential application
            # Start with eta as initial input
            z = eta  # Initialize with natural parameters
            
            # Apply MLP sequentially across time steps
            for i in range(self.config.num_time_steps):
                # Convert discrete step to continuous time t ∈ [0, 1]
                t = i / (self.config.num_time_steps - 1) if self.config.num_time_steps > 1 else 0.0
                
                # Create time embedding for continuous time t
                embed_dim = min(self.config.time_embed_dim, 16)  # Limit embedding dimension
                freqs = jnp.linspace(0, 1, embed_dim // 2)
                time_embed = jnp.concatenate([
                    jnp.sin(2 * jnp.pi * freqs * t),
                    jnp.cos(2 * jnp.pi * freqs * t)
                ])
                time_embed_batch = jnp.broadcast_to(time_embed, (eta.shape[0], time_embed.shape[0]))
                
                # Apply MLP with current z as input
                # MLP input: [eta, z, time_embedding]
                z = mlp(eta, time_embed_batch, z=z, 
                       previous_output=None, training=False)
            
            return z
    
    def get_noise_schedule(self) -> jnp.ndarray:
        """Get noise schedule for diffusion process as continuous time function."""
        if self.config.noise_schedule == "noprop_ct":
            return jnp.linspace(0, self.config.max_noise, self.config.num_time_steps)
        elif self.config.noise_schedule == "flow_matching":
            # Flow matching uses fixed noise, return constant schedule
            return jnp.full(self.config.num_time_steps, self.config.flow_matching_sigma)
        else:
            raise ValueError(f"Unknown noise schedule: {self.config.noise_schedule}. Available: 'noprop_ct', 'flow_matching'")
    
    def get_gamma_at_time(self, t: float) -> float:
        """Get γ(t) at continuous time t ∈ [0, 1]."""
        if self.config.noise_schedule == "noprop_ct":
            # For NoProp-CT noise schedule, use linear γ(t) = t
            return t
        elif self.config.noise_schedule == "flow_matching":
            # For flow matching, γ(t) is not directly used in the same way
            # We'll return a constant value for consistency
            return 0.0
        else:
            raise ValueError(f"Unknown noise schedule: {self.config.noise_schedule}. Available: 'noprop_ct', 'flow_matching'")
    
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
        noise = self.config.flow_matching_sigma * jnp.ones_like(mean)
        
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
    
    def compute_internal_loss(self, params: dict, eta: jnp.ndarray, 
                            predicted_stats: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """No internal loss for noprop-CT (losses are computed per layer)."""
        return 0.0

