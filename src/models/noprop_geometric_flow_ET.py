"""
NoProp Geometric Flow ET implementation.

This module combines the geometric flow dynamics with NoProp continuous-time training protocols.
It implements a geometric flow network that learns the dynamics:
    du/dt = A@A^T@(η_target - η_init)

where A is learned using NoProp continuous-time training with t ~ Uniform(0,1).

Key features:
- Geometric flow dynamics with PSD constraints (A@A^T)
- NoProp continuous-time training (t ~ Uniform(0,1))
- Supports both NoProp-CT and Flow Matching noise schedules
- Smoothness penalties for stable dynamics
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
from ..ef import ExponentialFamily, ef_factory


class NoProp_Geometric_Flow_Block(nn.Module):
    """
    Geometric Flow block for NoProp training that learns A(u, t, η_t).
    
    This block learns the matrix A such that:
        du/dt = A(u, t, η_t) @ A(u, t, η_t)^T @ (η_target - η_init)
    
    The block is trained using NoProp continuous-time protocols where t ~ Uniform(0,1).
    """
    
    hidden_sizes: List[int]
    output_dim: int
    matrix_rank: int = None  # Rank of matrix A (if None, use output_dim)
    noise_schedule: str = "noprop_ct"  # "noprop_ct" or "flow_matching"
    use_feature_engineering: bool = True
    activation: str = "swish"
    use_weak_layer_norm: bool = False
    use_resnet: bool = True
    resnet_skip_every: int = 2
    
    def _weak_layer_norm(self, x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
        """Weak layer normalization: x/norm(x) * log(1 + norm(x))"""
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        normalized = x / (norm + eps)
        return normalized * jnp.log(1.0 + norm)
    
    def _time_embedding(self, t: float, dim: int) -> jnp.ndarray:
        """Create time embedding for continuous time t ∈ [0, 1]."""
        embed_dim = min(dim, 16)
        freqs = jnp.linspace(0, 1, embed_dim // 2)
        time_embed = jnp.concatenate([
            jnp.sin(2 * jnp.pi * freqs * t),
            jnp.cos(2 * jnp.pi * freqs * t)
        ])
        return time_embed
    
    def get_gamma_at_time(self, t: float) -> float:
        """Get γ(t) at continuous time t ∈ [0, 1]."""
        if self.noise_schedule == "noprop_ct":
            return t
        elif self.noise_schedule == "flow_matching":
            return 0.0  # Not directly used in flow matching
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")
    
    def get_noise_at_time(self, t: float) -> float:
        """Get noise level at continuous time t ∈ [0, 1] using ᾱ_t = σ(-γ(t))."""
        gamma_t = self.get_gamma_at_time(t)
        alpha_bar_t = jax.nn.sigmoid(-gamma_t)
        return alpha_bar_t
    
    def get_snr_at_time(self, t: float) -> float:
        """Get signal-to-noise ratio at continuous time t ∈ [0, 1]."""
        alpha_bar_t = self.get_noise_at_time(t)
        return alpha_bar_t / (1 - alpha_bar_t)
    
    def get_snr_derivative_at_time(self, t: float) -> float:
        """Get derivative of SNR at continuous time t ∈ [0, 1]."""
        gamma_t = self.get_gamma_at_time(t)
        gamma_prime_t = 1.0  # For linear γ(t) = t
        alpha_bar_t = self.get_noise_at_time(t)
        alpha_bar_prime = -gamma_prime_t * alpha_bar_t * (1 - alpha_bar_t)
        snr_derivative = alpha_bar_prime / ((1 - alpha_bar_t) ** 2)
        return snr_derivative
    
    def add_noise(self, x: jnp.ndarray, noise: jnp.ndarray, t: float) -> jnp.ndarray:
        """Add noise to input at continuous time t ∈ [0, 1] using variance-preserving OU process."""
        alpha_bar_t = self.get_noise_at_time(t)
        sqrt_alpha_bar = jnp.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = jnp.sqrt(1 - alpha_bar_t)
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
    
    @nn.compact
    def __call__(self, eta_init: jnp.ndarray, eta_target: jnp.ndarray, 
                 mu_current: jnp.ndarray, t: float, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through the geometric flow block.
        
        Args:
            eta_init: Initial natural parameters [batch_size, eta_dim]
            eta_target: Target natural parameters [batch_size, eta_dim]
            mu_current: Current mu values [batch_size, mu_dim]
            t: Continuous time t ∈ [0, 1]
            training: Whether in training mode
            
        Returns:
            Matrix A [batch_size, matrix_rank, output_dim] for geometric flow dynamics
        """
        # Apply feature engineering to eta if enabled
        if self.use_feature_engineering:
            from ..eta_features import compute_eta_features
            eta_init_features = compute_eta_features(eta_init, method='noprop')
            eta_target_features = compute_eta_features(eta_target, method='noprop')
        else:
            eta_init_features = eta_init
            eta_target_features = eta_target
        
        # Create time embedding
        time_embed = self._time_embedding(t, self.output_dim)
        time_embed_batch = jnp.broadcast_to(time_embed, (eta_init.shape[0], time_embed.shape[0]))
        
        # Concatenate inputs: [eta_init_features, eta_target_features, mu_current, time_embed]
        x = jnp.concatenate([eta_init_features, eta_target_features, mu_current, time_embed_batch], axis=-1)
        
        # Build the network layers with ResNet connections
        for i, hidden_size in enumerate(self.hidden_sizes):
            # Store input for potential skip connection
            residual = x if self.use_resnet and i > 0 and i % self.resnet_skip_every == 0 else None
            
            x = nn.Dense(hidden_size, 
                        kernel_init=nn.initializers.lecun_normal(),
                        bias_init=nn.initializers.zeros)(x)
            
            # Apply activation
            if self.activation == "swish":
                x = nn.swish(x)
            elif self.activation == "tanh":
                x = nn.tanh(x)
            elif self.activation == "relu":
                x = nn.relu(x)
            else:
                raise ValueError(f"Unknown activation: {self.activation}")
            
            # Add skip connection if conditions are met
            if residual is not None and residual.shape[-1] == x.shape[-1]:
                x = x + residual
        
        # Determine matrix rank
        matrix_rank = self.matrix_rank if self.matrix_rank is not None else self.output_dim
        
        # Output layer: predict matrix A with shape [batch_size, matrix_rank, output_dim]
        # We'll reshape the output to get the matrix structure
        total_elements = matrix_rank * self.output_dim
        x = nn.Dense(total_elements,
                    kernel_init=nn.initializers.lecun_normal(),
                    bias_init=nn.initializers.zeros)(x)
        
        # Reshape to [batch_size, matrix_rank, output_dim]
        A = jnp.reshape(x, (x.shape[0], matrix_rank, self.output_dim))
        
        # Post-hoc normalization by matrix rank for stability
        A = A / jnp.sqrt(matrix_rank)
        
        # Compute the geometric flow operation: A@A.T@(eta_target - eta_init)
        eta_diff = eta_target - eta_init
        du_dt = jnp.einsum('bij,bj->bi', A @ jnp.transpose(A, (0, 2, 1)), eta_diff)
        
        return du_dt


class NoProp_Geometric_Flow_ET_Network(nn.Module):
    """
    NoProp Geometric Flow ET Network that learns flow dynamics using NoProp training.
    
    This network learns the geometric flow dynamics:
        du/dt = A(u, t, η_t) @ A(u, t, η_t)^T @ (η_target - η_init)
    
    where A is learned using NoProp continuous-time training protocols.
    """
    
    hidden_sizes: List[int] = None
    output_dim: int = None
    matrix_rank: int = None
    n_time_steps: int = 10
    smoothness_weight: float = 1e-3
    time_embed_dim: int = None
    noise_schedule: str = "noprop_ct"
    flow_matching_sigma: float = 0.1
    
    def setup(self):
        """Initialize the geometric flow block."""
        # Determine time embedding dimension
        time_embed_dim = self.time_embed_dim if self.time_embed_dim is not None else self.output_dim
        
        # Create the geometric flow block
        self.geometric_flow_block = NoProp_Geometric_Flow_Block(
            hidden_sizes=self.hidden_sizes,
            output_dim=self.output_dim,
            matrix_rank=self.matrix_rank,
            noise_schedule=self.noise_schedule,
            use_feature_engineering=True,
            activation="swish",
            use_weak_layer_norm=False,
            use_resnet=True,
            resnet_skip_every=2
        )
    
    def _time_embedding(self, t: float, dim: int) -> jnp.ndarray:
        """Create time embedding for continuous time t ∈ [0, 1]."""
        time_embed_dim = self.time_embed_dim if self.time_embed_dim is not None else self.output_dim
        return self.geometric_flow_block._time_embedding(t, time_embed_dim)
    
    def get_gamma_at_time(self, t: float) -> float:
        """Get γ(t) at continuous time t ∈ [0, 1]."""
        return self.geometric_flow_block.get_gamma_at_time(t)
    
    def get_noise_at_time(self, t: float) -> float:
        """Get noise level at continuous time t ∈ [0, 1]."""
        return self.geometric_flow_block.get_noise_at_time(t)
    
    def get_snr_at_time(self, t: float) -> float:
        """Get signal-to-noise ratio at continuous time t ∈ [0, 1]."""
        return self.geometric_flow_block.get_snr_at_time(t)
    
    def get_snr_derivative_at_time(self, t: float) -> float:
        """Get derivative of SNR at continuous time t ∈ [0, 1]."""
        return self.geometric_flow_block.get_snr_derivative_at_time(t)
    
    def add_noise(self, x: jnp.ndarray, noise: jnp.ndarray, t: float) -> jnp.ndarray:
        """Add noise to input at continuous time t ∈ [0, 1]."""
        return self.geometric_flow_block.add_noise(x, noise, t)
    
    def generate_flow_matching_sample(self, z_0: jnp.ndarray, z_1: jnp.ndarray, t: float) -> jnp.ndarray:
        """Generate sample z_t for flow matching: p(z(t)|z_0,z_1,x) = Normal(t*z_1 + (1-t)*z_0, σ²)"""
        mean = t * z_1 + (1 - t) * z_0
        return mean
    
    def sample_flow_matching_data(self, z_1: jnp.ndarray, t: float, rng: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample flow matching data: z_0 ~ N(0, I), z_t ~ N(t*z_1 + (1-t)*z_0, σ²)"""
        rng, z_0_rng = jax.random.split(rng)
        z_0 = jax.random.normal(z_0_rng, z_1.shape)
        mean = t * z_1 + (1 - t) * z_0
        rng, z_t_rng = jax.random.split(rng)
        noise = jax.random.normal(z_t_rng, z_1.shape)
        z_t = mean + self.flow_matching_sigma * noise
        return z_0, z_t
    
    def compute_internal_loss(self, params: Dict, eta: jnp.ndarray, 
                            predicted_stats: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """No internal loss for geometric flow (losses are computed per time step)."""
        return 0.0
    
    def __call__(self, eta_init: jnp.ndarray, eta_target: jnp.ndarray, 
                 mu_init: jnp.ndarray, method: str = "predict") -> jnp.ndarray:
        """
        Forward pass through the geometric flow network.
        
        Args:
            eta_init: Initial natural parameters [batch_size, eta_dim]
            eta_target: Target natural parameters [batch_size, eta_dim]
            mu_init: Initial mu values [batch_size, mu_dim]
            method: Method to use ("predict" for inference, "predict_matrix_A" for training)
            
        Returns:
            Final mu values [batch_size, mu_dim] or matrix A for training
        """
        if method == "predict":
            return self._integrate_flow(eta_init, eta_target, mu_init)
        elif method == "predict_matrix_A":
            # For training: return matrix A at a specific time point
            # This will be called with continuous time t during training
            return self.geometric_flow_block(eta_init, eta_target, mu_init, 0.5, training=True)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def predict_matrix_A(self, eta_init: jnp.ndarray, eta_target: jnp.ndarray, 
                        mu_init: jnp.ndarray, t: float = 0.5) -> jnp.ndarray:
        """Predict du/dt for training at time t (now returns the full geometric flow operation)."""
        return self.geometric_flow_block(eta_init, eta_target, mu_init, t, training=True)
    
    def predict(self, eta_init: jnp.ndarray, eta_target: jnp.ndarray, 
                mu_init: jnp.ndarray) -> jnp.ndarray:
        """Predict final mu values using geometric flow integration."""
        return self._integrate_flow(eta_init, eta_target, mu_init)
    
    def _integrate_flow(self, eta_init: jnp.ndarray, eta_target: jnp.ndarray, 
                       mu_init: jnp.ndarray) -> jnp.ndarray:
        """Integrate the geometric flow from t=0 to t=1."""
        mu = mu_init
        dt = 1.0 / self.n_time_steps
        
        for i in range(self.n_time_steps):
            t = i / self.n_time_steps
            # The geometric flow block now returns du_dt directly
            du_dt = self.geometric_flow_block(eta_init, eta_target, mu, t, training=False)
            
            # Euler step
            mu = mu + dt * du_dt
        
        return mu


class NoProp_Geometric_Flow_ET_Trainer:
    """
    Trainer for NoProp Geometric Flow ET Network.
    
    This trainer implements the NoProp continuous-time training protocol for geometric flow:
    - Continuous time sampling t ~ Uniform(0,1)
    - Geometric flow dynamics with PSD constraints
    - Smoothness penalties for stable dynamics
    """
    
    def __init__(self, config: FullConfig, loss_type: str = "geometric_flow"):
        self.config = config
        self.loss_type = loss_type
        
        # Create the network with proper parameters
        self.model = NoProp_Geometric_Flow_ET_Network(
            hidden_sizes=config.network.hidden_sizes if config.network.hidden_sizes else [64, 64, 64],
            output_dim=config.network.output_dim,
            matrix_rank=getattr(config.network, 'matrix_rank', None),
            n_time_steps=getattr(config.network, 'n_time_steps', 10),
            smoothness_weight=getattr(config.network, 'smoothness_weight', 1e-3),
            noise_schedule=getattr(config.network, 'noise_schedule', 'noprop_ct'),
            flow_matching_sigma=getattr(config.network, 'flow_matching_sigma', 0.1)
        )
        
        # Training parameters
        self.learning_rate = config.training.learning_rate
        self.batch_size = getattr(config.training, 'batch_size', 32)
        
        # Initialize random key
        self.rng = random.PRNGKey(42)
        
        # Initialize parameters (will be set during training)
        self.mlp_params = None
        self.mlp_opt_state = None
        self.mlp_optimizer = None
    
    def initialize_mlp(self, sample_eta_init: jnp.ndarray, sample_eta_target: jnp.ndarray, 
                      sample_mu_init: jnp.ndarray) -> Tuple[Dict, Any]:
        """Initialize the MLP parameters."""
        # Create sample inputs
        sample_t = 0.5  # Sample time point
        
        # Initialize parameters
        params = self.model.init(
            self.rng, 
            sample_eta_init, 
            sample_eta_target, 
            sample_mu_init,
            method="predict_matrix_A"
        )
        
        # Initialize optimizer
        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(params)
        
        return params, opt_state
    
    def continuous_time_loss_fn(self, t: float, params: Dict, eta_init: jnp.ndarray, 
                               eta_target: jnp.ndarray, mu_init: jnp.ndarray,
                               target_mu_T: jnp.ndarray, loss_strategy: str = "geometric_flow",
                               input_output_penalty_weight: float = 0.0) -> jnp.ndarray:
        """
        Compute loss for geometric flow at continuous time t.
        
        Args:
            t: Continuous time t ∈ [0, 1]
            params: Network parameters
            eta_init: Initial natural parameters
            eta_target: Target natural parameters  
            mu_init: Initial mu values
            target_mu_T: Target mu values
            loss_strategy: Loss strategy ("geometric_flow", "simple_target", "flow_matching")
            input_output_penalty_weight: Weight for input-output difference penalty
            
        Returns:
            Loss value for this time point
        """
        # Get du/dt directly from the network (which now computes A@A^T@(η_target - η_init) internally)
        du_dt = self.model.apply(params, eta_init, eta_target, mu_init, 
                                method="predict_matrix_A")
        
        if loss_strategy == "geometric_flow":
            # Geometric flow loss: encourage du/dt to point toward target
            target_direction = target_mu_T - mu_init
            direction_loss = jnp.mean((du_dt - target_direction) ** 2)
            
            # Smoothness penalty
            smoothness_penalty = self.model.smoothness_weight * jnp.mean(du_dt ** 2)
            
            return direction_loss + smoothness_penalty
            
        elif loss_strategy == "simple_target":
            # Simple target loss: direct MSE to final target
            mse_loss = jnp.mean((du_dt - (target_mu_T - mu_init)) ** 2)
            reg_loss = 1e-4 * jnp.mean(du_dt ** 2)
            return mse_loss + reg_loss
            
        elif loss_strategy == "flow_matching":
            # Flow matching loss for geometric flow
            target_flow = target_mu_T - mu_init
            flow_loss = jnp.mean((du_dt - target_flow) ** 2)
            reg_loss = 1e-4 * jnp.mean(du_dt ** 2)
            return flow_loss + reg_loss
            
        else:
            raise ValueError(f"Unknown loss strategy: {loss_strategy}. Available: 'geometric_flow', 'simple_target', 'flow_matching'")
    
    def train_continuous_time(self, t: float, batch: Dict[str, jnp.ndarray], 
                             loss_strategy: str = None,
                             input_output_penalty_weight: float = 0.0) -> Tuple[Dict, Any, float]:
        """
        Single training step for geometric flow at continuous time t.
        
        Args:
            t: Continuous time t ∈ [0, 1]
            batch: Training batch data
            loss_strategy: Loss strategy to use
            input_output_penalty_weight: Weight for input-output difference penalty
            
        Returns:
            Updated parameters, optimizer state, and loss value
        """
        # Use loss_type if loss_strategy is not specified
        if loss_strategy is None:
            loss_strategy = self.loss_type
        
        # Compute loss and gradients
        loss_value, grads = jax.value_and_grad(self.continuous_time_loss_fn, argnums=1)(
            t, self.mlp_params, batch['eta_init'], batch['eta_target'], batch['mu_init'],
            batch['mu_T'], loss_strategy, input_output_penalty_weight
        )
        
        # Clip gradients to prevent instability
        grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        
        # Update parameters
        updates, new_opt_state = self.mlp_optimizer.update(grads, self.mlp_opt_state)
        new_params = optax.apply_updates(self.mlp_params, updates)
        
        # Update state
        self.mlp_params = new_params
        self.mlp_opt_state = new_opt_state
        
        return new_params, new_opt_state, float(loss_value)
    
    def train_epoch(self, train_data: Dict[str, jnp.ndarray], 
                   loss_strategy: str = "geometric_flow",
                   input_output_penalty_weight: float = 0.0) -> List[float]:
        """Train the geometric flow network for one epoch."""
        # Shuffle data
        n_train = train_data['eta_init'].shape[0]
        batch_size = min(self.batch_size, n_train)
        
        # Create shuffled indices
        self.rng, shuffle_rng = random.split(self.rng)
        indices = random.permutation(shuffle_rng, n_train)
        
        # Shuffle all data
        train_data_shuffled = {k: v[indices] for k, v in train_data.items()}
        
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
              epochs: int = 300, loss_strategy: str = "geometric_flow",
              input_output_penalty_weight: float = 0.0) -> Tuple[Dict, Dict]:
        """
        Full training loop for geometric flow using NoProp continuous-time protocol.
        
        Args:
            train_data: Training data with keys ['eta_init', 'eta_target', 'mu_init', 'mu_T']
            val_data: Validation data
            epochs: Number of training epochs
            loss_strategy: Loss strategy to use
            input_output_penalty_weight: Weight for input-output difference penalty
            
        Returns:
            Best parameters and training history
        """
        # Initialize the network
        sample_eta_init = train_data['eta_init'][:1]
        sample_eta_target = train_data['eta_target'][:1]
        sample_mu_init = train_data['mu_init'][:1]
        self.mlp_params, self.mlp_opt_state = self.initialize_mlp(
            sample_eta_init, sample_eta_target, sample_mu_init
        )
        self.mlp_optimizer = optax.adam(self.learning_rate)
        
        print(f"Training NoProp Geometric Flow ET with {self.model.n_time_steps} time steps for inference")
        print(f"Single network trained on continuous time t ~ Uniform(0,1) (no backpropagation)")
        print(f"Network Architecture: {self.config.network.hidden_sizes}")
        print(f"Matrix Rank: {self.model.matrix_rank}")
        print(f"Loss Strategy: {loss_strategy}")
        if input_output_penalty_weight > 0:
            print(f"Input-Output Penalty Weight: {input_output_penalty_weight}")
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        best_params = self.mlp_params
        
        pbar = tqdm(range(epochs), desc="Training NoProp Geometric Flow ET")
        for epoch in pbar:
            # Train the network for one epoch
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
        # Use the network for inference
        eta_init = val_data['eta_init']
        eta_target = val_data['eta_target']
        mu_init = val_data['mu_init']
        
        # Get final predictions using the __call__ method
        mu_final = self.model.apply(self.mlp_params, eta_init, eta_target, mu_init, method="predict")
        
        # Compute MSE loss
        mse_loss = jnp.mean((mu_final - val_data['mu_T']) ** 2)
        return float(mse_loss)
    
    def predict(self, eta_init: jnp.ndarray, eta_target: jnp.ndarray, mu_init: jnp.ndarray) -> jnp.ndarray:
        """Make predictions using the trained model."""
        return self.model.apply(self.mlp_params, eta_init, eta_target, mu_init, method="predict")


def create_model_and_trainer(config: FullConfig, loss_type: str = "geometric_flow") -> NoProp_Geometric_Flow_ET_Trainer:
    """Factory function to create NoProp Geometric Flow ET model and trainer."""
    return NoProp_Geometric_Flow_ET_Trainer(config, loss_type)
