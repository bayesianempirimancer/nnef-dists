"""
Geometric Flow ET implementation with dedicated architecture.

This module provides a standalone Geometric Flow ET model that learns flow dynamics 
for exponential families. It implements a geometric flow network that learns the dynamics:
    du/dt = A@A^T@(η_target - η_init)

where:
- A = NN(u, t, η_t) with η_t = η_init*(1-t) + η_target*t  
- u(0) = μ_T_init (from nearest analytical point)
- u(1) = μ_T_target (goal)

The A@A^T structure ensures positive semidefiniteness, and smoothness penalties
encourage stable dynamics.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import random, grad
from typing import Dict, Any, Tuple, Optional, List
import optax
import time
from tqdm import tqdm

from ..base_model import BaseNeuralNetwork, BaseTrainer
from ..config import FullConfig, NetworkConfig
from ..ef import ExponentialFamily


class Geometric_Flow_ET_Network(BaseNeuralNetwork):
    """
    Geometric Flow ET Network that learns flow dynamics for exponential families.
    
    The network learns A(u, t, η_t) such that:
        du/dt = A(u, t, η_t) @ A(u, t, η_t)^T @ (η_target - η_init)
    
    Key features:
    - Geometric flow dynamics with PSD constraints
    - Smoothness penalties for stable dynamics
    - Minimal time steps due to expected smoothness
    """
    
    matrix_rank: int = None  # Rank of matrix A (if None, use output dimension)
    n_time_steps: int = 10  # Increased time steps for better integration
    smoothness_weight: float = 1e-3  # Penalty for large du/dt
    time_embed_dim: int = None  # Will be set to match eta/mu dimensions
    max_freq: float = 10.0  # Maximum frequency for time embedding
    
    def _weak_layer_norm(self, x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
        """
        Weak layer normalization: x/norm(x) * log(1 + norm(x))
        
        This normalizes the input but preserves magnitude information,
        making it less aggressive than standard layer normalization.
        
        Args:
            x: Input tensor [batch_size, features]
            eps: Small constant for numerical stability
            
        Returns:
            Weakly normalized tensor
        """
        # Compute L2 norm along the last dimension
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        
        # Normalize and scale by log(1 + norm)
        normalized = x / (norm + eps)
        scaled = normalized * jnp.log(1 + norm)
        
        return scaled
    
    def _time_embedding(self, t: float, target_dim: int) -> jnp.ndarray:
        """
        Create Fourier time embedding with dimensions matching eta/mu.
        
        Args:
            t: Time value ∈ [0, 1]
            target_dim: Target dimension for the embedding (should match eta/mu dim)
            
        Returns:
            Time embedding [target_dim,]
        """
        # Use target_dim as the embedding dimension
        embed_dim = target_dim
        
        # Create frequency schedule - use fewer frequencies for better generalization
        n_freqs = max(1, embed_dim // 4)  # Use 1/4 of the dimension for frequencies
        freqs = jnp.logspace(0, jnp.log10(self.max_freq), n_freqs)
        
        # Create sin and cos embeddings
        sin_embeddings = jnp.sin(2 * jnp.pi * freqs * t)
        cos_embeddings = jnp.cos(2 * jnp.pi * freqs * t)
        
        # Combine sin and cos
        freq_embeddings = jnp.concatenate([sin_embeddings, cos_embeddings])
        
        # If we need more dimensions, repeat and add phase shifts
        if len(freq_embeddings) < embed_dim:
            # Repeat the embeddings with different phase shifts
            n_repeats = (embed_dim + len(freq_embeddings) - 1) // len(freq_embeddings)
            repeated_embeddings = jnp.tile(freq_embeddings, n_repeats)
            embeddings = repeated_embeddings[:embed_dim]
        else:
            # Truncate if we have too many
            embeddings = freq_embeddings[:embed_dim]
        
        # Add a linear component for time progression
        linear_component = jnp.linspace(0, t, embed_dim)
        
        # Combine frequency and linear components
        final_embedding = embeddings + 0.1 * linear_component
        
        return final_embedding
    
    def _diagonal_favorable_init(self, mu_dim: int, matrix_rank: int):
        """
        Initialize matrix A such that A@A.T is closer to a diagonal matrix.
        
        Strategy:
        1. Initialize A with small random values
        2. Add a diagonal bias to make A@A.T more diagonal
        3. Scale appropriately to avoid vanishing/exploding gradients
        
        Args:
            mu_dim: Dimension of the output space
            matrix_rank: Rank of matrix A
            
        Returns:
            Initialization function for Flax Dense layer
        """
        def init_fn(key, shape, dtype=jnp.float32):
            # Standard small random initialization
            std = 0.1  # Small standard deviation
            random_part = random.normal(key, shape, dtype) * std
            
            # Create a matrix that will make A@A.T more diagonal
            # We want A to have structure that leads to diagonal A@A.T
            if len(shape) == 2:  # [input_dim, output_dim]
                input_dim, output_dim = shape
                
                # Reshape to [input_dim, mu_dim, matrix_rank]
                if output_dim == mu_dim * matrix_rank:
                    # Create initialization that favors diagonal A@A.T
                    # Strategy: make A have strong diagonal components
                    A_reshaped = random_part.reshape(input_dim, mu_dim, matrix_rank)
                    
                    # Add diagonal bias: for each i, make A[i,i,:] larger
                    for i in range(min(mu_dim, matrix_rank)):
                        # Add extra weight to diagonal elements
                        diagonal_bias = 0.5  # Additional weight for diagonal
                        A_reshaped = A_reshaped.at[:, i, i].add(diagonal_bias)
                    
                    return A_reshaped.reshape(input_dim, output_dim)
                else:
                    return random_part
            else:
                return random_part
        
        return init_fn
    
    
    @nn.compact  
    def __call__(self, eta_init: jnp.ndarray, eta_target: jnp.ndarray, 
                 mu_init: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Geometric flow computation using inherited ET network architecture.
        
        Args:
            eta_init: Initial natural parameters [batch_size, eta_dim]
            eta_target: Target natural parameters [batch_size, eta_dim]
            mu_init: Initial expectations [batch_size, mu_dim]
            training: Whether in training mode
            
        Returns:
            mu_target: Predicted expectations at eta_target [batch_size, mu_dim]
        """
        batch_size = eta_init.shape[0]
        eta_dim = eta_init.shape[1]
        mu_dim = mu_init.shape[1]
        
        if self.matrix_rank is None:
            matrix_rank = mu_dim
        else:
            matrix_rank = self.matrix_rank
        
        # Simple forward Euler integration with minimal time steps
        dt = 1.0 / self.n_time_steps
        u_current = mu_init
        
        # Store derivative norms squared for smoothness penalty
        derivative_norms_squared = []
        
        for i in range(self.n_time_steps):
            t = i * dt
            
            # Fourier time embedding with dimensions matching mu
            t_embed = self._time_embedding(t, mu_dim)  # [mu_dim,]
            t_embed_batch = jnp.broadcast_to(t_embed, (batch_size, mu_dim))
            
            # Network input: [u, sin/cos(t), η_init, η_target]
            # This gives the network direct access to both endpoints instead of interpolated values
            net_input = jnp.concatenate([u_current, t_embed_batch, eta_init, eta_target], axis=-1)
            
            # Use inherited ET network architecture to predict matrix A
            A_flat = self.predict_matrix_A(net_input, training)
            A = A_flat.reshape(batch_size, mu_dim, matrix_rank)
            
            # Compute flow matrix: Σ = A @ A^T (guaranteed PSD)
            Sigma = A@A.mT
            
            # Direction vector
            delta_eta = eta_target - eta_init
            
            # Handle dimension mismatch
            if eta_dim != mu_dim:
                delta_eta_proj = nn.Dense(mu_dim, name=f'eta_proj_step_{i}')(delta_eta)
            else:
                delta_eta_proj = delta_eta
            
            # Flow field: du/dt = Σ @ (η_target - η_init)
            du_dt = jnp.matmul(Sigma, delta_eta_proj[:, :, None]).squeeze(-1)
            
            # Store norm squared for smoothness penalty (memory efficient)
            du_dt_norm_squared = jnp.sum(du_dt ** 2, axis=-1)  # [batch_size,]
            derivative_norms_squared.append(du_dt_norm_squared)
            
            # Forward Euler step
            u_current = u_current + dt * du_dt
        
        # Store derivative norms squared for potential smoothness loss computation
        if training:
            self.sow('intermediates', 'derivative_norms_squared', jnp.array(derivative_norms_squared))
        
        return u_current
    
    def _mlp_forward(self, net_input: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """MLP architecture forward pass for matrix A prediction."""
        x = net_input
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            # Use LeCun normal initialization and zero bias for better conditioning
            x = nn.Dense(hidden_size, name=f'mlp_hidden_{i}',
                        kernel_init=nn.initializers.lecun_normal(),
                        bias_init=nn.initializers.zeros)(x)
            x = nn.swish(x)
            if getattr(self.config, 'use_layer_norm', True):
                x = self._weak_layer_norm(x)
        return x
    
    def _glu_forward(self, net_input: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """GLU architecture forward pass for matrix A prediction."""
        x = net_input
        # Input projection with LeCun normal initialization
        x = nn.Dense(self.config.hidden_sizes[0], name='glu_input_proj',
                    kernel_init=nn.initializers.lecun_normal(),
                    bias_init=nn.initializers.zeros)(x)
        x = jnp.tanh(x)
        
        # GLU blocks with residual connections
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            residual = x
            if residual.shape[-1] != hidden_size:
                residual = nn.Dense(hidden_size, name=f'glu_residual_proj_{i}',
                                   kernel_init=nn.initializers.lecun_normal(),
                                   bias_init=nn.initializers.zeros)(residual)
            
            # GLU layer with LeCun normal initialization
            linear1 = nn.Dense(hidden_size, name=f'glu_linear1_{i}',
                              kernel_init=nn.initializers.lecun_normal(),
                              bias_init=nn.initializers.zeros)(x)
            linear2 = nn.Dense(hidden_size, name=f'glu_linear2_{i}',
                              kernel_init=nn.initializers.lecun_normal(),
                              bias_init=nn.initializers.zeros)(x)
            gate = nn.sigmoid(linear1)
            glu_out = gate * linear2
            
            # Residual connection
            x = residual + glu_out
            x = nn.swish(x)
            if getattr(self.config, 'use_layer_norm', True):
                x = self._weak_layer_norm(x)
        
        return x
    
    def _quadratic_forward(self, net_input: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Quadratic ResNet architecture forward pass for matrix A prediction."""
        x = net_input
        
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            # Residual connection with LeCun normal initialization
            residual = x
            if residual.shape[-1] != hidden_size:
                residual = nn.Dense(hidden_size, name=f'quad_residual_proj_{i}',
                                   kernel_init=nn.initializers.lecun_normal(),
                                   bias_init=nn.initializers.zeros)(residual)
            
            # Quadratic transformation with LeCun normal initialization
            linear_out = nn.Dense(hidden_size, name=f'quad_linear_{i}',
                                 kernel_init=nn.initializers.lecun_normal(),
                                 bias_init=nn.initializers.zeros)(x)
            quadratic_out = nn.Dense(hidden_size, name=f'quad_quadratic_{i}',
                                    kernel_init=nn.initializers.lecun_normal(),
                                    bias_init=nn.initializers.zeros)(x * x)
            
            # Combine linear and quadratic terms
            x = residual + linear_out + quadratic_out
            x = nn.swish(x)
            if getattr(self.config, 'use_layer_norm', True):
                x = self._weak_layer_norm(x)
        
        return x
    
    @nn.compact
    def predict_matrix_A(self, net_input: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Predict matrix A using the inherited ET network architecture.
        
        Args:
            net_input: Network input [batch_size, input_dim]
            training: Whether in training mode
            
        Returns:
            A: Matrix [batch_size, mu_dim, matrix_rank]
        """
        batch_size = net_input.shape[0]
        
        # Use inherited ET network architecture
        architecture = getattr(self.config, 'architecture', 'mlp')
        if architecture == "mlp":
            A_flat = self._mlp_forward(net_input, training)
        elif architecture == "glu":
            A_flat = self._glu_forward(net_input, training)
        elif architecture == "quadratic":
            A_flat = self._quadratic_forward(net_input, training)
        else:
            raise ValueError(f"Architecture {architecture} not supported for geometric flow")
        
        # Calculate dimensions from input
        # Input format: [u, t_embed, eta_init, eta_target]
        # where u and t_embed are mu_dim, eta_init and eta_target are eta_dim
        # So total input dim = 2 * mu_dim + 2 * eta_dim
        # For 3D case: eta_dim = 12, mu_dim = 12, so total = 2*12 + 2*12 = 48
        total_input_dim = net_input.shape[1]
        
        # For 3D case, we know eta_dim = mu_dim = 12
        # So: total_input_dim = 2 * 12 + 2 * 12 = 48
        # Therefore: mu_dim = eta_dim = total_input_dim / 4
        mu_dim = total_input_dim // 4
        eta_dim = mu_dim  # For multivariate normal, eta_dim = mu_dim
        
        if self.matrix_rank is None:
            matrix_rank = mu_dim
        else:
            matrix_rank = self.matrix_rank
        
        # Final layer to get matrix A elements with LeCun normal initialization
        A_flat = nn.Dense(mu_dim * matrix_rank, name='matrix_A_output',
                         kernel_init=nn.initializers.lecun_normal(),
                         bias_init=nn.initializers.zeros)(A_flat)
        
        # Reshape to matrix form
        A = A_flat.reshape(batch_size, mu_dim, matrix_rank)
        
        # Post-hoc normalization by matrix rank to control magnitude
        A = A / jnp.sqrt(matrix_rank)
        
        return A
    
    def compute_flow_field(self, u: jnp.ndarray, t: float, eta_init: jnp.ndarray, 
                          eta_target: jnp.ndarray) -> jnp.ndarray:
        """
        Compute flow field du/dt = A@A^T@(η_target - η_init).
        
        Args:
            u: Current expectation state [batch_size, mu_dim]
            t: Current time ∈ [0,1]
            eta_init: Initial natural parameters [batch_size, eta_dim]
            eta_target: Target natural parameters [batch_size, eta_dim]
            
        Returns:
            du_dt: Flow field [batch_size, mu_dim]
        """
        # Prepare network input
        batch_size = u.shape[0]
        mu_dim = u.shape[1]
        t_embed = self._time_embedding(t, mu_dim)
        t_embed_batch = jnp.broadcast_to(t_embed, (batch_size, mu_dim))
        
        # Network input: [u, sin/cos(t), η_init, η_target]
        # This gives the network direct access to both endpoints instead of interpolated values
        net_input = jnp.concatenate([u, t_embed_batch, eta_init, eta_target], axis=-1)
        
        # Predict matrix A
        A_flat = self.predict_matrix_A(net_input, training=True)
        mu_dim = u.shape[1]
        matrix_rank = self.matrix_rank if self.matrix_rank is not None else mu_dim
        A = A_flat.reshape(batch_size, mu_dim, matrix_rank)
        
        # Compute flow matrix: Σ = A @ A^T
        Sigma = jnp.matmul(A, jnp.transpose(A, (0, 2, 1)))  # [batch_size, mu_dim, mu_dim]
        
        # Direction vector
        delta_eta = eta_target - eta_init  # [batch_size, eta_dim]
        
        # Handle dimension mismatch if eta_dim != mu_dim
        eta_dim = delta_eta.shape[-1]
        mu_dim = u.shape[-1]
        
        if eta_dim != mu_dim:
            # Learn a projection matrix
            delta_eta_proj = nn.Dense(mu_dim, name='eta_projection')(delta_eta)
        else:
            delta_eta_proj = delta_eta
        
        # Flow field: du/dt = Σ @ (η_target - η_init)
        du_dt = jnp.matmul(Sigma, delta_eta_proj[:, :, None]).squeeze(-1)
        
        return du_dt
    
    def __call__(self, eta_init: jnp.ndarray, eta_target: jnp.ndarray, 
                 mu_init: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Integrate flow from (eta_init, mu_init) to eta_target.
        
        Args:
            eta_init: Initial natural parameters [batch_size, eta_dim]
            eta_target: Target natural parameters [batch_size, eta_dim]
            mu_init: Initial expectations [batch_size, mu_dim]
            training: Whether in training mode
            
        Returns:
            mu_target: Predicted expectations at eta_target [batch_size, mu_dim]
        """
        # Simple forward Euler integration with intermediate collection
        dt = 1.0 / self.n_time_steps
        u_current = mu_init
        batch_size, mu_dim = mu_init.shape
        eta_dim = eta_init.shape[-1]
        matrix_rank = self.matrix_rank or mu_dim
        
        # Store derivative norms squared for smoothness penalty
        derivative_norms_squared = []
        
        for i in range(self.n_time_steps):
            t = i * dt
            
            # Fourier time embedding with dimensions matching mu
            t_embed = self._time_embedding(t, mu_dim)  # [mu_dim,]
            t_embed_batch = jnp.broadcast_to(t_embed, (batch_size, mu_dim))
            
            # Network input: [u, sin/cos(t), η_init, η_target]
            # This gives the network direct access to both endpoints instead of interpolated values
            net_input = jnp.concatenate([u_current, t_embed_batch, eta_init, eta_target], axis=-1)
            
            # Use inherited ET network architecture to predict matrix A
            A_flat = self.predict_matrix_A(net_input, training)
            A = A_flat.reshape(batch_size, mu_dim, matrix_rank)
            
            # Compute flow matrix: Σ = A @ A^T (guaranteed PSD)
            Sigma = A@A.mT
            
            # Direction vector
            delta_eta = eta_target - eta_init
            
            # Handle dimension mismatch
            if eta_dim != mu_dim:
                delta_eta_proj = nn.Dense(mu_dim, name=f'eta_proj_step_{i}')(delta_eta)
            else:
                delta_eta_proj = delta_eta
            
            # Flow field: du/dt = Σ @ (η_target - η_init)
            du_dt = jnp.matmul(Sigma, delta_eta_proj[:, :, None]).squeeze(-1)
            
            # Store norm squared for smoothness penalty (memory efficient)
            du_dt_norm_squared = jnp.sum(du_dt ** 2, axis=-1)  # [batch_size,]
            derivative_norms_squared.append(du_dt_norm_squared)
            
            # Forward Euler step
            u_current = u_current + dt * du_dt
        
        # Store derivative norms squared for potential smoothness loss computation
        if training:
            self.sow('intermediates', 'derivative_norms_squared', jnp.array(derivative_norms_squared))
        
        return u_current
    
    def compute_internal_loss(self, params: Dict, eta: jnp.ndarray, 
                            predicted_stats: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Compute smoothness penalty for geometric flow dynamics.
        
        This implements the smoothness penalty that encourages stable flow dynamics
        by penalizing large derivatives du/dt during the ODE integration.
        """
        if not training or self.smoothness_weight <= 0:
            return 0.0
            
        # For geometric flow, we need to compute smoothness penalty
        # This requires running a forward pass to collect intermediate derivatives
        try:
            # Run forward pass with intermediate collection
            _, intermediates = self.apply(
                params, eta, eta, predicted_stats,  # Use predicted_stats as mu_init
                training=True, mutable=['intermediates']
            )
            
            # Smoothness penalty: penalize large derivatives du/dt
            if 'derivative_norms_squared' in intermediates and self.smoothness_weight > 0:
                derivative_norms_squared = intermediates['derivative_norms_squared']  # tuple of [n_time_steps, batch_size]
                
                # Convert tuple to array and compute penalty for large derivatives
                derivative_norms_squared_array = jnp.array(derivative_norms_squared)
                smoothness_loss = jnp.mean(derivative_norms_squared_array)
                
                return self.smoothness_weight * smoothness_loss
            
        except Exception:
            # If intermediate collection fails, return 0 (graceful degradation)
            pass
            
        return 0.0


class Geometric_Flow_ET_Trainer(BaseTrainer):
    """
    Trainer for Geometric Flow ET Networks.
    
    Inherits from ETTrainer but implements flow-based training with:
    - Analytical initialization using find_nearest_analytical_point
    - Smoothness penalties for stable dynamics
    - Geometric constraints via A@A^T structure
    """
    
    def __init__(self, config: FullConfig, matrix_rank: int = None, 
                 n_time_steps: int = 10, smoothness_weight: float = 1e-3):
        model = Geometric_Flow_ET_Network(
            config=config.network,
            matrix_rank=matrix_rank,
            n_time_steps=n_time_steps,
            smoothness_weight=smoothness_weight
        )
        super().__init__(model, config)
        self.matrix_rank = matrix_rank
        self.n_time_steps = n_time_steps
        self.smoothness_weight = smoothness_weight
        self.ef_instance = None
    
    def set_exponential_family(self, ef_instance):
        """Set the exponential family instance for analytical point computation."""
        self.ef_instance = ef_instance
    
    def create_flow_batch(self, eta_targets: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Create training batch using nearest analytical points.
        
        Args:
            eta_targets: Target natural parameters [batch_size, eta_dim]
            
        Returns:
            Batch with eta_init, mu_init, eta_target, mu_target
        """
        if self.ef_instance is None:
            raise ValueError("Must call set_exponential_family() before creating batches")
        
        batch_size = eta_targets.shape[0]
        eta_inits = []
        mu_inits = []
        mu_targets = []
        
        for i in range(batch_size):
            # Find nearest analytical point
            eta_init_dict, mu_init_dict = self.ef_instance.find_nearest_analytical_point(eta_targets[i])
            
            eta_init = self.ef_instance.flatten_stats_or_eta(eta_init_dict)
            mu_init = self.ef_instance.flatten_stats_or_eta(mu_init_dict)
            
            # Compute true target μ
            eta_target_dict = self.ef_instance.unflatten_stats_or_eta(eta_targets[i])
            eta1 = jnp.real(eta_target_dict['x'])  # Ensure real
            eta2 = jnp.real(eta_target_dict['xxT'])  # Ensure real
            
            Sigma = jnp.real(-0.5 * jnp.linalg.inv(eta2))
            mu = jnp.real(jnp.linalg.solve(eta2, -0.5 * eta1))
            E_xx = jnp.real(Sigma + jnp.outer(mu, mu))
            
            mu_target = self.ef_instance.flatten_stats_or_eta({'x': mu, 'xxT': E_xx})
            
            eta_inits.append(eta_init)
            mu_inits.append(mu_init)
            mu_targets.append(mu_target)
        
        return {
            'eta_init': jnp.array(eta_inits),
            'mu_init': jnp.array(mu_inits),
            'eta_target': eta_targets,
            'mu_target': jnp.array(mu_targets)
        }
    
    def geometric_flow_loss(self, params: Dict, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Compute loss for geometric flow training with smoothness penalty.
        
        Loss = ||u(1) - μ_target||² + smoothness_penalty * ||du/dt||²
        """
        # Flow prediction with intermediate derivatives
        u_pred, intermediates = self.model.apply(
            params,
            batch['eta_init'],
            batch['eta_target'], 
            batch['mu_init'],
            training=True,
            mutable=['intermediates']
        )
        
        # Primary loss: endpoint error
        endpoint_loss = jnp.mean((u_pred - batch['mu_target']) ** 2)
        
        total_loss = endpoint_loss
        
        # Smoothness penalty: penalize large derivatives du/dt
        if 'derivative_norms_squared' in intermediates and self.smoothness_weight > 0:
            derivative_norms_squared = intermediates['derivative_norms_squared']  # tuple of [n_time_steps, batch_size]
            
            # Convert tuple to array and compute penalty for large derivatives
            derivative_norms_squared_array = jnp.array(derivative_norms_squared)
            smoothness_loss = jnp.mean(derivative_norms_squared_array)
            
            total_loss += self.smoothness_weight * smoothness_loss
        
        # Optional: Additional regularization
        reg_weight = getattr(self.config.training, 'regularization_weight', 1e-5)
        if reg_weight > 0:
            # L2 regularization on parameters
            l2_reg = 0.0
            for param in jax.tree.leaves(params):
                l2_reg += jnp.sum(param ** 2)
            total_loss += reg_weight * l2_reg
        
        return jnp.real(total_loss)
    
    def train_step(self, params: Dict, opt_state: Any, batch: Dict[str, jnp.ndarray], 
                   optimizer: Any) -> Tuple[Dict, Any, float]:
        """Single training step with geometric flow loss."""
        loss_value, grads = jax.value_and_grad(self.geometric_flow_loss)(params, batch)
        
        # Gradient clipping
        grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, float(loss_value)
    
    def train(self, eta_targets_train: jnp.ndarray, 
              eta_targets_val: Optional[jnp.ndarray] = None,
              epochs: int = 200, learning_rate: float = 1e-3,
              batch_size: int = 32) -> Tuple[Dict, Dict]:
        """
        Train the geometric flow network (no progress bars - handled by training script).
        """
        if self.ef_instance is None:
            raise ValueError("Must call set_exponential_family() before training")
        
        # Initialize model
        rng = random.PRNGKey(42)
        dummy_batch = self.create_flow_batch(eta_targets_train[:2])
        params = self.model.init(
            rng,
            dummy_batch['eta_init'],
            dummy_batch['eta_target'],
            dummy_batch['mu_init']
        )
        
        # Optimizer
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)
        
        # Training loop with progress bar
        history = {'train_loss': [], 'val_loss': []}
        best_params = params
        best_loss = float('inf')
        n_train = eta_targets_train.shape[0]
        
        # Print training details
        print(f"Training {self.model.__class__.__name__} for {epochs} epochs")
        print(f"Architecture: {self.model.config.hidden_sizes}")
        print(f"Matrix rank: {getattr(self.model, 'matrix_rank', 'None')}")
        print(f"Time steps: {getattr(self.model, 'n_time_steps', 'None')}")
        print(f"Parameters: {sum(p.size for p in jax.tree_util.tree_flatten(params)[0]):,}")
        
        pbar = tqdm(range(epochs), desc="Training Geometric Flow ET")
        for epoch in pbar:
            # Training
            train_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_train, batch_size):
                eta_batch = eta_targets_train[i:i+batch_size]
                batch = self.create_flow_batch(eta_batch)
                params, opt_state, batch_loss = self.train_step(params, opt_state, batch, optimizer)
                train_loss += batch_loss
                n_batches += 1
            
            train_loss /= n_batches
            history['train_loss'].append(train_loss)
            
            # Validation
            if eta_targets_val is not None:
                val_batch = self.create_flow_batch(eta_targets_val)
                val_loss = float(self.geometric_flow_loss(params, val_batch))
                history['val_loss'].append(val_loss)
                
                # Update progress bar with loss information
                pbar.set_postfix({
                    'train_loss': f'{train_loss:.6f}',
                    'val_loss': f'{val_loss:.6f}',
                    'best': f'{best_loss:.6f}'
                })
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = params
            else:
                # Update progress bar with training loss only
                pbar.set_postfix({
                    'train_loss': f'{train_loss:.6f}',
                    'best': f'{best_loss:.6f}'
                })
                
                # No validation data, use training loss for early stopping
                if train_loss < best_loss:
                    best_loss = train_loss
                    best_params = params
        
        return best_params, history
    
    def predict(self, params: Dict, eta_targets: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Make predictions using the trained geometric flow network.
        
        Args:
            params: Trained model parameters
            eta_targets: Target natural parameters [batch_size, eta_dim]
            
        Returns:
            Dictionary with predictions and intermediate results
        """
        if self.ef_instance is None:
            raise ValueError("Must call set_exponential_family() before prediction")
        
        # Create flow batch (this gives us the analytical starting points)
        batch = self.create_flow_batch(eta_targets)
        
        # Flow prediction
        mu_pred = self.model.apply(
            params,
            batch['eta_init'],
            batch['eta_target'],
            batch['mu_init'],
            training=False
        )
        
        return {
            'mu_predicted': mu_pred,
            'mu_target_true': batch['mu_target'],
            'eta_init': batch['eta_init'],
            'mu_init': batch['mu_init'],
            'flow_distances': jnp.linalg.norm(batch['eta_target'] - batch['eta_init'], axis=1)
        }
    
    def evaluate(self, params: Dict, eta_targets: jnp.ndarray) -> Dict[str, float]:
        """Evaluate the trained model."""
        predictions = self.predict(params, eta_targets)
        
        # Compute metrics
        mse = float(jnp.mean((predictions['mu_predicted'] - predictions['mu_target_true']) ** 2))
        mae = float(jnp.mean(jnp.abs(predictions['mu_predicted'] - predictions['mu_target_true'])))
        
        # Component-wise errors
        errors_by_component = jnp.mean((predictions['mu_predicted'] - predictions['mu_target_true']) ** 2, axis=0)
        
        return {
            'mse': mse,
            'mae': mae,
            'mean_flow_distance': float(jnp.mean(predictions['flow_distances'])),
            'component_errors': errors_by_component
        }


def create_model_and_trainer(config: FullConfig, matrix_rank: int = None, 
                           n_time_steps: int = 10, smoothness_weight: float = 1e-3):
    """Factory function to create Geometric Flow ET model and trainer."""
    return Geometric_Flow_ET_Trainer(
        config=config,
        matrix_rank=matrix_rank,
        n_time_steps=n_time_steps,
        smoothness_weight=smoothness_weight
    )
