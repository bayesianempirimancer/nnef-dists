"""
LogZ Network - A unified neural network for learning log normalizers.

This module provides a flexible neural network architecture specifically designed
for learning log normalizers A(η) of exponential families. The network outputs
a scalar log normalizer whose gradients and Hessians provide the mean and 
covariance of sufficient statistics.

Key features:
- Flexible architecture (MLP, GLU, Quadratic ResNet)
- Automatic gradient/Hessian computation via JAX
- Numerically stable training with regularization
- Support for various activation functions
- Layer normalization for stability
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import random, grad, hessian, jacfwd
from typing import Dict, Any, Tuple, Optional, Union, List
import optax
import time

from ..base_model import BaseNeuralNetwork, BaseTrainer
from ..config import FullConfig
from ..ef import ExponentialFamily


class LogZNetwork(BaseNeuralNetwork):
    """
    Unified LogZ Network for learning log normalizers.
    
    This network can be configured to use different architectures:
    - MLP: Standard multi-layer perceptron
    - GLU: Gated linear unit architecture
    - Quadratic: Quadratic ResNet architecture
    
    The network outputs a scalar log normalizer A(η) whose derivatives
    provide the moments of the exponential family distribution.
    """
    
    architecture: str = "mlp"  # "mlp", "glu", "quadratic"
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through the logZ network.
        
        Args:
            eta: Natural parameters [batch_size, eta_dim]
            training: Whether in training mode
            
        Returns:
            Log normalizer values [batch_size,]
        """
        # Apply feature engineering if enabled
        if hasattr(self.config, 'use_feature_engineering') and self.config.use_feature_engineering:
            from ..eta_features import compute_eta_features
            x = compute_eta_features(eta, method='default')
        else:
            x = eta
        
        if self.architecture == "mlp":
            x = self._mlp_forward(x, training)
        elif self.architecture == "glu":
            x = self._glu_forward(x, training)
        elif self.architecture == "quadratic":
            x = self._quadratic_forward(x, training)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        # Final projection to scalar log normalizer
        x = nn.Dense(1, name='logZ_output')(x)
        return jnp.squeeze(x, axis=-1)
    
    def _mlp_forward(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        """MLP architecture forward pass."""
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            x = nn.Dense(hidden_size, name=f'mlp_hidden_{i}')(x)
            x = nn.swish(x)
            if getattr(self.config, 'use_layer_norm', True):
                x = nn.LayerNorm(name=f'mlp_layer_norm_{i}')(x)
            
            dropout_rate = getattr(self.config, 'dropout_rate', 0.0)
            if dropout_rate > 0:
                x = nn.Dropout(rate=dropout_rate, deterministic=not training)(x)
        
        return x
    
    def _glu_forward(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        """GLU architecture forward pass."""
        # Input projection
        x = nn.Dense(self.config.hidden_sizes[0], name='glu_input_proj')(x)
        x = nn.swish(x)
        
        # GLU blocks
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            residual = x
            if residual.shape[-1] != hidden_size:
                residual = nn.Dense(hidden_size, name=f'glu_residual_proj_{i}')(residual)
            
            # GLU layer
            linear1 = nn.Dense(hidden_size, name=f'glu_linear1_{i}')(x)
            linear2 = nn.Dense(hidden_size, name=f'glu_linear2_{i}')(x)
            gate = nn.sigmoid(linear1)
            glu_out = gate * linear2
            
            # Residual connection
            x = residual + glu_out
            
            # Layer normalization
            if getattr(self.config, 'use_layer_norm', True):
                x = nn.LayerNorm(name=f'glu_layer_norm_{i}')(x)
            
            dropout_rate = getattr(self.config, 'dropout_rate', 0.0)
            if dropout_rate > 0:
                x = nn.Dropout(rate=dropout_rate, deterministic=not training)(x)
        
        return x
    
    def _quadratic_forward(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        """Quadratic ResNet architecture forward pass."""
        # Input projection
        if len(self.config.hidden_sizes) > 0:
            x = nn.Dense(self.config.hidden_sizes[0], name='quad_input_proj')(x)
        else:
            x = nn.Dense(64, name='quad_input_proj')(x)
        
        # Quadratic residual blocks
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            x = self._quadratic_residual_block(x, hidden_size, i, training)
        
        return x
    
    def _quadratic_residual_block(self, x: jnp.ndarray, hidden_size: int, 
                                 block_idx: int, training: bool) -> jnp.ndarray:
        """Single quadratic residual block."""
        # Store input for residual connection
        residual = x
        if residual.shape[-1] != hidden_size:
            residual = nn.Dense(hidden_size, name=f'quad_residual_proj_{block_idx}')(residual)
        
        # Linear transformation
        linear_out = nn.Dense(hidden_size, name=f'quad_linear_{block_idx}')(x)
        linear_out = nn.swish(linear_out)
        
        # Quadratic transformation with smaller initialization
        quadratic_out = nn.Dense(hidden_size, 
                                kernel_init=nn.initializers.normal(stddev=0.01),
                                name=f'quad_quadratic_{block_idx}')(x)
        quadratic_out = nn.swish(quadratic_out)
        
        # Combine: y = residual + Ax + (Bx)x (updated formula)
        output = residual + linear_out - residual * quadratic_out
        
        # Layer normalization
        if getattr(self.config, 'use_layer_norm', True):
            output = nn.LayerNorm(name=f'quad_layer_norm_{block_idx}')(output)
        
        return output


class LogZTrainer(BaseTrainer):
    """
    Trainer for LogZ Networks with gradient/Hessian computation.
    
    This trainer specializes in training networks that output log normalizers,
    using automatic differentiation to compute gradients (mean) and Hessians
    (covariance) for loss computation.
    """
    
    def __init__(self, config: FullConfig, architecture: str = "mlp", 
                 hessian_method: str = 'diagonal', adaptive_weights: bool = True):
        model = LogZNetwork(config=config.network, architecture=architecture)
        super().__init__(model, config)
        self.architecture = architecture
        self.hessian_method = hessian_method
        self.adaptive_weights = adaptive_weights
        self.epoch = 0
    
    def compute_gradient(self, params: Dict, eta: jnp.ndarray) -> jnp.ndarray:
        """Compute gradient of log normalizer (mean of sufficient statistics)."""
        def single_log_normalizer(eta_single):
            return self.model.apply(params, eta_single[None, :], training=False)[0]
        
        # Compute gradient for each sample
        gradients = jax.vmap(jax.grad(single_log_normalizer))(eta)
        return gradients
    
    def compute_hessian(self, params: Dict, eta: jnp.ndarray) -> jnp.ndarray:
        """Compute Hessian of log normalizer (covariance of sufficient statistics)."""
        def single_log_normalizer(eta_single):
            return self.model.apply(params, eta_single[None, :], training=False)[0]
        
        if self.hessian_method == 'diagonal':
            # Only compute diagonal elements for efficiency
            def hessian_diag(eta_single):
                hess = jax.hessian(single_log_normalizer)(eta_single)
                return jnp.diag(hess)
            hessians = jax.vmap(hessian_diag)(eta)
            return hessians
        elif self.hessian_method == 'full':
            # Compute full Hessian (expensive)
            hessians = jax.vmap(jax.hessian(single_log_normalizer))(eta)
            return hessians
        else:
            raise ValueError(f"Unknown hessian_method: {self.hessian_method}")
    
    def logZ_loss_fn(self, params: Dict, eta: jnp.ndarray, 
                     target_mean: jnp.ndarray, target_cov: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Compute loss based on log normalizer derivatives.
        
        Args:
            params: Model parameters
            eta: Natural parameters
            target_mean: Target mean of sufficient statistics
            target_cov: Target covariance of sufficient statistics (optional)
            
        Returns:
            Loss value
        """
        # Compute network predictions
        predicted_mean = self.compute_gradient(params, eta)
        
        # Mean loss
        mean_loss = jnp.mean((predicted_mean - target_mean) ** 2)
        
        total_loss = mean_loss
        
        # Add covariance loss if target covariance is provided
        if target_cov is not None:
            predicted_cov = self.compute_hessian(params, eta)
            
            if self.hessian_method == 'diagonal':
                # Only compare diagonal elements
                cov_loss = jnp.mean((predicted_cov - jnp.diag(target_cov)) ** 2)
            else:
                # Compare full covariance matrices
                cov_loss = jnp.mean((predicted_cov - target_cov) ** 2)
            
            # Weight covariance loss
            cov_weight = 0.1 if not self.adaptive_weights else max(0.01, 1.0 / (1.0 + 0.1 * self.epoch))
            total_loss += cov_weight * cov_loss
        
        return total_loss
    
    def train_step(self, params: Dict, opt_state: Any, batch: Dict[str, jnp.ndarray], 
                   optimizer: Any) -> Tuple[Dict, Any, float]:
        """Single training step with epoch tracking."""
        loss_value, grads = jax.value_and_grad(self.logZ_loss_fn)(
            params, batch['eta'], batch['mean'], batch.get('cov')
        )
        
        # Gradient clipping for stability
        grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, float(loss_value)
    
    def train(self, train_data: Dict[str, jnp.ndarray], 
              val_data: Optional[Dict[str, jnp.ndarray]] = None,
              epochs: int = 300, learning_rate: float = 1e-3) -> Tuple[Dict, Dict]:
        """Train the LogZ network."""
        # Initialize model
        rng = random.PRNGKey(42)
        params = self.model.init(rng, train_data['eta'][:1])
        
        # Optimizer
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)
        
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        best_params = params
        best_loss = float('inf')
        
        print(f"Training LogZ Network ({self.architecture}) for {epochs} epochs")
        print(f"Hessian method: {self.hessian_method}")
        print(f"Adaptive weights: {self.adaptive_weights}")
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Training step
            train_loss = 0.0
            batch_size = 32
            n_train = train_data['eta'].shape[0]
            
            # Mini-batch training
            for i in range(0, n_train, batch_size):
                batch = {
                    'eta': train_data['eta'][i:i+batch_size],
                    'mean': train_data['mean'][i:i+batch_size],
                    'cov': train_data.get('cov', train_data['mean'][i:i+batch_size])  # Fallback
                }
                
                params, opt_state, batch_loss = self.train_step(params, opt_state, batch, optimizer)
                train_loss += batch_loss
            
            train_loss /= (n_train // batch_size)
            history['train_loss'].append(train_loss)
            
            # Validation
            if val_data is not None:
                val_loss = self.logZ_loss_fn(params, val_data['eta'], 
                                           val_data['mean'], val_data.get('cov'))
                history['val_loss'].append(float(val_loss))
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = params
            
            # Progress reporting
            if epoch % 50 == 0 or epoch < 10:
                if val_data is not None:
                    print(f"  Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
                else:
                    print(f"  Epoch {epoch}: Train Loss = {train_loss:.6f}")
        
        return best_params, history
    
    def predict(self, params: Dict, eta: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Make predictions using the trained model."""
        predicted_mean = self.compute_gradient(params, eta)
        predicted_cov = self.compute_hessian(params, eta)
        
        return {
            'mean': predicted_mean,
            'covariance': predicted_cov
        }
    
    def evaluate(self, params: Dict, test_data: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """Evaluate the model on test data."""
        predictions = self.predict(params, test_data['eta'])
        
        # Mean evaluation
        mean_mse = jnp.mean((predictions['mean'] - test_data['mean']) ** 2)
        mean_mae = jnp.mean(jnp.abs(predictions['mean'] - test_data['mean']))
        
        results = {
            'mean_mse': float(mean_mse),
            'mean_mae': float(mean_mae)
        }
        
        # Covariance evaluation if available
        if 'cov' in test_data:
            if self.hessian_method == 'diagonal':
                cov_mse = jnp.mean((predictions['covariance'] - jnp.diag(test_data['cov'])) ** 2)
            else:
                cov_mse = jnp.mean((predictions['covariance'] - test_data['cov']) ** 2)
            
            results['cov_mse'] = float(cov_mse)
        
        return results


def create_logZ_network_and_trainer(config: FullConfig, architecture: str = "mlp") -> LogZTrainer:
    """Factory function to create LogZ network and trainer."""
    return LogZTrainer(config, architecture=architecture)
