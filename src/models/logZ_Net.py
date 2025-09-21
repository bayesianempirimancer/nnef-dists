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
from tqdm import tqdm

from ..base_model import BaseNeuralNetwork, BaseTrainer
from ..config import FullConfig, NetworkConfig
from ..ef import ExponentialFamily


class LogZNetwork(nn.Module):
    """
    Unified LogZ Network for learning log normalizers.
    
    This network can be configured to use different architectures:
    - MLP: Standard multi-layer perceptron
    - GLU: Gated linear unit architecture
    - Quadratic: Quadratic ResNet architecture
    
    The network outputs a scalar log normalizer A(η) whose derivatives
    provide the moments of the exponential family distribution.
    """
    
    config: NetworkConfig
    architecture: str = "mlp"  # Default architecture
    
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
    
    def compute_internal_loss(self, params: Dict, eta: jnp.ndarray, 
                            predicted_mean: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Compute model-specific internal losses (e.g., regularization, smoothness penalties).
        
        Base implementation returns 0. Subclasses can override to add specific penalties.
        
        Args:
            params: Model parameters
            eta: Natural parameters [batch_size, eta_dim]
            predicted_mean: Predicted mean statistics [batch_size, mu_dim] 
            training: Whether in training mode
            
        Returns:
            Internal loss value (scalar)
        """
        return 0.0
    
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
    
    def get_parameter_count(self, params: Dict) -> int:
        """Count total number of parameters."""
        return sum(x.size for x in jax.tree_leaves(params))


class LogZTrainer(BaseTrainer):
    """
    Trainer for LogZ Networks with gradient/Hessian computation.
    
    This trainer specializes in training networks that output log normalizers,
    using automatic differentiation to compute gradients (mean) and Hessians
    (covariance) for loss computation.
    """
    
    def __init__(self, config: FullConfig, architecture: str = "mlp", 
                 loss_type: str = "mse_mean_only", hessian_method: str = 'diagonal', 
                 adaptive_weights: bool = True, l1_reg_weight: float = 1e-4):
        model = LogZNetwork(config=config.network)
        super().__init__(model, config)
        self.architecture = architecture
        self.loss_type = loss_type
        self.adaptive_weights = adaptive_weights
        self.l1_reg_weight = l1_reg_weight
        self.epoch = 0
        
        # Determine if we need Hessian computation based on loss type
        self.needs_hessian = loss_type in ["mse_mean_and_cov", "mse_mean_and_diag_cov", "KLqp", "KLqp_diag", "KLpq", "KLpq_diag"]
        self.hessian_method = "diagonal" if loss_type in ["mse_mean_and_diag_cov", "KLqp_diag", "KLpq_diag"] else "full"
        # Precompiled functions for efficiency (will be set after model initialization)
        self._compiled_gradient_fn = None
        self._compiled_hessian_fn = None
    
    def _compile_functions(self):
        """Precompile gradient and Hessian functions for efficiency."""
        # Precompile the function structure (not bound to specific params)
        def batch_gradient_fn(params, eta_batch):
            def single_log_normalizer(eta_single):
                return self.model.apply(params, eta_single[None, :], training=False)[0]
            return jax.vmap(jax.grad(single_log_normalizer))(eta_batch)
        
        self._compiled_gradient_fn = jax.jit(batch_gradient_fn)
        
        # Precompile Hessian function based on method
        if self.hessian_method == 'diagonal':
            def batch_hessian_diag_fn(params, eta_batch):
                def single_log_normalizer(eta_single):
                    return self.model.apply(params, eta_single[None, :], training=False)[0]
                def diagonal_hessian_fn(eta_single):
                    grad_fn = jax.grad(single_log_normalizer)
                    return jnp.diag(jax.jacfwd(grad_fn)(eta_single))
                return jax.vmap(diagonal_hessian_fn)(eta_batch)
            self._compiled_hessian_fn = jax.jit(batch_hessian_diag_fn)
        else:
            def batch_hessian_full_fn(params, eta_batch):
                def single_log_normalizer(eta_single):
                    return self.model.apply(params, eta_single[None, :], training=False)[0]
                return jax.vmap(jax.hessian(single_log_normalizer))(eta_batch)
            self._compiled_hessian_fn = jax.jit(batch_hessian_full_fn)
    
    def compute_gradient(self, params: Dict, eta: jnp.ndarray) -> jnp.ndarray:
        """Compute gradient of log normalizer (mean of sufficient statistics)."""
        # Use precompiled function if available, otherwise compile on first use
        if self._compiled_gradient_fn is None:
            self._compile_functions()
        
        return self._compiled_gradient_fn(params, eta)
    
    def compute_hessian(self, params: Dict, eta: jnp.ndarray) -> jnp.ndarray:
        """Compute Hessian of log normalizer (covariance of sufficient statistics)."""
        # Use precompiled function if available, otherwise compile on first use
        if self._compiled_hessian_fn is None:
            self._compile_functions()
        
        return self._compiled_hessian_fn(params, eta)
    
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

        # Always compute mean prediction and loss
        predicted_mean = self.compute_gradient(params, eta)
        mean_loss = jnp.mean((predicted_mean - target_mean) ** 2)
        total_loss = mean_loss
        
        # Add covariance loss based on loss type
        if self.loss_type == "mse_mean_only":
            # Only use mean loss - no Hessian needed
            pass
        elif self.loss_type in ["mse_mean_and_cov", "mse_mean_and_diag_cov"]:             
            if target_cov is None:
                raise ValueError("Target covariance is required for mse_mean_and_cov or mse_mean_and_diag_cov loss type")
            # Compute covariance loss
            predicted_cov = self.compute_hessian(params, eta)
            
            if self.hessian_method == 'diagonal':
                # Only compare diagonal elements
                if target_cov.ndim == 3:  # Full covariance matrix [batch, dim, dim]
                    target_diag = jnp.diagonal(target_cov, axis1=-2, axis2=-1)
                else:  # Already diagonal [batch, dim]
                    target_diag = target_cov
                cov_loss = jnp.mean((predicted_cov - target_diag) ** 2)
            else:
                # Compare full covariance matrices
                cov_loss = jnp.mean((predicted_cov - target_cov) ** 2)
            
            # Weight covariance loss
            cov_weight = 0.1 if not self.adaptive_weights else max(0.01, 1.0 / (1.0 + 0.1 * self.epoch))
            total_loss += cov_weight * cov_loss
        
        # Add model-specific internal losses (e.g., smoothness penalties, regularization)
        internal_loss = self.model.compute_internal_loss(params, eta, predicted_mean, training=True)
        total_loss += internal_loss
        
        # Add L1 regularization if enabled
        if self.l1_reg_weight > 0.0:
            l1_reg = 0.0
            for param in jax.tree_leaves(params):
                l1_reg += jnp.sum(jnp.abs(param))
            total_loss += self.l1_reg_weight * l1_reg
        
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
        print(f"Loss type: {self.loss_type}")
        if self.needs_hessian:
            print(f"Hessian method: {self.hessian_method}")
        print(f"Adaptive weights: {self.adaptive_weights}")
        
        # Start timing
        start_time = time.time()
        
        # Training loop with progress bar
        with tqdm(range(epochs), desc="Training", unit="epoch") as pbar:
            for epoch in pbar:
                self.epoch = epoch
                
                # Training step
                train_loss = 0.0
                batch_size = 32
                n_train = train_data['eta'].shape[0]
                
                # Mini-batch training
                for i in range(0, n_train, batch_size):
                    batch = {
                        'eta': train_data['eta'][i:i+batch_size],
                        'mean': train_data['mean'][i:i+batch_size]
                    }
                    # Only add covariance if it exists in training data
                    if 'cov' in train_data:
                        batch['cov'] = train_data['cov'][i:i+batch_size]
                    
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
                    
                    # Update progress bar
                    pbar.set_postfix({'Train Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}'})
                else:
                    pbar.set_postfix({'Train Loss': f'{train_loss:.4f}'})
                
                # Detailed progress reporting
                if epoch % 50 == 0 or epoch < 10:
                    if val_data is not None:
                        print(f"  Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
                    else:
                        print(f"  Epoch {epoch}: Train Loss = {train_loss:.6f}")
        
        # Calculate total training time
        total_training_time = time.time() - start_time
        print(f"\n✓ Training completed in {total_training_time:.1f}s")
        
        # Add timing to history
        history['total_training_time'] = total_training_time
        
        return best_params, history
    
    def predict(self, params: Dict, eta: jnp.ndarray, compute_covariance: bool = False) -> Dict[str, jnp.ndarray]:
        """Make predictions using the trained model."""
        predicted_mean = self.compute_gradient(params, eta)
        
        result = {'mean': predicted_mean}
        
        if compute_covariance:
            predicted_cov = self.compute_hessian(params, eta)
            result['covariance'] = predicted_cov
        
        return result
    
    def evaluate(self, params: Dict, test_data: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """Evaluate the model on test data."""
        # Only compute covariance if test data has covariance information
        compute_cov = 'cov' in test_data
        predictions = self.predict(params, test_data['eta'], compute_covariance=compute_cov)
        
        # Mean evaluation
        mean_mse = jnp.mean((predictions['mean'] - test_data['mean']) ** 2)
        mean_mae = jnp.mean(jnp.abs(predictions['mean'] - test_data['mean']))
        
        results = {
            'mean_mse': float(mean_mse),
            'mean_mae': float(mean_mae)
        }
        
        # Covariance evaluation if available
        if compute_cov:
            if self.hessian_method == 'diagonal':
                if test_data['cov'].ndim == 3:  # Full covariance matrix
                    target_diag = jnp.diagonal(test_data['cov'], axis1=1, axis2=2)
                else:  # Already diagonal
                    target_diag = test_data['cov']
                cov_mse = jnp.mean((predictions['covariance'] - target_diag) ** 2)
            else:
                cov_mse = jnp.mean((predictions['covariance'] - test_data['cov']) ** 2)
            
            results['cov_mse'] = float(cov_mse)
        
        return results


def create_logZ_network_and_trainer(config: FullConfig, architecture: str = "mlp", l1_reg_weight: float = 1e-4) -> LogZTrainer:
    """Factory function to create LogZ network and trainer."""
    return LogZTrainer(config, architecture=architecture, l1_reg_weight=l1_reg_weight)
