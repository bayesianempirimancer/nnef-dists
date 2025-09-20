"""
Convex Neural Network for log normalizer A(η).

This module implements an Input Convex Neural Network (ICNN) that parameterizes
convex functions, ensuring that the learned log normalizer A(η) maintains 
convexity properties essential for exponential family distributions.

Key features:
- Non-negative weights in hidden layers to maintain convexity
- Convex activation functions (ReLU, Softplus)
- Skip connections from input to all hidden layers
- Guaranteed convex output with respect to input η

Theoretical foundation:
- For exponential family: p(x|η) = h(x) exp(ηᵀT(x) - A(η))
- A(η) must be convex to ensure valid probability distribution
- E[T(X)] = ∇A(η) (gradient of log normalizer)
- Cov[T(X)] = ∇²A(η) (Hessian of log normalizer, positive definite)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import random, grad, hessian, jacfwd
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import optax
import time

from ..base_model import BaseNeuralNetwork, BaseTrainer
from ..config import FullConfig
from ..ef import ExponentialFamily


class ConvexLayer(nn.Module):
    """
    A single layer in the Input Convex Neural Network.
    
    Maintains convexity by:
    1. Non-negative weights from previous layer
    2. Skip connections from input with unrestricted weights
    3. Convex activation function
    """
    
    hidden_size: int
    use_bias: bool = True
    activation: str = "relu"
    
    @nn.compact
    def __call__(self, z_prev: jnp.ndarray, x_input: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through convex layer.
        
        Args:
            z_prev: Output from previous layer [batch_size, prev_hidden_size]
            x_input: Original input (skip connection) [batch_size, input_dim]
            training: Whether in training mode
            
        Returns:
            Layer output [batch_size, hidden_size]
        """
        # Non-negative weights from previous layer (maintains convexity)
        if z_prev is not None:
            W_z = self.param('W_z', 
                           nn.initializers.uniform(scale=0.1), 
                           (z_prev.shape[-1], self.hidden_size))
            # Ensure non-negative weights for convexity
            W_z_nonneg = nn.softplus(W_z)  # Always positive
            z_term = jnp.dot(z_prev, W_z_nonneg)
        else:
            z_term = 0.0
        
        # Skip connection from input (unrestricted weights)
        W_x = self.param('W_x',
                        nn.initializers.xavier_uniform(),
                        (x_input.shape[-1], self.hidden_size))
        x_term = jnp.dot(x_input, W_x)
        
        # Bias term
        if self.use_bias:
            b = self.param('b', nn.initializers.zeros, (self.hidden_size,))
            output = z_term + x_term + b
        else:
            output = z_term + x_term
        
        # Apply convex activation function
        if self.activation == "relu":
            output = nn.relu(output)
        elif self.activation == "softplus":
            output = nn.softplus(output)
        elif self.activation == "elu":
            # ELU is convex for x >= 0, approximately convex overall
            output = nn.elu(output)
        elif self.activation == "leaky_relu":
            output = nn.leaky_relu(output, negative_slope=0.01)
        elif self.activation == "tanh":
            # Fall back to ReLU for convexity (tanh is not convex)
            output = nn.relu(output)
        elif self.activation == "swish":
            # Fall back to ReLU for convexity (swish is not convex)
            output = nn.relu(output)
        elif self.activation == "gelu":
            # Fall back to ReLU for convexity (gelu is not convex)
            output = nn.relu(output)
        else:
            # Default to ReLU for any unknown activation
            output = nn.relu(output)
        
        return output


class ConvexNeuralNetworkLogZ(BaseNeuralNetwork):
    """
    Input Convex Neural Network for learning log normalizer A(η).
    
    Architecture ensures that the output is convex with respect to input η,
    which is essential for the log normalizer of exponential families.
    
    Architecture:
    1. Input layer with unrestricted weights
    2. Multiple convex layers with non-negative weights and skip connections
    3. Final output layer with non-negative weights (scalar output)
    """
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through convex neural network.
        
        Args:
            eta: Natural parameters [batch_size, eta_dim]
            training: Whether in training mode
            
        Returns:
            Log normalizer values [batch_size,]
        """
        # Apply feature engineering if enabled - use convex_only features for convex networks
        if hasattr(self.config, 'use_feature_engineering') and self.config.use_feature_engineering:
            from ..eta_features import compute_eta_features
            x_input = compute_eta_features(eta, method='convex_only')
        else:
            x_input = eta
            
        batch_size = eta.shape[0]
        
        # Store original input for skip connections
        original_input = x_input
        
        # First layer (no previous z, only input)
        if len(self.config.hidden_sizes) > 0:
            z = ConvexLayer(
                hidden_size=self.config.hidden_sizes[0],
                activation=self.config.activation,
                name='convex_layer_0'
            )(None, original_input, training=training)
        else:
            # If no hidden layers, go directly to output
            z = original_input
        
        # Hidden convex layers with skip connections
        for i, hidden_size in enumerate(self.config.hidden_sizes[1:], 1):
            z = ConvexLayer(
                hidden_size=hidden_size,
                activation=self.config.activation,
                name=f'convex_layer_{i}'
            )(z, original_input, training=training)
        
        # Final output layer (scalar, non-negative weights to maintain convexity)
        if len(self.config.hidden_sizes) > 0:
            # Non-negative weights for final layer
            W_final_z = self.param('W_final_z',
                                 nn.initializers.uniform(scale=0.1),
                                 (z.shape[-1], 1))
            W_final_z_nonneg = nn.softplus(W_final_z)
            
            # Skip connection from input (unrestricted)
            W_final_x = self.param('W_final_x',
                                 nn.initializers.xavier_uniform(),
                                 (original_input.shape[-1], 1))
            
            # Final bias
            b_final = self.param('b_final', nn.initializers.zeros, (1,))
            
            # Combine terms
            log_normalizer = (jnp.dot(z, W_final_z_nonneg) + 
                            jnp.dot(original_input, W_final_x) + 
                            b_final)
        else:
            # Direct linear mapping if no hidden layers
            W_direct = self.param('W_direct',
                                nn.initializers.xavier_uniform(),
                                (original_input.shape[-1], 1))
            b_direct = self.param('b_direct', nn.initializers.zeros, (1,))
            log_normalizer = jnp.dot(original_input, W_direct) + b_direct
        
        # Ensure scalar output
        return jnp.squeeze(log_normalizer, axis=-1)


def convex_gradient_computation(model, params, eta, eps=1e-8):
    """
    Compute gradient of convex log normalizer w.r.t. natural parameters.
    
    Args:
        model: ConvexNeuralNetworkLogZ
        params: Model parameters
        eta: Natural parameters [batch_size, eta_dim]
        eps: Small regularization parameter
        
    Returns:
        Gradient (mean) [batch_size, eta_dim]
    """
    def single_log_normalizer(eta_single):
        return model.apply(params, eta_single[None, :], training=False)[0]
    
    # Compute gradient using automatic differentiation
    grad_fn = grad(single_log_normalizer)
    gradients = jax.vmap(grad_fn)(eta)
    
    # The convex property should naturally provide stable gradients,
    # but we can still add small clipping for numerical safety
    gradients = jnp.clip(gradients, -50.0, 50.0)
    
    return gradients


def convex_hessian_computation(model, params, eta, method='diagonal', eps=1e-8):
    """
    Compute Hessian of convex log normalizer w.r.t. natural parameters.
    
    For convex functions, the Hessian should be positive semi-definite.
    
    Args:
        model: ConvexNeuralNetworkLogZ
        params: Model parameters
        eta: Natural parameters [batch_size, eta_dim]
        method: 'diagonal', 'full'
        eps: Regularization parameter
        
    Returns:
        Hessian (covariance) [batch_size, eta_dim] or [batch_size, eta_dim, eta_dim]
    """
    def single_log_normalizer(eta_single):
        return model.apply(params, eta_single[None, :], training=False)[0]
    
    if method == 'diagonal':
        # Compute diagonal of Hessian using forward-mode AD
        def diagonal_hessian_fn(eta_single):
            hess_diag = jnp.diag(hessian(single_log_normalizer)(eta_single))
            return hess_diag
        
        hessians = jax.vmap(diagonal_hessian_fn)(eta)
        
        # For convex functions, Hessian diagonal should be non-negative
        # Add small regularization to ensure positive definiteness
        hessians = jnp.maximum(hessians, eps)
        
        return hessians
        
    elif method == 'full':
        # Full Hessian computation
        hess_fn = hessian(single_log_normalizer)
        hessians = jax.vmap(hess_fn)(eta)
        
        # For convex functions, Hessian should be positive semi-definite
        # Add regularization to diagonal to ensure positive definiteness
        eye = jnp.eye(hessians.shape[-1])
        hessians = hessians + eps * eye[None, :, :]
        
        return hessians
    
    else:
        raise ValueError(f"Unknown Hessian method: {method}")


def convex_log_normalizer_loss_fn(model, params, eta_batch, mean_batch, cov_batch=None,
                                 loss_weights=None, hessian_method='diagonal',
                                 regularization=1e-6, epoch=0):
    """
    Loss function for convex neural network log normalizer.
    
    Includes convexity-aware regularization and stability measures.
    """
    if loss_weights is None:
        loss_weights = {'mean_weight': 1.0, 'cov_weight': 0.1}
    
    # Compute network predictions
    log_normalizer = model.apply(params, eta_batch, training=True)
    
    # Compute gradient (mean)
    network_mean = convex_gradient_computation(model, params, eta_batch, regularization)
    
    # Mean loss
    mean_diff = network_mean - mean_batch
    mean_loss = jnp.mean(jnp.square(mean_diff))
    
    total_loss = loss_weights['mean_weight'] * mean_loss
    
    # Covariance loss (if requested and data available)
    if cov_batch is not None and loss_weights['cov_weight'] > 0:
        try:
            # Compute Hessian (covariance)
            network_hessian = convex_hessian_computation(
                model, params, eta_batch, method=hessian_method, eps=regularization
            )
            
            if hessian_method == 'diagonal':
                # For diagonal approximation
                empirical_diag = jnp.diagonal(cov_batch, axis1=1, axis2=2)
                cov_loss = jnp.mean(jnp.square(network_hessian - empirical_diag))
            else:
                # For full Hessian
                cov_loss = jnp.mean(jnp.square(network_hessian - cov_batch))
            
            total_loss += loss_weights['cov_weight'] * cov_loss
            
        except Exception as e:
            print(f"Warning: Convex Hessian computation failed: {e}")
            pass
    
    # Convexity-preserving regularization
    # Add small L2 regularization on log normalizer values
    convexity_reg = regularization * jnp.mean(jnp.square(log_normalizer))
    total_loss += convexity_reg
    
    return total_loss


class ConvexNeuralNetworkLogZTrainer(BaseTrainer):
    """Trainer for Convex Neural Network Log Normalizer."""
    
    def __init__(self, config: FullConfig, hessian_method='diagonal', 
                 loss_weights=None, use_curriculum=True):
        model = ConvexNeuralNetworkLogZ(config=config.network)
        super().__init__(model, config)
        self.hessian_method = hessian_method
        self.loss_weights = loss_weights or {'mean_weight': 1.0, 'cov_weight': 0.1}
        self.use_curriculum = use_curriculum
        self.epoch = 0
    
    def loss_fn(self, params: Dict, batch: Dict[str, jnp.ndarray], training: bool = True) -> jnp.ndarray:
        """Compute convex log normalizer loss for a batch."""
        # Use curriculum learning if enabled
        if self.use_curriculum:
            if self.epoch < 30:
                current_weights = {'mean_weight': 1.0, 'cov_weight': 0.0}
            elif self.epoch < 80:
                progress = (self.epoch - 30) / 50
                current_weights = {'mean_weight': 1.0, 'cov_weight': 0.1 * progress}
            else:
                current_weights = {'mean_weight': 1.0, 'cov_weight': 0.1}
        else:
            current_weights = self.loss_weights
        
        return convex_log_normalizer_loss_fn(
            self.model, params,
            batch['eta'], batch['mean'],
            batch.get('cov'),
            loss_weights=current_weights,
            hessian_method=self.hessian_method,
            epoch=self.epoch
        )
    
    def train_step(self, params: Dict, opt_state: Any, batch: Dict[str, jnp.ndarray],
                   optimizer: Any) -> Tuple[Dict, Any, float]:
        """Single training step with convexity-aware gradient processing."""
        loss_value, grads = jax.value_and_grad(self.loss_fn)(params, batch, training=True)
        
        # Moderate gradient clipping (convex functions should have stable gradients)
        grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        # Ensure non-negative weights are maintained after update
        # (This is handled by the softplus activation in the layers)
        
        return params, opt_state, loss_value
    
    def train(self, train_data: Dict[str, jnp.ndarray], val_data: Dict[str, jnp.ndarray]) -> Tuple[Dict, Dict]:
        """Training loop with convex-specific optimizations."""
        # Initialize
        sample_input = train_data['eta'][:1]
        params, opt_state = self.initialize_model(sample_input)
        optimizer = self.create_optimizer()
        
        # Training state
        best_val_loss = float('inf')
        patience_counter = 0
        best_params = params
        history = {'train_loss': [], 'val_loss': [], 'mean_loss': [], 'cov_loss': []}
        
        tc = self.config.training
        batch_size = tc.batch_size
        n_train = train_data['eta'].shape[0]
        
        print(f"Training Convex Neural Network LogZ for {tc.num_epochs} epochs")
        print(f"Hessian method: {self.hessian_method}")
        print(f"Curriculum learning: {self.use_curriculum}")
        print(f"Architecture: {self.config.network.hidden_sizes}")
        print(f"Parameters: {self.model.get_parameter_count(params):,}")
        
        start_time = time.time()
        
        for epoch in range(tc.num_epochs):
            self.epoch = epoch
            
            # Shuffle training data
            self.rng, shuffle_rng = random.split(self.rng)
            perm = random.permutation(shuffle_rng, n_train)
            
            train_losses = []
            mean_losses = []
            cov_losses = []
            
            # Training batches
            for i in range(0, n_train, batch_size):
                batch_idx = perm[i:i + batch_size]
                batch = {
                    'eta': train_data['eta'][batch_idx],
                    'mean': train_data['mean'][batch_idx],
                    'cov': train_data.get('cov', None)
                }
                
                params, opt_state, loss = self.train_step(params, opt_state, batch, optimizer)
                train_losses.append(loss)
                
                # Track individual loss components
                network_mean = convex_gradient_computation(self.model, params, batch['eta'])
                mean_loss = float(jnp.mean(jnp.square(network_mean - batch['mean'])))
                mean_losses.append(mean_loss)
                
                if batch.get('cov') is not None and epoch >= 30:  # Only after curriculum warmup
                    try:
                        hessian_pred = convex_hessian_computation(
                            self.model, params, batch['eta'], self.hessian_method
                        )
                        if self.hessian_method == 'diagonal':
                            empirical_diag = jnp.diagonal(batch['cov'], axis1=1, axis2=2)
                            cov_loss = float(jnp.mean(jnp.square(hessian_pred - empirical_diag)))
                        else:
                            cov_loss = float(jnp.mean(jnp.square(hessian_pred - batch['cov'])))
                        cov_losses.append(cov_loss)
                    except:
                        cov_losses.append(0.0)
                else:
                    cov_losses.append(0.0)
            
            avg_train_loss = float(jnp.mean(jnp.array(train_losses)))
            avg_mean_loss = float(jnp.mean(jnp.array(mean_losses)))
            avg_cov_loss = float(jnp.mean(jnp.array(cov_losses)))
            
            history['train_loss'].append(avg_train_loss)
            history['mean_loss'].append(avg_mean_loss)
            history['cov_loss'].append(avg_cov_loss)
            
            # Validation
            if epoch % tc.validation_freq == 0:
                val_batch = {
                    'eta': val_data['eta'],
                    'mean': val_data['mean'],
                    'cov': val_data.get('cov', None)
                }
                val_loss = float(self.loss_fn(params, val_batch, training=False))
                history['val_loss'].append(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss - tc.min_delta:
                    best_val_loss = val_loss
                    best_params = params
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if tc.early_stopping and patience_counter >= tc.patience:
                    print(f"    Early stopping at epoch {epoch}")
                    break
                
                if epoch % (tc.validation_freq * 2) == 0:
                    # Get current curriculum weights
                    if self.use_curriculum:
                        if epoch < 30:
                            current_weights = {'mean_weight': 1.0, 'cov_weight': 0.0}
                        elif epoch < 80:
                            progress = (epoch - 30) / 50
                            current_weights = {'mean_weight': 1.0, 'cov_weight': 0.1 * progress}
                        else:
                            current_weights = {'mean_weight': 1.0, 'cov_weight': 0.1}
                    else:
                        current_weights = self.loss_weights
                    
                    print(f"    Epoch {epoch:3d}: Train={avg_train_loss:.4f}, Val={val_loss:.4f}, "
                          f"Mean={avg_mean_loss:.4f}, Cov={avg_cov_loss:.4f}, "
                          f"Weights={current_weights}")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.1f}s")
        
        return best_params, history
    
    def evaluate_with_derivatives(self, params: Dict, test_data: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """Evaluate convex model by comparing derivatives to empirical statistics."""
        # Get predictions
        log_normalizer = self.model.apply(params, test_data['eta'], training=False)
        
        # Compute derivatives
        network_mean = convex_gradient_computation(self.model, params, test_data['eta'])
        
        # Mean metrics
        mean_mse = float(jnp.mean(jnp.square(network_mean - test_data['mean'])))
        mean_mae = float(jnp.mean(jnp.abs(network_mean - test_data['mean'])))
        
        metrics = {
            'mean_mse': mean_mse,
            'mean_mae': mean_mae,
            'log_normalizer_std': float(jnp.std(log_normalizer))
        }
        
        # Covariance metrics (if available)
        if 'cov' in test_data:
            try:
                network_hessian = convex_hessian_computation(
                    self.model, params, test_data['eta'], method=self.hessian_method
                )
                
                if self.hessian_method == 'diagonal':
                    empirical_diag = jnp.diagonal(test_data['cov'], axis1=1, axis2=2)
                    cov_mse = float(jnp.mean(jnp.square(network_hessian - empirical_diag)))
                    cov_mae = float(jnp.mean(jnp.abs(network_hessian - empirical_diag)))
                else:
                    cov_mse = float(jnp.mean(jnp.square(network_hessian - test_data['cov'])))
                    cov_mae = float(jnp.mean(jnp.abs(network_hessian - test_data['cov'])))
                
                metrics.update({
                    'cov_mse': cov_mse,
                    'cov_mae': cov_mae
                })
                
            except Exception as e:
                print(f"Warning: Could not compute covariance metrics: {e}")
        
        return metrics


def create_convex_nn_log_normalizer_trainer(config: FullConfig,
                                          hessian_method='diagonal',
                                          loss_weights=None,
                                          use_curriculum=True) -> ConvexNeuralNetworkLogZTrainer:
    """Factory function to create convex neural network log normalizer trainer."""
    return ConvexNeuralNetworkLogZTrainer(
        config, hessian_method, loss_weights, use_curriculum
    )


# Example usage and testing functions

def test_convex_nn_on_gaussian():
    """Test convex neural network on 1D Gaussian case."""
    from ..ef import GaussianNatural1D
    
    # Create simple test case
    ef = GaussianNatural1D()
    
    # Test parameters
    eta_test = jnp.array([[1.0, -0.5], [2.0, -1.0]])  # Natural parameters for Gaussian
    
    # Create a convex neural network model
    config = FullConfig()
    config.network.hidden_sizes = [32, 16]
    config.network.output_dim = 1  # Log normalizer is scalar
    config.network.activation = "relu"  # Convex activation
    
    model = ConvexNeuralNetworkLogZ(config=config.network)
    
    # Initialize
    rng = random.PRNGKey(42)
    params = model.init(rng, eta_test)
    
    print(f"Convex Neural Network parameters: {sum(x.size for x in jax.tree_leaves(params))}")
    
    # Test forward pass
    log_norm = model.apply(params, eta_test)
    print(f"Log normalizer outputs: {log_norm}")
    
    # Test gradient computation
    mean_pred = convex_gradient_computation(model, params, eta_test)
    print(f"Predicted mean: {mean_pred}")
    
    # Test Hessian computation
    hessian_pred = convex_hessian_computation(model, params, eta_test, method='diagonal')
    print(f"Predicted Hessian (diagonal): {hessian_pred}")
    
    return model, params


if __name__ == "__main__":
    test_convex_nn_on_gaussian()
