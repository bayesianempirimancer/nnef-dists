"""
Numerically stable implementation of log normalizer networks.

This module extends the basic log normalizer approach with additional
numerical stability measures for gradient and Hessian computations.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import random, grad, hessian, jacfwd
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import optax
import time

from .base_model import BaseNeuralNetwork, BaseTrainer
from .config import FullConfig
from .ef import ExponentialFamily


class LogNormalizerNetwork(BaseNeuralNetwork):
    """Basic log normalizer network that outputs a scalar log normalizer."""
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through log normalizer network.
        
        Args:
            eta: Natural parameters [batch_size, eta_dim]
            training: Whether in training mode
            
        Returns:
            Log normalizer values [batch_size,]
        """
        x = eta
        
        # Hidden layers
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            x = nn.Dense(hidden_size, name=f'hidden_{i}')(x)
            x = nn.swish(x)
            if getattr(self.config, 'use_layer_norm', True):
                x = nn.LayerNorm(name=f'layer_norm_{i}')(x)
        
        # Output layer - scalar log normalizer
        x = nn.Dense(1, name='output')(x)
        return jnp.squeeze(x, axis=-1)


def stable_gradient_computation(model, params, eta, eps=1e-6):
    """
    Numerically stable gradient computation with regularization.
    
    Args:
        model: LogNormalizerNetwork
        params: Model parameters
        eta: Natural parameters [batch_size, eta_dim]
        eps: Regularization parameter
    
    Returns:
        Stable gradient (mean) [batch_size, eta_dim]
    """
    def single_log_normalizer(eta_single):
        return model.apply(params, eta_single[None, :], training=False)[0]
    
    # Add small perturbation to avoid numerical issues
    eta_perturbed = eta + eps * jnp.ones_like(eta)
    
    # Compute gradient
    grad_fn = grad(single_log_normalizer)
    gradients = jax.vmap(grad_fn)(eta_perturbed)
    
    # Clip extreme values for stability
    gradients = jnp.clip(gradients, -10.0, 10.0)
    
    return gradients


def stable_hessian_computation(model, params, eta, method='diagonal', eps=1e-6):
    """
    Numerically stable Hessian computation.
    
    Args:
        model: LogNormalizerNetwork
        params: Model parameters
        eta: Natural parameters [batch_size, eta_dim]
        method: 'diagonal', 'full', or 'lanczos'
        eps: Regularization parameter
    
    Returns:
        Stable Hessian (covariance) approximation
    """
    def single_log_normalizer(eta_single):
        return model.apply(params, eta_single[None, :], training=False)[0]
    
    if method == 'diagonal':
        # Diagonal approximation with numerical stability
        def diagonal_hessian_fn(eta_single):
            # Use finite differences for better numerical stability
            def finite_diff_hessian(eta_single):
                grad_fn = grad(single_log_normalizer)
                gradients = grad_fn(eta_single)
                
                # Compute diagonal elements using finite differences
                diag_hessian = jnp.zeros_like(eta_single)
                for i in range(len(eta_single)):
                    eta_plus = eta_single.at[i].add(eps)
                    eta_minus = eta_single.at[i].add(-eps)
                    
                    grad_plus = grad_fn(eta_plus)
                    grad_minus = grad_fn(eta_minus)
                    
                    diag_hessian = diag_hessian.at[i].set(
                        (grad_plus[i] - grad_minus[i]) / (2 * eps)
                    )
                
                return diag_hessian
            
            return finite_diff_hessian(eta_single)
        
        hessians = jax.vmap(diagonal_hessian_fn)(eta)
        
        # Ensure positive definiteness (diagonal elements should be positive)
        hessians = jnp.maximum(hessians, eps)
        
        return hessians
    
    elif method == 'full':
        # Full Hessian with regularization
        hess_fn = hessian(single_log_normalizer)
        hessians = jax.vmap(hess_fn)(eta)
        
        # Add regularization to diagonal
        eye = jnp.eye(hessians.shape[-1])
        hessians = hessians + eps * eye[None, :, :]
        
        # Ensure positive definiteness using Cholesky decomposition
        try:
            # Try Cholesky decomposition
            L = jnp.linalg.cholesky(hessians)
            # Reconstruct positive definite matrix
            hessians = L @ jnp.transpose(L, (0, 2, 1))
        except:
            # If Cholesky fails, use eigenvalue clipping
            eigenvals, eigenvecs = jnp.linalg.eigh(hessians)
            eigenvals = jnp.maximum(eigenvals, eps)
            hessians = eigenvecs @ jnp.diag(eigenvals) @ jnp.transpose(eigenvecs, (0, 2, 1))
        
        return hessians
    
    elif method == 'lanczos':
        # Lanczos approximation for large-scale Hessians
        # This is a more advanced method for very large parameter dimensions
        raise NotImplementedError("Lanczos Hessian approximation not yet implemented")
    
    else:
        raise ValueError(f"Unknown Hessian method: {method}")


def adaptive_loss_weights(epoch: int, warmup_epochs: int = 50) -> Dict[str, float]:
    """
    Adaptive loss weights that start with mean-only training and gradually
    introduce covariance loss.
    
    Args:
        epoch: Current training epoch
        warmup_epochs: Number of epochs to warm up with mean-only training
    
    Returns:
        Dict with loss weights
    """
    if epoch < warmup_epochs:
        return {'mean_weight': 1.0, 'cov_weight': 0.0}
    else:
        # Gradually increase covariance weight
        progress = min(1.0, (epoch - warmup_epochs) / warmup_epochs)
        cov_weight = 0.1 * progress
        return {'mean_weight': 1.0, 'cov_weight': cov_weight}


def stable_log_normalizer_loss_fn(model, params, eta_batch, mean_batch, cov_batch=None, 
                                 loss_weights=None, hessian_method='diagonal', 
                                 regularization=1e-6, epoch=0):
    """
    Numerically stable version of log normalizer loss function.
    
    Includes adaptive loss weights and stability measures.
    """
    if loss_weights is None:
        loss_weights = adaptive_loss_weights(epoch)
    
    # Compute network predictions
    log_normalizer = model.apply(params, eta_batch, training=True)
    
    # Compute stable gradient (mean)
    network_mean = stable_gradient_computation(model, params, eta_batch, regularization)
    
    # Mean loss with clipping
    mean_diff = network_mean - mean_batch
    mean_diff = jnp.clip(mean_diff, -5.0, 5.0)  # Prevent extreme gradients
    mean_loss = jnp.mean(jnp.square(mean_diff))
    
    total_loss = loss_weights['mean_weight'] * mean_loss
    
    # Covariance loss (if requested and data available)
    if cov_batch is not None and loss_weights['cov_weight'] > 0:
        try:
            # Compute stable Hessian (covariance)
            network_hessian = stable_hessian_computation(
                model, params, eta_batch, method=hessian_method, eps=regularization
            )
            
            if hessian_method == 'diagonal':
                # For diagonal approximation
                empirical_diag = jnp.diagonal(cov_batch, axis1=1, axis2=2)
                
                # Clip extreme values
                network_diag = jnp.clip(network_hessian, eps, 10.0)
                empirical_diag = jnp.clip(empirical_diag, eps, 10.0)
                
                # Use relative error for better numerical stability
                relative_error = jnp.abs(network_diag - empirical_diag) / (empirical_diag + eps)
                cov_loss = jnp.mean(relative_error)
                
            else:
                # For full Hessian
                # Use Frobenius norm for stability
                diff = network_hessian - cov_batch
                cov_loss = jnp.mean(jnp.linalg.norm(diff, axis=(1, 2)))
            
            total_loss += loss_weights['cov_weight'] * cov_loss
            
        except Exception as e:
            # If Hessian computation fails, continue without covariance loss
            print(f"Warning: Stable Hessian computation failed: {e}")
            pass
    
    # Add regularization term to prevent overfitting
    regularization_loss = regularization * jnp.mean(jnp.square(log_normalizer))
    total_loss += regularization_loss
    
    return total_loss


class StableLogNormalizerTrainer(BaseTrainer):
    """Trainer for LogNormalizerNetwork with numerical stability measures."""
    
    def __init__(self, config: FullConfig, hessian_method='diagonal', 
                 adaptive_weights=True):
        model = LogNormalizerNetwork(config=config.network)
        super().__init__(model, config)
        self.hessian_method = hessian_method
        self.adaptive_weights = adaptive_weights
        self.epoch = 0
    
    def loss_fn(self, params: Dict, batch: Dict[str, jnp.ndarray], training: bool = True) -> jnp.ndarray:
        """Compute stable log normalizer loss for a batch."""
        return stable_log_normalizer_loss_fn(
            self.model, params, 
            batch['eta'], batch['mean'],
            batch.get('cov'), 
            loss_weights=adaptive_loss_weights(self.epoch) if self.adaptive_weights else None,
            hessian_method=self.hessian_method,
            epoch=self.epoch
        )
    
    def train_step(self, params: Dict, opt_state: Any, batch: Dict[str, jnp.ndarray], 
                   optimizer: Any) -> Tuple[Dict, Any, float]:
        """Single training step with epoch tracking."""
        loss_value, grads = jax.value_and_grad(self.loss_fn)(params, batch, training=True)
        
        # Gradient clipping for stability
        grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss_value
    
    def train(self, train_data: Dict[str, jnp.ndarray], val_data: Dict[str, jnp.ndarray]) -> Tuple[Dict, Dict]:
        """Training loop with stability measures."""
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
        
        print(f"Training Stable LogNormalizerNetwork for {tc.num_epochs} epochs")
        print(f"Hessian method: {self.hessian_method}")
        print(f"Adaptive weights: {self.adaptive_weights}")
        print(f"Architecture: {self.config.network.hidden_sizes}")
        print(f"Parameters: {self.model.get_parameter_count(params):,}")
        
        start_time = time.time()
        
        for epoch in range(tc.num_epochs):
            self.epoch = epoch  # Update epoch for adaptive weights
            
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
                mean_loss = float(jnp.mean(jnp.square(
                    stable_gradient_computation(self.model, params, batch['eta']) - batch['mean']
                )))
                mean_losses.append(mean_loss)
                
                if batch.get('cov') is not None:
                    try:
                        hessian_pred = stable_hessian_computation(
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
                    current_weights = adaptive_loss_weights(epoch)
                    print(f"    Epoch {epoch:3d}: Train={avg_train_loss:.4f}, Val={val_loss:.4f}, "
                          f"Mean={avg_mean_loss:.4f}, Cov={avg_cov_loss:.4f}, "
                          f"Weights={current_weights}")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.1f}s")
        
        return best_params, history


def create_stable_log_normalizer_trainer(config: FullConfig, 
                                       hessian_method='diagonal',
                                       adaptive_weights=True) -> StableLogNormalizerTrainer:
    """Factory function to create stable log normalizer trainer."""
    return StableLogNormalizerTrainer(config, hessian_method, adaptive_weights)
