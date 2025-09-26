"""
LogZ Network Base Classes - Core training functionality for LogZ models.

This module provides the base training functionality for LogZ networks that learn
log normalizers A(η) of exponential families. The network outputs a scalar log 
normalizer whose gradients and Hessians provide the mean and covariance of 
sufficient statistics.

Note: Architecture-specific LogZ networks are now in individual model files:
- MLP: src/models/mlp_logZ.py
- GLU: src/models/glu_logZ.py  
- Quadratic ResNet: src/models/quadratic_resnet_logZ.py
- Convex NN: src/models/convex_nn_logZ.py
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, Any, Tuple, Optional
import optax
from tqdm import tqdm

from ..base_model import BaseTrainer
from ..config import FullConfig
from .utils.gradient_hessian_utils import LogNormalizerDerivatives


class LogZTrainer(BaseTrainer):
    """
    Base trainer for LogZ Networks with gradient/Hessian computation.
    
    This trainer provides the core training functionality for networks that output log normalizers,
    using automatic differentiation to compute gradients (mean) and Hessians (covariance) 
    for loss computation.
    
    Architecture-specific implementations should inherit from this class and provide
    their own model instances.
    """
    
    def __init__(self, model, config: FullConfig, 
                 loss_type: str = "mse_mean_only", hessian_method: str = 'diagonal', 
                 adaptive_weights: bool = True, l1_reg_weight: float = 1e-4):
        """
        Initialize LogZ trainer.
        
        Args:
            model: The LogZ network model (should be an instance of BaseNeuralNetwork)
            config: Full configuration object
            loss_type: Type of loss function to use
            hessian_method: Method for computing Hessian ('diagonal' or 'full')
            adaptive_weights: Whether to use adaptive weighting for covariance loss
            l1_reg_weight: Weight for L1 regularization
        """
        super().__init__(model, config)
        self.loss_type = loss_type
        self.adaptive_weights = adaptive_weights
        self.l1_reg_weight = l1_reg_weight
        self.epoch = 0
        
        # Determine if we need Hessian computation based on loss type
        self.needs_hessian = loss_type in ["mse_mean_and_cov", "mse_mean_and_diag_cov", "KLqp", "KLqp_diag", "KLpq", "KLpq_diag"]
        self.hessian_method = "diagonal" if loss_type in ["mse_mean_and_diag_cov", "KLqp_diag", "KLpq_diag"] else hessian_method
        
        # Initialize gradient/hessian utilities
        self.derivatives = LogNormalizerDerivatives(
            model_apply_fn=model.apply,
            hessian_method=self.hessian_method,
            compile_functions=True
        )
    
    
    def compute_gradient(self, params: Dict, eta: jnp.ndarray) -> jnp.ndarray:
        """Compute gradient of log normalizer (expectation of sufficient statistics)."""
        return self.derivatives.predict_mean(params, eta)
    
    def compute_hessian(self, params: Dict, eta: jnp.ndarray) -> jnp.ndarray:
        """Compute Hessian of log normalizer (covariance of sufficient statistics)."""
        return self.derivatives.predict_covariance(params, eta)
    
    def loss_fn(self, params: Dict, eta: jnp.ndarray, 
                target_mu_T: jnp.ndarray, target_cov_TT: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Compute loss based on log normalizer derivatives.
        
        Args:
            params: Model parameters
            eta: Natural parameters
            target_mu_T: Target mean of sufficient statistics
            target_cov_tt: Target covariance of sufficient statistics (optional)
            
        Returns:
            Loss value
        """
        # Always compute mean prediction and loss
        predicted_mean = self.compute_gradient(params, eta)
        total_loss = 0.0
        
        # Add covariance loss based on loss type
        if self.loss_type == "mse_mean_only":
            mse_loss = jnp.mean((predicted_mean - target_mu_T) ** 2)
            total_loss += mse_loss
        elif self.loss_type in ["mse_mean_and_cov", "mse_mean_and_diag_cov"]:             
            if target_cov_TT is None:
                raise ValueError("Target covariance is required for mse_mean_and_cov or mse_mean_and_diag_cov loss type")
            # Compute covariance loss
            predicted_cov = self.compute_hessian(params, eta)
            
            if self.hessian_method == 'diagonal':
                # Only compare diagonal elements
                if target_cov_TT.ndim == 3:  # Full covariance matrix [batch, dim, dim]
                    target_diag = jnp.diagonal(target_cov_TT, axis1=-2, axis2=-1)
                else:  # Already diagonal [batch, dim]
                    target_diag = target_cov_TT
                cov_loss = jnp.mean((predicted_cov - target_diag) ** 2)
            else:
                # Compare full covariance matrices
                cov_loss = jnp.mean((predicted_cov - target_cov_TT) ** 2)
            
            # Weight covariance loss
            cov_weight = 0.1 if not self.adaptive_weights else max(0.01, 1.0 / (1.0 + 0.1 * self.epoch))
            total_loss += cov_weight * cov_loss
        
        # Add model-specific internal losses (e.g., smoothness penalties, regularization)
        internal_loss = self.model.compute_internal_loss(params, eta, predicted_mean, training=True)
        total_loss += internal_loss
        
        # Add L1 regularization if enabled
        if self.l1_reg_weight > 0.0:
            l1_reg = 0.0
            for param in jax.tree.leaves(params):
                l1_reg += jnp.sum(jnp.abs(param))
            total_loss += self.l1_reg_weight * l1_reg
        
        return total_loss
    
    def train_step(self, params: Dict, opt_state: Any, batch: Dict[str, jnp.ndarray], 
                   optimizer: Any) -> Tuple[Dict, Any, float]:
        """Single training step with epoch tracking."""
        loss_value, grads = jax.value_and_grad(self.loss_fn)(
            params, batch['eta'], batch['mu_T'], batch.get('cov_TT')
        )
        
        # Update parameters
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, float(loss_value)
    
    def train(self, train_data: Dict[str, jnp.ndarray], 
              val_data: Optional[Dict[str, jnp.ndarray]] = None,
              epochs: int = 300, learning_rate: float = 1e-3) -> Tuple[Dict, Dict]:
        """
        Train the LogZ network.
        
        Args:
            train_data: Training data with 'eta', 'mu_T', and optionally 'cov_TT' keys
            val_data: Validation data (optional)
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            
        Returns:
            Tuple of (best_params, training_history)
        """
        # Initialize optimizer
        optimizer = optax.adam(learning_rate)
        
        # Initialize parameters
        self.rng, init_rng = random.split(self.rng)
        # Ensure consistent initialization shape for gradient computation
        init_batch = jnp.expand_dims(train_data['eta'][0], axis=0)  # Shape (1, input_dim)
        params = self.model.init(init_rng, init_batch, training=True)
        opt_state = optimizer.init(params)
        
        # Training loop
        best_params = params
        best_loss = float('inf')
        patience_counter = 0
        training_history = {'train_loss': [], 'val_loss': []}
        
        pbar = tqdm(range(epochs), desc="Training LogZ Network")
        for epoch in pbar:
            self.epoch = epoch  # Update epoch for adaptive weighting
            
            # Training step
            params, opt_state, train_loss = self.train_step(
                params, opt_state, train_data, optimizer
            )
            training_history['train_loss'].append(train_loss)
            
            # Validation
            if val_data is not None:
                val_loss = float(self.loss_fn(
                    params, val_data['eta'], val_data['mu_T'], val_data.get('cov_TT')
                ))
                training_history['val_loss'].append(val_loss)
                
                # Update progress bar with both train and validation loss
                pbar.set_postfix({
                    'train_loss': f'{train_loss:.6f}',
                    'val_loss': f'{val_loss:.6f}'
                })
            else:
                # Update progress bar with training loss only
                pbar.set_postfix({'train_loss': f'{train_loss:.6f}'})
                
                # No validation data, use training loss for early stopping
                if train_loss < best_loss:
                    best_loss = train_loss
                    best_params = params
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.training.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
        
        return best_params, training_history
    
    def predict(self, params: Dict, eta: jnp.ndarray, compute_covariance: bool = False) -> Dict[str, jnp.ndarray]:
        """
        Make predictions using the trained model.
        
        Args:
            params: Model parameters
            eta: Natural parameters [batch_size, eta_dim]
            compute_covariance: Whether to compute covariance predictions
            
        Returns:
            Dictionary containing:
            - 'stats': Predicted means [batch_size, eta_dim]
            - 'covariance': Predicted covariances with shape:
              * Diagonal method: [batch_size, eta_dim] (diagonal elements only)
              * Full method: [batch_size, eta_dim, eta_dim] (full covariance matrix)
        """
        predicted_mean = self.compute_gradient(params, eta)
        
        result = {'stats': predicted_mean}
        
        if compute_covariance:
            predicted_cov = self.compute_hessian(params, eta)
            result['covariance'] = predicted_cov
        
        return result
    
    def evaluate(self, params: Dict, test_data: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """Evaluate the model on test data."""
        # Only compute covariance if test data has covariance information
        compute_cov = 'cov' in test_data
        predictions = self.predict(params, test_data['eta'], compute_covariance=compute_cov)
        
        # Compute statistics metrics
        stats_mse = float(jnp.mean((predictions['stats'] - test_data['mu_T']) ** 2))
        stats_mae = float(jnp.mean(jnp.abs(predictions['stats'] - test_data['mu_T'])))
        
        result = {
            'mse': stats_mse,
            'mae': stats_mae,
            'component_errors': jnp.mean((predictions['stats'] - test_data['mu_T']) ** 2, axis=0)
        }
        
        # Compute covariance metrics if available
        if compute_cov and 'covariance' in predictions:
            predicted_cov = predictions['covariance']
            test_cov = test_data['cov']
            
            # Handle different Hessian methods
            hessian_method = self.derivatives.get_hessian_method()
            
            if hessian_method == 'diagonal':
                # Diagonal Hessian returns shape (batch_size, eta_dim)
                # Test data covariance should also be diagonal for comparison
                if test_cov.ndim == 3:  # Full covariance matrix (batch_size, eta_dim, eta_dim)
                    # Extract diagonal elements for comparison
                    test_cov_diag = jnp.diagonal(test_cov, axis1=-2, axis2=-1)
                    cov_mse = float(jnp.mean((predicted_cov - test_cov_diag) ** 2))
                else:  # Already diagonal
                    cov_mse = float(jnp.mean((predicted_cov - test_cov) ** 2))
            else:  # Full Hessian method
                # Full Hessian returns shape (batch_size, eta_dim, eta_dim)
                cov_mse = float(jnp.mean((predicted_cov - test_cov) ** 2))
            
            result['cov_mse'] = cov_mse
        
        return result