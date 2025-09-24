"""
ET Network Base Classes - Core training functionality for ET models.

This module provides the base training functionality for ET (Exponential Family) networks
that directly predict expected sufficient statistics. The architecture-specific implementations
are now in individual model files in the src/models/ directory.

Note: Architecture-specific ET networks are now in individual model files:
- MLP: src/models/mlp_ET.py
- GLU: src/models/glu_ET.py  
- Quadratic ResNet: src/models/quadratic_resnet_ET.py
- Invertible NN: src/models/invertible_nn_ET.py
- NoProp-CT: src/models/noprop_ct_ET.py
- Glow: src/models/glow_net_ET.py
- Geometric Flow: src/models/geometric_flow_net.py
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, Any, Tuple, Optional
import optax
from tqdm import tqdm
import time

from ..base_model import BaseTrainer
from ..config import FullConfig


class ETTrainer(BaseTrainer):
    """
    Base trainer for ET Networks that directly predict expected statistics.
    
    This trainer provides the core training functionality for networks that directly output
    expected sufficient statistics E[T(X)|η] without requiring gradient/Hessian computation.
    
    Architecture-specific implementations should inherit from this class and provide
    their own model instances.
    """
    
    def __init__(self, model, config: FullConfig, l1_reg_weight: float = 1e-4):
        """
        Initialize ET trainer.
        
        Args:
            model: The ET network model (should be an instance of BaseNeuralNetwork)
            config: Full configuration object
            l1_reg_weight: Weight for L1 regularization
        """
        super().__init__(model, config)
        self.l1_reg_weight = l1_reg_weight
    
    def loss_fn(self, params: Dict, eta: jnp.ndarray, 
                target_mu_T: jnp.ndarray) -> jnp.ndarray:
        """
        Compute loss for expected statistics prediction.
        
        Args:
            params: Model parameters
            eta: Natural parameters
            target_mu_T: Target expected sufficient statistics
            
        Returns:
            Loss value
        """
        # Compute network predictions
        predicted_mu_T = self.model.apply(params, eta, training=True)
        
        # MSE loss
        mse_loss = jnp.mean((predicted_mu_T - target_mu_T) ** 2)
        total_loss = mse_loss
        
        # Add model-specific internal losses (e.g., smoothness penalties, regularization)
        internal_loss = self.model.compute_internal_loss(params, eta, predicted_mu_T, training=True)
        total_loss += internal_loss
        
        # L1 regularization on parameters (configurable, default off)
        if self.l1_reg_weight > 0.0:
            l1_reg = 0.0
            for param in jax.tree.leaves(params):
                l1_reg += jnp.sum(jnp.abs(param))
            total_loss += self.l1_reg_weight * l1_reg
        
        return total_loss
    
    def train_step(self, params: Dict, opt_state: Any, batch: Dict[str, jnp.ndarray], 
                   optimizer: Any) -> Tuple[Dict, Any, float]:
        """Single training step."""
        loss_value, grads = jax.value_and_grad(self.loss_fn)(
            params, batch['eta'], batch['mu_T']
        )
        
        # Update parameters
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, float(loss_value)
    
    def train(self, train_data: Dict[str, jnp.ndarray], 
              val_data: Optional[Dict[str, jnp.ndarray]] = None,
              epochs: int = 300, learning_rate: float = None) -> Tuple[Dict, Dict]:
        """
        Train the ET network.
        
        Args:
            train_data: Training data with 'eta' and 'mu_T' keys
            val_data: Validation data (optional)
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer (if None, uses config)
            
        Returns:
            Tuple of (best_params, training_history)
        """
        # Use advanced optimizer from BaseTrainer if available, otherwise fallback
        if hasattr(self, 'config') and self.config is not None:
            # Update epochs in config if different
            if epochs != self.config.training.num_epochs:
                self.config.training.num_epochs = epochs
            # Override learning rate if provided
            if learning_rate is not None:
                self.config.training.learning_rate = learning_rate
            optimizer = self.create_optimizer()
        else:
            # Fallback to simple optimizer
            lr = learning_rate if learning_rate is not None else 1e-3
            optimizer = optax.adam(lr)
        
        # Initialize parameters
        self.rng, init_rng = random.split(self.rng)
        params = self.model.init(init_rng, train_data['eta'][:1])
        opt_state = optimizer.init(params)
        
        # Training loop
        best_params = params
        best_loss = float('inf')
        patience_counter = 0
        training_history = {'train_loss': [], 'val_loss': []}
        
        pbar = tqdm(range(epochs), desc="Training ET Network")
        for epoch in pbar:
            # Training step
            params, opt_state, train_loss = self.train_step(
                params, opt_state, train_data, optimizer
            )
            training_history['train_loss'].append(train_loss)
            
            # Validation
            if val_data is not None:
                val_loss = float(self.loss_fn(params, val_data['eta'], val_data['mu_T']))
                training_history['val_loss'].append(val_loss)
                
                # Update progress bar with loss information
                pbar.set_postfix({
                    'train_loss': f'{train_loss:.6f}',
                    'val_loss': f'{val_loss:.6f}'
                })
                
                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = params
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.training.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
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
    
    def predict(self, params: Dict, eta: jnp.ndarray) -> jnp.ndarray:
        """Make predictions using the trained model."""
        return self.model.apply(params, eta, training=False)
    
    def evaluate(self, params: Dict, test_data: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """Evaluate the model on test data."""
        predictions = self.predict(params, test_data['eta'])
        
        # Compute metrics
        mse = float(jnp.mean((predictions - test_data['mu_T']) ** 2))
        mae = float(jnp.mean(jnp.abs(predictions - test_data['mu_T'])))
        
        # Component-wise errors
        errors_by_component = jnp.mean((predictions - test_data['mu_T']) ** 2, axis=0)
        
        return {
            'mse': mse,
            'mae': mae,
            'component_errors': errors_by_component
        }

    def benchmark_inference(self, params: Dict, eta_data, num_runs=10, batch_size=1000):
        """Benchmark inference time using larger batches for better accuracy."""
        # Warm-up run to ensure compilation is complete
        self.predict(params, eta_data[:1])
        
        # Use a reasonable batch size for timing (not too large to avoid memory issues)
        batch_size = min(batch_size, len(eta_data))
        batch_data = eta_data[:batch_size]
        
        # Measure inference time over multiple runs on the batch
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = self.predict(params, batch_data)
            elapsed = time.perf_counter() - start_time
            # Ensure we never have negative times (safeguard against clock issues)
            times.append(max(0.0, elapsed))
        
        # Return statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Calculate per-sample timing by normalizing by batch size
        per_sample_time = avg_time / batch_size
        samples_per_second = batch_size / avg_time
        
        return {
            'avg_inference_time': per_sample_time,
            'min_inference_time': min_time / batch_size,
            'max_inference_time': max_time / batch_size,
            'inference_per_sample': per_sample_time,
            'samples_per_second': samples_per_second,
            'batch_size_used': batch_size,
            'total_batch_time': avg_time
        }