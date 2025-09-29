"""
Training implementation for NoProp Geometric Flow ET models.

This module implements a custom training loop for NoProp geometric flow models
that uses continuous-time training protocols instead of standard backpropagation.
"""

from typing import Dict, Any, Optional, Tuple, List
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import time

# BaseTrainer and TrainingMetrics removed - using BaseETTrainer instead
from .base_et_trainer import BaseETTrainer
from ..models.noprop_geometric_flow_et_net import NoProp_Geometric_Flow_ET_Network
from ..configs.noprop_geometric_flow_et_config import NoProp_Geometric_Flow_ET_Config


class NoPropGeometricFlowTrainer(BaseETTrainer):
    """
    Trainer for NoProp Geometric Flow ET models using continuous-time training.
    
    This implements a custom training loop with:
    - Continuous-time sampling t ~ Uniform(0,1)
    - NoProp training protocols
    - Multiple loss strategies (flow_matching, geometric_flow, simple_target)
    - No time integration during training
    """
    
    def __init__(self, model: NoProp_Geometric_Flow_ET_Network, config: NoProp_Geometric_Flow_ET_Config):
        super().__init__(model, config)
        self.model = model
        self.config = config
        
        # Create optimizer
        self.optimizer = optax.adam(config.learning_rate)
        
        # Training state
        self.trainerstate = None
    
    def compute_loss(self, params: Dict, batch: Dict[str, jnp.ndarray], 
                    training: bool = True) -> Tuple[float, Dict[str, Any]]:
        """
        Compute NoProp loss using the specified loss strategy.
        
        Args:
            params: Model parameters
            batch: Training batch with 'eta_init', 'eta_target', 'mu_init', 't' keys
            training: Whether in training mode
            
        Returns:
            Tuple of (total_loss, additional_metrics)
        """
        eta_init = batch['eta_init']
        eta_target = batch['eta_target']
        mu_init = batch['mu_init']
        t = batch['t']
        
        # Use the model's NoProp loss computation
        loss = self.model.compute_noprop_loss(params, eta_init, eta_target, mu_init, t, training=training)
        
        metrics = {
            'noprop_loss': float(loss),
            'loss_type': self.config.loss_type,
            'noise_schedule': self.config.noise_schedule
        }
        
        return loss, metrics
    
    def train_step(self, params: Dict, opt_state: Any, batch: Dict[str, jnp.ndarray]) -> Tuple[Dict, Any, Dict]:
        """
        Single NoProp training step with continuous-time sampling.
        
        Args:
            params: Model parameters
            opt_state: Optimizer state
            batch: Training batch
            
        Returns:
            Tuple of (updated_params, updated_opt_state, metrics)
        """
        # Sample random time points for NoProp training
        batch_size = batch['eta_init'].shape[0]
        self.rng, time_rng = jax.random.split(self.rng)
        t_samples = jax.random.uniform(time_rng, (batch_size,))
        
        # Update batch with sampled times
        batch_with_times = batch.copy()
        batch_with_times['t'] = t_samples
        
        # Compute loss and gradients
        (loss, loss_metrics), grads = jax.value_and_grad(
            self.compute_loss, has_aux=True
        )(params, batch_with_times, training=True)
        
        # Apply gradients
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        # Create metrics
        metrics = {
            'train_loss': float(loss),
            'learning_rate': self.config.learning_rate,
            **loss_metrics
        }
        
        return new_params, new_opt_state, metrics
    
    def _initialize_trainerstate(self, sample_batch: Dict[str, jnp.ndarray]) -> train_state.TrainState:
        """Initialize training state with sample batch."""
        # Initialize model parameters
        self.rng, init_rng = jax.random.split(self.rng)
        params = self.model.init(
            init_rng, 
            sample_batch['eta_init'], 
            sample_batch['eta_target'], 
            sample_batch['mu_init'], 
            sample_batch['t']
        )
        
        # Initialize optimizer state
        opt_state = self.optimizer.init(params)
        
        return train_state.TrainState(
            step=0,
            apply_fn=self.model.apply,
            params=params,
            tx=self.optimizer,
            opt_state=opt_state
        )
    
    def create_trainerbatch(self, eta_init: jnp.ndarray, eta_target: jnp.ndarray, 
                            mu_init: jnp.ndarray, t: Optional[jnp.ndarray] = None) -> Dict[str, jnp.ndarray]:
        """
        Create training batch for NoProp geometric flow.
        
        Args:
            eta_init: Initial natural parameters [batch_size, eta_dim]
            eta_target: Target natural parameters [batch_size, eta_dim]
            mu_init: Initial expectations [batch_size, mu_dim]
            t: Time points [batch_size] (if None, will be sampled randomly)
            
        Returns:
            Training batch
        """
        batch_size = eta_init.shape[0]
        
        if t is None:
            # Sample random time points
            self.rng, time_rng = jax.random.split(self.rng)
            t = jax.random.uniform(time_rng, (batch_size,))
        
        return {
            'eta_init': eta_init,
            'eta_target': eta_target,
            'mu_init': mu_init,
            't': t
        }
    
    def predict(self, params: Dict, eta: jnp.ndarray) -> jnp.ndarray:
        """
        Make predictions using the trained model (integrates over time).
        
        Args:
            params: Model parameters
            eta: Input natural parameters
            
        Returns:
            Predicted expectations
        """
        return self.model.apply(params, eta, method=self.model.predict)
    
    def train_continuous_time(self, params: Dict, eta_init: jnp.ndarray, eta_target: jnp.ndarray,
                            mu_init: jnp.ndarray, t: jnp.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Train at specific continuous time points (NoProp-specific method).
        
        Args:
            params: Model parameters
            eta_init: Initial natural parameters
            eta_target: Target natural parameters
            mu_init: Initial expectations
            t: Time points
            
        Returns:
            Tuple of (loss, metrics)
        """
        batch = self.create_trainerbatch(eta_init, eta_target, mu_init, t)
        return self.compute_loss(params, batch, training=True)
    
    def save_model(self, path: str):
        """Save model following Hugging Face conventions."""
        import pickle
        import os
        
        os.makedirs(path, exist_ok=True)
        
        # Save model parameters
        with open(os.path.join(path, 'model_params.pkl'), 'wb') as f:
            pickle.dump(self.trainerstate.params, f)
        
        # Save config
        with open(os.path.join(path, 'config.json'), 'w') as f:
            import json
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save training state
        with open(os.path.join(path, 'trainerstate.pkl'), 'wb') as f:
            pickle.dump(self.trainerstate, f)
    
    def load_model(self, path: str):
        """Load model following Hugging Face conventions."""
        import pickle
        import json
        import os
        
        # Load config
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config_dict = json.load(f)
        
        # Load model parameters
        with open(os.path.join(path, 'model_params.pkl'), 'rb') as f:
            params = pickle.load(f)
        
        # Load training state
        with open(os.path.join(path, 'trainerstate.pkl'), 'rb') as f:
            self.trainerstate = pickle.load(f)
        
        return params
