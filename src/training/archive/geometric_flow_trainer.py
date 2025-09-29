"""
Training implementation for regular Geometric Flow ET models.

This module implements the standard Hugging Face training pattern for
regular geometric flow models using standard backpropagation.
"""

from typing import Dict, Any, Optional, Tuple, List
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import time

# BaseTrainer and TrainingMetrics removed - using BaseETTrainer instead
from .base_et_trainer import BaseETTrainer
from ..models.geometric_flow_et_net import Geometric_Flow_ET_Network
from ..configs.geometric_flow_et_config import Geometric_Flow_ET_Config


class GeometricFlowTrainer(BaseETTrainer):
    """
    Trainer for regular Geometric Flow ET models using standard backpropagation.
    
    This follows the standard Hugging Face training pattern with:
    - Standard backpropagation
    - Smoothness penalties
    - Standard optimization (Adam, etc.)
    - Full time integration during training
    """
    
    def __init__(self, model: Geometric_Flow_ET_Network, config: Geometric_Flow_ET_Config):
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
        Compute geometric flow loss with smoothness penalty.
        
        Args:
            params: Model parameters
            batch: Training batch with 'eta' and 'mu_T' keys
            training: Whether in training mode
            
        Returns:
            Tuple of (total_loss, additional_metrics)
        """
        eta = batch['eta']
        mu_target = batch['mu_T']
        
        # Forward pass through model
        mu_predicted = self.model.apply(params, eta, training=training)
        
        # Main prediction loss (MSE)
        prediction_loss = jnp.mean((mu_predicted - mu_target) ** 2)
        
        # Smoothness penalty (from model intermediates)
        smoothness_loss = 0.0
        if training:
            # Get derivative norms from model intermediates
            intermediates = self.model.apply(params, eta, training=training, capture_intermediates=True)
            if 'intermediates' in intermediates and 'derivative_norms_squared' in intermediates['intermediates']:
                derivative_norms = intermediates['intermediates']['derivative_norms_squared']
                smoothness_loss = jnp.mean(derivative_norms)
        
        # Total loss
        total_loss = prediction_loss + self.config.smoothness_weight * smoothness_loss
        
        metrics = {
            'prediction_loss': float(prediction_loss),
            'smoothness_loss': float(smoothness_loss),
            'total_loss': float(total_loss)
        }
        
        return total_loss, metrics
    
    def train_step(self, params: Dict, opt_state: Any, batch: Dict[str, jnp.ndarray]) -> Tuple[Dict, Any, Dict]:
        """
        Single training step with standard backpropagation.
        
        Args:
            params: Model parameters
            opt_state: Optimizer state
            batch: Training batch
            
        Returns:
            Tuple of (updated_params, updated_opt_state, metrics)
        """
        # Compute loss and gradients
        (loss, loss_metrics), grads = jax.value_and_grad(
            self.compute_loss, has_aux=True
        )(params, batch, training=True)
        
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
        params = self.model.init(init_rng, sample_batch['eta'])
        
        # Initialize optimizer state
        opt_state = self.optimizer.init(params)
        
        return train_state.TrainState(
            step=0,
            apply_fn=self.model.apply,
            params=params,
            tx=self.optimizer,
            opt_state=opt_state
        )
    
    def create_trainerbatch(self, eta_targets: jnp.ndarray, mu_targets: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Create training batch for geometric flow.
        
        Args:
            eta_targets: Target natural parameters [batch_size, eta_dim]
            mu_targets: Target expectations [batch_size, mu_dim]
            
        Returns:
            Training batch
        """
        return {
            'eta': eta_targets,
            'mu_T': mu_targets
        }
    
    def predict(self, params: Dict, eta: jnp.ndarray) -> jnp.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            params: Model parameters
            eta: Input natural parameters
            
        Returns:
            Predicted expectations
        """
        return self.model.apply(params, eta, training=False)
    
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
