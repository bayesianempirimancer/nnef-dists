"""
Base model class for all neural network architectures.

This provides a consistent interface for all model types with standardized
training, evaluation, and configuration handling.
"""

import abc
from typing import Dict, Any, Tuple, Optional
import jax
import jax.numpy as jnp
from jax import random
import optax
import flax.linen as nn
from pathlib import Path
import json
import time

from .config import FullConfig, NetworkConfig, TrainingConfig


class BaseNeuralNetwork(nn.Module):
    """Abstract base class for all neural network models."""
    
    config: NetworkConfig
    
    @abc.abstractmethod
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Forward pass through the network."""
        pass
    
    def get_parameter_count(self, params: Dict) -> int:
        """Count total number of parameters."""
        return sum(x.size for x in jax.tree.leaves(params))


class BaseTrainer:
    """Base trainer class with common training functionality."""
    
    def __init__(self, model: BaseNeuralNetwork, config: FullConfig):
        self.model = model
        self.config = config
        self.rng = random.PRNGKey(config.experiment.random_seed)
        
    def create_optimizer(self) -> optax.GradientTransformation:
        """Create optimizer based on configuration."""
        tc = self.config.training
        
        # Create learning rate schedule if requested
        if tc.use_lr_schedule:
            if tc.lr_schedule_type == "exponential":
                lr_schedule = optax.exponential_decay(
                    init_value=tc.learning_rate,
                    transition_steps=tc.lr_decay_steps,
                    decay_rate=tc.lr_decay_rate
                )
            elif tc.lr_schedule_type == "cosine":
                lr_schedule = optax.cosine_decay_schedule(
                    init_value=tc.learning_rate,
                    decay_steps=tc.lr_decay_steps
                )
            else:
                lr_schedule = tc.learning_rate
        else:
            lr_schedule = tc.learning_rate
        
        # Create base optimizer
        if tc.optimizer == "adam":
            base_opt = optax.adam(learning_rate=lr_schedule)
        elif tc.optimizer == "adamw":
            base_opt = optax.adamw(learning_rate=lr_schedule, weight_decay=tc.weight_decay)
        elif tc.optimizer == "sgd":
            base_opt = optax.sgd(learning_rate=lr_schedule)
        elif tc.optimizer == "rmsprop":
            base_opt = optax.rmsprop(learning_rate=lr_schedule)
        else:
            raise ValueError(f"Unknown optimizer: {tc.optimizer}")
        
        # Add gradient clipping
        optimizer = optax.chain(
            optax.clip_by_global_norm(tc.gradient_clip_norm),
            base_opt
        )
        
        return optimizer
    
    def initialize_model(self, sample_input: jnp.ndarray) -> Tuple[Dict, optax.OptState]:
        """Initialize model parameters and optimizer state."""
        self.rng, init_rng = random.split(self.rng)
        params = self.model.init(init_rng, sample_input)
        
        optimizer = self.create_optimizer()
        opt_state = optimizer.init(params)
        
        return params, opt_state
    
    def loss_fn(self, params: Dict, batch: Dict[str, jnp.ndarray], training: bool = True) -> jnp.ndarray:
        """Compute loss for a batch."""
        predictions = self.model.apply(params, batch['eta'], training=training)
        return jnp.mean(jnp.square(predictions - batch['mu_T']))
    
    def train_step(self, params: Dict, opt_state: optax.OptState, batch: Dict[str, jnp.ndarray], 
                   optimizer: optax.GradientTransformation) -> Tuple[Dict, optax.OptState, float]:
        """Single training step."""
        loss_value, grads = jax.value_and_grad(self.loss_fn)(params, batch, training=True)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value
    
    def eval_step(self, params: Dict, batch: Dict[str, jnp.ndarray]) -> float:
        """Single evaluation step."""
        return self.loss_fn(params, batch, training=False)
    
    def train(self, train_data: Dict[str, jnp.ndarray], val_data: Dict[str, jnp.ndarray]) -> Tuple[Dict, Dict]:
        """Full training loop with early stopping."""
        # Initialize
        sample_input = train_data['eta'][:1]
        params, opt_state = self.initialize_model(sample_input)
        optimizer = self.create_optimizer()
        
        # Training state
        best_val_loss = float('inf')
        patience_counter = 0
        best_params = params
        history = {'train_loss': [], 'val_loss': []}
        
        tc = self.config.training
        batch_size = tc.batch_size
        n_train = train_data['eta'].shape[0]
        
        print(f"Training {self.model.__class__.__name__} for {tc.num_epochs} epochs")
        print(f"Architecture: {self.config.network.hidden_sizes}")
        print(f"Parameters: {self.model.get_parameter_count(params):,}")
        
        start_time = time.time()
        
        for epoch in range(tc.num_epochs):
            # Shuffle training data
            self.rng, shuffle_rng = random.split(self.rng)
            perm = random.permutation(shuffle_rng, n_train)
            
            train_losses = []
            
            # Training batches
            for i in range(0, n_train, batch_size):
                batch_idx = perm[i:i + batch_size]
                batch = {
                    'eta': train_data['eta'][batch_idx],
                    'mu_T': train_data['mu_T'][batch_idx]
                }
                
                params, opt_state, loss = self.train_step(params, opt_state, batch, optimizer)
                train_losses.append(loss)
            
            avg_train_loss = float(jnp.mean(jnp.array(train_losses)))
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            if epoch % tc.validation_freq == 0:
                val_batch = {'eta': val_data['eta'], 'mu_T': val_data['mu_T']}
                val_loss = float(self.eval_step(params, val_batch))
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
                    print(f"    Epoch {epoch:3d}: Train={avg_train_loss:.2f}, Val={val_loss:.2f}")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.1f}s")
        
        return best_params, history
    
    def evaluate(self, params: Dict, test_data: Dict[str, jnp.ndarray], 
                 ground_truth: Optional[jnp.ndarray] = None) -> Dict[str, float]:
        """Evaluate model on test data."""
        # Get predictions
        predictions = self.model.apply(params, test_data['eta'], training=False)
        
        # Compute metrics
        mse = float(jnp.mean(jnp.square(predictions - test_data['mu_T'])))
        mae = float(jnp.mean(jnp.abs(predictions - test_data['mu_T'])))
        
        metrics = {'mse': mse, 'mae': mae}
        
        # Ground truth comparison if available
        if ground_truth is not None:
            gt_mse = float(jnp.mean(jnp.square(predictions - ground_truth)))
            gt_mae = float(jnp.mean(jnp.abs(predictions - ground_truth)))
            metrics.update({'ground_truth_mse': gt_mse, 'ground_truth_mae': gt_mae})
        
        return metrics
    
    def save_model(self, params: Dict, save_path: Path):
        """Save model parameters and configuration."""
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save parameters
        import pickle
        with open(save_path / "params.pkl", "wb") as f:
            pickle.dump(params, f)
        
        # Save configuration
        with open(save_path / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, save_path: Path) -> Dict:
        """Load model parameters."""
        import pickle
        with open(save_path / "params.pkl", "rb") as f:
            params = pickle.load(f)
        return params


def create_standard_mlp(config: NetworkConfig) -> BaseNeuralNetwork:
    """Create a standard MLP with the given configuration."""
    
    class StandardMLP(BaseNeuralNetwork):
        
        @nn.compact
        def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
            # Feature engineering if requested
            if self.config.use_feature_engineering:
                from .eta_features import compute_eta_features
                x = compute_eta_features(x, method='default')
            
            # Hidden layers
            for i, hidden_size in enumerate(self.config.hidden_sizes):
                x = nn.Dense(hidden_size, name=f'hidden_{i}')(x)
                
                # Batch/Layer normalization
                if self.config.use_batch_norm:
                    x = nn.BatchNorm(use_running_average=not training, name=f'bn_{i}')(x)
                elif self.config.use_layer_norm:
                    x = nn.LayerNorm(name=f'ln_{i}')(x)
                
                # Activation
                if self.config.activation == "tanh":
                    x = nn.tanh(x)
                elif self.config.activation == "relu":
                    x = nn.relu(x)
                elif self.config.activation == "swish":
                    x = nn.swish(x)
                elif self.config.activation == "gelu":
                    x = nn.gelu(x)
                
                # Dropout
                if self.config.dropout_rate > 0:
                    x = nn.Dropout(rate=self.config.dropout_rate, deterministic=not training)(x)
                
                # Skip connections for very deep networks
                if self.config.skip_connections and i > 0 and x.shape[-1] == self.config.hidden_sizes[i-1]:
                    # Add residual connection if dimensions match
                    pass  # Would need to store previous layer output
            
            # Output layer
            x = nn.Dense(self.config.output_dim, name='output')(x)
            return x
    
    return StandardMLP(config=config)
