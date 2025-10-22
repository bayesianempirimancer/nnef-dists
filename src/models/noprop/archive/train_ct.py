#!/usr/bin/env python3
"""
NoProp CT Training Script

This script provides training for the NoProp CT (Continuous-Time) model.
The CT model uses a diffusion-based training protocol with (z, x, t) signature.

Usage:
    python src/models/noprop/train_ct.py --data data/multivariate_normal_tril_data_10000.pkl --epochs 100
"""

import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import time

# Handle imports for both module usage and direct script execution
import sys
if __name__ == "__main__":
    # When run as script, add project root to path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    # Use absolute imports when running as script
    from src.models.base_training_config import BaseTrainingConfig
    from src.models.base_trainer import BaseETTrainer
else:
    # Use relative imports when used as module
    from ..base_training_config import BaseTrainingConfig
    from ..base_trainer import BaseETTrainer

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pickle

# Model imports
if __name__ == "__main__":
    # When run as script, use absolute imports
    from src.models.noprop.ct import Config, NoPropCT
else:
    # When used as module, use relative imports
    from .ct import Config, NoPropCT


class TrainingConfig:
    """Training configuration for NoProp CT model."""
    
    def __init__(self, num_epochs=200, batch_size=256, learning_rate=1e-3, output_dir="artifacts", dropout_epochs=180):
        # === BASIC TRAINING PARAMETERS ===
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.output_dir = output_dir
        self.dropout_epochs = dropout_epochs
        
        # === CT SPECIFIC TRAINING PARAMETERS ===
        self.num_timesteps = 20
        self.integration_method = "euler"
        self.reg_weight = 0.0
        
        # === MODEL ARCHITECTURE ===
        self.model_hidden_dims = (64, 64, 64)
        self.model_dropout_rate = 0.0
        self.time_embed_dim = 64
        self.time_embed_method = "sinusoidal"
        
        # === NOISE SCHEDULE ===
        self.noise_schedule = "linear"


class NoPropCTTrainer:
    """Trainer for NoProp CT model using built-in methods."""
    
    def __init__(self, config: TrainingConfig, input_dim: int, output_dim: int):
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ct_model = NoPropCT(config=self._create_model_config())
    
    def _create_model_config(self) -> Config:
        """Create model config from training config."""
        return Config(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            z_shape=(self.output_dim,),
            x_shape=(self.input_dim,),
            noise_schedule=self.config.noise_schedule,
            time_embed_dim=self.config.time_embed_dim,
            time_embed_method=self.config.time_embed_method,
            num_timesteps=self.config.num_timesteps,
            integration_method=self.config.integration_method,
            reg_weight=self.config.reg_weight,
            model_type="conditional_resnet",
            model_hidden_dims=self.config.model_hidden_dims,
            model_dropout_rate=self.config.model_dropout_rate
        )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train NoProp CT model")
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Output directory")
    parser.add_argument("--dropout-epochs", type=int, default=180, help="Number of epochs to use dropout")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    with open(args.data, 'rb') as f:
        data = pickle.load(f)
    
    eta = data['eta']
    mu_T = data['mu_T']
    cov_TT = data['cov_TT']
    ess = data['ess']
    
    print(f"Data shapes: eta={eta.shape}, mu_T={mu_T.shape}")
    
    # Create training config
    config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        dropout_epochs=args.dropout_epochs
    )
    
    # Create trainer
    trainer = NoPropCTTrainer(config, eta.shape[1], mu_T.shape[1])
    
    # Prepare data
    train_data = (eta, mu_T, cov_TT, ess)
    
    # Initialize model
    key = jr.PRNGKey(42)
    params = trainer.ct_model.init(key, mu_T[:1], eta[:1], jnp.array([0.5]))
    
    print(f"Model initialized with {sum(x.size for x in jax.tree.leaves(params)):,} parameters")
    
    # Create optimizer
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)
    
    print(f"Starting training for {config.num_epochs} epochs...")
    
    # Create a simple loss function that doesn't use JIT
    def simple_loss_fn(params, x, target, key, use_dropout=True):
        # Sample random timesteps
        batch_size = x.shape[0]
        t = jr.uniform(key, (batch_size,), minval=0.0, maxval=1.0)
        
        # Get model output with dropout control
        model_output = trainer.ct_model.apply(params, target, x, t, training=use_dropout)
        
        # Simple MSE loss for now
        mse_loss = jnp.mean((model_output - target) ** 2)
        return mse_loss
    
    for epoch in range(config.num_epochs):
        # Determine if dropout should be active
        use_dropout = epoch < config.dropout_epochs
        
        # Sample batch
        batch_indices = jr.choice(key, eta.shape[0], (config.batch_size,), replace=False)
        batch_eta = eta[batch_indices]
        batch_mu_T = mu_T[batch_indices]
        
        # Compute loss and gradients
        key, subkey = jr.split(key)
        loss, grads = jax.value_and_grad(simple_loss_fn)(params, batch_eta, batch_mu_T, subkey, use_dropout)
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        if epoch % 20 == 0:
            dropout_status = "with dropout" if use_dropout else "no dropout"
            print(f"Epoch {epoch:3d}: Loss = {loss:.6f} ({dropout_status})")
    
    print("Training completed!")
    
    # Save model
    output_path = Path(config.output_dir) / "noprop_ct_model.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump({
            'params': params,
            'config': config,
            'model_config': trainer.ct_model.config
        }, f)
    
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()
