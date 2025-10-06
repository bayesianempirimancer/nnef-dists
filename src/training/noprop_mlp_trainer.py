#!/usr/bin/env python3
"""
NoProp MLP Training Script

This script provides a command-line interface for training NoProp MLP models.
It uses the NoProp training protocol with a simplified MLP architecture instead
of the Fisher flow field.

Usage:
    python src/training/noprop_mlp_trainer.py --data data/training_data.pkl --epochs 100
    python src/training/noprop_mlp_trainer.py --data data/my_data.pkl --hidden-sizes 64 64 64
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import time

# Handle imports for both module usage and direct script execution
if __name__ == "__main__":
    # When run as script, add project root to path
    import sys
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import jax.random as random
import optax

from src.configs.base_training_config import BaseTrainingConfig
from src.configs.noprop_mlp_config import NoProp_MLP_Config, create_noprop_mlp_config_from_args, add_noprop_mlp_arguments
from src.training.base_et_trainer import BaseETTrainer
from src.training.flow_model_trainer import FlowModelTrainer
from src.models.noprop_mlp_net import NoProp_MLP_Network
from scripts.plotting.plot_learning_curves import create_enhanced_learning_plot


class NoPropMLPTrainer:
    """
    Trainer for NoProp MLP models using the NoProp training protocol.
    
    This implements a custom training loop with:
    - Continuous-time sampling t ~ Uniform(0,1)
    - NoProp training protocols (noprop, flow_matching)
    - No time integration during training (only during inference)
    - Custom loss computation for NoProp training
    """
    
    def __init__(self, model: NoProp_MLP_Network, config: NoProp_MLP_Config, training_config: BaseTrainingConfig = None):
        self.model = model
        self.config = config
        self.training_config = training_config
        
        # Create optimizer
        learning_rate = training_config.learning_rate if training_config else 0.001
        self.optimizer = optax.adamw(
            learning_rate=learning_rate,
            weight_decay=0.0001,
            b1=0.9,
            b2=0.999,
            eps=1e-8
        )
        
        # Initialize random key
        self.rng = jax.random.PRNGKey(42)
    
    def train(self, 
              train_eta: jnp.ndarray, 
              train_mu_T: jnp.ndarray,
              val_eta: jnp.ndarray, 
              val_mu_T: jnp.ndarray,
              num_epochs: int,
              dropout_epochs: int = 0,
              learning_rate: Optional[float] = None,
              batch_size: Optional[int] = None,
              eval_steps: int = 10,
              save_steps: Optional[int] = None,
              output_dir: str = None) -> Dict[str, Any]:
        """
        Train the NoProp MLP model using the NoProp training protocol.
        """
        print(f"Starting NoProp MLP training for {num_epochs} epochs...")
        print(f"Training data: {train_eta.shape[0]} samples")
        print(f"Validation data: {val_eta.shape[0]} samples")
        print(f"Loss type: {self.config.loss_type}")
        print(f"Batch size: {batch_size if batch_size else 'Full batch'}")
        print("-" * 60)
        
        # Initialize model parameters
        self.rng, init_rng = jax.random.split(self.rng)
        # NoProp models need (z, eta, t) for initialization
        z_sample = jnp.zeros_like(train_mu_T[:1])
        t_sample = jnp.array(0.0)
        params = self.model.init(init_rng, z_sample, train_eta[:1], t_sample, training=True)
        
        # Initialize optimizer
        opt_state = self.optimizer.init(params)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training step
            self.rng, train_rng = jax.random.split(self.rng)
            
            # Create batch
            if batch_size and batch_size < len(train_eta):
                # Mini-batch training
                batch_indices = jax.random.choice(
                    train_rng, len(train_eta), shape=(batch_size,), replace=False
                )
                eta_batch = train_eta[batch_indices]
                mu_T_batch = train_mu_T[batch_indices]
            else:
                # Full batch training
                eta_batch = train_eta
                mu_T_batch = train_mu_T
            
            # Compute loss and gradients
            def loss_func(params):
                # Split RNG for both noise and dropout
                noise_rng, dropout_rng = jax.random.split(train_rng, 2)
                rngs = {'noise': noise_rng, 'dropout': dropout_rng}
                return self.model.loss(params, eta_batch, mu_T_batch, training=True, rngs=rngs)
            
            loss, grads = jax.value_and_grad(loss_func)(params)
            
            # Update parameters
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            train_losses.append(float(loss))
            
            # Validation
            if epoch % eval_steps == 0 or epoch == num_epochs - 1:
                val_rng = jax.random.PRNGKey(42)  # Fixed seed for validation
                val_noise_rng, val_dropout_rng = jax.random.split(val_rng, 2)
                val_rngs = {'noise': val_noise_rng, 'dropout': val_dropout_rng}
                val_loss = self.model.loss(params, val_eta, val_mu_T, training=False, rngs=val_rngs)
                val_losses.append(float(val_loss))
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                print(f"Epoch {epoch:3d}: Train Loss = {loss:.6f}, Val Loss = {val_loss:.6f}")
            else:
                print(f"Epoch {epoch:3d}: Train Loss = {loss:.6f}")
        
        # Compute inference time
        start_time = time.time()
        _ = self.model.predict(params, val_eta[:100], n_time_steps=20)
        inference_time = time.time() - start_time
        
        # Count parameters
        param_count = sum(x.size for x in jax.tree.leaves(params))
        
        results = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'params': params,
            'optimizer_state': opt_state,
            'epochs': num_epochs,
            'inference_time': inference_time,
            'inference_time_per_sample': inference_time / 100,
            'param_count': param_count,
            'best_val_loss': best_val_loss,
            'final_train_loss': train_losses[-1] if train_losses else 0.0,
            'final_val_loss': val_losses[-1] if val_losses else 0.0
        }
        
        print(f"\nTraining completed!")
        print(f"Final train loss: {results['final_train_loss']:.6f}")
        print(f"Final val loss: {results['final_val_loss']:.6f}")
        print(f"Best val loss: {results['best_val_loss']:.6f}")
        print(f"Model parameters: {param_count:,}")
        
        return results


def load_training_data(data_path: str) -> Tuple[Dict[str, Any], int, int]:
    """Load training data from pickle file."""
    print(f"Loading training data from: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    eta_dim = data['train']['eta'].shape[1]
    mu_dim = data['train']['mu_T'].shape[1]
    
    print(f"Loaded data with dimensions: eta_dim={eta_dim}, mu_dim={mu_dim}")
    print(f"Train data shapes: eta {data['train']['eta'].shape}, mu_T {data['train']['mu_T'].shape}")
    print(f"Val data shapes: eta {data['val']['eta'].shape}, mu_T {data['val']['mu_T'].shape}")
    
    return data, eta_dim, mu_dim


def create_configs_from_args(args, eta_dim: int, mu_dim: int) -> Tuple[NoProp_MLP_Config, BaseTrainingConfig]:
    """Create model and training configurations from command line arguments."""
    print("\nCreating configurations...")
    
    # Create model config using the config file
    model_config = create_noprop_mlp_config_from_args(args, eta_dim, mu_dim)

    # Create training config using the centralized method
    training_config = BaseETTrainer.create_training_config_from_args(args)
    
    return model_config, training_config


def train_model(model_config: NoProp_MLP_Config, training_config: BaseTrainingConfig, 
                data: Dict[str, Any], args, output_dir: str) -> Dict[str, Any]:
    """Train the NoProp MLP model."""
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Model Configuration:")
    print(f"  Type: noprop_mlp")
    print(f"  Architecture: {model_config.get_architecture_summary()}")
    print(f"  Supports dropout: {model_config.supports_dropout}")
    print(f"  Dropout rate: {model_config.dropout_rate}")
    print(f"  Time embed dim: {model_config.time_embed_dim}")
    print(f"  Time embed freq range: [{model_config.time_embed_min_freq}, {model_config.time_embed_max_freq}]")
    print(f"  Eta embed dim: {model_config.eta_embed_dim}")
    print(f"  Loss type: {model_config.loss_type}")
    print(f"\nTraining Configuration:")
    print(f"  Optimizer: {training_config.optimizer.upper()}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Loss function: {training_config.loss_function.upper()}")
    print(f"  Mini-batching: {training_config.use_mini_batching}")
    print(f"  Random sampling: {training_config.random_batch_sampling}")
    print(f"  Eval steps: {training_config.eval_steps}")
    print("="*60)
    
    # Create model and trainer
    model = NoProp_MLP_Network(config=model_config)
    trainer = NoPropMLPTrainer(model, model_config, training_config)
    
    # Train the model
    results = trainer.train(
        train_eta=data['train']['eta'],
        train_mu_T=data['train']['mu_T'],
        val_eta=data['val']['eta'],
        val_mu_T=data['val']['mu_T'],
        num_epochs=args.epochs,
        dropout_epochs=args.dropout_epochs,
        learning_rate=training_config.learning_rate,
        batch_size=training_config.batch_size,
        eval_steps=training_config.eval_steps,
        save_steps=getattr(args, 'save_steps', None),
        output_dir=output_dir
    )
    
    return results


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    # Start with base parser
    parser = BaseETTrainer.create_base_argument_parser("NoProp MLP Training Script")
    
    # Add NoProp MLP specific arguments
    parser = add_noprop_mlp_arguments(parser)
    
    return parser.parse_args()


def main():
    """Main training function."""
    print("NoProp MLP Training Script")
    print("="*60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Load training data
    print("1. Loading training data...")
    data, eta_dim, mu_dim = load_training_data(args.data)
    
    # Create configurations
    print("2. Creating configurations...")
    model_config, training_config = create_configs_from_args(args, eta_dim, mu_dim)
    
    # Set up output directory
    print("3. Setting up output directory...")
    if args.output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"artifacts/noprop_mlp_{timestamp}"
    else:
        output_dir = args.output_dir
    print(f"Output directory: {output_dir}")
    
    # Train model
    print("4. Training model...")
    results = train_model(model_config, training_config, data, args, output_dir)
    
    # Save results
    print("5. Saving results...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save training results
    with open(f"{output_dir}/training_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    print(f"Training results saved to: {output_dir}/training_results.pkl")
    
    # Save model config
    import json
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump(model_config.to_dict(), f, indent=2)
    print(f"Model config saved to: {output_dir}/config.json")
    
    # Save model parameters
    with open(f"{output_dir}/model_params.pkl", 'wb') as f:
        pickle.dump(results['params'], f)
    print(f"Model parameters saved to: {output_dir}/model_params.pkl")
    
    # Generate plots
    print("6. Generating training plots...")
    try:
        # Generate enhanced learning curves plot
        from scripts.load_model_and_data import load_model_and_data
        config, results, data, model, params, metadata = load_model_and_data(str(output_dir), args.data)
        save_path = Path(output_dir) / "learning_errors_enhanced.png"
        create_enhanced_learning_plot(config, results, data, model, params, metadata, save_path)
        print(f"Training plots saved to: {output_dir}")
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    print("\n" + "="*60)
    print("✅ SUCCESS! NoProp MLP training completed")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
