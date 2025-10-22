#!/usr/bin/env python3
"""
NoProp Flow Matching Training Script

This script provides a command-line interface for training NoProp Flow Matching models.
It uses the NoProp training protocol with a simplified CRN-MLP architecture

Usage:
    python src/models/noprop/train_fm.py --data data/training_data.pkl --epochs 100
    python src/models/noprop/train_fm.py --data data/my_data.pkl --hidden-sizes 64 64 64
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
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

# Model imports
if __name__ == "__main__":
    # When run as script, use absolute imports
    from src.models.noprop.fm import Config, NoPropFM
else:
    # When used as module, use relative imports
    from .fm import Config, NoPropFM


class NoPropFMTrainer:
    """
    Trainer for NoProp Flow Matching models using the NoProp training protocol.
    
    This implements a custom training loop with:
    - Continuous-time sampling t ~ Uniform(0,1)
    - Flow Matching training protocols
    - No time integration during training (only during inference)
    - Custom loss computation for NoProp Flow Matching training
    """
    
    def __init__(self, model: NoPropFM, config: Config, training_config: BaseTrainingConfig = None):
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
              dropout_epochs: Optional[int] = None,
              learning_rate: Optional[float] = None,
              batch_size: Optional[int] = None,
              eval_steps: int = 10,
              save_steps: Optional[int] = None,
              output_dir: str = None) -> Dict[str, Any]:
        """
        Train the NoProp MLP model using the NoProp training protocol.
        """
        # Determine effective dropout epochs based on new logic:
        # - If dropout_epochs is None: use dropout for entire training
        # - If dropout_epochs is specified: use dropout for first dropout_epochs, then turn off
        if dropout_epochs is None:
            effective_dropout_epochs = num_epochs  # Use dropout for entire training
        else:
            effective_dropout_epochs = dropout_epochs  # Use specified dropout epochs
        
        print(f"Starting NoProp Flow Matching training for {num_epochs} epochs...")
        print(f"Training data: {train_eta.shape[0]} samples")
        print(f"Validation data: {val_eta.shape[0]} samples")
        print(f"Loss type: {self.config.loss_type}")
        print(f"Batch size: {batch_size if batch_size else 'Full batch'}")
        print(f"Dropout epochs: {effective_dropout_epochs}")
        print("-" * 60)
        
        # Initialize model parameters
        self.rng, init_rng = jax.random.split(self.rng)
        # NoProp models need (z, eta, t) for initialization
        z_sample = jnp.zeros_like(train_mu_T[:1])
        t_sample = jnp.array([0.0])  # Make it an array with batch dimension
        params = self.model.init(init_rng, z_sample, train_eta[:1], t_sample, training=True)
        
        # Initialize optimizer
        opt_state = self.optimizer.init(params)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Determine if dropout should be active
            use_dropout = epoch < effective_dropout_epochs
            
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
                # For FM model, use compute_loss method with (x, target, key) signature
                return self.model.compute_loss(params, eta_batch, mu_T_batch, train_rng)
            
            (loss, metrics), grads = jax.value_and_grad(loss_func, has_aux=True)(params)
            
            # Update parameters
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            train_losses.append(float(loss))
            
            # Validation - compute every epoch
            val_rng = jax.random.PRNGKey(42)  # Fixed seed for validation
            val_loss, val_metrics = self.model.compute_loss(params, val_eta, val_mu_T, val_rng)
            val_losses.append(float(val_loss))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            # Print detailed info every eval_steps, otherwise just basic info
            if epoch % eval_steps == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch:3d}: Train Loss = {loss:.6f}, Val Loss = {val_loss:.6f}")
            else:
                print(f"Epoch {epoch:3d}: Train Loss = {loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Compute inference time
        start_time = time.time()
        _ = self.model.predict(params, val_eta[:100], num_steps=20)
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



def create_configs_from_args(args, eta_dim: int, mu_dim: int) -> Tuple[Config, BaseTrainingConfig]:
    """Create model and training configurations from command line arguments."""
    print("\nCreating configurations...")
    
    # Create model config with all required values
    config_kwargs = {
        'input_dim': eta_dim,
        'output_dim': mu_dim,
    }
    
    # Set hidden sizes from command line arguments
    if hasattr(args, 'hidden_sizes'):
        config_kwargs['hidden_sizes'] = tuple(args.hidden_sizes)
    if hasattr(args, 'dropout_rate'):
        config_kwargs['model_dropout_rate'] = args.dropout_rate
    if hasattr(args, 'activation'):
        config_kwargs['activation'] = args.activation
    # Flow Matching doesn't use noise schedules
    
    # Add missing attributes for base config compatibility (only if they exist in Config)
    # Note: Config class inherits from BaseConfig, so it should have these fields
    # But we need to be careful about which fields actually exist
    
    # Create the model config with all values
    model_config = Config(**config_kwargs)

    # Create training config using the centralized method
    training_config = BaseETTrainer.create_training_config_from_args(args)
    
    return model_config, training_config


def train_model(model_config: Config, training_config: BaseTrainingConfig, 
                data: Dict[str, Any], output_dir: Path,
                epochs: int, dropout_epochs: Optional[int]) -> Dict[str, Any]:
    """
    Train the NoProp MLP model.
    
    Args:
        model_config: Model configuration
        training_config: Training configuration
        data: Training data dictionary
        output_dir: Output directory for saving results
        epochs: Total number of epochs
        dropout_epochs: Number of epochs with dropout (None = use dropout for entire training)
        
    Returns:
        Training results dictionary
    """
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Model Configuration:")
    print(f"  Type: noprop_fm")
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
    
    if dropout_epochs is None:
        print(f"\nStarting training for {epochs} epochs with dropout for entire training...")
    else:
        print(f"\nStarting training for {epochs} epochs with {dropout_epochs} dropout epochs...")
    
    # Create Flow Matching model (no noise schedule needed for FM)
    model = NoPropFM(config=model_config)
    trainer = NoPropFMTrainer(model, model_config, training_config)
    
    print(f"Model will be saved to: {output_dir}")
    
    # Train the model
    results = trainer.train(
        train_eta=data['train']['eta'],
        train_mu_T=data['train']['mu_T'],
        val_eta=data['val']['eta'],
        val_mu_T=data['val']['mu_T'],
        num_epochs=epochs,
        dropout_epochs=dropout_epochs,
        learning_rate=training_config.learning_rate,
        batch_size=training_config.batch_size,
        eval_steps=training_config.eval_steps,
        save_steps=getattr(training_config, 'save_steps', None),
        output_dir=str(output_dir)
    )
    
    return results, model


def add_noprop_fm_arguments(parser):
    """Add NoProp Flow Matching specific arguments to parser."""
    # Add model-specific arguments
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64, 64], 
                       help="Hidden layer sizes (default: 64 64 64)")
    parser.add_argument("--dropout-rate", type=float, default=0.0, 
                       help="Dropout rate (default: 0.0)")
    parser.add_argument("--activation", type=str, default="swish", 
                       help="Activation function (default: swish)")
    # Flow Matching doesn't use noise schedules, so remove that argument
    return parser


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    # Start with base parser
    parser = BaseETTrainer.create_base_argument_parser("NoProp Flow Matching Training Script")
    
    # Add NoProp Flow Matching specific arguments
    parser = add_noprop_fm_arguments(parser)
    
    return parser.parse_args()


def main():
    """Main training function."""
    print("NoProp Flow Matching Training Script")
    print("="*60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Load training data
    print("1. Loading training data...")
    data, eta_dim, mu_dim = BaseETTrainer.load_training_data(args.data)
    
    # Create configurations
    print("2. Creating configurations...")
    model_config, training_config = create_configs_from_args(args, eta_dim, mu_dim)
    
    # Set up output directory
    print("3. Setting up output directory...")
    if args.output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"artifacts/noprop_fm_{timestamp}"
    else:
        output_dir = args.output_dir
    print(f"Output directory: {output_dir}")
    
    # Train model
    print("4. Training model...")
    epochs = args.epochs if hasattr(args, 'epochs') and args.epochs is not None else 100
    dropout_epochs = args.dropout_epochs if hasattr(args, 'dropout_epochs') and args.dropout_epochs is not None else None
    results, model = train_model(
        model_config, 
        training_config, 
        data, 
        Path(output_dir),
        epochs=epochs, 
        dropout_epochs=dropout_epochs
    )
    
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
        # Use the specialized plotting function for diffusion models
        from src.utils.plotting_fm import create_learning_plot_fm
        
        # Prepare data for plotting
        plot_data = {
            'train': {
                'eta': data['train']['eta'],
                'mu_T': data['train']['mu_T'],
                'cov_TT': data['train']['cov_TT'],
                'ess': data['train']['ess']
            },
            'val': {
                'eta': data['val']['eta'],
                'mu_T': data['val']['mu_T'],
                'cov_TT': data['val']['cov_TT'],
                'ess': data['val']['ess']
            },
            'test': {
                'eta': data['test']['eta'],
                'mu_T': data['test']['mu_T'],
                'cov_TT': data['test']['cov_TT'],
                'ess': data['test']['ess']
            }
        }
        
        # Create metadata
        metadata = {
            'total_expected_MSE_train': None,  # Will be calculated from cov_TT and ess
            'total_expected_MSE_val': None
        }
        
        # Create comprehensive learning plot for diffusion model
        plot_path = Path(output_dir) / "learning_analysis.png"
        create_learning_plot_fm(
            config=model_config.__dict__,
            results=results,
            data=plot_data,
            model=model,
            params=results['params'],
            metadata=metadata,
            save_path=str(plot_path)
        )
        
        print(f"Comprehensive learning analysis plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ SUCCESS! NoProp Flow Matching training completed")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
