#!/usr/bin/env python3
"""
NoProp Geometric Flow ET Training Script

This script provides a command-line interface for training NoProp Geometric Flow ET models.
It handles all file I/O, data loading, configuration setup, and architecture construction,
while implementing the diffusion-like NoProp training protocol.

Usage:
    python src/training/noprop_geometric_flow_et_trainer.py --data data/training_data.pkl --epochs 100
    python src/training/noprop_geometric_flow_et_trainer.py --data data/my_data.pkl --n-time-steps 20 --loss-type flow_matching
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
from src.configs.noprop_geometric_flow_et_config import NoProp_Geometric_Flow_ET_Config, create_noprop_geometric_flow_et_config
from src.training.base_et_trainer import BaseETTrainer
from src.models.noprop_geometric_flow_et_net import NoProp_Geometric_Flow_ET_Network
from scripts.plotting.plot_learning_curves import create_enhanced_learning_plot


class NoPropGeometricFlowETTrainer(BaseETTrainer):
    """
    Trainer for NoProp Geometric Flow ET models using diffusion-like training protocols.
    
    This implements a custom training loop with:
    - Continuous-time sampling t ~ Uniform(0,1)
    - NoProp training protocols (flow_matching, geometric_flow, simple_target)
    - No time integration during training (only during inference)
    - Custom loss computation for diffusion-like training
    """
    
    def __init__(self, model: NoProp_Geometric_Flow_ET_Network, config: NoProp_Geometric_Flow_ET_Config, training_config: BaseTrainingConfig = None):
        super().__init__(model, config)
        self.model = model
        self.config = config
        self.training_config = training_config
        
        # Create optimizer with learning rate from training config
        learning_rate = training_config.learning_rate if training_config else 0.05
        self.optimizer = optax.adamw(
            learning_rate=learning_rate,
            weight_decay=0.0001,
            b1=0.9,
            b2=0.999,
            eps=1e-8
        )
        
        # Training state
        self.trainerstate = None
    
    def compute_loss(self, params: Dict, batch: Dict[str, jnp.ndarray], 
                    training: bool = True) -> Tuple[float, Dict[str, Any]]:
        """
        Compute NoProp loss using the specified loss strategy.
        
        Args:
            params: Model parameters
            batch: Training batch with 'eta' and 'mu_T' keys
            training: Whether in training mode
            
        Returns:
            Tuple of (total_loss, additional_metrics)
        """
        eta = batch['eta']
        mu_T = batch['mu_T']
        # Note: t is generated internally by the model's loss functions
        
        # Create RNG for dropout if training
        rngs = {}
        if training:
            self.rng, dropout_rng = jax.random.split(self.rng)
            rngs['dropout'] = dropout_rng
        
        # Use the model's loss method (which handles both NoProp and flow matching)
        loss = self.model.loss(params, eta, mu_T, training=training, rngs=rngs)
        
        metrics = {
            'noprop_loss': loss,
            'loss_type': self.config.loss_type
        }
        
        return loss, metrics
    
    def train_step(self, params: Dict, opt_state: Any, batch: Dict[str, jnp.ndarray]) -> Tuple[Dict, Any, Dict]:
        """
        Single NoProp training step.
        
        Args:
            params: Model parameters
            opt_state: Optimizer state
            batch: Training batch
            
        Returns:
            Tuple of (updated_params, updated_opt_state, metrics)
        """
        # Compute loss and gradients (time sampling is handled internally by the model)
        (loss, loss_metrics), grads = jax.value_and_grad(
            self.compute_loss, has_aux=True
        )(params, batch, training=True)
        
        # Apply gradients
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        # Combine metrics
        metrics = {
            'loss': loss,
            **loss_metrics
        }
        
        return new_params, new_opt_state, metrics
    
    def _save_checkpoint(self, params: Dict, opt_state: Any, epoch: int, 
                        train_loss: float, val_loss: float, output_dir: str):
        """
        Save training checkpoint.
        
        Args:
            params: Model parameters
            opt_state: Optimizer state
            epoch: Current epoch
            train_loss: Current training loss
            val_loss: Current validation loss
            output_dir: Directory to save to
        """
        checkpoint_dir = Path(output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'params': params,
            'opt_state': opt_state,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config.__dict__
        }
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pkl"
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)
        
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    def train(self, 
              train_eta: jnp.ndarray, 
              train_mu_T: jnp.ndarray,
              val_eta: jnp.ndarray, 
              val_mu_T: jnp.ndarray,
              epochs: int,
              batch_size: int = 32,
              eval_steps: int = 10,
              dropout_epochs: int = None,
              save_steps: int = None,
              output_dir: str = None) -> Dict[str, Any]:
        """
        Train the NoProp model using diffusion-like training protocol.
        
        Args:
            train_eta: Training natural parameters
            train_mu_T: Training target statistics
            val_eta: Validation natural parameters
            val_mu_T: Validation target statistics
            epochs: Number of training epochs
            batch_size: Batch size for training
            eval_steps: Steps between evaluations
            dropout_epochs: Epochs to use dropout (None = use throughout)
            save_steps: Steps between checkpoint saves (None = no checkpoints)
            output_dir: Directory to save results
            
        Returns:
            Training results dictionary
        """
        print(f"Starting NoProp training for {epochs} epochs...")
        print(f"Training data: {train_eta.shape[0]} samples")
        print(f"Validation data: {val_eta.shape[0]} samples")
        learning_rate = self.training_config.learning_rate if self.training_config else 0.05
        print(f"Learning rate: {learning_rate}")
        print(f"Batch size: {batch_size}")
        print(f"Loss type: {self.config.loss_type}")
        print(f"Dropout epochs: {dropout_epochs if dropout_epochs else 'All epochs'}")
        print(f"Validation every: {eval_steps} epochs")
        print(f"Progress will be shown every epoch with time tracking")
        print("-" * 80)
        
        # Initialize model parameters
        self.rng, init_rng = jax.random.split(self.rng)
        params = self.model.init(init_rng, train_eta[:1], train_eta[:1], train_mu_T[:1], 0.0)
        
        # Initialize optimizer state
        opt_state = self.optimizer.init(params)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            # Determine if dropout should be used
            use_dropout = dropout_epochs is None or epoch < dropout_epochs
            
            # Training step
            epoch_train_losses = []
            
            # Create batches
            n_train = train_eta.shape[0]
            indices = jax.random.permutation(self.rng, n_train)
            self.rng, _ = jax.random.split(self.rng)
            
            for i in range(0, n_train, batch_size):
                batch_indices = indices[i:i+batch_size]
                
                # Create NoProp batch
                batch = {
                    'eta': train_eta[batch_indices],
                    'mu_T': train_mu_T[batch_indices]
                }
                
                # Training step
                params, opt_state, metrics = self.train_step(params, opt_state, batch)
                epoch_train_losses.append(metrics['loss'])
            
            avg_train_loss = jnp.mean(jnp.array(epoch_train_losses))
            train_losses.append(float(avg_train_loss))
            
            # Validation
            if epoch % eval_steps == 0 or epoch == epochs - 1:
                # For validation, we need to predict final mu from eta
                val_predictions = self.model.predict(params, val_eta)
                val_loss = jnp.mean((val_predictions - val_mu_T) ** 2)
                val_losses.append(float(val_loss))
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                dropout_status = "ON" if use_dropout else "OFF"
                elapsed_time = time.time() - start_time
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch:3d}/{epochs}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f} (Dropout: {dropout_status}) [Time: {elapsed_time:.1f}s, Epoch: {epoch_time:.1f}s]")
                
                # Save checkpoint if requested
                if save_steps is not None and output_dir is not None and epoch % save_steps == 0:
                    self._save_checkpoint(params, opt_state, epoch, avg_train_loss, val_loss, output_dir)
            else:
                # Show progress every epoch for better monitoring
                dropout_status = "ON" if use_dropout else "OFF"
                elapsed_time = time.time() - start_time
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch:3d}/{epochs}: Train Loss = {avg_train_loss:.6f} (Dropout: {dropout_status}) [Time: {elapsed_time:.1f}s, Epoch: {epoch_time:.1f}s]")
                
                # Save checkpoint if requested (even without validation)
                if save_steps is not None and output_dir is not None and epoch % save_steps == 0:
                    # Use last validation loss if available, otherwise use train loss
                    last_val_loss = val_losses[-1] if val_losses else avg_train_loss
                    self._save_checkpoint(params, opt_state, epoch, avg_train_loss, last_val_loss, output_dir)
        
        training_time = time.time() - start_time
        
        # Compute inference time
        print("\nComputing inference time...")
        inference_start = time.time()
        test_batch = val_eta[:100]
        _ = self.model.predict(params, test_batch)
        inference_time = time.time() - inference_start
        inference_time_per_sample = inference_time / 100
        
        # Count parameters
        param_count = sum(x.size for x in jax.tree.leaves(params))
        
        print(f"  Inference time: {inference_time:.6f} seconds for batch of 100")
        print(f"  Inference time per sample: {inference_time_per_sample:.6f} seconds")
        print(f"  Model parameters: {param_count:,}")
        
        # Prepare results
        results = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1] if val_losses else None,
            'training_time': training_time,
            'inference_time': inference_time,
            'inference_time_per_sample': inference_time_per_sample,
            'parameter_count': param_count,
            'config': self.config.to_dict(),
            'model_params': params
        }
        
        print(f"\nTraining Results:")
        print(f"  Training time: {training_time:.2f} seconds")
        print(f"  Final train loss: {train_losses[-1]:.6f}")
        print(f"  Final val loss: {val_losses[-1] if val_losses else 'N/A':.6f}")
        print(f"  Best val loss: {best_val_loss:.6f}")
        
        return results


def load_training_data(data_path: str) -> tuple[Dict[str, Any], int, int]:
    """
    Load training data from pickle file.
    
    Args:
        data_path: Path to the pickle file containing training data
        
    Returns:
        Tuple of (data_dict, eta_dim, mu_dim)
    """
    print(f"Loading training data from: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    eta_dim = data['train']['eta'].shape[1]
    mu_dim = data['train']['mu_T'].shape[1]
    
    print(f"Loaded data with dimensions: eta_dim={eta_dim}, mu_dim={mu_dim}")
    print(f"Train data shapes: eta {data['train']['eta'].shape}, mu_T {data['train']['mu_T'].shape}")
    print(f"Val data shapes: eta {data['val']['eta'].shape}, mu_T {data['val']['mu_T'].shape}")
    
    return data, eta_dim, mu_dim


def create_configs_from_args(args, eta_dim: int, mu_dim: int) -> tuple[NoProp_Geometric_Flow_ET_Config, BaseTrainingConfig]:
    """
    Create model and training configurations from command line arguments.
    
    Args:
        args: Parsed command line arguments
        eta_dim: Input dimension (natural parameters)
        mu_dim: Output dimension (target statistics)
        
    Returns:
        Tuple of (model_config, training_config)
    """
    print("\nCreating configurations...")
    
    # Create model config with NoProp-specific parameters
    model_kwargs = {
        'input_dim': eta_dim,
        'output_dim': mu_dim,
        'x_shape': (eta_dim,)  # Set x_shape based on eta dimension
    }
    
    # Handle temporal embedding disable flag
    if hasattr(args, 'disable_temporal_embedding') and args.disable_temporal_embedding:
        model_kwargs['time_embed_dim'] = 1  # Force constant embedding
    
    # Parse model configuration arguments
    model_attributes = ['n_time_steps', 'smoothness_weight', 'matrix_rank', 'time_embed_dim',
                       'time_embed_min_freq', 'time_embed_max_freq',
                       'architecture', 'hidden_sizes', 'activation', 'use_layer_norm', 
                       'layer_norm_type', 'initialization_method', 'dropout_rate',
                       'loss_type']
    for attribute in model_attributes:
        if hasattr(args, attribute) and getattr(args, attribute) is not None:
            model_kwargs[attribute] = getattr(args, attribute)
    
    # Create model configuration
    model_config = create_noprop_geometric_flow_et_config(**model_kwargs)

    # Create training config using the centralized method
    training_config = BaseETTrainer.create_training_config_from_args(args)
    
    return model_config, training_config


def train_model(model_config: NoProp_Geometric_Flow_ET_Config, training_config: BaseTrainingConfig, 
                data: Dict[str, Any], args, output_dir: str) -> Dict[str, Any]:
    """
    Train the NoProp Geometric Flow ET model using the provided configurations and data.
    
    Args:
        model_config: Model configuration
        training_config: Training configuration
        data: Training data dictionary
        args: Command line arguments
        output_dir: Output directory for saving results
        
    Returns:
        Training results dictionary
    """
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Model Configuration:")
    print(f"  Type: {model_config.model_type}")
    print(f"  Architecture: {model_config.get_architecture_summary()}")
    print(f"  Supports dropout: {model_config.supports_dropout}")
    print(f"  Dropout rate: {model_config.dropout_rate}")
    print(f"  Time steps: {model_config.n_time_steps}")
    print(f"  Smoothness weight: {model_config.smoothness_weight}")
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
    model = NoProp_Geometric_Flow_ET_Network(config=model_config)
    trainer = NoPropGeometricFlowETTrainer(model, model_config, training_config)
    
    # Train the model
    results = trainer.train(
        train_eta=data['train']['eta'],
        train_mu_T=data['train']['mu_T'],
        val_eta=data['val']['eta'],
        val_mu_T=data['val']['mu_T'],
        epochs=args.epochs,
        batch_size=training_config.batch_size,
        eval_steps=training_config.eval_steps,
        dropout_epochs=args.dropout_epochs,
        save_steps=getattr(args, 'save_steps', None),
        output_dir=output_dir
    )
    
    return results


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    # Start with base parser
    parser = BaseETTrainer.create_base_argument_parser("NoProp Geometric Flow ET Training Script")
    
    # Add model-specific arguments
    parser.add_argument("--n-time-steps", type=int,
                       help="Number of time steps for ODE integration (default from config)")
    parser.add_argument("--smoothness-weight", type=float,
                       help="Weight for smoothness penalty (default from config)")
    parser.add_argument("--matrix-rank", type=int,
                       help="Rank of the flow matrix (default from config)")
    parser.add_argument("--time-embed-dim", type=int,
                       help="Time embedding dimension (default from config)")
    parser.add_argument("--time-embed-min-freq", type=float,
                       help="Minimum frequency for log frequency time embedding (default from config)")
    parser.add_argument("--time-embed-max-freq", type=float,
                       help="Maximum frequency for log frequency time embedding (default from config)")
    parser.add_argument("--disable-temporal-embedding", action="store_true",
                       help="Disable temporal embedding (force time_embed_dim=1, use constant embedding)")
    parser.add_argument("--architecture", type=str, choices=["mlp"],
                       help="Network architecture (default from config)")
    parser.add_argument("--hidden-sizes", type=int, nargs="+",
                       help="Hidden layer sizes (default from config)")
    parser.add_argument("--activation", type=str, choices=["relu", "gelu", "swish", "tanh", "none", "linear"],
                       help="Activation function (default from config)")
    parser.add_argument("--use-layer-norm", action="store_true",
                       help="Use layer normalization (default from config)")
    parser.add_argument("--layer-norm-type", type=str,
                       choices=["none", "weak_layer_norm", "rms_norm", "group_norm", "instance_norm",
                               "weight_norm", "spectral_norm", "adaptive_norm", "pre_norm", "post_norm"],
                       help="Type of layer normalization (default from config)")
    parser.add_argument("--dropout-rate", type=float,
                       help="Dropout rate (default from config)")
    parser.add_argument("--initialization-method", type=str,
                       choices=["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "lecun_normal"],
                       help="Weight initialization method (default from config)")
    
    # NoProp-specific arguments
    parser.add_argument("--loss-type", type=str, choices=["flow_matching", "geometric_flow", "simple_target"],
                       help="NoProp loss type (default from config)")
    
    return parser.parse_args()


def main():
    """Main training function."""
    print("NoProp Geometric Flow ET Training Script")
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
        output_dir = f"artifacts/noprop_geometric_flow_et_{timestamp}"
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
        pickle.dump(results['model_params'], f)
    print(f"Model parameters saved to: {output_dir}/model_params.pkl")
    
    # Generate plots
    print("6. Generating plots...")
    try:
        # Generate enhanced learning curves plot
        from scripts.load_model_and_data import load_model_and_data
        config, results, data, model, params, metadata = load_model_and_data(str(output_dir), args.data)
        save_path = Path(output_dir) / "learning_errors_enhanced.png"
        create_enhanced_learning_plot(config, results, data, model, params, metadata, save_path)
        print("✅ Training plots generated successfully!")
    except Exception as e:
        print(f"⚠️  Warning: Could not generate plots: {e}")
    
    print("\n" + "="*60)
    print("✅ SUCCESS! NoProp Geometric Flow ET training completed")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
