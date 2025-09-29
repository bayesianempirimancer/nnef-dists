#!/usr/bin/env python3
"""
Quadratic ET Training Script

This script provides a command-line interface for training Quadratic ET models.
It handles all file I/O, data loading, configuration setup, and architecture construction,
while delegating the core training logic to BaseETTrainer.

Usage:
    python src/training/quadratic_et_trainer.py --data data/training_data.pkl --epochs 100
    python src/training/quadratic_et_trainer.py --data data/my_data.pkl --hidden-sizes 128 64 --use-activation
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional

# Handle imports for both module usage and direct script execution
if __name__ == "__main__":
    # When run as script, add project root to path
    import sys
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

import jax.numpy as jnp
import jax.random as random

from src.configs.base_training_config import BaseTrainingConfig
from src.configs.quadratic_et_config import Quadratic_ET_Config, create_quadratic_et_config, create_quadratic_training_config_from_args
from src.training.base_et_trainer import BaseETTrainer
from src.models.quadratic_et_net import Quadratic_ET_Network
from scripts.plotting.generate_plots import generate_plots


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
    
    # Extract dimensions
    eta_dim = data['train']['eta'].shape[-1]
    mu_dim = data['train']['mu_T'].shape[-1]
    
    print(f"Loaded data with dimensions: eta_dim={eta_dim}, mu_dim={mu_dim}")
    print(f"Train data shapes: eta {data['train']['eta'].shape}, mu_T {data['train']['mu_T'].shape}")
    print(f"Val data shapes: eta {data['val']['eta'].shape}, mu_T {data['val']['mu_T'].shape}")
    
    return data, eta_dim, mu_dim


def create_configs_from_args(args, eta_dim: int, mu_dim: int) -> tuple[Quadratic_ET_Config, BaseTrainingConfig]:
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
    
    # Create model config with Quadratic-specific parameters
    model_kwargs = {
        'input_dim': eta_dim,
        'output_dim': mu_dim
    }
    
    # Parse model configuration arguments
    model_attributes = ['hidden_sizes', 'activation', 'use_layer_norm', 
                       'use_quadratic_norm', 'dropout_rate', 'num_resnet_blocks', 
                       'share_parameters', 'weight_residual', 'residual_weight', 'initialization_method']
    for attribute in model_attributes:
        if hasattr(args, attribute) and getattr(args, attribute) is not None:
            model_kwargs[attribute] = getattr(args, attribute)
    
    # Provide default hidden_sizes if not specified
    if 'hidden_sizes' not in model_kwargs:
        model_kwargs['hidden_sizes'] = [16]
    
    # Create model configuration
    model_config = create_quadratic_et_config(**model_kwargs)

    # Create training config using the quadratic-specific method (defaults to RMSprop)
    training_config = create_quadratic_training_config_from_args(args)
    
    return model_config, training_config


def train_model(model_config: Quadratic_ET_Config, training_config: BaseTrainingConfig, 
                data: Dict[str, Any], args, output_dir: str) -> Dict[str, Any]:
    """
    Train the Quadratic ET model using the provided configurations and data.
    
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
    model = Quadratic_ET_Network(config=model_config)
    trainer = BaseETTrainer(model, model_config)
    
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
        save_steps=training_config.save_steps,
        output_dir=output_dir,
        training_config=training_config
    )
    
    # Save the model and results
    trainer.save_model(output_dir, results)
    
    return results


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Quadratic ET Training Script")
    
    # Required arguments
    parser.add_argument("--data", type=str, required=True,
                       help="Path to training data pickle file")
    parser.add_argument("--epochs", type=int, required=True,
                       help="Number of training epochs")
    
    # Optional arguments with defaults
    parser.add_argument("--dropout-epochs", type=int, default=0,
                       help="Number of epochs to use dropout (default: 0)")
    parser.add_argument("--output-dir", type=str,
                       help="Output directory for results (default: auto-generated)")
    
    # Optimizer arguments
    parser.add_argument("--learning-rate", type=float,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int,
                       help="Batch size")
    parser.add_argument("--optimizer", type=str, 
                       choices=["adam", "adamw", "sgd", "rmsprop"],
                       help="Optimizer type")
    parser.add_argument("--weight-decay", type=float,
                       help="Weight decay")
    parser.add_argument("--beta1", type=float,
                       help="Adam beta1 parameter")
    parser.add_argument("--beta2", type=float,
                       help="Adam beta2 parameter")
    parser.add_argument("--eps", type=float,
                       help="Adam epsilon parameter")
    parser.add_argument("--loss-function", type=str, 
                       choices=["mse", "mae", "huber", "model_specific"],
                       help="Loss function")
    parser.add_argument("--l1-reg-weight", type=float,
                       help="L1 regularization weight")
    
    # Model architecture arguments
    parser.add_argument("--hidden-sizes", type=int, nargs="+",
                       help="Hidden layer sizes for quadratic blocks")
    parser.add_argument("--activation", type=str, choices=["relu", "gelu", "swish", "tanh", "none", "linear"],
                       help="Activation function")
    parser.add_argument("--use-layer-norm", action="store_true",
                       help="Use layer normalization")
    parser.add_argument("--use-quadratic-norm", action="store_true",
                       help="Use quadratic normalization")
    parser.add_argument("--dropout-rate", type=float,
                       help="Dropout rate")
    parser.add_argument("--num-resnet-blocks", type=int,
                       help="Number of ResNet blocks")
    parser.add_argument("--share-parameters", action="store_true",
                       help="Share parameters across ResNet blocks")
    parser.add_argument("--weight-residual", action="store_true",
                       help="Weight residual connections")
    parser.add_argument("--residual-weight", type=float,
                       help="Weight for residual connections")
    parser.add_argument("--initialization-method", type=str,
                       choices=["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "lecun_normal"],
                       help="Weight initialization method")
    
    # Training control arguments
    parser.add_argument("--use-mini-batching", action="store_true",
                       help="Enable mini-batching")
    parser.add_argument("--no-mini-batching", dest="use_mini_batching", action="store_false",
                       help="Disable mini-batching")
    parser.add_argument("--random-batch-sampling", action="store_true",
                       help="Use random batch sampling")
    parser.add_argument("--sequential-batch-sampling", dest="random_batch_sampling", action="store_false",
                       help="Use sequential batch sampling")
    parser.add_argument("--eval-steps", type=int,
                       help="Evaluation frequency in epochs")
    parser.add_argument("--save-steps", type=int,
                       help="Model saving frequency in epochs")
    parser.add_argument("--early-stopping-patience", type=int,
                       help="Early stopping patience in epochs")
    parser.add_argument("--early-stopping-min-delta", type=float,
                       help="Early stopping minimum delta")
    parser.add_argument("--log-frequency", type=int,
                       help="Logging frequency in epochs")
    parser.add_argument("--random-seed", type=int,
                       help="Random seed")    
    # Plotting arguments
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip generating plots")
    parser.add_argument("--plot-data", type=str,
                       help="Data file for plotting (default: same as training data)")
    
    return parser.parse_args()


def main():
    """Main training function."""
    print("Quadratic ET Training Script")
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
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Auto-generate output directory name
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"artifacts/quadratic_et_{timestamp}")
    
    print(f"Output directory: {output_dir}")
    
    # Train model
    print("4. Training model...")
    results = train_model(model_config, training_config, data, args, str(output_dir))
    
    # Generate plots
    if not args.no_plots:
        print("5. Generating plots...")
        plot_data_path = args.plot_data if args.plot_data else args.data
        generate_plots(output_dir, plot_data_path)
        print("✅ Training plots generated successfully!")
    else:
        print("5. Skipping plots (--no-plots specified)")
    
    print("\n" + "="*60)
    print("✅ SUCCESS! Quadratic ET training completed")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
