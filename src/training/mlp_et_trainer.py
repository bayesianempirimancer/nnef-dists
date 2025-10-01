#!/usr/bin/env python3
"""
MLP ET Training Script

This script provides a command-line interface for training MLP ET models.
It handles all file I/O, data loading, configuration setup, and architecture construction,
while delegating the core training logic to BaseETTrainer.

Usage:
    python src/training/mlp_et_trainer.py --data data/training_data.pkl --epochs 100
    python src/training/mlp_et_trainer.py --data data/my_data.pkl --hidden-sizes 128 64 --dropout-rate 0.2
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
from src.configs.mlp_et_config import MLP_ET_Config, create_mlp_et_config
from src.training.base_et_trainer import BaseETTrainer
from src.models.mlp_et_net import MLP_ET_Network
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
    
    # Extract data dimensions
    eta_dim = data['train']['eta'].shape[-1]
    mu_dim = data['train']['mu_T'].shape[-1]
    
    print(f"Loaded data with dimensions: eta_dim={eta_dim}, mu_dim={mu_dim}")
    print(f"Train data shapes: eta {data['train']['eta'].shape}, mu_T {data['train']['mu_T'].shape}")
    print(f"Val data shapes: eta {data['val']['eta'].shape}, mu_T {data['val']['mu_T'].shape}")
    
    return data, eta_dim, mu_dim


def create_configs_from_args(args, eta_dim: int, mu_dim: int) -> tuple[MLP_ET_Config, BaseTrainingConfig]:
    """
    Create model and training configurations from command line arguments.
    Only passes explicitly provided arguments, letting config defaults handle the rest.
    
    Args:
        args: Parsed command line arguments
        eta_dim: Input dimension (eta)
        mu_dim: Output dimension (mu_T)
        
    Returns:
        Tuple of (model_config, training_config)
    """
    print("\nCreating configurations...")
    
    # Build model config kwargs - only include explicitly provided arguments
    model_kwargs = {
        'input_dim': eta_dim,
        'output_dim': mu_dim,
    }
    
    # Parse model configuration arguments
    model_attributes = ['hidden_sizes', 'activation', 'dropout_rate', 'num_resnet_blocks', 'initialization_method']
    for attribute in model_attributes:
        if hasattr(args, attribute) and getattr(args, attribute) is not None:
            model_kwargs[attribute] = getattr(args, attribute)
    
    # Create model configuration
    model_config = create_mlp_et_config(**model_kwargs)
    
    # Create training configuration using BaseETTrainer method
    training_config = BaseETTrainer.create_training_config_from_args(args)
    
    # Print configuration summary
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
    
    return model_config, training_config


def train_model(
    model_config: MLP_ET_Config,
    training_config: BaseTrainingConfig,
    data: Dict[str, Any],
    output_dir: Path,
    epochs: int,
    dropout_epochs: int
) -> Dict[str, Any]:
    """
    Train the model using BaseETTrainer.
    
    Args:
        model_config: Model configuration
        training_config: Training configuration
        data: Training data
        output_dir: Output directory for saving results
        epochs: Total number of epochs
        dropout_epochs: Number of epochs with dropout
        
    Returns:
        Training results dictionary
    """
    print(f"\nStarting training for {epochs} epochs with {dropout_epochs} dropout epochs...")
    
    # Create model and trainer
    model = MLP_ET_Network(config=model_config)
    trainer = BaseETTrainer(model, model_config)  # Note: trainer still uses model config for now
    
    print(f"Model will be saved to: {output_dir}")
    
    # Start training
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
        output_dir=output_dir,
        training_config=training_config
    )
    
    # Save the model and results
    trainer.save_model(str(output_dir), results)
    
    return results


def main():
    """Main training script."""
    # Start with base parser
    parser = BaseETTrainer.create_base_argument_parser("Train MLP ET models with configurable architecture and training parameters")
    
    # Add epilog with examples
    parser.epilog = """
Examples:
  # Basic training with default settings
  python src/training/mlp_et_trainer.py --data data/training_data.pkl --epochs 100

  # Custom architecture and training parameters
  python src/training/mlp_et_trainer.py --data data/my_data.pkl --hidden-sizes 128 64 32 --dropout-rate 0.2 --learning-rate 0.001

  # Training with specific output directory
  python src/training/mlp_et_trainer.py --data data/training_data.pkl --output-dir artifacts/my_experiment --epochs 200 --dropout-epochs 100
    """
    
    # Add model-specific arguments
    parser.add_argument("--hidden-sizes", type=int, nargs="+",
                       help="Hidden layer sizes (default from config)")
    parser.add_argument("--activation", type=str, choices=["relu", "gelu", "swish", "tanh"],
                       help="Activation function (default from config)")
    parser.add_argument("--dropout-rate", type=float,
                       help="Dropout rate (default from config)")
    parser.add_argument("--num-resnet-blocks", type=int,
                       help="Number of ResNet blocks (default from config)")
    parser.add_argument("--initialization-method", type=str,
                       choices=["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "lecun_normal"],
                       help="Weight initialization method (default from config)")
    
    args = parser.parse_args()
    
    # Set random seed (use default if not provided)
    random_seed = args.random_seed if hasattr(args, 'random_seed') and args.random_seed is not None else 42
    random_key = random.PRNGKey(random_seed)
    
    print("MLP ET Training Script")
    print("="*60)
    
    # Load data
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
        output_dir = Path("artifacts") / f"mlp_et_{timestamp}"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Train model
    print("4. Training model...")
    epochs = args.epochs if hasattr(args, 'epochs') and args.epochs is not None else 100
    dropout_epochs = args.dropout_epochs if hasattr(args, 'dropout_epochs') and args.dropout_epochs is not None else 50
    results = train_model(
        model_config, 
        training_config, 
        data, 
        output_dir,
        epochs=epochs, 
        dropout_epochs=dropout_epochs
    )
    
    # Generate plots (default to True unless --no-plots is used)
    if not args.no_plots:
        print("5. Generating plots...")
        generate_plots(output_dir, args.data)
    else:
        print("5. Skipping plot generation (explicitly disabled)")
    
    print("\n" + "="*60)
    print("✅ SUCCESS! MLP ET training completed")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()