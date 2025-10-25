"""
Training script for NoProp models (CT, FM, DF).

This script provides a single interface to train any of the three model types
with identical protocols for fair comparison.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple

import jax
import jax.numpy as jnp

# Add the project root to the path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.models.base_trainer import BaseETTrainer
from src.models.base_training_config import BaseTrainingConfig
from src.models.noprop.trainer import NoPropTrainer, create_config
from src.models.noprop.ct import NoPropCT
from src.models.noprop.fm import NoPropFM
from src.models.noprop.df import NoPropDF
from src.embeddings.noise_schedules import (
    LinearNoiseSchedule, SigmoidNoiseSchedule, CosineNoiseSchedule,
    SimpleLearnableNoiseSchedule, LearnableNoiseSchedule
)


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common arguments for all model types."""
    
    # Data arguments
    parser.add_argument("--data", type=str, required=True,
                       help="Path to training data file (.pkl)")
    
    # Model selection
    parser.add_argument("--model-type", type=str, required=True,
                       choices=["CT", "FM", "DF"],
                       help="Model type to train (CT, FM, or DF)")
    parser.add_argument("--model", type=str, default="ConditionalResnet_MLP",
                       choices=["ConditionalResnet_MLP", "NaturalFlow", "GeometricFlow", "PotentialFlow"],
                       help="Model architecture to use (default: ConditionalResnet_MLP)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=400,
                       help="Number of training epochs (default: 400)")
    parser.add_argument("--dropout-epochs", type=int, default=300,
                       help="Number of epochs with dropout (default: 300)")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate (default: 0.001)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size (default: None, uses full dataset)")
    parser.add_argument("--eval-steps", type=int, default=10,
                       help="Steps between detailed evaluation (default: 10)")
    
    # Model-specific arguments
    parser.add_argument("--loss-type", type=str, default="mse",
                       choices=["mse", "snr_weighted_mse"],
                       help="Loss function type (default: mse)")
    parser.add_argument("--noise-schedule", type=str, default="linear",
                       choices=["linear", "sigmoid", "cosine", "simple_learnable", "learnable"],
                       help="Noise schedule type (default: linear)")
    parser.add_argument("--sigma-t", type=float, default=0.1,
                       help="Noise level for FM/DF models (default: 0.1)")
    parser.add_argument("--reg-weight", type=float, default=0.0,
                       help="Regularization weight (default: 0.0)")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: auto-generated)")
    
    return parser


def create_model(model_type: str, model: str, config, z_shape: Tuple[int, ...], x_ndims: int = 1):
    """Create the specified model type."""
    
    if model_type.upper() == "CT":
        # Use specified model architecture for CT
        return NoPropCT(config=config, z_shape=z_shape, x_ndims=x_ndims, model=model, noise_schedule=config.noise_schedule)
    
    elif model_type.upper() == "FM":
        # Use specified model architecture for FM
        return NoPropFM(config=config, z_shape=z_shape, x_ndims=x_ndims, model=model)
    
    elif model_type.upper() == "DF":
        # Use specified model architecture for DF
        return NoPropDF(config=config, z_shape=z_shape, x_ndims=x_ndims, model=model)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="NoProp Model Training")
    parser = add_common_arguments(parser)
    args = parser.parse_args()
    
    print("NoProp Model Training Script")
    print("="*60)
    print(f"Model type: {args.model_type}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Dropout epochs: {args.dropout_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    
    # Load training data
    print("\n1. Loading training data...")
    data, eta_dim, mu_dim = BaseETTrainer.load_training_data(args.data)
    
    # Set batch size to full dataset if not specified (disable minibatching by default)
    if args.batch_size is None:
        args.batch_size = len(data['train']['eta'])
        print(f"Batch size set to full dataset size: {args.batch_size}")
    
    # Standardize targets (mu_T) to help with training
    print("Standardizing targets...")
    
    # Standardize targets (mu_T)
    train_mu_T_mean = jnp.mean(data['train']['mu_T'], axis=0)
    train_mu_T_std = jnp.std(data['train']['mu_T'], axis=0)
    
    # Apply target standardization to all splits
    data['train']['mu_T'] = (data['train']['mu_T'] - train_mu_T_mean) / train_mu_T_std
    data['val']['mu_T'] = (data['val']['mu_T'] - train_mu_T_mean) / train_mu_T_std
    if 'test' in data:
        data['test']['mu_T'] = (data['test']['mu_T'] - train_mu_T_mean) / train_mu_T_std
    
    print(f"Target standardization: mean={train_mu_T_mean}, std={train_mu_T_std}")
    
    # Store standardization parameters for later use
    standardization_params = {
        'mu_T_mean': train_mu_T_mean,
        'mu_T_std': train_mu_T_std
    }
    
    # Create model configuration
    print("\n2. Creating model configuration...")
    config_kwargs = {
        'loss_type': args.loss_type,
        'noise_schedule': args.noise_schedule,
        'sigma_t': args.sigma_t,
        'reg_weight': args.reg_weight,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
    }
    
    config = create_config(args.model_type, **config_kwargs)
    print(f"Configuration: {config}")
    
    # Set up output directory
    print("\n3. Setting up output directory...")
    if args.output_dir is None:
        # Extract dataset name from data file path
        data_file = Path(args.data)
        dataset_name = data_file.stem  # Get filename without extension
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"artifacts/{args.model_type.lower()}_{dataset_name}_{timestamp}"
    else:
        output_dir = args.output_dir
    print(f"Output directory: {output_dir}")
    
    # Create model
    print("\n4. Creating model...")
    z_shape = (eta_dim,)
    x_ndims = 1  # x_ndims = 1 for the cases we're considering
    
    model = create_model(args.model_type, args.model, config, z_shape, x_ndims)
    print(f"Model created: {type(model).__name__}")
    
    # Create trainer
    print("\n5. Creating trainer...")
    trainer = NoPropTrainer(model)
    
    # Train model
    print("\n6. Training model...")
    results = trainer.train(
        train_eta=data['train']['eta'],
        train_mu_T=data['train']['mu_T'],
        val_eta=data['val']['eta'],
        val_mu_T=data['val']['mu_T'],
        num_epochs=args.epochs,
        test_eta=data.get('test', {}).get('eta'),
        test_mu_T=data.get('test', {}).get('mu_T'),
        dropout_epochs=args.dropout_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        eval_steps=args.eval_steps,
        output_dir=output_dir
    )
    
    # Add standardization parameters to results
    results['standardization_params'] = standardization_params
    
    # Save results and generate plots
    print("\n7. Saving results and generating plots...")
    trainer.save_results(results, output_dir)
    
    print(f"\nSUCCESS! {args.model_type} training completed")
    print(f"Results saved to: {output_dir}")
    print(f"Final validation MSE: {results['final_val_mse']:.6f}")
    if results['final_test_mse'] is not None:
        print(f"Final test MSE: {results['final_test_mse']:.6f}")


if __name__ == "__main__":
    main()
