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
import time
from pathlib import Path
from typing import Dict, Any, Optional

import jax.random as jr

# Handle imports for both module usage and direct script execution
import sys
if __name__ == "__main__":
    # When run as script, add project root to path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    # Use absolute imports when running as script
    from src.models.base_training_config import BaseTrainingConfig
    from src.models.base_trainer import BaseETTrainer
    from scripts.plotting.plot_learning_curves import create_enhanced_learning_plot
else:
    # Use relative imports when used as module
    from ..base_training_config import BaseTrainingConfig
    from ..base_trainer import BaseETTrainer
    from ....scripts.plotting.plot_learning_curves import create_enhanced_learning_plot


# Model imports
if __name__ == "__main__":
    # When run as script, use absolute imports
    from src.models.mlp_et.model import Config, MLP_ET_Net
else:
    # When used as module, use relative imports
    from .model import Config, MLP_ET_Net


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

class TrainingConfig(BaseTrainingConfig):
    """
    MLP ET specific training configuration.
    
    This configuration inherits from BaseTrainingConfig and adds any MLP ET
    specific training parameters. Currently uses all base defaults.
    """
    
    # MLP ET specific training switches
    training_switches = [
        'learning_rate', 'batch_size', 'weight_decay', 'l1_reg_weight', 'early_stopping_patience', 
        'log_frequency', 'random_seed', 'optimizer'
    ]


# ============================================================================
# CONFIGURATION CREATION
# ============================================================================



def train_model(
    model_config: Config,
    training_config: TrainingConfig,
    data: Dict[str, Any],
    output_dir: Path,
    epochs: int,
    dropout_epochs: Optional[int]
) -> Dict[str, Any]:
    """
    Train the model using BaseETTrainer.
    
    Args:
        model_config: Model configuration
        training_config: Training configuration
        data: Training data
        output_dir: Output directory for saving results
        epochs: Total number of epochs
        dropout_epochs: Number of epochs with dropout (None = use dropout for entire training)
        
    Returns:
        Training results dictionary
    """
    if dropout_epochs is None:
        print(f"\nStarting training for {epochs} epochs with dropout for entire training...")
    else:
        print(f"\nStarting training for {epochs} epochs with {dropout_epochs} dropout epochs...")
    
    # Create model and trainer
    model = MLP_ET_Net(config=model_config)
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
    parser.add_argument("--activation", type=str, choices=["relu", "gelu", "swish", "swiglu", "switanh", "softplusglu", "tanh"],
                       help="Activation function (default from config)")
    parser.add_argument("--dropout-rate", type=float,
                       help="Dropout rate (default from config)")
    parser.add_argument("--num-resnet-blocks", type=int,
                       help="Number of ResNet blocks (default from config)")
    parser.add_argument("--initialization-method", type=str,
                       choices=["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "lecun_normal"],
                       help="Weight initialization method (default from config)")
    parser.add_argument("--embedding-type", type=str,
                       choices=["default", "minimal", "polynomial", "advanced", "convex_only"],
                       help="Eta embedding type (default from config)")
    
    args = parser.parse_args()
    
    # Set random seed (use default if not provided)
    random_seed = args.random_seed if hasattr(args, 'random_seed') and args.random_seed is not None else 42
    random_key = jr.PRNGKey(random_seed)
    
    print("MLP ET Training Script")
    print("="*60)
    
    # Load data
    print("1. Loading training data...")
    data, eta_dim, mu_dim = BaseETTrainer.load_training_data(args.data)
    
    # Extract distribution name from data metadata
    ef_distribution_name = "unknown"
    if "metadata" in data and "ef_distribution_name" in data["metadata"]:
        ef_distribution_name = data["metadata"]["ef_distribution_name"]
        print(f"Detected distribution: {ef_distribution_name}")
    else:
        print("Warning: Could not detect distribution name from data metadata")
    
    # Create configurations
    print("2. Creating configurations...")
    
    # Create model configuration using base class method
    args.input_dim = eta_dim
    args.output_dim = mu_dim
    model_config = Config.create_from_args(args)
    
    # Create training configuration using model-specific config
    training_config = TrainingConfig.create_from_args(args)
    
    # Print configuration summary
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Model Configuration:")
    print(f"  Name: {model_config.model_name}")
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
    
    # Set up output directory
    print("3. Setting up output directory...")
    if args.output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Include distribution name in output directory
        output_dir = Path("artifacts") / f"mlp_et_{ef_distribution_name}_{timestamp}"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Train model
    print("4. Training model...")
    epochs = args.epochs if hasattr(args, 'epochs') and args.epochs is not None else 100
    dropout_epochs = args.dropout_epochs if hasattr(args, 'dropout_epochs') and args.dropout_epochs is not None else None
    tic = time.time()
    results = train_model(
        model_config, 
        training_config, 
        data, 
        output_dir,
        epochs=epochs, 
        dropout_epochs=dropout_epochs
    )
    tock = time.time() - tic
    # Generate plots (default to True unless --no-plots is used)
    if not args.no_plots:
        print("5. Generating plots...")
        # Load the trained model and params for plotting
        model = MLP_ET_Net(config=model_config)
        import pickle
        with open(output_dir / "model_params.pkl", "rb") as f:
            params = pickle.load(f)
        
        # Generate learning curves plot
        save_path = Path(output_dir) / "learning_plot.png"
        
        # Generate predictions for plotting
        import jax.numpy as jnp
        import numpy as np
        
        def make_predictions_batch(model, params, eta_data, batch_size=100):
            """Make predictions on data in batches."""
            predictions = []
            for i in range(0, len(eta_data), batch_size):
                batch = eta_data[i:i + batch_size]
                batch_jnp = jnp.array(batch)
                pred = model.apply(params, batch_jnp, training=False)
                # Handle tuple return (some models return (prediction, aux_loss))
                if isinstance(pred, tuple):
                    pred = pred[0]
                predictions.append(np.array(pred))
            return np.concatenate(predictions, axis=0)
        
        # Generate predictions for all datasets
        train_pred = make_predictions_batch(model, params, data['train']['eta'])
        val_pred = make_predictions_batch(model, params, data['val']['eta'])
        test_pred = make_predictions_batch(model, params, data['test']['eta'])
        
        # Call the scripts plotting function
        create_enhanced_learning_plot(
            results=results,
            train_pred=train_pred,
            val_pred=val_pred,
            test_pred=test_pred,
            train_mu_T=data['train']['mu_T'],
            val_mu_T=data['val']['mu_T'],
            test_mu_T=data['test']['mu_T'],
            output_path=str(save_path),
            model_name=model_config.model_name
        )
    else:
        print("5. Skipping plot generation (explicitly disabled)")
    
    print("\n" + "="*60)
    print("✅ SUCCESS! MLP ET training completed in {tock:.2f} seconds")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()