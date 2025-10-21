#!/usr/bin/env python3
"""
GLU ET Training Script

This script provides a command-line interface for training GLU ET models.
It handles all file I/O, data loading, configuration setup, and architecture construction,
while delegating the core training logic to BaseETTrainer.

Usage:
    python src/training/glu_et_trainer.py --data data/training_data.pkl --epochs 100
    python src/training/glu_et_trainer.py --data data/my_data.pkl --hidden-sizes 128 64 --gate-activation sigmoid
"""

import argparse
from pathlib import Path
from typing import Dict, Any, Optional

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

import jax.random as jr

# Model imports
if __name__ == "__main__":
    # When run as script, use absolute imports
    from src.models.glu_et.model import Config, GLU_ET_Net
else:
    # When used as module, use relative imports
    from .model import Config, GLU_ET_Net


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

class TrainingConfig(BaseTrainingConfig):
    """
    GLU ET specific training configuration.
    
    This configuration inherits from BaseTrainingConfig and adds any GLU ET
    specific training parameters. Currently uses all base defaults.
    """
    
    # GLU ET specific training switches
    training_switches = [
        'learning_rate', 'batch_size', 'weight_decay', 'l1_reg_weight', 'early_stopping_patience', 
        'log_frequency', 'random_seed', 'optimizer'
    ]


# ============================================================================
# TRAINING FUNCTION
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
    Train the GLU ET model using the provided configurations and data.
    
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
    """
    Train the model using BaseETTrainer.
    
    Args:
        model_config: Configuration for the model architecture.
        training_config: Configuration for the training process.
        data: Dictionary containing training and evaluation data.
        output_dir: Directory to save model checkpoints and results.
        epochs: Number of training epochs.
        dropout_epochs: Number of epochs to apply dropout.
        
    Returns:
        A dictionary containing training results (e.g., learning curves).
    """
    
    # Initialize model and trainer
    model = GLU_ET_Net(config=model_config)
    trainer = BaseETTrainer(model, training_config)
    
    # Train the model
    print(f"4. Training {model_config.model_name} for {epochs} epochs...")
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
    
    return results


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train GLU ET models with configurable architecture and training parameters",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--data", type=str, required=True,
                        help="Path to training data pickle file")
    parser.add_argument("--epochs", type=int, required=True,
                        help="Number of training epochs")
    
    # Optional arguments
    parser.add_argument("--dropout-epochs", type=int, default=None,
                        help="Number of epochs to use dropout (default: use dropout for entire training)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results (default: auto-generated)")
    
    # Training parameters (from BaseTrainingConfig)
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Learning rate (default from config)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default from config)")
    parser.add_argument("--optimizer", type=str, choices=['adam', 'adamw', 'sgd', 'rmsprop'], default=None,
                        help="Optimizer type (default from config)")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Weight decay (default from config)")
    parser.add_argument("--beta1", type=float, default=None,
                        help="Adam beta1 parameter (default from config)")
    parser.add_argument("--beta2", type=float, default=None,
                        help="Adam beta2 parameter (default from config)")
    parser.add_argument("--eps", type=float, default=None,
                        help="Adam epsilon parameter (default from config)")
    parser.add_argument("--loss-function", type=str, choices=['mse', 'mae', 'huber', 'model_specific'], default=None,
                        help="Loss function (default from config)")
    parser.add_argument("--l1-reg-weight", type=float, default=None,
                        help="L1 regularization weight (default from config)")
    parser.add_argument("--use-mini-batching", action="store_true",
                        help="Use mini-batching (default from config)")
    parser.add_argument("--no-mini-batching", action="store_false", dest="use_mini_batching",
                        help="Disable mini-batching (default from config)")
    parser.add_argument("--random-batch-sampling", action="store_true",
                        help="Use random batch sampling (default from config)")
    parser.add_argument("--sequential-batch-sampling", action="store_false", dest="random_batch_sampling",
                        help="Use sequential batch sampling (default from config)")
    parser.add_argument("--eval-steps", type=int, default=None,
                        help="Steps between evaluations (default from config)")
    parser.add_argument("--save-steps", type=int, default=None,
                        help="Steps between model saves (default from config)")
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                        help="Epochs to wait before early stopping (default from config)")
    parser.add_argument("--early-stopping-min-delta", type=float, default=None,
                        help="Minimum change to qualify as improvement (default from config)")
    parser.add_argument("--log-frequency", type=int, default=None,
                        help="Steps between logging (default from config)")
    parser.add_argument("--random-seed", type=int, default=None,
                        help="Random seed for reproducibility (default from config)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots")
    parser.add_argument("--plot-data", type=str, default=None,
                        help="Path to data file for plotting (default: same as training data)")
    
    # Model parameters (from Config)
    parser.add_argument("--hidden-sizes", type=int, nargs='+', default=None,
                        help="Hidden layer sizes (default from config)")
    parser.add_argument("--activation", type=str, choices=['relu', 'gelu', 'swish', 'swiglu', 'switanh', 'softplusglu', 'tanh'], default=None,
                        help="Activation function (default from config)")
    parser.add_argument("--gate-activation", type=str, choices=['sigmoid', 'tanh', 'relu', 'swish'], default=None,
                        help="Gate activation function for GLU (default from config)")
    parser.add_argument("--dropout-rate", type=float, default=None,
                        help="Dropout rate (default from config)")
    parser.add_argument("--num-resnet-blocks", type=int, default=None,
                        help="Number of ResNet blocks (default from config)")
    parser.add_argument("--initialization-method", type=str, choices=['xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'lecun_normal'], default=None,
                        help="Weight initialization method (default from config)")
    parser.add_argument("--embedding-type", type=str, choices=['default', 'minimal', 'polynomial', 'advanced', 'convex_only'], default=None,
                        help="Eta embedding type (default from config)")
    
    args = parser.parse_args()
    
    # Load data
    print("1. Loading training data...")
    data, eta_dim, mu_dim = BaseETTrainer.load_training_data(args.data)
    
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
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path(f"artifacts/{model_config.model_name}_{timestamp}")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output directory: {output_dir}")
    
    # Save configurations
    print("4. Saving configurations...")
    model_config.save(output_dir)
    training_config.save(output_dir)
    
    # Train the model
    results = train_model(model_config, training_config, data, output_dir, args.epochs, args.dropout_epochs)
    
    # Generate plots (default to True unless --no-plots is used)
    if not args.no_plots:
        print("5. Generating plots...")
        # Generate enhanced learning curves plot
        from scripts.load_model_and_data import load_model_and_data
        config, results, data, model, params, metadata = load_model_and_data(str(output_dir), args.data)
        save_path = Path(output_dir) / "learning_errors_enhanced.png"
        create_enhanced_learning_plot(config, results, data, model, params, metadata, save_path)
    else:
        print("5. Skipping plot generation (explicitly disabled)")
    
    print("\n" + "="*60)
    print("✅ SUCCESS! GLU ET training completed")




if __name__ == "__main__":
    main()
