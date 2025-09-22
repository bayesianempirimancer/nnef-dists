#!/usr/bin/env python3
"""
Training script for Geometric Flow ET neural networks.

This script trains networks that learn flow dynamics to predict expected sufficient statistics
E[T(X)] using the geometric flow approach:
    du/dt = A@A^T@(η_target - η_init)

Usage:
    python scripts/training/train_geometric_flow_ET.py --config configs/multivariate_3d_large.yaml
    python scripts/training/train_geometric_flow_ET.py --config configs/multivariate_3d_large.yaml --plot-only
"""

import argparse
import sys
from pathlib import Path
import pickle
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import standardized plotting functions
sys.path.append(str(Path(__file__).parent.parent))
from plot_training_results import plot_training_results, plot_model_comparison, save_results_summary

from src.config import FullConfig
from src.ef import ef_factory
from src.models.geometric_flow_net import create_model_and_trainer

def create_simple_config(eta_dim):
    """Create a simple configuration for geometric flow training."""
    from src.config import NetworkConfig, TrainingConfig
    
    network_config = NetworkConfig(
        hidden_sizes=[128, 64, 32],
        use_layer_norm=True,
        dropout_rate=0.0,
        output_dim=eta_dim  # Use inferred dimension
    )
    
    training_config = TrainingConfig(
        num_epochs=150,
        learning_rate=1e-3,
        batch_size=16
    )
    
    return FullConfig(network=network_config, training=training_config)


def train_geometric_flow_et(save_dir: str, plot_only: bool = False):
    """Train geometric flow ET network on 3D Gaussian data."""
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data from standard data file
    data_file = Path("data/easy_3d_gaussian.pkl")
    if not data_file.exists():
        print("Easy 3D Gaussian dataset not found. Please run:")
        print("python scripts/generate_normal_data.py --difficulty Easy --dim 3")
        return
        
    print("Loading training data from standard data file...")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    eta_train = data['train']['eta']
    mu_train = data['train']['mu_T']
    eta_val = data['val']['eta']
    mu_val = data['val']['mu_T']
    
    # Infer dimensions from metadata or data
    if 'metadata' in data and 'eta_dim' in data['metadata']:
        eta_dim = data['metadata']['eta_dim']
        print(f"Using eta_dim from metadata: {eta_dim}")
        
        # Print additional metadata info if available
        if 'ef_distribution_name' in data['metadata']:
            print(f"Exponential family: {data['metadata']['ef_distribution_name']}")
        if 'x_shape' in data['metadata']:
            print(f"Data shape x: {data['metadata']['x_shape']}")
        if 'x_dim' in data['metadata']:
            print(f"Data dimension: {data['metadata']['x_dim']}")
    else:
        eta_dim = eta_train.shape[-1]
        print(f"Inferred eta_dim from data shape: {eta_dim}")
    
    # Create configuration
    config = create_simple_config(eta_dim)
    print(f"Using simple configuration for geometric flow training")
    
    # Purge cov_TT to save memory
    if "cov_TT" in data["train"]: del data["train"]["cov_TT"]
    if "cov_TT" in data["val"]: del data["val"]["cov_TT"]
    if "cov_TT" in data["test"]: del data["test"]["cov_TT"]
    import gc; gc.collect()
    print("✅ Purged cov_TT elements from memory for optimization")
    
    print(f"Training data: {eta_train.shape[0]} samples")
    print(f"Validation data: {eta_val.shape[0]} samples")
    print(f"η dimension: {eta_train.shape[1]}")
    print(f"μ dimension: {mu_train.shape[1]}")
    
    # Create exponential family instance
    ef = ef_factory("multivariate_normal", x_shape=(3,))
    
    # Create and configure trainer
    trainer = create_model_and_trainer(
        config=config,
        matrix_rank=8,  # Reduced rank for efficiency
        n_time_steps=3,  # Minimal due to smoothness
        smoothness_weight=1e-3
    )
    
    # Set exponential family for analytical point computation
    trainer.set_exponential_family(ef)
    
    print(f"Geometric Flow ET Network Configuration:")
    print(f"  Architecture: {getattr(trainer.config.network, 'architecture', 'mlp')}")
    print(f"  Matrix rank: {trainer.matrix_rank}")
    print(f"  Time steps: {trainer.n_time_steps}")
    print(f"  Smoothness weight: {trainer.smoothness_weight}")
    print(f"  Time embedding dim: {trainer.model.time_embed_dim}")
    print(f"  Max frequency: {trainer.model.max_freq}")
    
    if plot_only:
        # Load existing model and create plots
        model_file = save_dir / "geometric_flow_et_params.pkl"
        history_file = save_dir / "geometric_flow_et_history.pkl"
        
        if model_file.exists() and history_file.exists():
            print("Loading existing model for plotting...")
            with open(model_file, 'rb') as f:
                params = pickle.load(f)
            with open(history_file, 'rb') as f:
                history = pickle.load(f)
            
            # Make predictions
            predictions_dict = trainer.predict(params, eta_val)
            predictions = predictions_dict['mu_predicted']
            
            # Store flow distances for plotting
            trainer.last_flow_distances = predictions_dict['flow_distances']
            
            # Create plots using standardized plotting function
            metrics = plot_training_results(
                trainer=trainer,
                eta_data=eta_val,
                ground_truth=mu_val,
                predictions=predictions,
                losses=history.get('train_loss', []),
                config=config,
                model_name="geometric_flow_ET",
                output_dir=str(save_dir),
                save_plots=True,
                show_plots=False
            )
        else:
            print("No existing model found. Run without --plot-only first.")
        return
    
    # Training
    print(f"\nStarting Geometric Flow ET training...")
    
    # Train using the model's specialized training method with progress tracking
    print(f"Training Geometric Flow Network")
    print(f"  Architecture: {getattr(trainer.config.network, 'architecture', 'mlp')}")
    print(f"  Matrix rank: {trainer.matrix_rank}")
    print(f"  Time steps: {trainer.n_time_steps}")
    print(f"  Training samples: {eta_train.shape[0]}")
    print(f"  η dimension: {eta_train.shape[1]}")
    print(f"  μ dimension: estimated 12 for 3D Gaussian")
    
    start_time = time.time()
    
    print("🚂 Starting geometric flow training...")
    params, history = trainer.train(
        eta_targets_train=eta_train,
        eta_targets_val=eta_val,
        epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate,
        batch_size=config.training.batch_size
    )
    
    training_time = time.time() - start_time
    print(f"\n✓ Geometric Flow Network training completed in {training_time:.1f}s")
    
    # Save model and history
    model_file = save_dir / "geometric_flow_et_params.pkl"
    history_file = save_dir / "geometric_flow_et_history.pkl"
    
    with open(model_file, 'wb') as f:
        pickle.dump(params, f)
    with open(history_file, 'wb') as f:
        pickle.dump(history, f)
    
    print(f"Model saved to: {model_file}")
    print(f"History saved to: {history_file}")
    
    # Evaluation
    print(f"\nEvaluating on validation set...")
    results = trainer.evaluate(params, eta_val)
    predictions_dict = trainer.predict(params, eta_val)
    predictions = predictions_dict['mu_predicted']
    
    # Store flow distances for plotting
    trainer.last_flow_distances = predictions_dict['flow_distances']
    
    print(f"Validation Results:")
    print(f"  MSE: {results['mse']:.8f}")
    print(f"  MAE: {results['mae']:.8f}")
    print(f"  Mean flow distance: {results['mean_flow_distance']:.6f}")
    
    # Component analysis
    print(f"  Component errors (linear terms):")
    for i in range(min(3, len(results['component_errors']))):
        print(f"    μ_{i+1}: {results['component_errors'][i]:.8f}")
    
    print(f"  Component errors (quadratic terms, first 3):")
    for i in range(3, min(6, len(results['component_errors']))):
        quad_i, quad_j = divmod(i-3, 3)
        print(f"    μ_{i+1} (x_{quad_i}x_{quad_j}): {results['component_errors'][i]:.8f}")
    
    # Create plots using standardized plotting function
    metrics = plot_training_results(
        trainer=trainer,
        eta_data=eta_val,
        ground_truth=mu_val,
        predictions=predictions,
        losses=history.get('train_loss', []),
        config=config,
        model_name="geometric_flow_ET",
        output_dir=str(save_dir),
        save_plots=True,
        show_plots=False
    )
    
    # Save evaluation results
    eval_file = save_dir / "geometric_flow_et_evaluation.pkl"
    eval_results = {
        'results': results,
        'predictions': predictions,
        'ground_truth': mu_val,
        'eta_data': eta_val,
        'training_time': training_time
    }
    
    with open(eval_file, 'wb') as f:
        pickle.dump(eval_results, f)
    
    print(f"Evaluation results saved to: {eval_file}")
    
    # Summary
    print(f"\n" + "="*60)
    print("GEOMETRIC FLOW ET NETWORK SUMMARY")
    print("="*60)
    print(f"✓ Training completed in {training_time:.1f}s")
    print(f"✓ Final MSE: {results['mse']:.2e}")
    print(f"✓ Final MAE: {results['mae']:.2e}")
    print(f"✓ Mean flow distance: {results['mean_flow_distance']:.4f}")
    print(f"✓ Used {trainer.n_time_steps} time steps with sinusoidal embeddings")
    
    if results['mse'] < 1e-4:
        print("🎉 EXCELLENT: Geometric flow learning highly successful!")
    elif results['mse'] < 1e-2:
        print("✅ GOOD: Reasonable geometric flow performance")
    else:
        print("⚠️ NEEDS WORK: Consider tuning hyperparameters")
    
    return params, history, results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Geometric Flow ET Network')
    parser.add_argument('--save-dir', default='artifacts/ET_models/geometric_flow_ET', 
                       help='Directory to save results')
    parser.add_argument('--plot-only', action='store_true', 
                       help='Only generate plots from existing results')
    
    args = parser.parse_args()
    
    print("Geometric Flow ET Network Training")
    print("="*45)
    print(f"Save directory: {args.save_dir}")
    print(f"Plot only: {args.plot_only}")
    
    try:
        results = train_geometric_flow_et(args.save_dir, args.plot_only)
        print(f"\n✓ Geometric Flow ET training completed successfully!")
        return 0
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
