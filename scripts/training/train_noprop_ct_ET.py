#!/usr/bin/env python3
"""
Training script for NoProp-CT (No Propagation Continuous Time) model.

This script trains, evaluates, and plots results for a NoProp-CT network
on the natural parameter to statistics mapping task.
"""

import sys
from pathlib import Path
import time
import json
import jax.numpy as jnp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import FullConfig
from src.models.noprop_ct_ET import create_model_and_trainer
from src.data_utils import load_3d_gaussian_data, compute_ground_truth_3d_tril
from plotting.model_comparison import create_comprehensive_report


# =============================================================================
# CONFIGURATION - Edit these parameters to modify the experiment
# =============================================================================

# Deep narrow NoProp-CT configuration
CONFIG = FullConfig()

# Network architecture - Deep narrow for continuous time
CONFIG.network.hidden_sizes = [96] * 8  # 8 time evolution layers x 96 units each
CONFIG.network.activation = "swish"  # Swish works well for continuous dynamics
CONFIG.network.use_feature_engineering = True
CONFIG.network.output_dim = 9  # For tril format

# NoProp-CT specific parameters
CONFIG.model_specific.num_time_steps = 10
CONFIG.model_specific.time_horizon = 1.0
CONFIG.model_specific.ode_solver = "euler"
CONFIG.model_specific.noise_scale = 0.01

# Training parameters optimized for CT dynamics
CONFIG.training.learning_rate = 6e-4  # Moderate LR for stability
CONFIG.training.num_epochs = 120
CONFIG.training.batch_size = 32
CONFIG.training.patience = 25
CONFIG.training.weight_decay = 1e-5
CONFIG.training.gradient_clip_norm = 0.8

# Experiment settings
CONFIG.experiment.experiment_name = "noprop_ct_deep_narrow"
CONFIG.experiment.output_dir = "artifacts/noprop_ct_results"

# Loss Function
LOSS_FUNCTION = "mse"  # Standard MSE loss

# =============================================================================


def main():
    """Main training and evaluation pipeline."""
    
    print("⏰ NOPROP-CT NETWORK TRAINING")
    print("=" * 50)
    
    config = CONFIG
    
    print(f"Experiment: {config.experiment.experiment_name}")
    print(f"Architecture: {len(config.network.hidden_sizes)} CT layers x {config.network.hidden_sizes[0]} units")
    print(f"Time Steps: {config.model_specific.num_time_steps}")
    print(f"Time Horizon: {config.model_specific.time_horizon}")
    print(f"ODE Solver: {config.model_specific.ode_solver}")
    print(f"Learning Rate: {config.training.learning_rate}")
    print(f"Batch Size: {config.training.batch_size}")
    print(f"Loss Function: {LOSS_FUNCTION}")
    
    # Load data
    print("\n📊 Loading data...")
    data_dir = Path("data")
    data, ef = load_3d_gaussian_data(data_dir, format="tril")
    
    # Create train/val/test splits
    train_data = {"eta": data["train_eta"], "y": data["train_y"]}
    val_data = {"eta": data["val_eta"], "y": data["val_y"]}
    
    # Create test split from validation data
    n_val = val_data["eta"].shape[0]
    n_test = min(n_val // 2, 500)
    
    test_data = {
        "eta": val_data["eta"][:n_test],
        "y": val_data["y"][:n_test]
    }
    
    val_data = {
        "eta": val_data["eta"][n_test:],
        "y": val_data["y"][n_test:]
    }
    
    print(f"Training samples: {train_data['eta'].shape[0]}")
    print(f"Validation samples: {val_data['eta'].shape[0]}")
    print(f"Test samples: {test_data['eta'].shape[0]}")
    print(f"Input dimension: {train_data['eta'].shape[1]}")
    print(f"Output dimension: {train_data['y'].shape[1]}")
    
    # Compute ground truth
    print("\n🎯 Computing ground truth...")
    ground_truth = compute_ground_truth_3d_tril(test_data['eta'], ef)
    empirical_mse = float(jnp.mean(jnp.square(test_data['y'] - ground_truth)))
    print(f"MCMC sampling error bound: {empirical_mse:.6f}")
    
    # Create model and trainer
    print("\n🏗️  Creating NoProp-CT Network...")
    trainer = create_model_and_trainer(config)
    
    # Train model
    print("\n🚂 Training model...")
    start_time = time.time()
    
    try:
        best_params, history = trainer.train(train_data, val_data)
        training_time = time.time() - start_time
        
        print(f"✅ Training completed in {training_time:.1f}s")
        
        # Evaluate model
        print("\n📈 Evaluating model...")
        metrics = trainer.evaluate(best_params, test_data, ground_truth)
        
        print(f"Test MSE (vs Empirical): {metrics['mse']:.2f}")
        print(f"Test MAE (vs Empirical): {metrics['mae']:.2f}")
        if 'ground_truth_mse' in metrics:
            print(f"Test MSE (vs Ground Truth): {metrics['ground_truth_mse']:.2f}")
            print(f"Test MAE (vs Ground Truth): {metrics['ground_truth_mae']:.2f}")
        
        # Prepare results
        param_count = trainer.model.get_parameter_count(best_params)
        
        results = {
            'NoProp-CT Network': {
                'status': 'success',
                'metrics': metrics,
                'history': history,
                'training_time': training_time,
                'architecture_info': {
                    'hidden_sizes': config.network.hidden_sizes,
                    'parameter_count': param_count,
                    'depth': len(config.network.hidden_sizes),
                    'width': config.network.hidden_sizes[0] if config.network.hidden_sizes else 0,
                    'time_steps': config.model_specific.num_time_steps,
                    'time_horizon': config.model_specific.time_horizon,
                    'activation': config.network.activation
                },
                'config': config.to_dict()
            }
        }
        
        # Save model
        if config.experiment.save_model:
            save_dir = Path(config.experiment.output_dir) / "model"
            trainer.save_model(best_params, save_dir)
        
        # Create plots and report
        if config.experiment.save_plots:
            print("\n📊 Creating plots and report...")
            output_dir = Path(config.experiment.output_dir)
            create_comprehensive_report(results, output_dir, config.experiment.experiment_name)
        
        # Save detailed results
        results_file = Path(config.experiment.output_dir) / "results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            def convert_for_json(obj):
                if hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_for_json(results), f, indent=2)
        
        print(f"📁 Results saved to {config.experiment.output_dir}")
        
        # Final summary
        print(f"\n🏆 FINAL RESULTS:")
        print(f"  Model: NoProp-CT Network")
        print(f"  Architecture: {len(config.network.hidden_sizes)} CT layers x {config.network.hidden_sizes[0]} units")
        print(f"  Time Steps: {config.model_specific.num_time_steps}")
        print(f"  Parameters: {param_count:,}")
        print(f"  Training Time: {training_time:.1f}s")
        print(f"  Best MSE (Ground Truth): {metrics.get('ground_truth_mse', metrics['mse']):.2f}")
        print(f"  MCMC Error Bound: {empirical_mse:.6f}")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        
        # Save failure information
        results = {
            'NoProp-CT Network': {
                'status': 'failed',
                'error': str(e),
                'training_time': time.time() - start_time,
                'config': config.to_dict()
            }
        }
        
        results_file = Path(config.experiment.output_dir) / "results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        raise


if __name__ == "__main__":
    main()
