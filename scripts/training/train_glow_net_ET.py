#!/usr/bin/env python3
"""
Training script for Glow Network ET model.

This script trains, evaluates, and plots results for a Glow Network ET
(normalizing flow with affine coupling) on the natural parameter to statistics mapping task.
"""

import sys
from pathlib import Path
import time
import json
import jax.numpy as jnp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_flow_config, FullConfig
from src.models.glow_net_ET import create_glow_et_model_and_trainer
from src.utils.data_utils import load_3d_gaussian_data, compute_ground_truth_3d_tril
from plotting.model_comparison import create_comprehensive_report


# =============================================================================
# CONFIGURATION - Edit these parameters to modify the experiment
# =============================================================================

# Model Configuration
MODEL_CONFIG = "flow"  # Use predefined flow config

# Custom configuration (used if MODEL_CONFIG is not in predefined configs)
CUSTOM_CONFIG = FullConfig()
CUSTOM_CONFIG.network.hidden_sizes = [64] * 4  # Base network layers
CUSTOM_CONFIG.network.activation = "tanh"
CUSTOM_CONFIG.network.use_feature_engineering = True
CUSTOM_CONFIG.network.output_dim = 9  # For tril format

# Flow-specific parameters
CUSTOM_CONFIG.model_specific.num_flow_layers = 30  # Very deep: 30 flow layers
CUSTOM_CONFIG.model_specific.flow_hidden_size = 64  # Narrow: 64 units per layer
CUSTOM_CONFIG.model_specific.num_timesteps = 100
CUSTOM_CONFIG.model_specific.beta_start = 1e-4
CUSTOM_CONFIG.model_specific.beta_end = 2e-2

CUSTOM_CONFIG.training.learning_rate = 1e-3
CUSTOM_CONFIG.training.num_epochs = 80
CUSTOM_CONFIG.training.batch_size = 32
CUSTOM_CONFIG.training.patience = 20

CUSTOM_CONFIG.experiment.experiment_name = "deep_flow_network_30x64"
CUSTOM_CONFIG.experiment.output_dir = "artifacts/deep_flow_results"

# Loss Function
LOSS_FUNCTION = "diffusion_mse"  # Diffusion-based MSE loss

# =============================================================================


def main():
    """Main training and evaluation pipeline."""
    
    print("🌊 DEEP FLOW NETWORK TRAINING")
    print("=" * 50)
    
    # Load configuration
    if MODEL_CONFIG == "flow":
        config = get_flow_config()
        # Override with our deep narrow settings
        config.model_specific.num_flow_layers = 30
        config.model_specific.flow_hidden_size = 64
        print(f"Using predefined flow config (customized)")
    else:
        config = CUSTOM_CONFIG
        print("Using custom configuration")
    
    # Override output settings
    config.experiment.experiment_name = config.experiment.experiment_name or "deep_flow_experiment"
    config.experiment.output_dir = config.experiment.output_dir or "artifacts/deep_flow_results"
    
    print(f"Experiment: {config.experiment.experiment_name}")
    print(f"Flow Layers: {config.model_specific.num_flow_layers}")
    print(f"Flow Hidden Size: {config.model_specific.flow_hidden_size}")
    print(f"Base Network: {config.network.hidden_sizes}")
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
    print("\n🏗️  Creating Deep Flow model...")
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
            'Deep Flow Network': {
                'status': 'success',
                'metrics': metrics,
                'history': history,
                'training_time': training_time,
                'architecture_info': {
                    'flow_layers': config.model_specific.num_flow_layers,
                    'flow_hidden_size': config.model_specific.flow_hidden_size,
                    'base_hidden_sizes': config.network.hidden_sizes,
                    'parameter_count': param_count,
                    'total_depth': config.model_specific.num_flow_layers + len(config.network.hidden_sizes),
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
            # Convert JAX arrays to lists for JSON serialization
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
        print(f"  Model: Deep Flow Network")
        print(f"  Architecture: {config.model_specific.num_flow_layers} flow layers x {config.model_specific.flow_hidden_size} units")
        print(f"  Base Network: {len(config.network.hidden_sizes)} layers x {config.network.hidden_sizes[0] if config.network.hidden_sizes else 0} units")
        print(f"  Total Parameters: {param_count:,}")
        print(f"  Training Time: {training_time:.1f}s")
        print(f"  Best MSE (Ground Truth): {metrics.get('ground_truth_mse', metrics['mse']):.2f}")
        print(f"  MCMC Error Bound: {empirical_mse:.6f}")
        
        # Performance analysis
        improvement_factor = empirical_mse / metrics.get('ground_truth_mse', metrics['mse']) if metrics.get('ground_truth_mse', metrics['mse']) > 0 else float('inf')
        if improvement_factor < 1:
            print(f"  🎯 Model exceeds MCMC bound by {1/improvement_factor:.1f}x")
        else:
            print(f"  📈 Model within {improvement_factor:.1f}x of MCMC bound")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        
        # Save failure information
        results = {
            'Deep Flow Network': {
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
