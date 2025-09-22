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
import numpy as np
import jax.numpy as jnp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_glow_config, FullConfig
from src.models.glow_net_ET import create_glow_et_model_and_trainer
# Import standardized data loading from template
sys.path.append(str(Path(__file__).parent))
from src.utils.data_utils import load_standardized_ep_data
from plot_training_results import plot_model_comparison, save_results_summary


# =============================================================================
# CONFIGURATION - Edit these parameters to modify the experiment
# =============================================================================

# Model Configuration
MODEL_CONFIG = "glow"  # Use predefined glow config

# Custom configuration (used if MODEL_CONFIG is not in predefined configs)
CUSTOM_CONFIG = FullConfig()
CUSTOM_CONFIG.network.hidden_sizes = [64] * 8  # Deep 8-layer base network
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
CUSTOM_CONFIG.experiment.output_dir = "artifacts/ET_models/glow_ET"

# Loss Function
LOSS_FUNCTION = "diffusion_mse"  # Diffusion-based MSE loss

# =============================================================================


def main():
    """Main training and evaluation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Glow ET models')
    parser.add_argument('--data_file', type=str, help='Path to data file (default: data/easy_3d_gaussian.pkl)')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    
    args = parser.parse_args()
    
    print("🌊 DEEP FLOW NETWORK TRAINING")
    print("=" * 50)
    
    # Load configuration
    if MODEL_CONFIG == "glow":
        config = get_glow_config()
        # Using the glow config with 50 layers (no override needed)
        print(f"Using predefined glow config with {config.model_specific.num_flow_layers} layers")
    else:
        config = CUSTOM_CONFIG
        print("Using custom configuration")
    
    # Override output settings
    config.experiment.experiment_name = config.experiment.experiment_name or "deep_flow_experiment"
    config.experiment.output_dir = config.experiment.output_dir or "artifacts/ET_models/glow_ET"
    
    print(f"Experiment: {config.experiment.experiment_name}")
    print(f"Flow Layers: {config.model_specific.num_flow_layers}")
    print(f"Flow Hidden Size: {config.model_specific.flow_hidden_size}")
    print(f"Base Network: {config.network.hidden_sizes}")
    print(f"Learning Rate: {config.training.learning_rate}")
    print(f"Batch Size: {config.training.batch_size}")
    print(f"Loss Function: {LOSS_FUNCTION}")
    
    # Load data using standardized template function (dimension-agnostic)
    print("\n📊 Loading data...")
    data_file = args.data_file if args.data_file else "data/easy_3d_gaussian.pkl"
    eta_data, ground_truth, metadata = load_standardized_ep_data(data_file)
    
    # Prepare data for BaseTrainer interface
    train_data = {"eta": eta_data, "mu_T": ground_truth}
    # Create validation split from training data for BaseTrainer
    val_split = int(0.8 * len(eta_data))
    val_data = {"eta": eta_data[val_split:], "mu_T": ground_truth[val_split:]}
    
    # Create test split from validation data
    n_val = val_data["eta"].shape[0]
    n_test = min(n_val // 2, 500)
    
    test_data = {
        "eta": val_data["eta"][:n_test],
        "mu_T": val_data["mu_T"][:n_test]
    }
    
    val_data = {
        "eta": val_data["eta"][n_test:],
        "mu_T": val_data["mu_T"][n_test:]
    }
    
    print(f"Training samples: {train_data['eta'].shape[0]}")
    print(f"Validation samples: {val_data['eta'].shape[0]}")
    print(f"Test samples: {test_data['eta'].shape[0]}")
    print(f"Input dimension: {train_data['eta'].shape[1]}")
    print(f"Output dimension: {train_data['y'].shape[1]}")
    
    # Use test data directly (ground truth is already in test_data['y'])
    print("\n🎯 Using test data as ground truth...")
    ground_truth = test_data['y']
    print(f"Ground truth shape: {ground_truth.shape}")
    
    # Create model and trainer
    print("\n🏗️  Creating Deep Flow model...")
    trainer = create_glow_et_model_and_trainer(config)
    
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
        
        # Benchmark inference time
        print("\n⏱️  Benchmarking inference time...")
        def benchmark_inference(trainer, params, test_data, num_runs=50):
            """Benchmark inference time with multiple runs for accuracy."""
            # Warm-up run to ensure compilation is complete
            _ = trainer.predict(params, test_data['eta'][:1])
            
            # Measure inference time over multiple runs
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                _ = trainer.predict(params, test_data['eta'])
                times.append(time.time() - start_time)
            
            # Return statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            return {
                'avg_inference_time': avg_time,
                'min_inference_time': min_time,
                'max_inference_time': max_time,
                'inference_per_sample': avg_time / len(test_data['eta']),
                'samples_per_second': len(test_data['eta']) / avg_time
            }
        
        inference_stats = benchmark_inference(trainer, best_params, test_data)
        print(f"Avg inference time: {inference_stats['avg_inference_time']:.4f}s ({inference_stats['samples_per_second']:.1f} samples/sec)")
        
        # Prepare results
        param_count = trainer.model.get_parameter_count(best_params)
        
        results = {
            'Deep Flow Network': {
                'status': 'success',
                'metrics': metrics,
                'history': history,
                'training_time': training_time,
                'avg_inference_time': inference_stats['avg_inference_time'],
                'inference_per_sample': inference_stats['inference_per_sample'],
                'samples_per_second': inference_stats['samples_per_second'],
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
        
        # Create comprehensive plots
        print("\n📊 Creating comprehensive plots...")
        output_dir = Path(config.experiment.output_dir)
        
        # Add predictions and ground truth to results for plotting
        results['Deep Flow Network']['predictions'] = np.array(predictions).tolist()
        results['Deep Flow Network']['ground_truth'] = np.array(ground_truth).tolist()
        
        # Create model comparison plots using standardized plotting function
        plot_model_comparison(
            results=results,
            output_dir=str(output_dir),
            save_plots=True,
            show_plots=False
        )
        
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
        
        # Save results summary using standardized function
        save_results_summary(
            results=results,
            output_dir=str(output_dir),
            experiment_name="glow_ET"
        )
        
        print(f"📁 Results saved to {config.experiment.output_dir}")
        
        # Final summary
        print(f"\n🏆 FINAL RESULTS:")
        print(f"  Model: Deep Flow Network")
        print(f"  Architecture: {config.model_specific.num_flow_layers} flow layers x {config.model_specific.flow_hidden_size} units")
        print(f"  Base Network: {len(config.network.hidden_sizes)} layers x {config.network.hidden_sizes[0] if config.network.hidden_sizes else 0} units")
        print(f"  Total Parameters: {param_count:,}")
        print(f"  Training Time: {training_time:.1f}s")
        print(f"  Inference Time: {inference_stats['samples_per_second']:.1f} samples/sec")
        print(f"  Best MSE (Ground Truth): {metrics.get('ground_truth_mse', metrics['mse']):.2f}")
        print(f"  MCMC Error Bound: Not calculated")
        
        # Performance analysis (empirical_mse not available)
        # improvement_factor = empirical_mse / metrics.get('ground_truth_mse', metrics['mse']) if metrics.get('ground_truth_mse', metrics['mse']) > 0 else float('inf')
        # if improvement_factor < 1:
        #     print(f"  🎯 Model exceeds MCMC bound by {1/improvement_factor:.1f}x")
        # else:
        #     print(f"  📈 Model within {improvement_factor:.1f}x of MCMC bound")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        
        # Save failure information
        results = {
            'Deep Flow Network': {
                'status': 'failed',
                'error': str(e),
                'training_time': time.time() - start_time,
                'avg_inference_time': None,
                'inference_per_sample': None,
                'samples_per_second': None,
                'config': config.to_dict()
            }
        }
        
        results_file = Path(config.experiment.output_dir) / "results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Glow ET models')
    parser.add_argument('--data_file', type=str, help='Path to data file (default: data/easy_3d_gaussian.pkl)')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    
    args = parser.parse_args()
    main()
