#!/usr/bin/env python3
"""
Training script for Quadratic ResNet model.

This script trains, evaluates, and plots results for a Quadratic ResNet
on the natural parameter to statistics mapping task.
"""

import sys
from pathlib import Path
import time
import json
import jax.numpy as jnp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_quadratic_config, FullConfig
from src.models.quadratic_resnet_ET import create_model_and_trainer
from src.utils.data_utils import load_3d_gaussian_data, compute_ground_truth_3d_tril
# from plotting.model_comparison import create_comprehensive_report  # TODO: Create plotting module


# =============================================================================
# CONFIGURATION - Edit these parameters to modify the experiment
# =============================================================================

# Model Configuration
MODEL_CONFIG = "custom"  # Use custom config for deep narrow

# Deep narrow quadratic ResNet configuration
CUSTOM_CONFIG = FullConfig()
CUSTOM_CONFIG.network.hidden_sizes = [96] * 10  # 10 layers x 96 units (deep narrow)
CUSTOM_CONFIG.network.activation = "tanh"
CUSTOM_CONFIG.network.use_feature_engineering = True
CUSTOM_CONFIG.network.residual_connections = True
CUSTOM_CONFIG.network.output_dim = 9  # For tril format

# Quadratic-specific parameters
CUSTOM_CONFIG.model_specific.use_quadratic_terms = True
CUSTOM_CONFIG.model_specific.quadratic_mixing = "adaptive"

CUSTOM_CONFIG.training.learning_rate = 8e-4
CUSTOM_CONFIG.training.num_epochs = 120
CUSTOM_CONFIG.training.batch_size = 32
CUSTOM_CONFIG.training.patience = 25
CUSTOM_CONFIG.training.weight_decay = 1e-5
CUSTOM_CONFIG.training.gradient_clip_norm = 0.8

CUSTOM_CONFIG.experiment.experiment_name = "quadratic_resnet_deep_narrow"
CUSTOM_CONFIG.experiment.output_dir = "artifacts/ET_models/quadratic_resnet_ET"

# Loss Function
LOSS_FUNCTION = "mse"  # Standard MSE loss

# =============================================================================


def main():
    """Main training and evaluation pipeline."""
    
    print("🔥 QUADRATIC RESNET TRAINING")
    print("=" * 50)
    
    # Load configuration
    if MODEL_CONFIG == "quadratic":
        config = get_quadratic_config()
        print(f"Using predefined quadratic config")
    else:
        config = CUSTOM_CONFIG
        print("Using custom deep narrow configuration")
    
    # Override output settings
    config.experiment.experiment_name = config.experiment.experiment_name or "quadratic_resnet_experiment"
    config.experiment.output_dir = config.experiment.output_dir or "artifacts/ET_models/quadratic_resnet_ET"
    
    print(f"Experiment: {config.experiment.experiment_name}")
    print(f"Architecture: {config.network.hidden_sizes}")
    print(f"Quadratic Terms: {config.model_specific.use_quadratic_terms}")
    print(f"Quadratic Mixing: {config.model_specific.quadratic_mixing}")
    print(f"Learning Rate: {config.training.learning_rate}")
    print(f"Batch Size: {config.training.batch_size}")
    print(f"Loss Function: {LOSS_FUNCTION}")
    
    # Load data
    print("\n📊 Loading data...")
    data_file = Path("data/easy_3d_gaussian.pkl")
    
    import pickle
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    # Create train/val/test splits with correct structure
    train_data = {"eta": data["train"]["eta"], "stats": data["train"]["mean"]}
    val_data = {"eta": data["val"]["eta"], "stats": data["val"]["mean"]}
    
    # Purge cov_tt to save memory
    if "cov" in data["train"]: del data["train"]["cov"]
    if "cov" in data["val"]: del data["val"]["cov"]
    if "cov" in data["test"]: del data["test"]["cov"]
    import gc; gc.collect()
    print("✅ Purged cov_tt elements from memory for optimization")
    
    # Create test split from validation data
    n_val = val_data["eta"].shape[0]
    n_test = min(n_val // 2, 500)
    
    test_data = {
        "eta": val_data["eta"][:n_test],
        "y": val_data["stats"][:n_test]
    }
    
    val_data = {
        "eta": val_data["eta"][n_test:],
        "stats": val_data["stats"][n_test:]
    }
    
    print(f"Training samples: {train_data['eta'].shape[0]}")
    print(f"Validation samples: {val_data['eta'].shape[0]}")
    print(f"Test samples: {test_data['eta'].shape[0]}")
    print(f"Input dimension: {train_data['eta'].shape[1]}")
    print(f"Output dimension: {train_data['stats'].shape[1]}")
    
    # Use test data directly (ground truth is already in test_data['y'])
    print("\n🎯 Using test data as ground truth...")
    ground_truth = test_data['y']
    print(f"Ground truth shape: {ground_truth.shape}")
    
    # Create model and trainer
    print("\n🏗️  Creating Quadratic ResNet model...")
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
            'Quadratic ResNet': {
                'status': 'success',
                'metrics': metrics,
                'history': history,
                'training_time': training_time,
                'architecture_info': {
                    'hidden_sizes': config.network.hidden_sizes,
                    'parameter_count': param_count,
                    'depth': len(config.network.hidden_sizes),
                    'activation': config.network.activation,
                    'quadratic_terms': config.model_specific.use_quadratic_terms
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
        print(f"  Model: Quadratic ResNet")
        print(f"  Architecture: {len(config.network.hidden_sizes)} layers x {config.network.hidden_sizes[0]} units")
        print(f"  Parameters: {param_count:,}")
        print(f"  Training Time: {training_time:.1f}s")
        print(f"  Best MSE (Ground Truth): {metrics.get('ground_truth_mse', metrics['mse']):.2f}")
        print(f"  MCMC Error Bound: {empirical_mse:.6f}")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        
        # Save failure information
        results = {
            'Quadratic ResNet': {
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
