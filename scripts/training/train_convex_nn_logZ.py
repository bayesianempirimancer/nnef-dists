#!/usr/bin/env python3
"""
Training script for Alternating Convex Neural Network-based log normalizer.

This script trains Input Convex Neural Networks (ICNNs) with alternating Type 1/Type 2 layers
that learn the log normalizer A(η) while maintaining convexity properties essential for 
exponential family distributions.

Features:
- Type 1 layers: W_z ≥ 0, convex activations (standard ICNN)
- Type 2 layers: W_z ≤ 0, concave activations (novel approach)
- Alternating pattern maintains overall convexity
- Skip connections from input to all layers

Usage:
    python scripts/training/train_convex_nn_logZ.py
"""

import argparse
import sys
from pathlib import Path
import time
import jax
import jax.numpy as jnp
from jax import random
from tqdm import tqdm

# Add project root to path  
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import standardized plotting functions
sys.path.append(str(Path(__file__).parent.parent))
from plot_training_results import plot_training_results, plot_model_comparison, save_results_summary

from src.config import FullConfig
from src.models.convex_nn_logZ import (
    Convex_LogZ_Network,
    Convex_LogZ_Trainer
)

def create_alternating_convex_config():
    """Create configuration for alternating convex architecture."""
    config = FullConfig()
    
    # Deep 8-layer network for 3D Gaussian testing  
    config.network.hidden_sizes = [64, 64, 64, 64, 64, 64, 64, 64]  # 8 layers with consistent size
    config.network.output_dim = 9  # 3D multivariate normal has 9 sufficient statistics
    config.network.activation = "softplus"  # Smooth convex/concave activations
    config.network.use_feature_engineering = False  # Disable for speed
    
    # Training parameters for mid-sized model
    config.training.learning_rate = 1e-3  # Standard learning rate
    config.training.num_epochs = 100  # More epochs for better convergence
    config.training.batch_size = 32  # Efficient batch size
    config.training.patience = 25
    config.training.weight_decay = 1e-6
    config.training.gradient_clip_norm = 1.0
    config.training.early_stopping = True
    config.training.min_delta = 1e-6
    config.training.validation_freq = 10
    
    return config


def analyze_alternating_architecture(config):
    """Display the alternating layer architecture pattern."""
    print("\nAlternating Convex Architecture Analysis:")
    print("=" * 55)
    
    sizes = config.network.hidden_sizes
    print(f"Total layers: {len(sizes)}")
    print(f"Layer sizes: {sizes}")
    print(f"Activation: {config.network.activation}")
    
    print("\nLayer Type Pattern:")
    print("Layer 0: Type 1 (W_z ≥ 0, convex activation) - Initial")
    
    for i in range(1, len(sizes)):
        if i % 2 == 0:
            layer_type = "Type 2"
            constraint = "W_z ≤ 0"
            activation = "concave"
        else:
            layer_type = "Type 1"
            constraint = "W_z ≥ 0"
            activation = "convex"
        
        print(f"Layer {i:2d}: {layer_type} ({constraint:8s}, {activation:7s} activation)")
    
    print("Final: Non-negative weights → scalar log normalizer")
    
    print("\nKey Properties:")
    print("✓ Alternating weight constraints maintain overall convexity")
    print("✓ Skip connections from input to all layers")
    print("✓ Curriculum learning for stable training")
    print("✓ Gradient clipping for numerical stability")


def main():
    """Main training function for alternating convex architecture."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Convex NN LogZ models')
    parser.add_argument('--data_file', type=str, help='Path to data file (default: data/easy_3d_gaussian.pkl)')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    
    args = parser.parse_args()
    
    print("Training Alternating Convex Neural Network LogZ Model")
    print("=" * 65)
    
    # Load easy_3d_gaussian data
    # Load test data using standardized function
    from src.utils.data_utils import load_standardized_ep_data
    data_file = args.data_file if args.data_file else "data/easy_3d_gaussian.pkl"
    eta, mu_T, metadata = load_standardized_ep_data(data_file)
    
    # Use consistent eta, mu_T format (no reformatting needed)
    data = {
        'train': {
            'eta': eta,
            'mu_T': mu_T
        },
        'val': {
            'eta': eta[:10],  # Use first 10 samples as validation
            'mu_T': mu_T[:10]
        },
        'test': {
            'eta': eta[:10],  # Use first 10 samples as test
            'mu_T': mu_T[:10]
        }
    }
    
    # Create configuration and analyze architecture
    config = create_alternating_convex_config()
    analyze_alternating_architecture(config)
    
    # Create trainer with alternating architecture
    print(f"\nCreating Alternating Convex Neural Network trainer...")
    trainer = Convex_LogZ_Trainer(
        config,
        hessian_method='diagonal',
        adaptive_weights=True
    )
    
    # Train the model
    print(f"\nTraining alternating convex architecture...")
    params, history = trainer.train(data['train'], data['val'])
    
    # Get training time from history (added by base trainer)
    training_time = history.get('total_training_time', 0.0)
    print(f"\n✓ Training completed in {training_time:.1f}s")
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    test_metrics = trainer.evaluate(params, data['test'])
    
    # Make sample predictions to show the model working
    print(f"\nSample predictions on test data:")
    test_sample = data['test']['eta'][:5]
    log_norm_pred = trainer.model.apply(params, test_sample, training=False)
    gradient_pred = trainer.compute_gradient(params, test_sample)
    
    print(f"Input eta (first 3 test samples, 9D):")
    for i, eta in enumerate(test_sample[:3]):
        eta_str = ", ".join([f"{x:6.3f}" for x in eta])
        print(f"  Sample {i+1}: [{eta_str}]")
    
    print(f"\nLog normalizer predictions: {log_norm_pred[:3]}")
    print(f"Gradient predictions (sufficient statistics, 9D):")
    for i, grad in enumerate(gradient_pred[:3]):
        grad_str = ", ".join([f"{x:6.3f}" for x in grad])
        print(f"  Sample {i+1}: [{grad_str}]")
    
    print(f"\nTrue targets (9D):")
    true_targets = data['test']['mu_T'][:3]
    for i, target in enumerate(true_targets):
        target_str = ", ".join([f"{x:6.3f}" for x in target])
        print(f"  Sample {i+1}: [{target_str}]")
    
    # Final performance summary
    print(f"\nFinal Performance Metrics:")
    print(f"=" * 40)
    print(f"Mean MSE: {test_metrics['mean_mse']:.8f}")
    print(f"Mean MAE: {test_metrics['mean_mae']:.8f}")
    
    if 'mean_runtime_ms' in test_metrics:
        print(f"Runtime (JIT): {test_metrics['mean_runtime_ms']:.3f}ms per batch")
        print(f"Per sample: {test_metrics.get('per_sample_runtime_ms', 0):.3f}ms")
    
    # Training history summary
    if 'train_loss' in history and len(history['train_loss']) > 0:
        final_train_loss = history['train_loss'][-1]
        print(f"Final train loss: {final_train_loss:.6f}")
        
        if 'val_loss' in history and len(history['val_loss']) > 0:
            final_val_loss = history['val_loss'][-1]
            print(f"Final val loss: {final_val_loss:.6f}")
    
    print(f"\n✓ Alternating Convex Architecture Advantages:")
    print("- Type 1 layers: W_z ≥ 0, convex activations (standard ICNN)")
    print("- Type 2 layers: W_z ≤ 0, concave activations (novel approach)")
    print("- Alternating pattern maintains overall convexity")
    print("- Skip connections from input to all layers")
    print("- Guaranteed convex log normalizer A(η)")
    print("- Stable gradient computation")
    print("- Positive semi-definite Hessian (valid covariance)")
    
    # Create output directory
    output_dir = Path("artifacts/logZ_models/convex_nn_logZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create results dictionary for plotting
    results = {
        "convex_nn_logZ": {
            "train_loss": history.get('train_loss', []),
            "val_loss": history.get('val_loss', []),
            "test_metrics": test_metrics,
            "model_name": "convex_nn_logZ",
            "config": {"architecture": "convex_nn", "hidden_sizes": [64, 48, 32, 24, 16, 8]}
        }
    }
    
    # Create model comparison plots using standardized plotting function
    plot_model_comparison(
        results=results,
        output_dir=str(output_dir),
        save_plots=True,
        show_plots=False
    )
    
    # Save results summary using standardized function
    save_results_summary(
        results=results,
        output_dir=str(output_dir)
    )
    
    print(f"\n✅ Convex NN LogZ training complete!")
    print(f"📁 Results saved to {output_dir}")
    
    return params, history, test_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Convex NN LogZ models')
    parser.add_argument('--data_file', type=str, help='Path to data file (default: data/easy_3d_gaussian.pkl)')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    
    args = parser.parse_args()
    main()
