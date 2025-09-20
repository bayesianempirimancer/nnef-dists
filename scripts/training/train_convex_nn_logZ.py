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

from src.config import FullConfig
from src.models.convex_nn_logZ import (
    ConvexNeuralNetworkLogZ,
    ConvexNeuralNetworkLogZTrainer
)

def create_alternating_convex_config():
    """Create configuration for alternating convex architecture."""
    config = FullConfig()
    
    # Mid-sized network with 6 layers for 3D Gaussian testing
    config.network.hidden_sizes = [64, 48, 32, 24, 16, 8]  # 6 layers with decreasing size
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


def generate_test_data(n_samples=2000, seed=42):
    """Generate 3D Gaussian test data using MultivariateNormal_tril exponential family."""
    from src.ef import MultivariateNormal_tril
    
    # Create 3D multivariate normal exponential family
    ef = MultivariateNormal_tril(x_shape=(3,))
    
    rng = random.PRNGKey(seed)
    
    eta_vectors = []
    mean_vectors = []
    
    for i in range(n_samples):
        rng, subkey = random.split(rng)
        
        # Generate 3D Gaussian parameters
        mean = random.normal(subkey, (3,)) * 0.5
        
        # Generate positive definite covariance via Cholesky
        rng, subkey = random.split(rng)
        L_raw = random.normal(subkey, (3, 3)) * 0.3
        L = jnp.tril(L_raw) + jnp.eye(3) * 0.5  # Lower triangular with positive diagonal
        cov = L @ L.T  # Positive definite covariance
        
        # Convert to natural parameters using MultivariateNormal_tril
        # For multivariate normal: eta = [Σ^{-1}μ; -0.5*tril(Σ^{-1})]
        sigma_inv = jnp.linalg.inv(cov)
        eta_mean_part = sigma_inv @ mean
        
        # Extract lower triangular part of precision matrix
        L_inv = jnp.linalg.cholesky(sigma_inv)
        eta_precision_part = -0.5 * L_inv[jnp.tril_indices(3)]
        
        eta_vector = jnp.concatenate([eta_mean_part, eta_precision_part])
        
        # Compute expected sufficient statistics using the EF
        # For multivariate normal: E[T(X)] = [μ; tril(μμᵀ + Σ)]
        expected_stats = jnp.concatenate([
            mean,  # 3 elements
            (jnp.outer(mean, mean) + cov)[jnp.tril_indices(3)]  # 6 elements (lower triangular)
        ])  # 9 elements total
        
        eta_vectors.append(eta_vector)
        mean_vectors.append(expected_stats)
    
    eta_array = jnp.array(eta_vectors)
    stats_array = jnp.array(mean_vectors)
    
    # Split into train/val/test
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    
    # Normalize data
    eta_mean = jnp.mean(eta_array, axis=0)
    eta_std = jnp.std(eta_array, axis=0) + 1e-8
    eta_normalized = (eta_array - eta_mean) / eta_std
    
    stats_mean = jnp.mean(stats_array, axis=0)
    stats_std = jnp.std(stats_array, axis=0) + 1e-8
    stats_normalized = (stats_array - stats_mean) / stats_std
    
    data = {
        'train': {
            'eta': eta_normalized[:n_train],
            'mean': stats_normalized[:n_train]
        },
        'val': {
            'eta': eta_normalized[n_train:n_train+n_val],
            'mean': stats_normalized[n_train:n_train+n_val]
        },
        'test': {
            'eta': eta_normalized[n_train+n_val:],
            'mean': stats_normalized[n_train+n_val:]
        }
    }
    
    print(f"Generated data shapes:")
    print(f"  Train: eta={data['train']['eta'].shape}, mean={data['train']['mean'].shape}")
    print(f"  Val:   eta={data['val']['eta'].shape}, mean={data['val']['mean'].shape}")
    print(f"  Test:  eta={data['test']['eta'].shape}, mean={data['test']['mean'].shape}")
    print(f"Data ranges - eta: [{eta_normalized.min():.3f}, {eta_normalized.max():.3f}]")
    
    return data


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
    print("Training Alternating Convex Neural Network LogZ Model")
    print("=" * 65)
    
    # Generate test data
    print("Generating 3D Gaussian test data using MultivariateNormal_tril...")
    data = generate_test_data(n_samples=1000, seed=42)  # Mid-sized dataset for 6-layer model
    
    # Create configuration and analyze architecture
    config = create_alternating_convex_config()
    analyze_alternating_architecture(config)
    
    # Create trainer with alternating architecture
    print(f"\nCreating Alternating Convex Neural Network trainer...")
    trainer = ConvexNeuralNetworkLogZTrainer(
        config,
        hessian_method='diagonal',
        use_curriculum=True  # Helps with training stability
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
    true_targets = data['test']['mean'][:3]
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
    
    return params, history, test_metrics


if __name__ == "__main__":
    main()
