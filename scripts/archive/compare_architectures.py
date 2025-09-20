#!/usr/bin/env python3
"""
Compare different neural network architectures for learning the eta -> y mapping.

This script tests:
1. Standard MLP
2. Division-aware MLP (with explicit division operations)
3. Simple analytical solution (for comparison)
"""

import sys
from pathlib import Path
import pickle
import jax
import jax.numpy as jnp
from jax import random
import optax
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ef import GaussianNatural1D
from src.model import nat2statMLP
from src.division_aware_mlp import DivisionAwareMomentNet
from scripts.run_noprop_ct_demo import load_existing_data


def analytical_solution(eta):
    """Analytical solution for Gaussian 1D: mu = -eta1/(2*eta2), sigma2 = -1/(2*eta2)"""
    eta1, eta2 = eta[:, 0], eta[:, 1]
    
    # Convert to mean and variance
    mu = -eta1 / (2 * eta2)
    sigma2 = -1 / (2 * eta2)
    
    # Expected sufficient statistics: E[x] = mu, E[x^2] = mu^2 + sigma^2
    E_x = mu
    E_x2 = mu**2 + sigma2
    
    return jnp.stack([E_x, E_x2], axis=-1)


def train_model(model, params, train_data, val_data, num_epochs=30, learning_rate=1e-3):
    """Train a model and return final parameters and history."""
    
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        eta_batch = train_data['eta']
        y_batch = train_data['y']
        
        def loss_fn(params):
            pred = model.apply(params, eta_batch)
            return jnp.mean(jnp.square(pred - y_batch))
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        train_loss = float(loss)
        
        # Validation
        val_pred = model.apply(params, val_data['eta'])
        val_loss = float(jnp.mean(jnp.square(val_pred - val_data['y'])))
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")
    
    return params, {'train_losses': train_losses, 'val_losses': val_losses}


def compare_architectures():
    """Compare different architectures."""
    
    print("🔬 Comparing Neural Network Architectures")
    print("=" * 50)
    
    # Load data
    ef = GaussianNatural1D()
    train_data, val_data, test_data = load_existing_data("gaussian_1d")
    
    print(f"Dataset: {train_data['eta'].shape[0]} training samples")
    print(f"Test set: {test_data['eta'].shape[0]} samples")
    
    # Test analytical solution first
    print("\n1. Analytical Solution (Ground Truth)")
    analytical_pred = analytical_solution(test_data['eta'])
    analytical_mse = float(jnp.mean(jnp.square(analytical_pred - test_data['y'])))
    analytical_mae = float(jnp.mean(jnp.abs(analytical_pred - test_data['y'])))
    
    print(f"  MSE: {analytical_mse:.6f}")
    print(f"  MAE: {analytical_mae:.6f}")
    print(f"  Note: This is the theoretical lower bound (MCMC sampling error)")
    
    # Standard MLP
    print("\n2. Standard MLP")
    rng = random.PRNGKey(42)
    standard_mlp = nat2statMLP(hidden_sizes=[64, 32], activation='tanh', output_dim=2)
    standard_params = standard_mlp.init(rng, test_data['eta'][:1])
    
    standard_params, standard_history = train_model(
        standard_mlp, standard_params, train_data, val_data, num_epochs=30
    )
    
    standard_pred = standard_mlp.apply(standard_params, test_data['eta'])
    standard_mse = float(jnp.mean(jnp.square(standard_pred - test_data['y'])))
    standard_mae = float(jnp.mean(jnp.abs(standard_pred - test_data['y'])))
    
    print(f"  Final MSE: {standard_mse:.6f}")
    print(f"  Final MAE: {standard_mae:.6f}")
    
    # Division-aware MLP
    print("\n3. Division-Aware MLP")
    rng = random.PRNGKey(43)
    division_mlp = DivisionAwareMomentNet(
        ef=ef, 
        hidden_sizes=(32, 16),
        use_division_layers=True,
        use_reciprocal_layers=True
    )
    division_params = division_mlp.init(rng, test_data['eta'][:1])
    
    division_params, division_history = train_model(
        division_mlp, division_params, train_data, val_data, num_epochs=30
    )
    
    division_pred = division_mlp.apply(division_params, test_data['eta'])
    division_mse = float(jnp.mean(jnp.square(division_pred - test_data['y'])))
    division_mae = float(jnp.mean(jnp.abs(division_pred - test_data['y'])))
    
    print(f"  Final MSE: {division_mse:.6f}")
    print(f"  Final MAE: {division_mae:.6f}")
    
    # Summary
    print("\n📊 Architecture Comparison Summary")
    print("=" * 40)
    print(f"{'Method':<20} {'MSE':<12} {'MAE':<12} {'Improvement':<15}")
    print("-" * 60)
    print(f"{'Analytical':<20} {analytical_mse:<12.6f} {analytical_mae:<12.6f} {'Baseline':<15}")
    print(f"{'Standard MLP':<20} {standard_mse:<12.6f} {standard_mae:<12.6f} {'{:.1f}x worse'.format(standard_mse/analytical_mse):<15}")
    print(f"{'Division-Aware':<20} {division_mse:<12.6f} {division_mae:<12.6f} {'{:.1f}x worse'.format(division_mse/analytical_mse):<15}")
    
    # Check if division-aware is better
    if division_mse < standard_mse:
        improvement = (standard_mse - division_mse) / standard_mse * 100
        print(f"\n✅ Division-aware MLP is {improvement:.1f}% better than standard MLP!")
    else:
        print(f"\n❌ Division-aware MLP is not better than standard MLP")
    
    # Generate comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training curves
    axes[0, 0].plot(standard_history['val_losses'], label='Standard MLP', alpha=0.8)
    axes[0, 0].plot(division_history['val_losses'], label='Division-Aware', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Validation MSE')
    axes[0, 0].set_title('Training Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_yscale('log')
    
    # Predictions vs targets - Standard MLP
    axes[0, 1].scatter(test_data['y'][:, 0], standard_pred[:, 0], alpha=0.6, s=20, label='E[x]')
    axes[0, 1].scatter(test_data['y'][:, 1], standard_pred[:, 1], alpha=0.6, s=20, label='E[x²]')
    axes[0, 1].plot([test_data['y'].min(), test_data['y'].max()], 
                   [test_data['y'].min(), test_data['y'].max()], 'r--', alpha=0.8)
    axes[0, 1].set_xlabel('True Values')
    axes[0, 1].set_ylabel('Predicted Values')
    axes[0, 1].set_title('Standard MLP Predictions')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Predictions vs targets - Division-aware MLP
    axes[1, 0].scatter(test_data['y'][:, 0], division_pred[:, 0], alpha=0.6, s=20, label='E[x]')
    axes[1, 0].scatter(test_data['y'][:, 1], division_pred[:, 1], alpha=0.6, s=20, label='E[x²]')
    axes[1, 0].plot([test_data['y'].min(), test_data['y'].max()], 
                   [test_data['y'].min(), test_data['y'].max()], 'r--', alpha=0.8)
    axes[1, 0].set_xlabel('True Values')
    axes[1, 0].set_ylabel('Predicted Values')
    axes[1, 0].set_title('Division-Aware MLP Predictions')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Error comparison
    standard_errors = jnp.abs(standard_pred - test_data['y'])
    division_errors = jnp.abs(division_pred - test_data['y'])
    
    axes[1, 1].hist(standard_errors.flatten(), bins=30, alpha=0.7, label='Standard MLP', density=True)
    axes[1, 1].hist(division_errors.flatten(), bins=30, alpha=0.7, label='Division-Aware', density=True)
    axes[1, 1].set_xlabel('Absolute Error')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save plots
    output_dir = Path("artifacts/architecture_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "architecture_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "architecture_comparison.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Plots saved to {output_dir}/")
    
    return {
        'analytical': {'mse': analytical_mse, 'mae': analytical_mae},
        'standard_mlp': {'mse': standard_mse, 'mae': standard_mae},
        'division_aware': {'mse': division_mse, 'mae': division_mae}
    }


if __name__ == "__main__":
    results = compare_architectures()
