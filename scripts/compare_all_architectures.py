#!/usr/bin/env python3
"""
Comprehensive comparison of all neural network architectures for learning eta -> y mapping.

This script tests:
1. Standard MLP
2. Division-aware MLP (with explicit division operations)
3. GLU-based MLP (with gating mechanisms)
4. Deep GLU with residual connections
5. Simple analytical solution (for comparison)
"""

import sys
from pathlib import Path
import pickle
import jax
import jax.numpy as jnp
from jax import random
import optax
import matplotlib.pyplot as plt
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ef import GaussianNatural1D
from src.model import nat2statMLP
from src.division_aware_mlp import DivisionAwareMomentNet
from src.glu_moment_net import GLUMomentNet, DeepGLUMomentNet, create_glu_train_state
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


def train_model(model, params, train_data, val_data, num_epochs=30, learning_rate=1e-3, name="Model"):
    """Train a model and return final parameters and history."""
    
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    train_losses = []
    val_losses = []
    
    print(f"  Training {name}...")
    start_time = time.time()
    
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
            print(f"    Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")
    
    training_time = time.time() - start_time
    print(f"  Training completed in {training_time:.2f}s")
    
    return params, {'train_losses': train_losses, 'val_losses': val_losses, 'training_time': training_time}


def compare_all_architectures():
    """Compare all neural network architectures."""
    
    print("🔬 Comprehensive Architecture Comparison")
    print("=" * 60)
    
    # Load data
    ef = GaussianNatural1D()
    train_data, val_data, test_data = load_existing_data("gaussian_1d")
    
    print(f"Dataset: {train_data['eta'].shape[0]} training samples")
    print(f"Test set: {test_data['eta'].shape[0]} samples")
    
    results = {}
    
    # Test analytical solution first
    print("\n1. Analytical Solution (Ground Truth)")
    analytical_pred = analytical_solution(test_data['eta'])
    analytical_mse = float(jnp.mean(jnp.square(analytical_pred - test_data['y'])))
    analytical_mae = float(jnp.mean(jnp.abs(analytical_pred - test_data['y'])))
    
    print(f"  MSE: {analytical_mse:.6f}")
    print(f"  MAE: {analytical_mae:.6f}")
    print(f"  Note: This is the theoretical lower bound (MCMC sampling error)")
    
    results['analytical'] = {
        'mse': analytical_mse, 
        'mae': analytical_mae,
        'training_time': 0.0
    }
    
    # Standard MLP
    print("\n2. Standard MLP")
    rng = random.PRNGKey(42)
    standard_mlp = nat2statMLP(hidden_sizes=[64, 32], activation='tanh', output_dim=2)
    standard_params = standard_mlp.init(rng, test_data['eta'][:1])
    
    standard_params, standard_history = train_model(
        standard_mlp, standard_params, train_data, val_data, num_epochs=30, name="Standard MLP"
    )
    
    standard_pred = standard_mlp.apply(standard_params, test_data['eta'])
    standard_mse = float(jnp.mean(jnp.square(standard_pred - test_data['y'])))
    standard_mae = float(jnp.mean(jnp.abs(standard_pred - test_data['y'])))
    
    results['standard_mlp'] = {
        'mse': standard_mse, 
        'mae': standard_mae,
        'training_time': standard_history['training_time']
    }
    
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
        division_mlp, division_params, train_data, val_data, num_epochs=30, name="Division-Aware MLP"
    )
    
    division_pred = division_mlp.apply(division_params, test_data['eta'])
    division_mse = float(jnp.mean(jnp.square(division_pred - test_data['y'])))
    division_mae = float(jnp.mean(jnp.abs(division_pred - test_data['y'])))
    
    results['division_aware'] = {
        'mse': division_mse, 
        'mae': division_mae,
        'training_time': division_history['training_time']
    }
    
    # GLU-based MLP
    print("\n4. GLU-based MLP")
    rng = random.PRNGKey(44)
    glu_config = {
        'model_type': 'glu',
        'hidden_sizes': (64, 32),
        'activation': 'tanh',
        'use_glu_layers': True,
        'glu_hidden_ratio': 2.0
    }
    glu_model, glu_params = create_glu_train_state(ef, glu_config, rng)
    
    glu_params, glu_history = train_model(
        glu_model, glu_params, train_data, val_data, num_epochs=30, name="GLU MLP"
    )
    
    glu_pred = glu_model.apply(glu_params, test_data['eta'])
    glu_mse = float(jnp.mean(jnp.square(glu_pred - test_data['y'])))
    glu_mae = float(jnp.mean(jnp.abs(glu_pred - test_data['y'])))
    
    results['glu'] = {
        'mse': glu_mse, 
        'mae': glu_mae,
        'training_time': glu_history['training_time']
    }
    
    # Deep GLU with residual connections
    print("\n5. Deep GLU with Residual Connections")
    rng = random.PRNGKey(45)
    deep_glu_config = {
        'model_type': 'deep_glu',
        'hidden_size': 64,
        'num_glu_layers': 4,
        'activation': 'tanh'
    }
    deep_glu_model, deep_glu_params = create_glu_train_state(ef, deep_glu_config, rng)
    
    deep_glu_params, deep_glu_history = train_model(
        deep_glu_model, deep_glu_params, train_data, val_data, num_epochs=30, name="Deep GLU"
    )
    
    deep_glu_pred = deep_glu_model.apply(deep_glu_params, test_data['eta'])
    deep_glu_mse = float(jnp.mean(jnp.square(deep_glu_pred - test_data['y'])))
    deep_glu_mae = float(jnp.mean(jnp.abs(deep_glu_pred - test_data['y'])))
    
    results['deep_glu'] = {
        'mse': deep_glu_mse, 
        'mae': deep_glu_mae,
        'training_time': deep_glu_history['training_time']
    }
    
    # Summary
    print("\n📊 Comprehensive Architecture Comparison Summary")
    print("=" * 60)
    print(f"{'Method':<25} {'MSE':<12} {'MAE':<12} {'Time(s)':<10} {'Improvement':<15}")
    print("-" * 80)
    print(f"{'Analytical':<25} {analytical_mse:<12.6f} {analytical_mae:<12.6f} {'0.0':<10} {'Baseline':<15}")
    print(f"{'Standard MLP':<25} {standard_mse:<12.6f} {standard_mae:<12.6f} {standard_history['training_time']:<10.1f} {'{:.1f}x worse'.format(standard_mse/analytical_mse):<15}")
    print(f"{'Division-Aware MLP':<25} {division_mse:<12.6f} {division_mae:<12.6f} {division_history['training_time']:<10.1f} {'{:.1f}x worse'.format(division_mse/analytical_mse):<15}")
    print(f"{'GLU MLP':<25} {glu_mse:<12.6f} {glu_mae:<12.6f} {glu_history['training_time']:<10.1f} {'{:.1f}x worse'.format(glu_mse/analytical_mse):<15}")
    print(f"{'Deep GLU':<25} {deep_glu_mse:<12.6f} {deep_glu_mae:<12.6f} {deep_glu_history['training_time']:<10.1f} {'{:.1f}x worse'.format(deep_glu_mse/analytical_mse):<15}")
    
    # Find best performing model
    neural_models = {k: v for k, v in results.items() if k != 'analytical'}
    best_model = min(neural_models.items(), key=lambda x: x[1]['mse'])
    
    print(f"\n🏆 Best Neural Network: {best_model[0].replace('_', ' ').title()}")
    print(f"   MSE: {best_model[1]['mse']:.6f}")
    print(f"   Training time: {best_model[1]['training_time']:.1f}s")
    
    # Check if GLU is better than standard MLP
    if glu_mse < standard_mse:
        improvement = (standard_mse - glu_mse) / standard_mse * 100
        print(f"\n✅ GLU MLP is {improvement:.1f}% better than standard MLP!")
    else:
        print(f"\n❌ GLU MLP is not better than standard MLP")
    
    if deep_glu_mse < standard_mse:
        improvement = (standard_mse - deep_glu_mse) / standard_mse * 100
        print(f"✅ Deep GLU is {improvement:.1f}% better than standard MLP!")
    
    # Generate comprehensive comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Training curves
    axes[0, 0].plot(standard_history['val_losses'], label='Standard MLP', alpha=0.8)
    axes[0, 0].plot(division_history['val_losses'], label='Division-Aware', alpha=0.8)
    axes[0, 0].plot(glu_history['val_losses'], label='GLU MLP', alpha=0.8)
    axes[0, 0].plot(deep_glu_history['val_losses'], label='Deep GLU', alpha=0.8)
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
    
    # Predictions vs targets - GLU MLP
    axes[0, 2].scatter(test_data['y'][:, 0], glu_pred[:, 0], alpha=0.6, s=20, label='E[x]')
    axes[0, 2].scatter(test_data['y'][:, 1], glu_pred[:, 1], alpha=0.6, s=20, label='E[x²]')
    axes[0, 2].plot([test_data['y'].min(), test_data['y'].max()], 
                   [test_data['y'].min(), test_data['y'].max()], 'r--', alpha=0.8)
    axes[0, 2].set_xlabel('True Values')
    axes[0, 2].set_ylabel('Predicted Values')
    axes[0, 2].set_title('GLU MLP Predictions')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Predictions vs targets - Deep GLU
    axes[1, 0].scatter(test_data['y'][:, 0], deep_glu_pred[:, 0], alpha=0.6, s=20, label='E[x]')
    axes[1, 0].scatter(test_data['y'][:, 1], deep_glu_pred[:, 1], alpha=0.6, s=20, label='E[x²]')
    axes[1, 0].plot([test_data['y'].min(), test_data['y'].max()], 
                   [test_data['y'].min(), test_data['y'].max()], 'r--', alpha=0.8)
    axes[1, 0].set_xlabel('True Values')
    axes[1, 0].set_ylabel('Predicted Values')
    axes[1, 0].set_title('Deep GLU Predictions')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Error comparison
    standard_errors = jnp.abs(standard_pred - test_data['y'])
    glu_errors = jnp.abs(glu_pred - test_data['y'])
    deep_glu_errors = jnp.abs(deep_glu_pred - test_data['y'])
    
    axes[1, 1].hist(standard_errors.flatten(), bins=30, alpha=0.7, label='Standard MLP', density=True)
    axes[1, 1].hist(glu_errors.flatten(), bins=30, alpha=0.7, label='GLU MLP', density=True)
    axes[1, 1].hist(deep_glu_errors.flatten(), bins=30, alpha=0.7, label='Deep GLU', density=True)
    axes[1, 1].set_xlabel('Absolute Error')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Performance comparison bar chart
    model_names = ['Standard\nMLP', 'Division\nAware', 'GLU\nMLP', 'Deep\nGLU']
    mse_values = [standard_mse, division_mse, glu_mse, deep_glu_mse]
    
    bars = axes[1, 2].bar(model_names, mse_values, alpha=0.7, color=['red', 'orange', 'blue', 'green'])
    axes[1, 2].set_ylabel('Test MSE')
    axes[1, 2].set_title('Performance Comparison')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, mse_values):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save plots
    output_dir = Path("artifacts/comprehensive_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "comprehensive_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "comprehensive_comparison.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Plots saved to {output_dir}/")
    
    return results


if __name__ == "__main__":
    results = compare_all_architectures()
