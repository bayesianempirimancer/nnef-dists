#!/usr/bin/env python3
"""
Test the quadratic ResNet architecture for learning division operations.

This script compares:
1. Standard MLP
2. Quadratic ResNet (y = x + Wx + (B*x)*x)
3. Adaptive Quadratic ResNet (with learnable mixing coefficients)
4. Analytical solution
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
from src.quadratic_resnet import QuadraticResNet, DeepAdaptiveQuadraticResNet, create_quadratic_train_state
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


def train_model(model, params, train_data, val_data, num_epochs=40, learning_rate=1e-3, name="Model"):
    """Train a model and return final parameters and history."""
    
    # Use gradient clipping for stability with quadratic terms
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adam(learning_rate)
    )
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


def test_quadratic_resnet():
    """Test quadratic ResNet architectures."""
    
    print("🧮 Testing Quadratic ResNet Architectures")
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
    
    results['analytical'] = {
        'mse': analytical_mse, 
        'mae': analytical_mae,
        'training_time': 0.0
    }
    
    # Standard MLP baseline
    print("\n2. Standard MLP (Baseline)")
    rng = random.PRNGKey(42)
    standard_mlp = nat2statMLP(hidden_sizes=[64, 32], activation='tanh', output_dim=2)
    standard_params = standard_mlp.init(rng, test_data['eta'][:1])
    
    standard_params, standard_history = train_model(
        standard_mlp, standard_params, train_data, val_data, num_epochs=40, name="Standard MLP"
    )
    
    standard_pred = standard_mlp.apply(standard_params, test_data['eta'])
    standard_mse = float(jnp.mean(jnp.square(standard_pred - test_data['y'])))
    standard_mae = float(jnp.mean(jnp.abs(standard_pred - test_data['y'])))
    
    results['standard_mlp'] = {
        'mse': standard_mse, 
        'mae': standard_mae,
        'training_time': standard_history['training_time']
    }
    
    # Quadratic ResNet
    print("\n3. Quadratic ResNet (y = x + Wx + (B*x)*x)")
    rng = random.PRNGKey(43)
    quad_config = {
        'model_type': 'quadratic',
        'hidden_size': 64,
        'num_layers': 8,
        'activation': 'tanh',
        'use_activation_between_layers': True
    }
    quad_model, quad_params = create_quadratic_train_state(ef, quad_config, rng)
    
    quad_params, quad_history = train_model(
        quad_model, quad_params, train_data, val_data, 
        num_epochs=40, learning_rate=8e-4, name="Quadratic ResNet"
    )
    
    quad_pred = quad_model.apply(quad_params, test_data['eta'])
    quad_mse = float(jnp.mean(jnp.square(quad_pred - test_data['y'])))
    quad_mae = float(jnp.mean(jnp.abs(quad_pred - test_data['y'])))
    
    results['quadratic'] = {
        'mse': quad_mse, 
        'mae': quad_mae,
        'training_time': quad_history['training_time']
    }
    
    # Deep Quadratic ResNet (more layers)
    print("\n4. Deep Quadratic ResNet (12 layers)")
    rng = random.PRNGKey(44)
    deep_quad_config = {
        'model_type': 'quadratic',
        'hidden_size': 64,
        'num_layers': 12,
        'activation': 'tanh',
        'use_activation_between_layers': True
    }
    deep_quad_model, deep_quad_params = create_quadratic_train_state(ef, deep_quad_config, rng)
    
    deep_quad_params, deep_quad_history = train_model(
        deep_quad_model, deep_quad_params, train_data, val_data, 
        num_epochs=40, learning_rate=6e-4, name="Deep Quadratic ResNet"
    )
    
    deep_quad_pred = deep_quad_model.apply(deep_quad_params, test_data['eta'])
    deep_quad_mse = float(jnp.mean(jnp.square(deep_quad_pred - test_data['y'])))
    deep_quad_mae = float(jnp.mean(jnp.abs(deep_quad_pred - test_data['y'])))
    
    results['deep_quadratic'] = {
        'mse': deep_quad_mse, 
        'mae': deep_quad_mae,
        'training_time': deep_quad_history['training_time']
    }
    
    # Adaptive Quadratic ResNet
    print("\n5. Adaptive Quadratic ResNet (learnable coefficients)")
    rng = random.PRNGKey(45)
    adaptive_config = {
        'model_type': 'adaptive_quadratic',
        'hidden_size': 64,
        'num_layers': 8,
        'activation': 'tanh'
    }
    adaptive_model, adaptive_params = create_quadratic_train_state(ef, adaptive_config, rng)
    
    adaptive_params, adaptive_history = train_model(
        adaptive_model, adaptive_params, train_data, val_data, 
        num_epochs=40, learning_rate=8e-4, name="Adaptive Quadratic ResNet"
    )
    
    adaptive_pred = adaptive_model.apply(adaptive_params, test_data['eta'])
    adaptive_mse = float(jnp.mean(jnp.square(adaptive_pred - test_data['y'])))
    adaptive_mae = float(jnp.mean(jnp.abs(adaptive_pred - test_data['y'])))
    
    results['adaptive_quadratic'] = {
        'mse': adaptive_mse, 
        'mae': adaptive_mae,
        'training_time': adaptive_history['training_time']
    }
    
    # Summary
    print("\n📊 Quadratic ResNet Comparison Summary")
    print("=" * 70)
    print(f"{'Method':<25} {'MSE':<12} {'MAE':<12} {'Time(s)':<10} {'vs Standard':<15}")
    print("-" * 80)
    print(f"{'Analytical':<25} {analytical_mse:<12.6f} {analytical_mae:<12.6f} {'0.0':<10} {'{:.0f}x better'.format(standard_mse/analytical_mse):<15}")
    print(f"{'Standard MLP':<25} {standard_mse:<12.6f} {standard_mae:<12.6f} {standard_history['training_time']:<10.1f} {'Baseline':<15}")
    print(f"{'Quadratic ResNet':<25} {quad_mse:<12.6f} {quad_mae:<12.6f} {quad_history['training_time']:<10.1f} {'{:.1f}% change'.format((quad_mse-standard_mse)/standard_mse*100):<15}")
    print(f"{'Deep Quadratic':<25} {deep_quad_mse:<12.6f} {deep_quad_mae:<12.6f} {deep_quad_history['training_time']:<10.1f} {'{:.1f}% change'.format((deep_quad_mse-standard_mse)/standard_mse*100):<15}")
    print(f"{'Adaptive Quadratic':<25} {adaptive_mse:<12.6f} {adaptive_mae:<12.6f} {adaptive_history['training_time']:<10.1f} {'{:.1f}% change'.format((adaptive_mse-standard_mse)/standard_mse*100):<15}")
    
    # Find best quadratic model
    quadratic_models = {k: v for k, v in results.items() if 'quadratic' in k}
    if quadratic_models:
        best_quad = min(quadratic_models.items(), key=lambda x: x[1]['mse'])
        print(f"\n🏆 Best Quadratic Model: {best_quad[0].replace('_', ' ').title()}")
        print(f"   MSE: {best_quad[1]['mse']:.6f}")
        
        if best_quad[1]['mse'] < standard_mse:
            improvement = (standard_mse - best_quad[1]['mse']) / standard_mse * 100
            print(f"✅ Quadratic ResNet is {improvement:.1f}% better than standard MLP!")
        else:
            print(f"❌ Quadratic ResNet did not outperform standard MLP")
    
    # Generate comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Training curves
    axes[0, 0].plot(standard_history['val_losses'], label='Standard MLP', alpha=0.8, linewidth=2)
    axes[0, 0].plot(quad_history['val_losses'], label='Quadratic ResNet', alpha=0.8, linewidth=2)
    axes[0, 0].plot(deep_quad_history['val_losses'], label='Deep Quadratic', alpha=0.8, linewidth=2)
    axes[0, 0].plot(adaptive_history['val_losses'], label='Adaptive Quadratic', alpha=0.8, linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Validation MSE')
    axes[0, 0].set_title('Training Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_yscale('log')
    
    # Standard MLP predictions
    axes[0, 1].scatter(test_data['y'][:, 0], standard_pred[:, 0], alpha=0.6, s=20, label='E[x]')
    axes[0, 1].scatter(test_data['y'][:, 1], standard_pred[:, 1], alpha=0.6, s=20, label='E[x²]')
    axes[0, 1].plot([test_data['y'].min(), test_data['y'].max()], 
                   [test_data['y'].min(), test_data['y'].max()], 'r--', alpha=0.8)
    axes[0, 1].set_xlabel('True Values')
    axes[0, 1].set_ylabel('Predicted Values')
    axes[0, 1].set_title('Standard MLP Predictions')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Best quadratic model predictions
    best_pred = quad_pred if quad_mse <= deep_quad_mse and quad_mse <= adaptive_mse else (
        deep_quad_pred if deep_quad_mse <= adaptive_mse else adaptive_pred
    )
    best_name = "Quadratic" if quad_mse <= deep_quad_mse and quad_mse <= adaptive_mse else (
        "Deep Quadratic" if deep_quad_mse <= adaptive_mse else "Adaptive Quadratic"
    )
    
    axes[0, 2].scatter(test_data['y'][:, 0], best_pred[:, 0], alpha=0.6, s=20, label='E[x]')
    axes[0, 2].scatter(test_data['y'][:, 1], best_pred[:, 1], alpha=0.6, s=20, label='E[x²]')
    axes[0, 2].plot([test_data['y'].min(), test_data['y'].max()], 
                   [test_data['y'].min(), test_data['y'].max()], 'r--', alpha=0.8)
    axes[0, 2].set_xlabel('True Values')
    axes[0, 2].set_ylabel('Predicted Values')
    axes[0, 2].set_title(f'Best {best_name} Predictions')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Error comparison
    standard_errors = jnp.abs(standard_pred - test_data['y'])
    quad_errors = jnp.abs(best_pred - test_data['y'])
    
    axes[1, 0].hist(standard_errors.flatten(), bins=30, alpha=0.7, label='Standard MLP', density=True)
    axes[1, 0].hist(quad_errors.flatten(), bins=30, alpha=0.7, label=f'{best_name}', density=True)
    axes[1, 0].set_xlabel('Absolute Error')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Performance comparison bar chart
    model_names = ['Standard\nMLP', 'Quadratic\nResNet', 'Deep\nQuadratic', 'Adaptive\nQuadratic']
    mse_values = [standard_mse, quad_mse, deep_quad_mse, adaptive_mse]
    colors = ['red', 'blue', 'green', 'orange']
    
    bars = axes[1, 1].bar(model_names, mse_values, alpha=0.7, color=colors)
    axes[1, 1].set_ylabel('Test MSE')
    axes[1, 1].set_title('Performance Comparison')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, mse_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Improvement over epochs
    epochs = range(len(standard_history['val_losses']))
    axes[1, 2].plot(epochs, jnp.array(standard_history['val_losses']) / analytical_mse, 
                   label='Standard MLP', alpha=0.8, linewidth=2)
    axes[1, 2].plot(epochs, jnp.array(quad_history['val_losses']) / analytical_mse, 
                   label='Quadratic ResNet', alpha=0.8, linewidth=2)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('MSE / Analytical MSE')
    axes[1, 2].set_title('Convergence to Theoretical Limit')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    axes[1, 2].set_yscale('log')
    
    plt.tight_layout()
    
    # Save plots
    output_dir = Path("artifacts/quadratic_resnet_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "quadratic_resnet_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "quadratic_resnet_comparison.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Plots saved to {output_dir}/")
    
    return results


if __name__ == "__main__":
    results = test_quadratic_resnet()
