#!/usr/bin/env python3
"""
Test the geometric loss function on our best architectures.

This script compares standard MSE training vs geometric KL loss training
on our top-performing architectures.
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

from src.ef import GaussianNatural1D, MultivariateNormal
from src.model import nat2statMLP
from src.quadratic_resnet import QuadraticResNet, DeepAdaptiveQuadraticResNet, create_quadratic_train_state
from src.geometric_loss import train_with_geometric_loss, evaluate_with_geometric_metrics, geometric_loss_fn
from scripts.run_noprop_ct_demo import load_existing_data


def analytical_solution_1d(eta):
    """Analytical solution for 1D Gaussian."""
    eta1, eta2 = eta[:, 0], eta[:, 1]
    mu = -eta1 / (2 * eta2)
    sigma2 = -1 / (2 * eta2)
    E_x = mu
    E_x2 = mu**2 + sigma2
    return jnp.stack([E_x, E_x2], axis=-1)


def train_standard_model(model, params, train_data, val_data, config, name="Model"):
    """Train model with standard MSE loss."""
    
    num_epochs = config.get('num_epochs', 100)
    learning_rate = config.get('learning_rate', 1e-3)
    batch_size = config.get('batch_size', 64)
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate)
    )
    opt_state = optimizer.init(params)
    
    best_params = params
    best_val_loss = float('inf')
    
    print(f"  Training {name} with standard MSE...")
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
        
        # Validation
        val_pred = model.apply(params, val_data['eta'])
        val_loss = float(jnp.mean(jnp.square(val_pred - val_data['y'])))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
        
        if epoch % 20 == 0:
            print(f"    Epoch {epoch}: Train={float(loss):.6f}, Val={val_loss:.6f}")
    
    training_time = time.time() - start_time
    print(f"  Standard training completed in {training_time:.1f}s")
    
    return best_params, {'training_time': training_time, 'best_val_loss': best_val_loss}


def test_geometric_loss():
    """Test geometric loss vs standard loss on 1D Gaussian."""
    
    print("🔬 TESTING GEOMETRIC LOSS FUNCTION")
    print("=" * 60)
    print("Comparing standard MSE vs geometric KL loss")
    
    # Load 1D data (which includes covariance)
    ef = GaussianNatural1D()
    train_data, val_data, test_data = load_existing_data("gaussian_1d")
    
    # Check if we have covariance data
    print(f"\nChecking data structure...")
    
    # Load the actual data file to get covariance
    data_file = Path("data/training_data_7a6d32f607c09d157b94129eff71e641.pkl")
    with open(data_file, 'rb') as f:
        full_data = pickle.load(f)
    
    # Add covariance to our data splits
    train_data['cov'] = full_data['train_cov']
    val_data['cov'] = full_data['val_cov'][:val_data['eta'].shape[0]]  # Match validation size
    test_data['cov'] = full_data['val_cov'][:test_data['eta'].shape[0]]  # Match test size
    
    print(f"Data shapes:")
    print(f"  Train eta: {train_data['eta'].shape}")
    print(f"  Train y: {train_data['y'].shape}")  
    print(f"  Train cov: {train_data['cov'].shape}")
    
    # Analytical baseline
    analytical_pred = analytical_solution_1d(test_data['eta'])
    analytical_mse = float(jnp.mean(jnp.square(analytical_pred - test_data['y'])))
    print(f"\\nAnalytical MSE baseline: {analytical_mse:.8f}")
    
    results = {}
    
    # Test 1: Standard MLP with standard loss
    print(f"\n{'='*60}")
    print("1. STANDARD MLP + STANDARD LOSS")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(42)
    standard_mlp = nat2statMLP(hidden_sizes=[64, 32], activation='tanh', output_dim=2)
    standard_params = standard_mlp.init(rng, test_data['eta'][:1])
    
    standard_config = {
        'num_epochs': 80,
        'learning_rate': 1e-3,
        'batch_size': 64
    }
    
    standard_params_mse, standard_history_mse = train_standard_model(
        standard_mlp, standard_params, train_data, val_data, 
        standard_config, "Standard MLP (MSE)"
    )
    
    # Evaluate with geometric metrics
    standard_pred = standard_mlp.apply(standard_params_mse, test_data['eta'])
    standard_mse = float(jnp.mean(jnp.square(standard_pred - test_data['y'])))
    
    results['standard_mse'] = {
        'name': 'Standard MLP (MSE)',
        'mse': standard_mse,
        'training_time': standard_history_mse['training_time']
    }
    
    print(f"  Final MSE: {standard_mse:.6f}")
    
    # Test 2: Standard MLP with geometric loss
    print(f"\n{'='*60}")
    print("2. STANDARD MLP + GEOMETRIC LOSS")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(43)
    standard_params_geo = standard_mlp.init(rng, test_data['eta'][:1])
    
    geometric_config = {
        'num_epochs': 80,
        'learning_rate': 5e-4,  # Lower LR for geometric loss
        'batch_size': 64,
        'kl_weight': 0.1,      # Start with smaller KL weight
        'mse_weight': 1.0,
        'regularization': 1e-5
    }
    
    standard_params_geo, standard_history_geo = train_with_geometric_loss(
        standard_mlp, standard_params_geo, train_data, val_data,
        geometric_config, "Standard MLP (Geometric)"
    )
    
    results['standard_geometric'] = evaluate_with_geometric_metrics(
        standard_mlp, standard_params_geo, test_data, "Standard MLP (Geometric)"
    )
    results['standard_geometric']['training_time'] = standard_history_geo['training_time']
    
    # Test 3: Quadratic ResNet with standard loss
    print(f"\n{'='*60}")
    print("3. QUADRATIC RESNET + STANDARD LOSS")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(44)
    quad_config = {
        'model_type': 'quadratic',
        'hidden_size': 64,
        'num_layers': 6,
        'activation': 'tanh'
    }
    quad_model, quad_params = create_quadratic_train_state(ef, quad_config, rng)
    
    quad_params_mse, quad_history_mse = train_standard_model(
        quad_model, quad_params, train_data, val_data,
        standard_config, "Quadratic ResNet (MSE)"
    )
    
    quad_pred = quad_model.apply(quad_params_mse, test_data['eta'])
    quad_mse = float(jnp.mean(jnp.square(quad_pred - test_data['y'])))
    
    results['quadratic_mse'] = {
        'name': 'Quadratic ResNet (MSE)',
        'mse': quad_mse,
        'training_time': quad_history_mse['training_time']
    }
    
    print(f"  Final MSE: {quad_mse:.6f}")
    
    # Test 4: Quadratic ResNet with geometric loss
    print(f"\n{'='*60}")
    print("4. QUADRATIC RESNET + GEOMETRIC LOSS")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(45)
    _, quad_params_geo = create_quadratic_train_state(ef, quad_config, rng)
    
    quad_params_geo, quad_history_geo = train_with_geometric_loss(
        quad_model, quad_params_geo, train_data, val_data,
        geometric_config, "Quadratic ResNet (Geometric)"
    )
    
    results['quadratic_geometric'] = evaluate_with_geometric_metrics(
        quad_model, quad_params_geo, test_data, "Quadratic ResNet (Geometric)"
    )
    results['quadratic_geometric']['training_time'] = quad_history_geo['training_time']
    
    # Summary
    print(f"\n{'='*60}")
    print("GEOMETRIC LOSS COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    print(f"{'Method':<30} {'MSE':<12} {'vs Analytical':<15} {'Time(s)':<10}")
    print("-" * 70)
    print(f"{'Analytical':<30} {analytical_mse:<12.8f} {'Baseline':<15} {'0.0':<10}")
    
    for key, result in results.items():
        ratio = result['mse'] / analytical_mse
        print(f"{result['name']:<30} {result['mse']:<12.6f} {ratio:<15.1f}x {result['training_time']:<10.1f}")
    
    # Check if geometric loss helps
    standard_improvement = (results['standard_mse']['mse'] - results['standard_geometric']['mse']) / results['standard_mse']['mse'] * 100
    quad_improvement = (results['quadratic_mse']['mse'] - results['quadratic_geometric']['mse']) / results['quadratic_mse']['mse'] * 100
    
    print(f"\\n📊 GEOMETRIC LOSS IMPACT:")
    print(f"  Standard MLP: {standard_improvement:+.1f}% change with geometric loss")
    print(f"  Quadratic ResNet: {quad_improvement:+.1f}% change with geometric loss")
    
    if standard_improvement > 0:
        print(f"  ✅ Geometric loss improves Standard MLP!")
    if quad_improvement > 0:
        print(f"  ✅ Geometric loss improves Quadratic ResNet!")
    
    # Create visualization
    create_geometric_loss_plots(results, analytical_mse)
    
    return results


def create_geometric_loss_plots(results, analytical_mse):
    """Create plots comparing geometric vs standard loss."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Performance comparison
    methods = list(results.keys())
    names = [results[m]['name'] for m in methods]
    mses = [results[m]['mse'] for m in methods]
    colors = ['red', 'orange', 'blue', 'purple']
    
    bars = axes[0, 0].bar(range(len(names)), mses, color=colors, alpha=0.7)
    axes[0, 0].axhline(y=analytical_mse, color='black', linestyle='--', label='Analytical')
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels([n.replace(' ', '\\n') for n in names], rotation=0)
    axes[0, 0].set_ylabel('Test MSE (log scale)')
    axes[0, 0].set_title('Geometric Loss vs Standard Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training time comparison
    times = [results[m]['training_time'] for m in methods]
    bars = axes[0, 1].bar(range(len(names)), times, color=colors, alpha=0.7)
    axes[0, 1].set_xticks(range(len(names)))
    axes[0, 1].set_xticklabels([n.replace(' ', '\\n') for n in names], rotation=0)
    axes[0, 1].set_ylabel('Training Time (s)')
    axes[0, 1].set_title('Training Time Comparison')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Improvement analysis
    architectures = ['Standard MLP', 'Quadratic ResNet']
    mse_improvements = []
    
    for arch in architectures:
        mse_key = arch.lower().replace(' ', '_') + '_mse'
        geo_key = arch.lower().replace(' ', '_') + '_geometric'
        
        if mse_key in results and geo_key in results:
            improvement = (results[mse_key]['mse'] - results[geo_key]['mse']) / results[mse_key]['mse'] * 100
            mse_improvements.append(improvement)
        else:
            mse_improvements.append(0)
    
    colors_imp = ['green' if x > 0 else 'red' for x in mse_improvements]
    bars = axes[1, 0].bar(range(len(architectures)), mse_improvements, color=colors_imp, alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 0].set_xticks(range(len(architectures)))
    axes[1, 0].set_xticklabels(architectures)
    axes[1, 0].set_ylabel('Improvement (%)')
    axes[1, 0].set_title('Geometric Loss Improvement')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary text
    axes[1, 1].axis('off')
    
    best_overall = min(results.items(), key=lambda x: x[1]['mse'])
    
    summary_text = "GEOMETRIC LOSS RESULTS\\n"
    summary_text += "=" * 25 + "\\n\\n"
    summary_text += f"🥇 Best Overall: {best_overall[1]['name']}\\n"
    summary_text += f"   MSE: {best_overall[1]['mse']:.6f}\\n"
    summary_text += f"   vs Analytical: {best_overall[1]['mse']/analytical_mse:.1f}x\\n\\n"
    
    summary_text += "KEY INSIGHTS:\\n"
    summary_text += "• Geometric loss uses empirical\\n  covariance structure\\n"
    summary_text += "• Network Jacobian estimates\\n  covariance via ∇_η μ_net\\n"
    summary_text += "• KL divergence respects\\n  natural geometry\\n"
    summary_text += "• Second-order information\\n  may stabilize training\\n"
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save plots
    output_dir = Path("artifacts/geometric_loss_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "geometric_loss_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "geometric_loss_comparison.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plots saved to {output_dir}/")


if __name__ == "__main__":
    print("🔬 Starting geometric loss evaluation...")
    
    try:
        results = test_geometric_loss()
        print("\\n✅ Geometric loss evaluation completed!")
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
