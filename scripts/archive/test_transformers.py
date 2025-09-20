#!/usr/bin/env python3
"""
Test Transformer-based architectures on the Gaussian datasets.

This script evaluates:
1. Standard Transformer (with standard MLP layers)
2. Quadratic Transformer (with quadratic ResNet layers) 
3. Pure Quadratic Transformer (specialized quadratic blocks)

We test on both 1D and 3D Gaussian datasets to see if the attention mechanism
combined with quadratic operations provides benefits.
"""

import sys
from pathlib import Path
import pickle
import jax
import jax.numpy as jnp
from jax import random
import optax
import matplotlib.pyplot as plt
import numpy as np
import time
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ef import GaussianNatural1D, MultivariateNormal
from src.transformer_moments import create_transformer_train_state
from scripts.run_noprop_ct_demo import load_existing_data


def load_3d_data():
    """Load 3D Gaussian data."""
    data_dir = Path("data")
    
    # Find 3D dataset
    for data_file in data_dir.glob("*.pkl"):
        try:
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            
            if data["train_eta"].shape[1] == 12 and data["train_eta"].shape[0] >= 2000:
                print(f"Using 3D dataset: {data_file.name}")
                
                train_data = {"eta": data["train_eta"], "y": data["train_y"]}
                val_data = {"eta": data["val_eta"], "y": data["val_y"]}
                
                # Create test split
                n_val = val_data["eta"].shape[0]
                n_test = min(n_val // 2, 200)
                
                test_data = {
                    "eta": val_data["eta"][:n_test],
                    "y": val_data["y"][:n_test]
                }
                
                val_data = {
                    "eta": val_data["eta"][n_test:],
                    "y": val_data["y"][n_test:]
                }
                
                return train_data, val_data, test_data
        except:
            continue
    
    raise FileNotFoundError("No suitable 3D dataset found")


def analytical_solution_1d(eta):
    """Analytical solution for 1D Gaussian."""
    eta1, eta2 = eta[:, 0], eta[:, 1]
    mu = -eta1 / (2 * eta2)
    sigma2 = -1 / (2 * eta2)
    E_x = mu
    E_x2 = mu**2 + sigma2
    return jnp.stack([E_x, E_x2], axis=-1)


def train_transformer(model, params, train_data, val_data, config, name="Transformer"):
    """Train a Transformer model."""
    
    num_epochs = config.get('num_epochs', 100)
    learning_rate = config.get('learning_rate', 1e-4)
    batch_size = config.get('batch_size', 64)
    
    # Use lower learning rate for Transformers
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate)
    )
    opt_state = optimizer.init(params)
    
    train_losses = []
    val_losses = []
    best_params = params
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    print(f"  Training {name}...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training
        n_train = train_data['eta'].shape[0]
        indices = jnp.arange(n_train)
        rng = random.PRNGKey(epoch)
        indices = random.permutation(rng, indices)
        
        epoch_train_loss = 0.0
        num_batches = 0
        
        for i in range(0, n_train, batch_size):
            batch_indices = indices[i:i+batch_size]
            eta_batch = train_data['eta'][batch_indices]
            y_batch = train_data['y'][batch_indices]
            
            def loss_fn(params):
                pred = model.apply(params, eta_batch, training=True, rngs={'dropout': random.PRNGKey(epoch)})
                return jnp.mean(jnp.square(pred - y_batch))
            
            loss, grads = jax.value_and_grad(loss_fn)(params)
            
            # Check for NaN
            if jnp.isnan(loss):
                print(f"    NaN loss at epoch {epoch}, stopping")
                break
                
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            epoch_train_loss += float(loss)
            num_batches += 1
        
        if num_batches == 0 or jnp.isnan(epoch_train_loss):
            break
            
        epoch_train_loss /= num_batches
        
        # Validation
        val_pred = model.apply(params, val_data['eta'], training=False)
        val_loss = float(jnp.mean(jnp.square(val_pred - val_data['y'])))
        
        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        
        # Best model tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 20 == 0:
            print(f"    Epoch {epoch:3d}: Train={epoch_train_loss:.6f}, Val={val_loss:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break
    
    training_time = time.time() - start_time
    print(f"  Training completed in {training_time:.1f}s, Best val: {best_val_loss:.6f}")
    
    return best_params, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'training_time': training_time,
        'best_val_loss': best_val_loss
    }


def evaluate_transformer(model, params, test_data, name="Transformer"):
    """Evaluate Transformer model."""
    
    pred = model.apply(params, test_data['eta'], training=False)
    
    mse = float(jnp.mean(jnp.square(pred - test_data['y'])))
    mae = float(jnp.mean(jnp.abs(pred - test_data['y'])))
    
    return {
        'name': name,
        'mse': mse,
        'mae': mae,
        'predictions': pred
    }


def test_transformers_1d():
    """Test Transformers on 1D Gaussian."""
    
    print("🤖 TESTING TRANSFORMERS ON 1D GAUSSIAN")
    print("=" * 60)
    
    # Load 1D data
    ef = GaussianNatural1D()
    train_data, val_data, test_data = load_existing_data("gaussian_1d")
    
    print(f"1D Dataset: {train_data['eta'].shape[0]} train, {test_data['eta'].shape[0]} test")
    
    # Analytical baseline
    analytical_pred = analytical_solution_1d(test_data['eta'])
    analytical_mse = float(jnp.mean(jnp.square(analytical_pred - test_data['y'])))
    print(f"Analytical MSE: {analytical_mse:.8f}")
    
    results_1d = {}
    
    # 1. Standard Transformer
    print(f"\n1. Standard Transformer")
    rng = random.PRNGKey(42)
    
    standard_config = {
        'model_type': 'transformer',
        'num_layers': 4,
        'num_heads': 4,
        'head_dim': 32,
        'mlp_dim': 128,
        'dropout_rate': 0.1,
        'activation': 'gelu'
    }
    
    model, params = create_transformer_train_state(ef, standard_config, rng)
    
    train_config = {
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'batch_size': 64
    }
    
    params, history = train_transformer(
        model, params, train_data, val_data, train_config, "Standard Transformer"
    )
    
    results_1d['standard'] = evaluate_transformer(model, params, test_data, "Standard Transformer")
    results_1d['standard']['training_time'] = history['training_time']
    
    # 2. Quadratic Transformer
    print(f"\n2. Quadratic Transformer")
    rng = random.PRNGKey(43)
    
    quad_config = {
        'model_type': 'quadratic_transformer',
        'num_layers': 4,
        'num_heads': 4,
        'head_dim': 32,
        'dropout_rate': 0.1,
        'activation': 'tanh',
        'use_adaptive_quadratic': True
    }
    
    model, params = create_transformer_train_state(ef, quad_config, rng)
    
    params, history = train_transformer(
        model, params, train_data, val_data, train_config, "Quadratic Transformer"
    )
    
    results_1d['quadratic'] = evaluate_transformer(model, params, test_data, "Quadratic Transformer")
    results_1d['quadratic']['training_time'] = history['training_time']
    
    # 3. Pure Quadratic Transformer
    print(f"\n3. Pure Quadratic Transformer")
    rng = random.PRNGKey(44)
    
    pure_config = {
        'model_type': 'pure_quadratic_transformer',
        'num_layers': 4,
        'num_heads': 4,
        'head_dim': 32,
        'dropout_rate': 0.1,
        'activation': 'tanh',
        'num_quadratic_layers': 2
    }
    
    model, params = create_transformer_train_state(ef, pure_config, rng)
    
    params, history = train_transformer(
        model, params, train_data, val_data, train_config, "Pure Quadratic Transformer"
    )
    
    results_1d['pure_quadratic'] = evaluate_transformer(model, params, test_data, "Pure Quadratic Transformer")
    results_1d['pure_quadratic']['training_time'] = history['training_time']
    
    # Summary
    print(f"\n📊 1D GAUSSIAN TRANSFORMER RESULTS")
    print("=" * 50)
    print(f"{'Model':<25} {'MSE':<12} {'vs Analytical':<15} {'Time(s)':<10}")
    print("-" * 65)
    
    for key, result in results_1d.items():
        ratio = result['mse'] / analytical_mse
        print(f"{result['name']:<25} {result['mse']:<12.6f} {ratio:<15.1f}x {result['training_time']:<10.1f}")
    
    return results_1d, analytical_mse


def test_transformers_3d():
    """Test Transformers on 3D Gaussian."""
    
    print(f"\n🧊 TESTING TRANSFORMERS ON 3D GAUSSIAN")
    print("=" * 60)
    
    # Load 3D data
    ef = MultivariateNormal(x_shape=(3,))
    train_data, val_data, test_data = load_3d_data()
    
    print(f"3D Dataset: {train_data['eta'].shape[0]} train, {test_data['eta'].shape[0]} test")
    
    results_3d = {}
    
    # 1. Standard Transformer
    print(f"\n1. Standard Transformer (3D)")
    rng = random.PRNGKey(45)
    
    standard_config = {
        'model_type': 'transformer',
        'num_layers': 3,  # Fewer layers for 3D
        'num_heads': 6,   # More heads for 12D
        'head_dim': 32,
        'mlp_dim': 256,
        'dropout_rate': 0.2,
        'activation': 'gelu'
    }
    
    model, params = create_transformer_train_state(ef, standard_config, rng)
    
    train_config = {
        'num_epochs': 80,
        'learning_rate': 5e-5,  # Lower LR for 3D
        'batch_size': 32
    }
    
    params, history = train_transformer(
        model, params, train_data, val_data, train_config, "Standard Transformer 3D"
    )
    
    results_3d['standard'] = evaluate_transformer(model, params, test_data, "Standard Transformer 3D")
    results_3d['standard']['training_time'] = history['training_time']
    
    # 2. Quadratic Transformer
    print(f"\n2. Quadratic Transformer (3D)")
    rng = random.PRNGKey(46)
    
    quad_config = {
        'model_type': 'quadratic_transformer',
        'num_layers': 3,
        'num_heads': 6,
        'head_dim': 32,
        'dropout_rate': 0.1,
        'activation': 'tanh',
        'use_adaptive_quadratic': True
    }
    
    model, params = create_transformer_train_state(ef, quad_config, rng)
    
    params, history = train_transformer(
        model, params, train_data, val_data, train_config, "Quadratic Transformer 3D"
    )
    
    results_3d['quadratic'] = evaluate_transformer(model, params, test_data, "Quadratic Transformer 3D")
    results_3d['quadratic']['training_time'] = history['training_time']
    
    # Summary
    print(f"\n📊 3D GAUSSIAN TRANSFORMER RESULTS")
    print("=" * 50)
    print(f"{'Model':<30} {'MSE':<12} {'Time(s)':<10}")
    print("-" * 55)
    
    for key, result in results_3d.items():
        print(f"{result['name']:<30} {result['mse']:<12.1f} {result['training_time']:<10.1f}")
    
    return results_3d


def create_transformer_plots(results_1d, results_3d, analytical_mse_1d):
    """Create comprehensive Transformer comparison plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1D Results
    names_1d = [r['name'] for r in results_1d.values()]
    mses_1d = [r['mse'] for r in results_1d.values()]
    colors = ['blue', 'red', 'green']
    
    bars = axes[0, 0].bar(range(len(names_1d)), mses_1d, color=colors, alpha=0.7)
    axes[0, 0].axhline(y=analytical_mse_1d, color='black', linestyle='--', label='Analytical')
    axes[0, 0].set_xticks(range(len(names_1d)))
    axes[0, 0].set_xticklabels([n.replace(' ', '\n') for n in names_1d], rotation=0)
    axes[0, 0].set_ylabel('MSE (log scale)')
    axes[0, 0].set_title('1D Gaussian: Transformer Comparison')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 3D Results
    names_3d = [r['name'] for r in results_3d.values()]
    mses_3d = [r['mse'] for r in results_3d.values()]
    
    bars = axes[0, 1].bar(range(len(names_3d)), mses_3d, color=colors[:len(names_3d)], alpha=0.7)
    axes[0, 1].set_xticks(range(len(names_3d)))
    axes[0, 1].set_xticklabels([n.replace(' ', '\n').replace('3D', '') for n in names_3d], rotation=0)
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].set_title('3D Gaussian: Transformer Comparison')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training time comparison
    times_1d = [r['training_time'] for r in results_1d.values()]
    times_3d = [r['training_time'] for r in results_3d.values()]
    
    x = np.arange(len(names_1d))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, times_1d, width, label='1D Gaussian', alpha=0.7)
    axes[1, 0].bar(x + width/2, times_3d[:len(times_1d)], width, label='3D Gaussian', alpha=0.7)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([n.split()[0] for n in names_1d])
    axes[1, 0].set_ylabel('Training Time (s)')
    axes[1, 0].set_title('Training Time Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary text
    axes[1, 1].axis('off')
    
    best_1d = min(results_1d.items(), key=lambda x: x[1]['mse'])
    best_3d = min(results_3d.items(), key=lambda x: x[1]['mse'])
    
    summary_text = "TRANSFORMER RESULTS\n"
    summary_text += "=" * 20 + "\n\n"
    summary_text += f"🥇 1D Winner: {best_1d[1]['name']}\n"
    summary_text += f"   MSE: {best_1d[1]['mse']:.6f}\n"
    summary_text += f"   vs Analytical: {best_1d[1]['mse']/analytical_mse_1d:.1f}x\n\n"
    
    summary_text += f"🥇 3D Winner: {best_3d[1]['name']}\n"
    summary_text += f"   MSE: {best_3d[1]['mse']:.1f}\n\n"
    
    summary_text += "KEY INSIGHTS:\n"
    summary_text += "• Self-attention captures\n  parameter dependencies\n"
    summary_text += "• Quadratic layers help with\n  division operations\n"
    summary_text += "• Hybrid approach combines\n  attention + polynomial approx\n"
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save plots
    output_dir = Path("artifacts/transformer_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "transformer_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "transformer_comparison.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"📊 Transformer plots saved to {output_dir}/")


def main():
    """Run comprehensive Transformer evaluation."""
    
    print("🤖 COMPREHENSIVE TRANSFORMER EVALUATION")
    print("=" * 80)
    print("Testing Standard, Quadratic, and Pure Quadratic Transformers")
    print("on both 1D and 3D Gaussian datasets")
    
    # Test on 1D
    results_1d, analytical_mse_1d = test_transformers_1d()
    
    # Test on 3D
    results_3d = test_transformers_3d()
    
    # Create visualizations
    create_transformer_plots(results_1d, results_3d, analytical_mse_1d)
    
    # Save results
    output_dir = Path("artifacts/transformer_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        '1d_results': {k: {
            'name': v['name'],
            'mse': v['mse'],
            'mae': v['mae'],
            'training_time': v['training_time']
        } for k, v in results_1d.items()},
        '3d_results': {k: {
            'name': v['name'],
            'mse': v['mse'],
            'mae': v['mae'],
            'training_time': v['training_time']
        } for k, v in results_3d.items()},
        'analytical_1d_mse': analytical_mse_1d
    }
    
    with open(output_dir / "transformer_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📁 All results saved to {output_dir}/")
    print("\n✅ Transformer evaluation completed!")


if __name__ == "__main__":
    main()
