#!/usr/bin/env python3
"""
Test the best neural network architectures on the 3D Gaussian dataset.

This script evaluates our top-performing models from the 1D case on the much more
challenging 3D Gaussian problem with 12-dimensional natural parameters.

The 3D Gaussian has natural parameters eta of shape (12,) representing:
- eta[0:3]: Linear terms (mean-related)  
- eta[3:12]: Quadratic terms (covariance-related, symmetric matrix)

Expected sufficient statistics are also 12-dimensional.
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
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ef import MultivariateNormal
from src.model import nat2statMLP
from src.glu_moment_net import GLUMomentNet, DeepGLUMomentNet, create_glu_train_state
from src.quadratic_resnet import QuadraticResNet, DeepAdaptiveQuadraticResNet, create_quadratic_train_state
from src.noprop_ct import NoPropCTMomentNet, NoPropCTConfig, train_noprop_ct_moment_net


def load_3d_gaussian_data():
    """Load the largest 3D Gaussian dataset."""
    
    # Find the largest 3D dataset (12-dimensional natural parameters)
    data_dir = Path("data")
    suitable_files = []
    
    for data_file in data_dir.glob("*.pkl"):
        try:
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            
            if data["train_eta"].shape[1] == 12:  # 3D Gaussian
                suitable_files.append((data_file, data["train_eta"].shape[0]))
        except Exception:
            continue
    
    if not suitable_files:
        raise FileNotFoundError("No 3D Gaussian datasets found!")
    
    # Choose the largest dataset
    best_file, n_samples = max(suitable_files, key=lambda x: x[1])
    
    print(f"Loading 3D Gaussian data from {best_file.name}")
    print(f"Dataset size: {n_samples} training samples")
    
    with open(best_file, 'rb') as f:
        data = pickle.load(f)
    
    # Create train/val/test splits
    train_data = {
        "eta": data["train_eta"],
        "y": data["train_y"]
    }
    
    val_data = {
        "eta": data["val_eta"], 
        "y": data["val_y"]
    }
    
    # Create test split from validation data
    n_val = val_data["eta"].shape[0]
    n_test = min(n_val // 2, 250)
    
    test_data = {
        "eta": val_data["eta"][:n_test],
        "y": val_data["y"][:n_test]
    }
    
    # Keep remaining as validation
    val_data = {
        "eta": val_data["eta"][n_test:],
        "y": val_data["y"][n_test:]
    }
    
    return train_data, val_data, test_data


def analytical_solution_3d(eta):
    """
    Analytical solution for 3D Gaussian natural parameters.
    
    For a 3D multivariate normal with natural parameterization:
    eta = [eta_linear, eta_quadratic] where eta_quadratic is symmetric
    
    This is much more complex than 1D case involving matrix operations.
    """
    batch_size = eta.shape[0]
    
    # Extract linear and quadratic parts
    eta_linear = eta[:, :3]  # Shape: (batch, 3)
    eta_quad_vec = eta[:, 3:]  # Shape: (batch, 9) - vectorized symmetric matrix
    
    # Reconstruct symmetric matrices from vectorized form
    # For 3x3 symmetric matrix: [a, b, c, b, d, e, c, e, f]
    eta_matrices = jnp.zeros((batch_size, 3, 3))
    
    # Fill symmetric matrix (assuming standard vectorization)
    eta_matrices = eta_matrices.at[:, 0, 0].set(eta_quad_vec[:, 0])  # a
    eta_matrices = eta_matrices.at[:, 0, 1].set(eta_quad_vec[:, 1])  # b  
    eta_matrices = eta_matrices.at[:, 0, 2].set(eta_quad_vec[:, 2])  # c
    eta_matrices = eta_matrices.at[:, 1, 0].set(eta_quad_vec[:, 1])  # b (symmetric)
    eta_matrices = eta_matrices.at[:, 1, 1].set(eta_quad_vec[:, 3])  # d
    eta_matrices = eta_matrices.at[:, 1, 2].set(eta_quad_vec[:, 4])  # e
    eta_matrices = eta_matrices.at[:, 2, 0].set(eta_quad_vec[:, 2])  # c (symmetric)  
    eta_matrices = eta_matrices.at[:, 2, 1].set(eta_quad_vec[:, 4])  # e (symmetric)
    eta_matrices = eta_matrices.at[:, 2, 2].set(eta_quad_vec[:, 5])  # f
    
    # For multivariate normal: eta_quad = -0.5 * Sigma^{-1}, eta_linear = Sigma^{-1} * mu
    # So: Sigma^{-1} = -2 * eta_quad, mu = -0.5 * eta_linear / eta_quad
    
    try:
        # Compute precision matrix (inverse covariance)
        precision = -2.0 * eta_matrices
        
        # Compute covariance matrix  
        covariance = jnp.linalg.inv(precision + 1e-6 * jnp.eye(3))  # Add regularization
        
        # Compute mean
        mean = jnp.einsum('bij,bj->bi', covariance, eta_linear)
        
        # Compute expected sufficient statistics
        # E[x] = mean
        # E[x x^T] = covariance + mean * mean^T (but vectorized)
        
        mean_outer = jnp.einsum('bi,bj->bij', mean, mean)
        second_moment = covariance + mean_outer
        
        # Vectorize the second moment matrix (same order as input)
        second_moment_vec = jnp.stack([
            second_moment[:, 0, 0],  # (0,0)
            second_moment[:, 0, 1],  # (0,1) 
            second_moment[:, 0, 2],  # (0,2)
            second_moment[:, 1, 1],  # (1,1)
            second_moment[:, 1, 2],  # (1,2)
            second_moment[:, 2, 2],  # (2,2)
        ], axis=-1)
        
        # Combine mean and second moments
        result = jnp.concatenate([mean, second_moment_vec], axis=-1)
        
        return result
        
    except:
        # If matrix operations fail, return zeros (will have high error)
        return jnp.zeros((batch_size, 12))


def train_model_3d(model, params, train_data, val_data, config, name="Model"):
    """Train a model on 3D data with appropriate scaling."""
    
    num_epochs = config.get('num_epochs', 150)  # Fewer epochs for 3D
    learning_rate = config.get('learning_rate', 5e-4)  # Lower LR for stability
    batch_size = config.get('batch_size', 32)  # Smaller batches for 3D
    
    # Create optimizer with gradient clipping (important for 3D)
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),  # Aggressive clipping for 3D
        optax.adam(learning_rate)
    )
    opt_state = optimizer.init(params)
    
    train_losses = []
    val_losses = []
    best_params = params
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 30  # Shorter patience for 3D
    
    print(f"  Training {name} for up to {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training with mini-batches
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
                pred = model.apply(params, eta_batch)
                return jnp.mean(jnp.square(pred - y_batch))
            
            loss, grads = jax.value_and_grad(loss_fn)(params)
            
            # Check for NaN gradients
            if jnp.any(jnp.isnan(jnp.concatenate([jnp.ravel(g) for g in jax.tree_leaves(grads)]))):
                print(f"    NaN gradients detected at epoch {epoch}, batch {i//batch_size}")
                break
                
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            epoch_train_loss += float(loss)
            num_batches += 1
        
        if num_batches == 0:
            break
            
        epoch_train_loss /= num_batches
        
        # Validation
        val_pred = model.apply(params, val_data['eta'])
        val_loss = float(jnp.mean(jnp.square(val_pred - val_data['y'])))
        
        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        
        # Early stopping and best model tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 20 == 0:
            print(f"    Epoch {epoch:3d}: Train={epoch_train_loss:.6f}, Val={val_loss:.6f}, Best={best_val_loss:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch} (patience={patience})")
            break
    
    training_time = time.time() - start_time
    print(f"  Training completed in {training_time:.1f}s, Best val loss: {best_val_loss:.6f}")
    
    return best_params, {
        'train_losses': train_losses, 
        'val_losses': val_losses, 
        'training_time': training_time,
        'best_val_loss': best_val_loss,
        'epochs_trained': len(train_losses)
    }


def evaluate_model_3d(model, params, test_data, name="Model"):
    """Evaluate model on 3D test data."""
    
    pred = model.apply(params, test_data['eta'])
    
    # Basic metrics
    mse = float(jnp.mean(jnp.square(pred - test_data['y'])))
    mae = float(jnp.mean(jnp.abs(pred - test_data['y'])))
    rmse = float(jnp.sqrt(mse))
    
    # Component-wise metrics (first 3 are means, last 9 are second moments)
    component_mse = jnp.mean(jnp.square(pred - test_data['y']), axis=0)
    mean_mse = jnp.mean(component_mse[:3])  # Mean components
    cov_mse = jnp.mean(component_mse[3:])   # Covariance components
    
    return {
        'name': name,
        'predictions': pred,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mean_mse': float(mean_mse),
        'cov_mse': float(cov_mse),
        'component_mse': component_mse.tolist()
    }


def test_3d_gaussian():
    """Test top architectures on 3D Gaussian dataset."""
    
    print("🧊 TESTING ON 3D GAUSSIAN DATASET")
    print("=" * 60)
    print("Challenge: 12-dimensional natural parameters → 12-dimensional sufficient statistics")
    print("Much more complex division operations involving matrix inversions!")
    
    # Load 3D data
    train_data, val_data, test_data = load_3d_gaussian_data()
    
    print(f"\nDataset sizes:")
    print(f"  Training: {train_data['eta'].shape[0]} samples")
    print(f"  Validation: {val_data['eta'].shape[0]} samples") 
    print(f"  Test: {test_data['eta'].shape[0]} samples")
    print(f"  Input dimension: {train_data['eta'].shape[1]}")
    print(f"  Output dimension: {train_data['y'].shape[1]}")
    
    # Create exponential family object
    ef = MultivariateNormal(x_shape=(3,))
    
    results = {}
    
    # 1. Analytical Solution (if possible)
    print(f"\n{'='*60}")
    print("1. ANALYTICAL SOLUTION (Ground Truth)")
    print(f"{'='*60}")
    
    try:
        analytical_pred = analytical_solution_3d(test_data['eta'])
        analytical_mse = float(jnp.mean(jnp.square(analytical_pred - test_data['y'])))
        analytical_mae = float(jnp.mean(jnp.abs(analytical_pred - test_data['y'])))
        
        print(f"  MSE: {analytical_mse:.8f}")
        print(f"  MAE: {analytical_mae:.8f}")
        
        results['analytical'] = {
            'name': 'Analytical Solution',
            'mse': analytical_mse,
            'mae': analytical_mae,
            'training_time': 0.0
        }
    except Exception as e:
        print(f"  ❌ Analytical solution failed: {e}")
        analytical_mse = float('inf')
        results['analytical'] = None
    
    # 2. Standard MLP (baseline)
    print(f"\n{'='*60}")
    print("2. STANDARD MLP (Baseline)")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(42)
    standard_mlp = nat2statMLP(
        hidden_sizes=[256, 128, 64], 
        activation='tanh', 
        output_dim=12
    )
    standard_params = standard_mlp.init(rng, test_data['eta'][:1])
    
    standard_config = {
        'num_epochs': 150,
        'learning_rate': 5e-4,
        'batch_size': 32
    }
    
    standard_params, standard_history = train_model_3d(
        standard_mlp, standard_params, train_data, val_data, 
        standard_config, "Standard MLP"
    )
    
    results['standard_mlp'] = evaluate_model_3d(
        standard_mlp, standard_params, test_data, "Standard MLP"
    )
    results['standard_mlp']['training_time'] = standard_history['training_time']
    
    # 3. Deep GLU (best from GLU family)
    print(f"\n{'='*60}")
    print("3. DEEP GLU WITH RESIDUAL CONNECTIONS")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(43)
    deep_glu_config = {
        'model_type': 'deep_glu',
        'hidden_size': 256,
        'num_glu_layers': 4,  # Fewer layers for 3D
        'activation': 'tanh'
    }
    deep_glu_model, deep_glu_params = create_glu_train_state(ef, deep_glu_config, rng)
    
    deep_glu_train_config = {
        'num_epochs': 150,
        'learning_rate': 3e-4,
        'batch_size': 32
    }
    
    deep_glu_params, deep_glu_history = train_model_3d(
        deep_glu_model, deep_glu_params, train_data, val_data,
        deep_glu_train_config, "Deep GLU"
    )
    
    results['deep_glu'] = evaluate_model_3d(
        deep_glu_model, deep_glu_params, test_data, "Deep GLU"
    )
    results['deep_glu']['training_time'] = deep_glu_history['training_time']
    
    # 4. Quadratic ResNet (winner from 1D)
    print(f"\n{'='*60}")
    print("4. QUADRATIC RESNET")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(44)
    quad_config = {
        'model_type': 'quadratic',
        'hidden_size': 256,
        'num_layers': 6,  # Fewer layers for 3D stability
        'activation': 'tanh',
        'use_activation_between_layers': True
    }
    quad_model, quad_params = create_quadratic_train_state(ef, quad_config, rng)
    
    quad_train_config = {
        'num_epochs': 150,
        'learning_rate': 3e-4,
        'batch_size': 32
    }
    
    quad_params, quad_history = train_model_3d(
        quad_model, quad_params, train_data, val_data,
        quad_train_config, "Quadratic ResNet"
    )
    
    results['quadratic'] = evaluate_model_3d(
        quad_model, quad_params, test_data, "Quadratic ResNet"
    )
    results['quadratic']['training_time'] = quad_history['training_time']
    
    # 5. Adaptive Quadratic ResNet (champion from 1D)
    print(f"\n{'='*60}")
    print("5. ADAPTIVE QUADRATIC RESNET (1D Champion)")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(45)
    adaptive_config = {
        'model_type': 'adaptive_quadratic',
        'hidden_size': 256,
        'num_layers': 6,  # Conservative for 3D
        'activation': 'tanh'
    }
    adaptive_model, adaptive_params = create_quadratic_train_state(ef, adaptive_config, rng)
    
    adaptive_train_config = {
        'num_epochs': 150,
        'learning_rate': 2e-4,  # Even lower LR
        'batch_size': 32
    }
    
    adaptive_params, adaptive_history = train_model_3d(
        adaptive_model, adaptive_params, train_data, val_data,
        adaptive_train_config, "Adaptive Quadratic ResNet"
    )
    
    results['adaptive_quadratic'] = evaluate_model_3d(
        adaptive_model, adaptive_params, test_data, "Adaptive Quadratic ResNet"
    )
    results['adaptive_quadratic']['training_time'] = adaptive_history['training_time']
    
    # Generate summary
    print(f"\n{'='*60}")
    print("3D GAUSSIAN RESULTS SUMMARY")
    print(f"{'='*60}")
    
    # Sort by MSE (exclude analytical if failed)
    neural_results = [(k, v) for k, v in results.items() if k != 'analytical' and v is not None]
    neural_results.sort(key=lambda x: x[1]['mse'])
    
    print(f"{'Rank':<4} {'Architecture':<25} {'MSE':<12} {'Mean MSE':<12} {'Cov MSE':<12} {'Time(s)':<10}")
    print("-" * 85)
    
    for rank, (key, result) in enumerate(neural_results, 1):
        print(f"{rank:<4} {result['name']:<25} {result['mse']:<12.6f} "
              f"{result.get('mean_mse', 0):<12.6f} {result.get('cov_mse', 0):<12.6f} "
              f"{result['training_time']:<10.1f}")
    
    if results['analytical'] is not None:
        print(f"\nAnalytical baseline: MSE = {results['analytical']['mse']:.6f}")
        best_neural = neural_results[0]
        gap = best_neural[1]['mse'] / results['analytical']['mse']
        print(f"Best neural network gap: {gap:.1f}x worse than analytical")
    
    # Key insights for 3D
    print(f"\n🎯 3D GAUSSIAN INSIGHTS:")
    
    best_model = neural_results[0]
    print(f"🥇 Best 3D model: {best_model[1]['name']}")
    print(f"   MSE: {best_model[1]['mse']:.6f}")
    
    # Compare with 1D performance
    print(f"\n📊 1D vs 3D Performance Comparison:")
    print(f"   • Problem complexity: 2D → 12D natural parameters (6x increase)")
    print(f"   • Expected difficulty: Much higher due to matrix operations")
    
    # Check if quadratic ResNets still dominate
    quad_models = [k for k in results.keys() if 'quadratic' in k and results[k] is not None]
    if quad_models:
        quad_mses = [results[k]['mse'] for k in quad_models]
        quad_avg = np.mean(quad_mses)
        standard_mse = results.get('standard_mlp', {}).get('mse', float('inf'))
        
        if quad_avg < standard_mse:
            improvement = standard_mse / quad_avg
            print(f"   • Quadratic ResNets still dominate: {improvement:.1f}x better than standard MLP")
        else:
            print(f"   • 3D complexity challenges quadratic ResNets")
    
    # Save results
    output_dir = Path("artifacts/3d_gaussian_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualization
    create_3d_plots(results, neural_results, output_dir)
    
    # Save JSON results
    results_for_json = {}
    for model_name, result in results.items():
        if result is not None:
            results_for_json[model_name] = {
                'name': result['name'],
                'mse': result['mse'],
                'mae': result.get('mae', 0),
                'training_time': result.get('training_time', 0),
                'mean_mse': result.get('mean_mse', 0),
                'cov_mse': result.get('cov_mse', 0)
            }
    
    with open(output_dir / "3d_results.json", 'w') as f:
        json.dump(results_for_json, f, indent=2)
    
    print(f"\n📁 3D results saved to {output_dir}/")
    
    return results


def create_3d_plots(results, neural_results, output_dir):
    """Create visualization for 3D results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Performance comparison
    names = [result[1]['name'] for result in neural_results]
    mses = [result[1]['mse'] for result in neural_results]
    colors = ['blue', 'green', 'red', 'purple', 'orange'][:len(names)]
    
    bars = axes[0, 0].bar(range(len(names)), mses, color=colors, alpha=0.7)
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Test MSE (log scale)')
    axes[0, 0].set_title('3D Gaussian Performance Comparison')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Mean vs Covariance MSE
    mean_mses = [result[1].get('mean_mse', 0) for result in neural_results]
    cov_mses = [result[1].get('cov_mse', 0) for result in neural_results]
    
    x = np.arange(len(names))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, mean_mses, width, label='Mean MSE', alpha=0.7)
    axes[0, 1].bar(x + width/2, cov_mses, width, label='Covariance MSE', alpha=0.7)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Component MSE')
    axes[0, 1].set_title('Mean vs Covariance Prediction Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Training time vs performance
    times = [result[1]['training_time'] for result in neural_results]
    axes[1, 0].scatter(times, mses, c=colors, s=100, alpha=0.7)
    for i, name in enumerate(names):
        axes[1, 0].annotate(name, (times[i], mses[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1, 0].set_xlabel('Training Time (s)')
    axes[1, 0].set_ylabel('Test MSE (log scale)')
    axes[1, 0].set_title('Efficiency vs Performance')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Summary text
    axes[1, 1].axis('off')
    
    best_model = neural_results[0]
    summary_text = f"3D GAUSSIAN RESULTS\n"
    summary_text += f"{'='*25}\n\n"
    summary_text += f"🥇 WINNER: {best_model[1]['name']}\n"
    summary_text += f"MSE: {best_model[1]['mse']:.6f}\n"
    summary_text += f"Training time: {best_model[1]['training_time']:.1f}s\n\n"
    
    summary_text += f"📊 CHALLENGE SCALING:\n"
    summary_text += f"• 1D: 2 → 2 dimensions\n"
    summary_text += f"• 3D: 12 → 12 dimensions\n"
    summary_text += f"• 6x parameter increase\n"
    summary_text += f"• Matrix operations required\n\n"
    
    summary_text += f"🏆 TOP 3:\n"
    for i, (k, result) in enumerate(neural_results[:3]):
        summary_text += f"{i+1}. {result['name']}\n"
        summary_text += f"   MSE: {result['mse']:.4f}\n"
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_dir / "3d_gaussian_results.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "3d_gaussian_results.pdf", bbox_inches='tight')
    plt.close()
    
    print("  📊 3D visualization plots generated!")


if __name__ == "__main__":
    print("🧊 Starting 3D Gaussian evaluation...")
    print("Testing our best architectures on the challenging 3D case!")
    print()
    
    try:
        results = test_3d_gaussian()
        print("\n✅ 3D Gaussian evaluation completed!")
    except Exception as e:
        print(f"\n❌ Error during 3D evaluation: {e}")
        raise
