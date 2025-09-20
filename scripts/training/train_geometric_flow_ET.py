#!/usr/bin/env python3
"""
Training script for Geometric Flow ET neural networks.

This script trains networks that learn flow dynamics to predict expected sufficient statistics
E[T(X)] using the geometric flow approach:
    du/dt = A@A^T@(η_target - η_init)

Usage:
    python scripts/training/train_geometric_flow_ET.py --config configs/multivariate_3d_large.yaml
    python scripts/training/train_geometric_flow_ET.py --config configs/multivariate_3d_large.yaml --plot-only
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import pickle
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import FullConfig
from src.ef import ef_factory
from src.models.geometric_flow_net import create_geometric_flow_et_network


def plot_geometric_flow_results(trainer, eta_data, ground_truth, predictions, history, config, save_dir):
    """Create comprehensive plots for geometric flow training results."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Training curves
    plt.subplot(3, 4, 1)
    if 'train_loss' in history:
        plt.plot(history['train_loss'], 'b-', linewidth=2, label='Train')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], 'r-', linewidth=2, label='Validation')
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Predictions vs Ground Truth (overall)
    plt.subplot(3, 4, 2)
    plt.scatter(ground_truth.flatten(), predictions.flatten(), alpha=0.6, s=15)
    min_val = min(ground_truth.min(), predictions.min())
    max_val = max(ground_truth.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title('Overall Predictions', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 3. Component-wise MSE
    plt.subplot(3, 4, 3)
    component_mse = np.mean((predictions - ground_truth) ** 2, axis=0)
    plt.bar(range(len(component_mse)), component_mse, alpha=0.7)
    plt.xlabel('Component Index')
    plt.ylabel('MSE')
    plt.title('Component-wise Errors', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # 4. Flow distances (if available)
    plt.subplot(3, 4, 4)
    if hasattr(trainer, 'last_flow_distances'):
        plt.hist(trainer.last_flow_distances, bins=20, alpha=0.7, color='orange')
        plt.xlabel('Flow Distance ||η₁ - η₀||')
        plt.ylabel('Frequency')
        plt.title('Flow Distance Distribution', fontsize=14, fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'Flow distances\nnot available', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=12)
        plt.title('Flow Distances', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 5-8. Component-wise predictions for linear terms (first 3)
    for i in range(min(3, ground_truth.shape[1])):
        plt.subplot(3, 4, 5 + i)
        plt.scatter(ground_truth[:, i], predictions[:, i], alpha=0.6, s=20)
        min_val = ground_truth[:, i].min()
        max_val = ground_truth[:, i].max()
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        plt.xlabel(f'True μ_{i+1}')
        plt.ylabel(f'Pred μ_{i+1}')
        plt.title(f'Linear Component {i+1}', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    # 9. Residuals analysis
    plt.subplot(3, 4, 9)
    residuals = predictions - ground_truth
    plt.hist(residuals.flatten(), bins=30, alpha=0.7, color='purple')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 10. Error vs flow distance (if available)
    plt.subplot(3, 4, 10)
    if hasattr(trainer, 'last_flow_distances'):
        prediction_errors = np.linalg.norm(predictions - ground_truth, axis=1)
        plt.scatter(trainer.last_flow_distances, prediction_errors, alpha=0.6, s=20)
        plt.xlabel('Flow Distance')
        plt.ylabel('Prediction Error')
        plt.title('Error vs Flow Distance', fontsize=12, fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'Flow distance\nanalysis\nnot available', ha='center', va='center',
                transform=plt.gca().transAxes, fontsize=10)
        plt.title('Error Analysis', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 11. Model architecture info
    plt.subplot(3, 4, 11)
    info_text = [
        f'Architecture: {trainer.architecture}',
        f'Matrix Rank: {trainer.matrix_rank}',
        f'Time Steps: {trainer.n_time_steps}',
        f'Smoothness Weight: {trainer.smoothness_weight}',
        f'Time Embed Dim: {trainer.model.time_embed_dim}',
        f'Max Frequency: {trainer.model.max_freq}',
        f'Final MSE: {np.mean((predictions - ground_truth) ** 2):.2e}',
        f'Final MAE: {np.mean(np.abs(predictions - ground_truth)):.2e}'
    ]
    
    for i, text in enumerate(info_text):
        plt.text(0.05, 0.9 - i*0.1, text, fontsize=11, transform=plt.gca().transAxes)
    plt.title('Model Configuration', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # 12. Time embedding visualization
    plt.subplot(3, 4, 12)
    # Show time embedding for a few time points
    t_points = jnp.linspace(0, 1, 11)
    embeddings = []
    for t in t_points:
        # Create dummy network instance to get embedding
        embed = trainer.model._time_embedding(float(t))
        embeddings.append(embed)
    
    embeddings = jnp.array(embeddings)
    plt.imshow(embeddings.T, aspect='auto', cmap='RdBu', interpolation='bilinear')
    plt.xlabel('Time Point')
    plt.ylabel('Embedding Dimension')
    plt.title('Time Embeddings', fontsize=12, fontweight='bold')
    plt.colorbar()
    
    plt.suptitle(f'Geometric Flow ET Network - 3D Multivariate Gaussian', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plots
    save_path = Path(save_dir) / 'geometric_flow_et_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Results plot saved to: {save_path}")
    
    plt.show()


def generate_3d_gaussian_data(n_samples: int = 1000, seed: int = 42):
    """Generate 3D multivariate Gaussian training data."""
    print(f"Generating {n_samples} 3D Gaussian samples...")
    
    rng = random.PRNGKey(seed)
    ef = ef_factory("multivariate_normal", x_shape=(3,))
    d = 3
    
    eta_samples = []
    mu_samples = []
    
    for i in range(n_samples):
        rng, subkey = random.split(rng)
        
        # Random linear part
        eta1 = random.normal(subkey, (d,)) * 0.4
        
        # Random negative definite quadratic part (direct construction, no eigenvalues!)
        rng, subkey = random.split(rng)
        
        # Method: A^T A is always positive semidefinite, so -A^T A - I is negative definite
        A = random.normal(subkey, (d, d)) * 0.4
        eta2 = -jnp.dot(A.T, A) - (0.5 + 0.5 * random.uniform(subkey)) * jnp.eye(d)
        
        # This is guaranteed to be negative definite and real (no eigenvalue computation needed!)
        
        # Compute true μ (ensure real values)
        Sigma = jnp.real(-0.5 * jnp.linalg.inv(eta2))
        mu = jnp.real(jnp.linalg.solve(eta2, -0.5 * eta1))
        E_xx = jnp.real(Sigma + jnp.outer(mu, mu))
        
        # Flatten
        eta_flat = ef.flatten_stats_or_eta({'x': eta1, 'xxT': eta2})
        mu_flat = ef.flatten_stats_or_eta({'x': mu, 'xxT': E_xx})
        
        eta_samples.append(eta_flat)
        mu_samples.append(mu_flat)
        
        if (i + 1) % 200 == 0:
            print(f"  Generated {i+1}/{n_samples}")
    
    return jnp.array(eta_samples), jnp.array(mu_samples)


def create_simple_config():
    """Create a simple configuration for geometric flow training."""
    from src.config import NetworkConfig, TrainingConfig
    
    network_config = NetworkConfig(
        hidden_sizes=[128, 64, 32],
        use_layer_norm=True,
        dropout_rate=0.0,
        output_dim=12  # For 3D Gaussian
    )
    
    training_config = TrainingConfig(
        num_epochs=150,
        learning_rate=1e-3,
        batch_size=16
    )
    
    return FullConfig(network=network_config, training=training_config)


def train_geometric_flow_et(save_dir: str, plot_only: bool = False):
    """Train geometric flow ET network on 3D Gaussian data."""
    
    # Create configuration
    config = create_simple_config()
    print(f"Using simple configuration for geometric flow training")
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate or load data (store in data directory, not artifacts)
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    data_file = data_dir / "geometric_flow_training_data.pkl"
    
    if plot_only and data_file.exists():
        print("Loading existing training data for plotting...")
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        eta_train, mu_train = data['eta_train'], data['mu_train']
        eta_val, mu_val = data['eta_val'], data['mu_val']
    else:
        print("Generating new training data...")
        # Generate training data
        eta_train, mu_train = generate_3d_gaussian_data(n_samples=800, seed=42)
        eta_val, mu_val = generate_3d_gaussian_data(n_samples=200, seed=123)
        
        # Save data to data directory
        data = {
            'eta_train': eta_train,
            'mu_train': mu_train,
            'eta_val': eta_val,
            'mu_val': mu_val
        }
        with open(data_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Training data saved to: {data_file} (in data/ directory)")
    
    print(f"Training data: {eta_train.shape[0]} samples")
    print(f"Validation data: {eta_val.shape[0]} samples")
    print(f"η dimension: {eta_train.shape[1]}")
    print(f"μ dimension: {mu_train.shape[1]}")
    
    # Create exponential family instance
    ef = ef_factory("multivariate_normal", x_shape=(3,))
    
    # Create and configure trainer
    trainer = create_geometric_flow_et_network(
        config=config,
        architecture="mlp",
        matrix_rank=8,  # Reduced rank for efficiency
        n_time_steps=3,  # Minimal due to smoothness
        smoothness_weight=1e-3
    )
    
    # Set exponential family for analytical point computation
    trainer.set_exponential_family(ef)
    
    print(f"Geometric Flow ET Network Configuration:")
    print(f"  Architecture: {trainer.architecture}")
    print(f"  Matrix rank: {trainer.matrix_rank}")
    print(f"  Time steps: {trainer.n_time_steps}")
    print(f"  Smoothness weight: {trainer.smoothness_weight}")
    print(f"  Time embedding dim: {trainer.model.time_embed_dim}")
    print(f"  Max frequency: {trainer.model.max_freq}")
    
    if plot_only:
        # Load existing model and create plots
        model_file = save_dir / "geometric_flow_et_params.pkl"
        history_file = save_dir / "geometric_flow_et_history.pkl"
        
        if model_file.exists() and history_file.exists():
            print("Loading existing model for plotting...")
            with open(model_file, 'rb') as f:
                params = pickle.load(f)
            with open(history_file, 'rb') as f:
                history = pickle.load(f)
            
            # Make predictions
            predictions_dict = trainer.predict(params, eta_val)
            predictions = predictions_dict['mu_predicted']
            
            # Store flow distances for plotting
            trainer.last_flow_distances = predictions_dict['flow_distances']
            
            # Create plots
            plot_geometric_flow_results(trainer, eta_val, mu_val, predictions, history, config, save_dir)
        else:
            print("No existing model found. Run without --plot-only first.")
        return
    
    # Training
    print(f"\nStarting Geometric Flow ET training...")
    start_time = time.time()
    
    params, history = trainer.train(
        eta_targets_train=eta_train,
        eta_targets_val=eta_val,
        epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate,
        batch_size=config.training.batch_size
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f}s")
    
    # Save model and history
    model_file = save_dir / "geometric_flow_et_params.pkl"
    history_file = save_dir / "geometric_flow_et_history.pkl"
    
    with open(model_file, 'wb') as f:
        pickle.dump(params, f)
    with open(history_file, 'wb') as f:
        pickle.dump(history, f)
    
    print(f"Model saved to: {model_file}")
    print(f"History saved to: {history_file}")
    
    # Evaluation
    print(f"\nEvaluating on validation set...")
    results = trainer.evaluate(params, eta_val)
    predictions_dict = trainer.predict(params, eta_val)
    predictions = predictions_dict['mu_predicted']
    
    # Store flow distances for plotting
    trainer.last_flow_distances = predictions_dict['flow_distances']
    
    print(f"Validation Results:")
    print(f"  MSE: {results['mse']:.8f}")
    print(f"  MAE: {results['mae']:.8f}")
    print(f"  Mean flow distance: {results['mean_flow_distance']:.6f}")
    
    # Component analysis
    print(f"  Component errors (linear terms):")
    for i in range(min(3, len(results['component_errors']))):
        print(f"    μ_{i+1}: {results['component_errors'][i]:.8f}")
    
    print(f"  Component errors (quadratic terms, first 3):")
    for i in range(3, min(6, len(results['component_errors']))):
        quad_i, quad_j = divmod(i-3, 3)
        print(f"    μ_{i+1} (x_{quad_i}x_{quad_j}): {results['component_errors'][i]:.8f}")
    
    # Create plots
    plot_geometric_flow_results(trainer, eta_val, mu_val, predictions, history, config, save_dir)
    
    # Save evaluation results
    eval_file = save_dir / "geometric_flow_et_evaluation.pkl"
    eval_results = {
        'results': results,
        'predictions': predictions,
        'ground_truth': mu_val,
        'eta_data': eta_val,
        'training_time': training_time
    }
    
    with open(eval_file, 'wb') as f:
        pickle.dump(eval_results, f)
    
    print(f"Evaluation results saved to: {eval_file}")
    
    # Summary
    print(f"\n" + "="*60)
    print("GEOMETRIC FLOW ET NETWORK SUMMARY")
    print("="*60)
    print(f"✓ Training completed in {training_time:.1f}s")
    print(f"✓ Final MSE: {results['mse']:.2e}")
    print(f"✓ Final MAE: {results['mae']:.2e}")
    print(f"✓ Mean flow distance: {results['mean_flow_distance']:.4f}")
    print(f"✓ Used {trainer.n_time_steps} time steps with sinusoidal embeddings")
    
    if results['mse'] < 1e-4:
        print("🎉 EXCELLENT: Geometric flow learning highly successful!")
    elif results['mse'] < 1e-2:
        print("✅ GOOD: Reasonable geometric flow performance")
    else:
        print("⚠️ NEEDS WORK: Consider tuning hyperparameters")
    
    return params, history, results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Geometric Flow ET Network')
    parser.add_argument('--save-dir', default='artifacts/geometric_flow_et', 
                       help='Directory to save results')
    parser.add_argument('--plot-only', action='store_true', 
                       help='Only generate plots from existing results')
    
    args = parser.parse_args()
    
    print("Geometric Flow ET Network Training")
    print("="*45)
    print(f"Save directory: {args.save_dir}")
    print(f"Plot only: {args.plot_only}")
    
    try:
        results = train_geometric_flow_et(args.save_dir, args.plot_only)
        print(f"\n✓ Geometric Flow ET training completed successfully!")
        return 0
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
