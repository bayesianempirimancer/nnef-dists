#!/usr/bin/env python3
"""
Plotting script for log normalizer network predictions vs ground truth.

This script trains both log normalizer approaches and creates comprehensive
plots showing how well they predict expected sufficient statistics compared
to ground truth values.
"""

import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import NetworkConfig, TrainingConfig, FullConfig
from ef import MultivariateNormal
from models.log_normalizer import (
    LogNormalizerNetwork, compute_log_normalizer_gradient,
    prepare_log_normalizer_data
)
from models.quadratic_resnet_log_normalizer import QuadraticResNetLogNormalizer


def generate_3d_gaussian_data(n_samples=500, seed=42):
    """Generate synthetic 3D Gaussian data with known ground truth."""
    rng = random.PRNGKey(seed)
    
    # Generate natural parameters for 3D Gaussian
    # Format: [eta_mean_1, eta_mean_2, eta_mean_3, eta_cov_11, eta_cov_12, ..., eta_cov_33]
    
    # Create base natural parameters
    eta_base = jnp.array([
        1.0, 0.5, -0.5,  # Mean parameters
        -2.0, 0.5, -0.3,  # Covariance row 1
        -1.5, 0.2,        # Covariance row 2  
        -2.5              # Covariance row 3 (diagonal)
    ])
    
    # Generate variations around the base
    eta_variations = random.normal(rng, (n_samples, 8)) * 0.3
    eta_batch = eta_base[None, :] + eta_variations
    
    # Pad to 12 parameters (3 mean + 9 covariance)
    # For 3D Gaussian with full covariance, we need 12 parameters
    eta_full = jnp.zeros((n_samples, 12))
    eta_full = eta_full.at[:, :8].set(eta_batch)
    
    # Compute ground truth expected sufficient statistics
    ground_truth_means = []
    ground_truth_covs = []
    
    for i in range(n_samples):
        # Extract mean and covariance parts
        eta_mean = eta_full[i, :3]
        
        # Create symmetric covariance matrix from natural parameters
        # For simplicity, use a structured approach
        cov_params = eta_full[i, 3:9]  # Use first 6 covariance parameters
        
        # Create a simple covariance structure
        sigma = jnp.array([
            [-cov_params[0], cov_params[1], cov_params[2]],
            [cov_params[1], -cov_params[3], cov_params[4]], 
            [cov_params[2], cov_params[4], -cov_params[5]]
        ])
        
        try:
            # Ensure positive definiteness
            eigenvals, eigenvecs = jnp.linalg.eigh(sigma)
            eigenvals = jnp.maximum(eigenvals, 0.1)  # Ensure positive
            sigma = eigenvecs @ jnp.diag(eigenvals) @ eigenvecs.T
            
            # Compute mean: μ = -Σ⁻¹ η₁
            sigma_inv = jnp.linalg.inv(sigma)
            mean_x = -sigma_inv @ eta_mean
            
            # Expected sufficient statistics
            # For 3D Gaussian: [E[x₁], E[x₂], E[x₃], E[x₁²], E[x₁x₂], E[x₁x₃], E[x₂²], E[x₂x₃], E[x₃²]]
            expected_stats = jnp.concatenate([
                mean_x,  # [E[x₁], E[x₂], E[x₃]]
                jnp.array([
                    mean_x[0]**2 + sigma[0,0],  # E[x₁²]
                    mean_x[0]*mean_x[1] + sigma[0,1],  # E[x₁x₂] 
                    mean_x[0]*mean_x[2] + sigma[0,2],  # E[x₁x₃]
                    mean_x[1]**2 + sigma[1,1],  # E[x₂²]
                    mean_x[1]*mean_x[2] + sigma[1,2],  # E[x₂x₃]
                    mean_x[2]**2 + sigma[2,2]   # E[x₃²]
                ])
            ])
            
            ground_truth_means.append(expected_stats)
            ground_truth_covs.append(sigma)
            
        except:
            # Fallback for singular matrices
            ground_truth_means.append(jnp.zeros(9))
            ground_truth_covs.append(jnp.eye(3))
    
    ground_truth_means = jnp.array(ground_truth_means)
    ground_truth_covs = jnp.array(ground_truth_covs)
    
    return eta_full, ground_truth_means, ground_truth_covs


def train_simple_model(model, eta_data, ground_truth_means, epochs=50):
    """Simple training loop for log normalizer models."""
    from jax import value_and_grad
    import optax
    
    # Prepare data
    train_data = prepare_log_normalizer_data(eta_data, ground_truth_means)
    
    # Initialize model
    rng = random.PRNGKey(123)
    params = model.init(rng, eta_data[:1])
    
    # Create optimizer
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)
    
    # Simple loss function
    def loss_fn(params, eta_batch, target_batch):
        # Forward pass
        log_norm = model.apply(params, eta_batch, training=True)
        
        # Compute gradient (mean prediction)
        def single_log_normalizer(eta_single):
            return model.apply(params, eta_single[None, :], training=False)[0]
        
        grad_fn = jax.grad(single_log_normalizer)
        network_mean = jax.vmap(grad_fn)(eta_batch)
        
        # MSE loss
        return jnp.mean(jnp.square(network_mean - target_batch))
    
    # Training loop
    for epoch in range(epochs):
        # Compute loss and gradients
        loss, grads = value_and_grad(loss_fn)(params, eta_data, ground_truth_means)
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}")
    
    return params


def plot_predictions_vs_ground_truth(eta_data, ground_truth, basic_pred, resnet_pred, save_path=None):
    """Create comprehensive plots comparing predictions to ground truth."""
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Statistic names for 3D Gaussian
    stat_names = [
        'E[x₁]', 'E[x₂]', 'E[x₃]',  # Mean statistics
        'E[x₁²]', 'E[x₁x₂]', 'E[x₁x₃]',  # Covariance statistics
        'E[x₂²]', 'E[x₂x₃]', 'E[x₃²]'
    ]
    
    n_stats = len(stat_names)
    
    # 1. Scatter plots for each statistic
    for i in range(n_stats):
        row = i // 3
        col = i % 3
        
        ax = plt.subplot(4, 3, i + 1)
        
        # Scatter plots
        ax.scatter(ground_truth[:, i], basic_pred[:, i], 
                  alpha=0.6, label='Basic LogNormalizer', s=20)
        ax.scatter(ground_truth[:, i], resnet_pred[:, i], 
                  alpha=0.6, label='Quadratic ResNet', s=20)
        
        # Perfect prediction line
        min_val = min(ground_truth[:, i].min(), basic_pred[:, i].min(), resnet_pred[:, i].min())
        max_val = max(ground_truth[:, i].max(), basic_pred[:, i].max(), resnet_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect')
        
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{stat_names[i]}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 2. Error comparison bar plot
    ax = plt.subplot(4, 3, 10)
    
    basic_errors = jnp.abs(basic_pred - ground_truth)
    resnet_errors = jnp.abs(resnet_pred - ground_truth)
    
    mean_errors_basic = jnp.mean(basic_errors, axis=0)
    mean_errors_resnet = jnp.mean(resnet_errors, axis=0)
    
    x = np.arange(len(stat_names))
    width = 0.35
    
    ax.bar(x - width/2, mean_errors_basic, width, label='Basic LogNormalizer', alpha=0.8)
    ax.bar(x + width/2, mean_errors_resnet, width, label='Quadratic ResNet', alpha=0.8)
    
    ax.set_xlabel('Statistics')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Prediction Error Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(stat_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Overall performance metrics
    ax = plt.subplot(4, 3, 11)
    
    # Compute overall MSE for each approach
    basic_mse = jnp.mean(jnp.square(basic_pred - ground_truth))
    resnet_mse = jnp.mean(jnp.square(resnet_pred - ground_truth))
    
    basic_mae = jnp.mean(jnp.abs(basic_pred - ground_truth))
    resnet_mae = jnp.mean(jnp.abs(resnet_pred - ground_truth))
    
    metrics = ['MSE', 'MAE']
    basic_values = [float(basic_mse), float(basic_mae)]
    resnet_values = [float(resnet_mse), float(resnet_mae)]
    
    x = np.arange(len(metrics))
    ax.bar(x - width/2, basic_values, width, label='Basic LogNormalizer', alpha=0.8)
    ax.bar(x + width/2, resnet_values, width, label='Quadratic ResNet', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Error Value')
    ax.set_title('Overall Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Error distribution
    ax = plt.subplot(4, 3, 12)
    
    # Flatten all errors
    basic_errors_flat = jnp.abs(basic_pred - ground_truth).flatten()
    resnet_errors_flat = jnp.abs(resnet_pred - ground_truth).flatten()
    
    ax.hist(basic_errors_flat, bins=30, alpha=0.7, label='Basic LogNormalizer', density=True)
    ax.hist(resnet_errors_flat, bins=30, alpha=0.7, label='Quadratic ResNet', density=True)
    
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    return fig


def main():
    """Main function to run the comparison and create plots."""
    print("Log Normalizer Prediction vs Ground Truth Comparison")
    print("=" * 60)
    
    # Generate synthetic data
    print("Generating 3D Gaussian data...")
    eta_data, ground_truth_means, ground_truth_covs = generate_3d_gaussian_data(n_samples=300, seed=42)
    
    print(f"Data shapes:")
    print(f"  Natural parameters: {eta_data.shape}")
    print(f"  Ground truth means: {ground_truth_means.shape}")
    print(f"  Ground truth covs: {ground_truth_covs.shape}")
    
    # Create models
    print("\nCreating models...")
    
    # Basic LogNormalizer
    basic_config = NetworkConfig()
    basic_config.hidden_sizes = [64, 32]
    basic_config.output_dim = 1
    basic_config.use_feature_engineering = True
    basic_config.use_batch_norm = False
    basic_config.use_layer_norm = False
    basic_config.activation = "tanh"
    
    basic_model = LogNormalizerNetwork(config=basic_config)
    
    # Quadratic ResNet
    resnet_config = NetworkConfig()
    resnet_config.hidden_sizes = [48, 32]
    resnet_config.output_dim = 1
    resnet_config.use_feature_engineering = True
    resnet_config.use_batch_norm = False
    resnet_config.use_layer_norm = False
    resnet_config.activation = "tanh"
    
    resnet_model = QuadraticResNetLogNormalizer(config=resnet_config)
    
    # Train models
    print("\nTraining Basic LogNormalizer...")
    basic_params = train_simple_model(basic_model, eta_data, ground_truth_means, epochs=30)
    
    print("\nTraining Quadratic ResNet...")
    resnet_params = train_simple_model(resnet_model, eta_data, ground_truth_means, epochs=30)
    
    # Make predictions
    print("\nMaking predictions...")
    
    # Basic model predictions
    def basic_single_log_normalizer(eta_single):
        return basic_model.apply(basic_params, eta_single[None, :], training=False)[0]
    
    basic_grad_fn = jax.grad(basic_single_log_normalizer)
    basic_predictions = jax.vmap(basic_grad_fn)(eta_data)
    
    # ResNet predictions
    def resnet_single_log_normalizer(eta_single):
        return resnet_model.apply(resnet_params, eta_single[None, :], training=False)[0]
    
    resnet_grad_fn = jax.grad(resnet_single_log_normalizer)
    resnet_predictions = jax.vmap(resnet_grad_fn)(eta_data)
    
    print(f"Prediction shapes:")
    print(f"  Basic: {basic_predictions.shape}")
    print(f"  ResNet: {resnet_predictions.shape}")
    print(f"  Ground truth: {ground_truth_means.shape}")
    
    # Create plots
    print("\nCreating comparison plots...")
    
    # Ensure same number of statistics
    min_stats = min(basic_predictions.shape[1], resnet_predictions.shape[1], ground_truth_means.shape[1])
    basic_pred_trimmed = basic_predictions[:, :min_stats]
    resnet_pred_trimmed = resnet_predictions[:, :min_stats]
    ground_truth_trimmed = ground_truth_means[:, :min_stats]
    
    plot_predictions_vs_ground_truth(
        eta_data, ground_truth_trimmed, basic_pred_trimmed, resnet_pred_trimmed,
        save_path="log_normalizer_predictions_comparison.png"
    )
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    basic_mse = float(jnp.mean(jnp.square(basic_pred_trimmed - ground_truth_trimmed)))
    resnet_mse = float(jnp.mean(jnp.square(resnet_pred_trimmed - ground_truth_trimmed)))
    
    basic_mae = float(jnp.mean(jnp.abs(basic_pred_trimmed - ground_truth_trimmed)))
    resnet_mae = float(jnp.mean(jnp.abs(resnet_pred_trimmed - ground_truth_trimmed)))
    
    print(f"Basic LogNormalizer:")
    print(f"  MSE: {basic_mse:.6f}")
    print(f"  MAE: {basic_mae:.6f}")
    
    print(f"\nQuadratic ResNet:")
    print(f"  MSE: {resnet_mse:.6f}")
    print(f"  MAE: {resnet_mae:.6f}")
    
    improvement_mse = (basic_mse - resnet_mse) / basic_mse * 100
    improvement_mae = (basic_mae - resnet_mae) / basic_mae * 100
    
    print(f"\nImprovement:")
    print(f"  MSE: {improvement_mse:.1f}%")
    print(f"  MAE: {improvement_mae:.1f}%")
    
    if resnet_mse < basic_mse:
        print(f"\n🏆 Quadratic ResNet wins with {improvement_mse:.1f}% better MSE!")
    else:
        print(f"\n🏆 Basic LogNormalizer wins with {-improvement_mse:.1f}% better MSE!")
    
    print(f"\nPlots saved to: log_normalizer_predictions_comparison.png")


if __name__ == "__main__":
    main()
