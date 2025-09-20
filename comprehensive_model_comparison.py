#!/usr/bin/env python3
"""
Comprehensive Model Comparison Script

This script tests all available models (both ET and logZ variants) and creates
a unified performance comparison figure showing training curves, predictions vs
ground truth, and performance metrics for all models.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, hessian, jacfwd, jacrev
import matplotlib.pyplot as plt
import flax.linen as nn
import optax
import time
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent / "src"))

def generate_simple_test_data(n_samples=800, seed=42):
    """Generate simple 3D Gaussian test data."""
    eta_vectors = []
    expected_stats = []
    
    for i in range(n_samples):
        # Generate random mean and covariance
        mean = random.normal(random.PRNGKey(seed + i), (3,)) * 1.0
        A = random.normal(random.PRNGKey(seed + i + 1000), (3, 3))
        covariance_matrix = A.T @ A + jnp.eye(3) * 0.01
        
        # Convert to natural parameters
        sigma_inv = jnp.linalg.inv(covariance_matrix)
        eta1 = sigma_inv @ mean  # η₁ = Σ⁻¹μ
        eta2_matrix = -0.5 * sigma_inv  # η₂ = -0.5Σ⁻¹
        eta_vector = jnp.concatenate([eta1, eta2_matrix.flatten()])
        eta_vectors.append(eta_vector)
        
        # Expected sufficient statistics
        expected_stat = jnp.concatenate([
            mean,  # μ (3 values)
            (jnp.outer(mean, mean) + covariance_matrix).flatten()  # μμᵀ + Σ (9 values)
        ])
        expected_stats.append(expected_stat)
    
    return jnp.array(eta_vectors), jnp.array(expected_stats)


class SimpleMLP_ET(nn.Module):
    """Simple MLP that directly predicts expected sufficient statistics (ET)."""
    hidden_sizes: list
    
    @nn.compact
    def __call__(self, x, training=True):
        for i, hidden_size in enumerate(self.hidden_sizes):
            x = nn.Dense(hidden_size, name=f'hidden_{i}')(x)
            x = nn.swish(x)
            if i < len(self.hidden_sizes) - 1:  # No activation on last layer
                x = nn.LayerNorm(name=f'layer_norm_{i}')(x)
        
        # Output layer - direct prediction of 12 sufficient statistics
        x = nn.Dense(12, name='output')(x)
        return x


class SimpleMLP_LogZ(nn.Module):
    """Simple MLP that outputs scalar log normalizer (logZ)."""
    hidden_sizes: list
    
    @nn.compact
    def __call__(self, x, training=True):
        for i, hidden_size in enumerate(self.hidden_sizes):
            x = nn.Dense(hidden_size, name=f'hidden_{i}')(x)
            x = nn.swish(x)
            if i < len(self.hidden_sizes) - 1:  # No activation on last layer
                x = nn.LayerNorm(name=f'layer_norm_{i}')(x)
        
        # Output layer - scalar log normalizer
        x = nn.Dense(1, name='output')(x)
        return jnp.squeeze(x, axis=-1)


class QuadraticResNet_LogZ(nn.Module):
    """Quadratic ResNet that outputs scalar log normalizer (logZ)."""
    hidden_sizes: list
    use_layer_norm: bool = True
    
    @nn.compact
    def __call__(self, x, training=True):
        # Input projection
        x = nn.Dense(self.hidden_sizes[0], name='input_proj')(x)
        x = nn.swish(x)
        
        # Quadratic residual blocks
        for i, hidden_size in enumerate(self.hidden_sizes):
            residual = x
            if residual.shape[-1] != hidden_size:
                residual = nn.Dense(hidden_size, name=f'residual_proj_{i}')(residual)
            
            # Linear transformation
            linear_out = nn.Dense(hidden_size, name=f'linear_{i}')(x)
            linear_out = nn.swish(linear_out)
            
            # Quadratic transformation
            quadratic_out = nn.Dense(hidden_size, 
                                   kernel_init=nn.initializers.normal(stddev=0.01), 
                                   name=f'quadratic_{i}')(x)
            quadratic_out = nn.swish(quadratic_out)
            
            # Quadratic mixing: y = x + Ax + (Bx)x
            x = residual + linear_out + (linear_out * quadratic_out)
            
            if self.use_layer_norm:
                x = nn.LayerNorm(name=f'layer_norm_{i}')(x)
        
        # Final scalar output
        x = nn.Dense(1, name='output')(x)
        return jnp.squeeze(x, axis=-1)


def train_model(model, eta_data, ground_truth, epochs=200, learning_rate=1e-3, model_type="ET"):
    """Train a model and return parameters, losses, and predictions."""
    print(f"Training {model_type} model...")
    
    # Initialize
    rng = random.PRNGKey(42)
    params = model.init(rng, eta_data[:1])
    
    # Optimizer
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)
    opt_state = optimizer.init(params)
    
    # Loss function
    def loss_fn(params, eta_batch, target_batch):
        if model_type == "ET":
            # Direct prediction loss
            predictions = model.apply(params, eta_batch, training=True)
            return jnp.mean(jnp.square(predictions - target_batch))
        else:  # logZ
            # Log normalizer approach - compute gradient for mean
            log_norm = model.apply(params, eta_batch, training=True)
            
            def single_log_normalizer(eta_single):
                return model.apply(params, eta_single[None, :], training=False)[0]
            
            grad_fn = jax.grad(single_log_normalizer)
            network_mean = jax.vmap(grad_fn)(eta_batch)
            mse_loss = jnp.mean(jnp.square(network_mean - target_batch))
            l2_loss = jnp.mean(jnp.square(log_norm))
            return mse_loss + 1e-6 * l2_loss
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        loss, grads = jax.value_and_grad(loss_fn)(params, eta_data, ground_truth)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        losses.append(float(loss))
        
        if epoch % 50 == 0 or epoch < 5:
            print(f"  Epoch {epoch}: Loss = {loss:.6f}")
    
    # Make predictions
    if model_type == "ET":
        predictions = model.apply(params, eta_data, training=False)
    else:  # logZ
        def single_log_normalizer(eta_single):
            return model.apply(params, eta_single[None, :], training=False)[0]
        grad_fn = jax.grad(single_log_normalizer)
        predictions = jax.vmap(grad_fn)(eta_data)
    
    return params, losses, predictions


def create_comprehensive_comparison_plot(models_data, eta_data, ground_truth):
    """Create a comprehensive comparison plot for all models."""
    
    n_models = len(models_data)
    fig = plt.figure(figsize=(20, 4 * n_models))
    
    # Model names and colors
    model_names = list(models_data.keys())
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, (model_name, data) in enumerate(models_data.items()):
        color = colors[i % len(colors)]
        params, losses, predictions = data
        
        # Calculate metrics
        mse = float(jnp.mean(jnp.square(predictions - ground_truth)))
        mae = float(jnp.mean(jnp.abs(predictions - ground_truth)))
        
        # Training curve
        ax1 = plt.subplot(n_models, 4, i * 4 + 1)
        ax1.plot(losses, color=color, linewidth=2, label=model_name)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{model_name} Training')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Predictions vs Ground Truth (mean statistics only)
        ax2 = plt.subplot(n_models, 4, i * 4 + 2)
        mean_gt = ground_truth[:, :3]  # First 3 are mean statistics
        mean_pred = predictions[:, :3]
        
        ax2.scatter(mean_gt[:, 0], mean_pred[:, 0], alpha=0.6, s=20, color=color, label='E[x₁]')
        ax2.scatter(mean_gt[:, 1], mean_pred[:, 1], alpha=0.6, s=20, color=color, marker='s', label='E[x₂]')
        ax2.scatter(mean_gt[:, 2], mean_pred[:, 2], alpha=0.6, s=20, color=color, marker='^', label='E[x₃]')
        
        # Perfect prediction line
        min_val = min(mean_gt.min(), mean_pred.min())
        max_val = max(mean_gt.max(), mean_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
        
        ax2.set_xlabel('Ground Truth')
        ax2.set_ylabel('Prediction')
        ax2.set_title(f'{model_name} Mean Stats')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Predictions vs Ground Truth (covariance statistics)
        ax3 = plt.subplot(n_models, 4, i * 4 + 3)
        cov_gt = ground_truth[:, 3:]  # Last 9 are covariance statistics
        cov_pred = predictions[:, 3:]
        
        ax3.scatter(cov_gt.flatten(), cov_pred.flatten(), alpha=0.6, s=10, color=color)
        
        # Perfect prediction line
        min_val = min(cov_gt.min(), cov_pred.min())
        max_val = max(cov_gt.max(), cov_pred.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
        
        ax3.set_xlabel('Ground Truth')
        ax3.set_ylabel('Prediction')
        ax3.set_title(f'{model_name} Covariance Stats')
        ax3.grid(True, alpha=0.3)
        
        # Performance summary
        ax4 = plt.subplot(n_models, 4, i * 4 + 4)
        ax4.text(0.1, 0.8, f'MSE: {mse:.6f}', transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.6, f'MAE: {mae:.6f}', transform=ax4.transAxes, fontsize=12)
        
        # Calculate R² for mean and covariance separately
        mean_r2 = 1 - jnp.sum((mean_pred - mean_gt)**2) / jnp.sum((mean_gt - jnp.mean(mean_gt))**2)
        cov_r2 = 1 - jnp.sum((cov_pred - cov_gt)**2) / jnp.sum((cov_gt - jnp.mean(cov_gt))**2)
        
        ax4.text(0.1, 0.4, f'Mean R²: {float(mean_r2):.3f}', transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.2, f'Cov R²: {float(cov_r2):.3f}', transform=ax4.transAxes, fontsize=12)
        
        ax4.set_title(f'{model_name} Performance')
        ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('comprehensive_model_comparison.png', dpi=150, bbox_inches='tight')
    print("Comprehensive comparison plot saved as: comprehensive_model_comparison.png")
    plt.close()


def main():
    """Main function for comprehensive model comparison."""
    print("Comprehensive Model Comparison")
    print("=" * 60)
    
    # Generate test data
    print("Generating test data...")
    eta_data, ground_truth = generate_simple_test_data(n_samples=400, seed=42)
    print(f"Data shapes: eta={eta_data.shape}, ground_truth={ground_truth.shape}")
    
    # Define models to test
    models = {
        'MLP_ET': SimpleMLP_ET(hidden_sizes=[64, 64]),
        'MLP_LogZ': SimpleMLP_LogZ(hidden_sizes=[64, 64]),
        'QuadResNet_LogZ': QuadraticResNet_LogZ(hidden_sizes=[64, 64]),
    }
    
    # Train all models
    models_data = {}
    for model_name, model in models.items():
        model_type = "ET" if "ET" in model_name else "LogZ"
        params, losses, predictions = train_model(
            model, eta_data, ground_truth, 
            epochs=200, learning_rate=1e-3, model_type=model_type
        )
        models_data[model_name] = (params, losses, predictions)
        print()
    
    # Create comprehensive comparison plot
    print("Creating comprehensive comparison plot...")
    create_comprehensive_comparison_plot(models_data, eta_data, ground_truth)
    
    # Performance summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    for model_name, (params, losses, predictions) in models_data.items():
        mse = float(jnp.mean(jnp.square(predictions - ground_truth)))
        mae = float(jnp.mean(jnp.abs(predictions - ground_truth)))
        final_loss = losses[-1]
        
        print(f"{model_name:15}: MSE={mse:.6f}, MAE={mae:.6f}, Final Loss={final_loss:.6f}")
    
    print("\n✅ Comprehensive comparison complete!")


if __name__ == "__main__":
    main()
