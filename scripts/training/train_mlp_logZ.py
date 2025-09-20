#!/usr/bin/env python3
"""
Training script for MLP-based log normalizer neural networks.

This script trains MLP networks that learn the log normalizer A(η)
and use automatic differentiation to compute the mean and covariance of
exponential family distributions.

Usage:
    python scripts/training/train_mlp_logZ.py --config configs/gaussian_1d.yaml
    python scripts/training/train_mlp_logZ.py --config configs/multivariate_3d.yaml
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Simple data generation function
def generate_simple_test_data(n_samples=400, seed=42):
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


# Simple configuration class
class SimpleConfig:
    def __init__(self, hidden_sizes, activation="swish"):
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.use_layer_norm = True

class SimpleMLPLogZ:
    """MLP Log Normalizer using official implementation."""
    
    def __init__(self, hidden_sizes=[64, 64]):
        self.hidden_sizes = hidden_sizes
        self.config = SimpleConfig(hidden_sizes=hidden_sizes, activation="swish")
    
    def create_model(self):
        """Create the model using the official implementation."""
        try:
            from src.models.mlp_logZ import MLPLogNormalizerTrainer
            # Create a dummy trainer to get the model
            trainer = MLPLogNormalizerTrainer(self.config)
            return trainer.model
        except ImportError:
            # Fallback to simplified implementation if import fails
            import flax.linen as nn
            
            class MLPLogZModel(nn.Module):
                hidden_sizes: list
                
                @nn.compact
                def __call__(self, x, training=True):
                    for i, hidden_size in enumerate(self.hidden_sizes):
                        x = nn.Dense(hidden_size, name=f'hidden_{i}')(x)
                        x = nn.swish(x)
                        if i < len(self.hidden_sizes) - 1:
                            x = nn.LayerNorm(name=f'layer_norm_{i}')(x)
                    
                    # Output layer - scalar log normalizer
                    x = nn.Dense(1, name='output')(x)
                    return jnp.squeeze(x, axis=-1)
            
            return MLPLogZModel(hidden_sizes=self.hidden_sizes)
    
    def train(self, eta_data, ground_truth, epochs=300, learning_rate=1e-3):
        """Train the model."""
        import optax
        import flax.linen as nn
        import jax
        
        model = self.create_model()
        
        # Initialize
        rng = random.PRNGKey(42)
        params = model.init(rng, eta_data[:1])
        
        # Optimizer
        optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)
        opt_state = optimizer.init(params)
        
        # Loss function
        def loss_fn(params, eta_batch, target_batch):
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
        
        return params, losses, model
    
    def predict(self, model, params, eta_data):
        """Make predictions using the trained model."""
        import jax
        
        def single_log_normalizer(eta_single):
            return model.apply(params, eta_single[None, :], training=False)[0]
        
        grad_fn = jax.grad(single_log_normalizer)
        predictions = jax.vmap(grad_fn)(eta_data)
        return predictions

def plot_training_results(trainer, eta_data, ground_truth, predictions, losses, config):
    """Create comprehensive plots for training results."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Learning curves
    ax1 = plt.subplot(2, 4, 1)
    plt.plot(losses, 'b-', linewidth=2)
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # 2. Predictions vs Ground Truth (scatter)
    ax2 = plt.subplot(2, 4, 2)
    plt.scatter(ground_truth, predictions, alpha=0.6, s=20)
    min_val = min(ground_truth.min(), predictions.min())
    max_val = max(ground_truth.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title('Predictions vs Ground Truth', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 3. Residuals
    ax3 = plt.subplot(2, 4, 3)
    residuals = predictions - ground_truth
    plt.scatter(ground_truth, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Ground Truth')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Ground Truth', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 4. Per-statistic performance
    ax4 = plt.subplot(2, 4, 4)
    stat_names = [f'E[T_{i}]' for i in range(ground_truth.shape[1])]
    mse_per_stat = np.mean((predictions - ground_truth) ** 2, axis=0)
    bars = plt.bar(range(len(stat_names)), mse_per_stat)
    plt.xlabel('Statistics')
    plt.ylabel('MSE')
    plt.title('MSE per Statistic', fontsize=14, fontweight='bold')
    plt.xticks(range(len(stat_names)), stat_names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 5. Log normalizer values
    ax5 = plt.subplot(2, 4, 5)
    log_normalizer_values = trainer.model.apply(trainer.params, eta_data, training=False)
    plt.hist(log_normalizer_values, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Log Normalizer A(η)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Log Normalizer Values', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 6. Gradient magnitudes
    ax6 = plt.subplot(2, 4, 6)
    gradients = trainer.compute_predictions(trainer.params, eta_data)
    gradient_magnitudes = np.linalg.norm(gradients, axis=1)
    plt.hist(gradient_magnitudes, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Gradient Magnitude ||∇A(η)||')
    plt.ylabel('Frequency')
    plt.title('Distribution of Gradient Magnitudes', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 7. Input distribution
    ax7 = plt.subplot(2, 4, 7)
    if eta_data.shape[1] <= 2:
        if eta_data.shape[1] == 1:
            plt.hist(eta_data[:, 0], bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Natural Parameters η')
        else:
            plt.scatter(eta_data[:, 0], eta_data[:, 1], alpha=0.6, s=20)
            plt.xlabel('η₁')
            plt.ylabel('η₂')
        plt.title('Input Distribution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, f'{eta_data.shape[1]}D input\n(too high for 2D plot)', 
                ha='center', va='center', transform=ax7.transAxes, fontsize=12)
        plt.title('Input Distribution', fontsize=14, fontweight='bold')
    
    # 8. Performance metrics
    ax8 = plt.subplot(2, 4, 8)
    mse = np.mean((predictions - ground_truth) ** 2)
    mae = np.mean(np.abs(predictions - ground_truth))
    r2 = 1 - np.sum((ground_truth - predictions) ** 2) / np.sum((ground_truth - np.mean(ground_truth)) ** 2)
    
    metrics_text = f"""
    MSE: {mse:.6f}
    MAE: {mae:.6f}
    R²: {r2:.4f}
    
    Final Loss: {losses[-1]:.6f}
    Parameters: {sum(x.size for x in jax.tree.leaves(trainer.params)):,}
    """
    
    plt.text(0.1, 0.5, metrics_text, transform=ax8.transAxes, 
             fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    plt.axis('off')
    plt.title('Performance Metrics', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot to artifacts directory
    artifacts_dir = Path(__file__).parent.parent.parent / "artifacts"
    model_dir = artifacts_dir / "logZ_models" / "mlp_logZ"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = model_dir / f"mlp_logZ_training_results_{config.network.exp_family}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Training results saved to: {output_path}")
    
    return fig

def main():
    """Main training function."""
    print("Training MLP LogZ Model")
    print("=" * 40)
    
    # Generate test data
    print("Generating test data...")
    eta_data, ground_truth = generate_simple_test_data(n_samples=400, seed=42)
    print(f"Data shapes: eta={eta_data.shape}, ground_truth={ground_truth.shape}")
    
    # Test different architectures
    architectures = {
        'MLP_LogZ_Small': [32, 32],
        'MLP_LogZ_Medium': [64, 64],
        'MLP_LogZ_Large': [128, 128],
        'MLP_LogZ_Deep': [64, 64, 64],
        'MLP_LogZ_Wide': [128, 64, 128],
    }
    
    results = {}
    
    for name, hidden_sizes in architectures.items():
        print(f"\nTraining {name} with hidden sizes: {hidden_sizes}")
        
        model = SimpleMLPLogZ(hidden_sizes=hidden_sizes)
        params, losses, trained_model = model.train(eta_data, ground_truth, epochs=300)
        predictions = model.predict(trained_model, params, eta_data)
        
        # Calculate metrics
        mse = float(jnp.mean(jnp.square(predictions - ground_truth)))
        mae = float(jnp.mean(jnp.abs(predictions - ground_truth)))
        
        # Create plots
        fig = plt.figure(figsize=(15, 10))
        
        # Training curve
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(losses, linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{name} Training Progress')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Mean statistics
        ax2 = plt.subplot(2, 3, 2)
        mean_gt = ground_truth[:, :3]
        mean_pred = predictions[:, :3]
        
        ax2.scatter(mean_gt[:, 0], mean_pred[:, 0], alpha=0.6, s=20, label='E[x₁]')
        ax2.scatter(mean_gt[:, 1], mean_pred[:, 1], alpha=0.6, s=20, marker='s', label='E[x₂]')
        ax2.scatter(mean_gt[:, 2], mean_pred[:, 2], alpha=0.6, s=20, marker='^', label='E[x₃]')
        
        min_val = min(mean_gt.min(), mean_pred.min())
        max_val = max(mean_gt.max(), mean_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
        
        ax2.set_xlabel('Ground Truth')
        ax2.set_ylabel('Prediction')
        ax2.set_title(f'{name} Mean Statistics')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Covariance statistics
        ax3 = plt.subplot(2, 3, 3)
        cov_gt = ground_truth[:, 3:]
        cov_pred = predictions[:, 3:]
        
        ax3.scatter(cov_gt.flatten(), cov_pred.flatten(), alpha=0.6, s=10)
        
        min_val = min(cov_gt.min(), cov_pred.min())
        max_val = max(cov_gt.max(), cov_pred.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
        
        ax3.set_xlabel('Ground Truth')
        ax3.set_ylabel('Prediction')
        ax3.set_title(f'{name} Covariance Statistics')
        ax3.grid(True, alpha=0.3)
        
        # Performance metrics
        ax4 = plt.subplot(2, 3, 4)
        mean_r2 = 1 - jnp.sum((mean_pred - mean_gt)**2) / jnp.sum((mean_gt - jnp.mean(mean_gt))**2)
        cov_r2 = 1 - jnp.sum((cov_pred - cov_gt)**2) / jnp.sum((cov_gt - jnp.mean(cov_gt))**2)
        
        ax4.text(0.1, 0.8, f'MSE: {mse:.6f}', transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.6, f'MAE: {mae:.6f}', transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.4, f'Mean R²: {float(mean_r2):.3f}', transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.2, f'Cov R²: {float(cov_r2):.3f}', transform=ax4.transAxes, fontsize=12)
        
        ax4.set_title(f'{name} Performance')
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        artifacts_dir = Path(__file__).parent.parent.parent / "artifacts"
        model_dir = artifacts_dir / "logZ_models" / "mlp_logZ"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(model_dir / f"{name.lower()}_results.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        results[name] = {'mse': mse, 'mae': mae, 'hidden_sizes': hidden_sizes}
        
        print(f"  Final MSE: {mse:.6f}, MAE: {mae:.6f}")
    
    # Summary
    print(f"\n{'='*60}")
    print("MLP LOGZ MODEL COMPARISON")
    print(f"{'='*60}")
    
    for name, result in results.items():
        print(f"{name:20}: MSE={result['mse']:.6f}, MAE={result['mae']:.6f}, "
              f"Architecture={result['hidden_sizes']}")
    
    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]['mse'])
    print(f"\n🏆 Best Model: {best_model[0]} with MSE={best_model[1]['mse']:.6f}")
    
    print("\n✅ MLP LogZ training complete!")


if __name__ == "__main__":
    main()
