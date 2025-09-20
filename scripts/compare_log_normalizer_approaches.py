#!/usr/bin/env python3
"""
Comparison script for log normalizer approaches on 3D Gaussian case.

This script compares:
1. Basic LogNormalizerNetwork (simple MLP)
2. QuadraticResNetLogNormalizer (quadratic residual network)

Both approaches learn the log normalizer A(η) and use automatic differentiation
to compute the mean and covariance of 3D multivariate Gaussian distributions.

The comparison evaluates:
- Training convergence
- Mean prediction accuracy
- Covariance prediction accuracy (diagonal approximation)
- Computational efficiency
- Model complexity
"""

import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import time
import json
import optax

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import NetworkConfig, TrainingConfig, FullConfig
from ef import MultivariateNormal
from models.log_normalizer import (
    LogNormalizerNetwork, LogNormalizerTrainer,
    compute_log_normalizer_gradient, compute_log_normalizer_hessian,
    prepare_log_normalizer_data, log_normalizer_loss_fn
)
from models.quadratic_resnet_log_normalizer import (
    QuadraticResNetLogNormalizer, QuadraticResNetLogNormalizerTrainer
)


def generate_3d_gaussian_data(n_samples=1000, seed=42):
    """Generate synthetic 3D Gaussian data for testing."""
    rng = random.PRNGKey(seed)
    
    # Generate random natural parameters
    # For 3D Gaussian: 3 mean parameters + 9 covariance parameters = 12 total
    eta_mean = random.normal(rng, (12,)) * 0.5  # Natural parameters
    
    # Generate batch of natural parameters around the mean
    eta_batch = eta_mean[None, :] + random.normal(random.PRNGKey(seed + 1), (n_samples, 12)) * 0.2
    
    # Ensure covariance part is negative definite (for integrability)
    # The covariance part is the last 9 parameters
    cov_params = eta_batch[:, 3:]  # Last 9 parameters
    cov_matrices = cov_params.reshape(n_samples, 3, 3)
    
    # Make symmetric and negative definite
    cov_matrices = (cov_matrices + jnp.transpose(cov_matrices, (0, 2, 1))) / 2
    # Ensure negative definiteness by subtracting a large positive diagonal
    eye = jnp.eye(3)[None, :, :]
    cov_matrices = cov_matrices - 2.0 * eye
    
    # Flatten back to natural parameter format
    eta_batch = jnp.concatenate([eta_batch[:, :3], cov_matrices.reshape(n_samples, 9)], axis=1)
    
    # Compute theoretical means for 3D Gaussian
    # For natural parameters η = [η₁, η₂] where η₁ is mean part and η₂ is covariance part
    # E[X] = -η₂⁻¹ η₁ and E[XXᵀ] = -η₂⁻¹ + E[X] E[X]ᵀ
    
    theoretical_means = []
    theoretical_covs = []
    
    for i in range(n_samples):
        eta1 = eta_batch[i, :3]  # Mean part
        eta2 = eta_batch[i, 3:].reshape(3, 3)  # Covariance part
        
        try:
            # E[X] = -η₂⁻¹ η₁
            eta2_inv = jnp.linalg.inv(eta2)
            mean_x = -eta2_inv @ eta1
            
            # E[XXᵀ] = -η₂⁻¹ + E[X] E[X]ᵀ
            mean_xxT = -eta2_inv + jnp.outer(mean_x, mean_x)
            
            # Flatten mean_xxT for storage
            mean_xxT_flat = mean_xxT.flatten()
            
            theoretical_means.append(jnp.concatenate([mean_x, mean_xxT_flat]))
            
            # Covariance of sufficient statistics (simplified)
            # For multivariate Gaussian, Cov[X] = -η₂⁻¹
            theoretical_covs.append(-eta2_inv)
            
        except:
            # If matrix is singular, use a default
            theoretical_means.append(jnp.zeros(12))
            theoretical_covs.append(jnp.eye(3))
    
    theoretical_means = jnp.array(theoretical_means)
    theoretical_covs = jnp.array(theoretical_covs)
    
    return eta_batch, theoretical_means, theoretical_covs


def create_network_configs():
    """Create configurations for both network types."""
    
    # Basic log normalizer config
    basic_config = NetworkConfig()
    basic_config.hidden_sizes = [128, 64, 32]
    basic_config.output_dim = 1
    basic_config.use_feature_engineering = True
    basic_config.use_batch_norm = False
    basic_config.use_layer_norm = False
    basic_config.activation = "tanh"
    basic_config.dropout_rate = 0.0
    
    # Quadratic ResNet config
    resnet_config = NetworkConfig()
    resnet_config.hidden_sizes = [96, 64, 32]  # Slightly smaller for fair comparison
    resnet_config.output_dim = 1
    resnet_config.use_feature_engineering = True
    resnet_config.use_batch_norm = False
    resnet_config.use_layer_norm = False
    resnet_config.activation = "tanh"
    resnet_config.dropout_rate = 0.0
    
    return basic_config, resnet_config


def train_model(model, config, train_data, val_data, test_data, model_name):
    """Train a single model and return results."""
    print(f"\nTraining {model_name}")
    print("=" * 50)
    
    # Create trainer
    if "ResNet" in model_name:
        trainer = QuadraticResNetLogNormalizerTrainer(
            config, hessian_method='diagonal', use_curriculum=True
        )
    else:
        trainer = LogNormalizerTrainer(
            config, hessian_method='diagonal',
            loss_weights={'mean_weight': 1.0, 'cov_weight': 0.1}
        )
    
    # Training configuration
    trainer.config.training.num_epochs = 100
    trainer.config.training.batch_size = 32
    trainer.config.training.validation_freq = 10
    trainer.config.training.early_stopping = True
    trainer.config.training.patience = 20
    
    # Train model
    start_time = time.time()
    params, history = trainer.train(train_data, val_data)
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    metrics = {}
    
    # Standard evaluation
    test_log_norm = trainer.model.apply(params, test_data['eta'], training=False)
    test_mean_pred = compute_log_normalizer_gradient(trainer.model, params, test_data['eta'])
    
    # Mean metrics
    mean_mse = float(jnp.mean(jnp.square(test_mean_pred - test_data['mean'])))
    mean_mae = float(jnp.mean(jnp.abs(test_mean_pred - test_data['mean'])))
    
    metrics.update({
        'mean_mse': mean_mse,
        'mean_mae': mean_mae,
        'log_normalizer_std': float(jnp.std(test_log_norm)),
        'training_time': training_time
    })
    
    # Hessian metrics (diagonal approximation)
    try:
        test_hessian_pred = compute_log_normalizer_hessian(
            trainer.model, params, test_data['eta'], method='diagonal'
        )
        
        # Compare with diagonal of theoretical covariance
        empirical_diag = jnp.diagonal(test_data['cov'], axis1=1, axis2=2)
        hessian_mse = float(jnp.mean(jnp.square(test_hessian_pred - empirical_diag)))
        hessian_mae = float(jnp.mean(jnp.abs(test_hessian_pred - empirical_diag)))
        
        metrics.update({
            'hessian_mse': hessian_mse,
            'hessian_mae': hessian_mae
        })
        
    except Exception as e:
        print(f"Warning: Hessian computation failed: {e}")
        metrics.update({'hessian_mse': float('inf'), 'hessian_mae': float('inf')})
    
    # Model complexity
    param_count = trainer.model.get_parameter_count(params)
    metrics['parameter_count'] = param_count
    
    print(f"Results for {model_name}:")
    print(f"  Mean MSE: {metrics['mean_mse']:.6f}")
    print(f"  Mean MAE: {metrics['mean_mae']:.6f}")
    print(f"  Hessian MSE: {metrics.get('hessian_mse', 'N/A')}")
    print(f"  Parameter count: {metrics['parameter_count']:,}")
    print(f"  Training time: {metrics['training_time']:.2f}s")
    
    return params, history, metrics


def plot_comparison_results(results, save_path=None):
    """Plot comparison results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping plots")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    models = list(results.keys())
    colors = ['blue', 'red', 'green', 'orange']
    
    # 1. Training loss curves
    for i, (model_name, result) in enumerate(results.items()):
        history = result['history']
        epochs = range(len(history['train_loss']))
        axes[0, 0].plot(epochs, history['train_loss'], 
                       label=f'{model_name} (train)', color=colors[i], alpha=0.7)
        if 'val_loss' in history and history['val_loss']:
            val_epochs = range(0, len(epochs), len(epochs) // len(history['val_loss']))[:len(history['val_loss'])]
            axes[0, 0].plot(val_epochs, history['val_loss'], 
                           label=f'{model_name} (val)', color=colors[i], linestyle='--', alpha=0.7)
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Mean prediction accuracy
    mean_mses = [results[model]['metrics']['mean_mse'] for model in models]
    bars = axes[0, 1].bar(models, mean_mses, color=colors[:len(models)], alpha=0.7)
    axes[0, 1].set_ylabel('Mean MSE')
    axes[0, 1].set_title('Mean Prediction Accuracy')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mse in zip(bars, mean_mses):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{mse:.2e}', ha='center', va='bottom')
    
    # 3. Hessian prediction accuracy
    hessian_mses = [results[model]['metrics'].get('hessian_mse', float('inf')) for model in models]
    valid_hessian = [mse != float('inf') for mse in hessian_mses]
    
    if any(valid_hessian):
        valid_models = [models[i] for i in range(len(models)) if valid_hessian[i]]
        valid_mses = [hessian_mses[i] for i in range(len(models)) if valid_hessian[i]]
        valid_colors = [colors[i] for i in range(len(models)) if valid_hessian[i]]
        
        bars = axes[1, 0].bar(valid_models, valid_mses, color=valid_colors, alpha=0.7)
        axes[1, 0].set_ylabel('Hessian MSE')
        axes[1, 0].set_title('Covariance Prediction Accuracy')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mse in zip(bars, valid_mses):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{mse:.2e}', ha='center', va='bottom')
    else:
        axes[1, 0].text(0.5, 0.5, 'Hessian computation failed', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Covariance Prediction Accuracy')
    
    # 4. Training efficiency (time vs accuracy)
    training_times = [results[model]['metrics']['training_time'] for model in models]
    mean_mses = [results[model]['metrics']['mean_mse'] for model in models]
    
    scatter = axes[1, 1].scatter(training_times, mean_mses, c=colors[:len(models)], 
                               s=100, alpha=0.7)
    axes[1, 1].set_xlabel('Training Time (s)')
    axes[1, 1].set_ylabel('Mean MSE')
    axes[1, 1].set_title('Training Efficiency')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add model labels
    for i, model in enumerate(models):
        axes[1, 1].annotate(model, (training_times[i], mean_mses[i]), 
                           xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def main():
    """Main comparison function."""
    print("Log Normalizer Approaches Comparison")
    print("=" * 60)
    print("Comparing Basic LogNormalizer vs Quadratic ResNet on 3D Gaussian")
    
    # Generate synthetic data
    print("\nGenerating synthetic 3D Gaussian data...")
    eta_data, mean_data, cov_data = generate_3d_gaussian_data(n_samples=1000, seed=42)
    
    # Split data
    n_train = 600
    n_val = 200
    n_test = 200
    
    train_eta = eta_data[:n_train]
    train_mean = mean_data[:n_train]
    train_cov = cov_data[:n_train]
    
    val_eta = eta_data[n_train:n_train+n_val]
    val_mean = mean_data[n_train:n_train+n_val]
    val_cov = cov_data[n_train:n_train+n_val]
    
    test_eta = eta_data[n_train+n_val:]
    test_mean = mean_data[n_train+n_val:]
    test_cov = cov_data[n_train+n_val:]
    
    print(f"Data shapes:")
    print(f"  Training: eta={train_eta.shape}, mean={train_mean.shape}, cov={train_cov.shape}")
    print(f"  Validation: eta={val_eta.shape}, mean={val_mean.shape}, cov={val_cov.shape}")
    print(f"  Test: eta={test_eta.shape}, mean={test_mean.shape}, cov={test_cov.shape}")
    
    # Prepare data for log normalizer training
    train_data = prepare_log_normalizer_data(train_eta, train_mean, train_cov)
    val_data = prepare_log_normalizer_data(val_eta, val_mean, val_cov)
    test_data = prepare_log_normalizer_data(test_eta, test_mean, test_cov)
    
    # Create network configurations
    basic_config, resnet_config = create_network_configs()
    
    # Create models
    basic_model = LogNormalizerNetwork(config=basic_config)
    resnet_model = QuadraticResNetLogNormalizer(config=resnet_config)
    
    # Initialize models to count parameters
    rng = random.PRNGKey(123)
    basic_params = basic_model.init(rng, train_eta[:1])
    resnet_params = resnet_model.init(rng, train_eta[:1])
    
    print(f"\nModel architectures:")
    print(f"  Basic LogNormalizer: {basic_model.get_parameter_count(basic_params):,} parameters")
    print(f"  Quadratic ResNet: {resnet_model.get_parameter_count(resnet_params):,} parameters")
    
    # Train both models
    results = {}
    
    # Train basic model
    basic_full_config = FullConfig()
    basic_full_config.network = basic_config
    basic_full_config.training = TrainingConfig()
    
    basic_params, basic_history, basic_metrics = train_model(
        basic_model, basic_full_config, train_data, val_data, test_data, "Basic LogNormalizer"
    )
    
    results["Basic LogNormalizer"] = {
        'params': basic_params,
        'history': basic_history,
        'metrics': basic_metrics
    }
    
    # Train ResNet model
    resnet_full_config = FullConfig()
    resnet_full_config.network = resnet_config
    resnet_full_config.training = TrainingConfig()
    
    resnet_params, resnet_history, resnet_metrics = train_model(
        resnet_model, resnet_full_config, train_data, val_data, test_data, "Quadratic ResNet"
    )
    
    results["Quadratic ResNet"] = {
        'params': resnet_params,
        'history': resnet_history,
        'metrics': resnet_metrics
    }
    
    # Summary comparison
    print(f"\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"\n{model_name}:")
        print(f"  Mean MSE: {metrics['mean_mse']:.6f}")
        print(f"  Mean MAE: {metrics['mean_mae']:.6f}")
        print(f"  Hessian MSE: {metrics.get('hessian_mse', 'N/A')}")
        print(f"  Parameters: {metrics['parameter_count']:,}")
        print(f"  Training time: {metrics['training_time']:.2f}s")
    
    # Determine winner
    basic_mse = results["Basic LogNormalizer"]['metrics']['mean_mse']
    resnet_mse = results["Quadratic ResNet"]['metrics']['mean_mse']
    
    if resnet_mse < basic_mse:
        improvement = (basic_mse - resnet_mse) / basic_mse * 100
        print(f"\n🏆 Quadratic ResNet wins!")
        print(f"   Improvement: {improvement:.2f}% better mean MSE")
    else:
        improvement = (resnet_mse - basic_mse) / resnet_mse * 100
        print(f"\n🏆 Basic LogNormalizer wins!")
        print(f"   Improvement: {improvement:.2f}% better mean MSE")
    
    # Save results
    output_dir = Path("results") / "log_normalizer_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    with open(output_dir / "comparison_metrics.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, result in results.items():
            serializable_results[model_name] = {
                'metrics': result['metrics'],
                'history': {k: [float(v) if hasattr(v, 'item') else v for v in values] 
                           for k, values in result['history'].items()}
            }
        json.dump(serializable_results, f, indent=2)
    
    # Create and save plots
    plot_comparison_results(results, output_dir / "comparison_plots.png")
    
    print(f"\nResults saved to {output_dir}")
    
    return results


if __name__ == "__main__":
    main()
