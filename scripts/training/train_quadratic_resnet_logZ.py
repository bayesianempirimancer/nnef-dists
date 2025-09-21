#!/usr/bin/env python3
"""
Training script for Quadratic ResNet log normalizer neural networks.

This script trains Quadratic ResNet networks that learn the log normalizer A(η)
and use automatic differentiation to compute the mean and covariance of
exponential family distributions.

Usage:
    python scripts/training/train_quadratic_resnet_logZ.py --config configs/gaussian_1d.yaml
    python scripts/training/train_quadratic_resnet_logZ.py --config configs/multivariate_3d.yaml
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from jax import random
# matplotlib import removed - now using standardized plotting

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import standardized plotting functions
sys.path.append(str(Path(__file__).parent.parent))
from plot_training_results import plot_training_results, plot_model_comparison, save_results_summary

# Data generation function removed - now using easy_3d_gaussian data


# Simple configuration class
class SimpleConfig:
    def __init__(self, hidden_sizes, activation="swish"):
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.use_layer_norm = True


class SimpleQuadraticResNetLogZ:
    """Quadratic ResNet Log Normalizer using official implementation."""
    
    def __init__(self, hidden_sizes=[64, 64]):
        self.hidden_sizes = hidden_sizes
        self.config = SimpleConfig(hidden_sizes=hidden_sizes, activation="swish")
    
    def create_model(self):
        """Create the model using the official implementation."""
        from src.models.quadratic_resnet_logZ import QuadraticResNetLogNormalizer
        # Create proper network config
        from src.config import NetworkConfig
        network_config = NetworkConfig()
        network_config.hidden_sizes = self.hidden_sizes
        network_config.activation = "swish"
        network_config.use_layer_norm = True
        network_config.output_dim = 1  # Log normalizer is scalar
        return QuadraticResNetLogNormalizer(config=network_config)
    
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


def main():
    """Main training function."""
    print("Training Quadratic ResNet LogZ Model")
    print("=" * 40)
    
    # Load test data from easy_3d_gaussian
    print("Loading test data from easy_3d_gaussian...")
    data_file = Path("data/easy_3d_gaussian.pkl")
    
    import pickle
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    eta_data = data["train"]["eta"]
    ground_truth = data["train"]["mean"]
    print(f"Data shapes: eta={eta_data.shape}, ground_truth={ground_truth.shape}")
    
    # Purge cov_tt to save memory
    if "cov" in data["train"]: del data["train"]["cov"]
    if "cov" in data["val"]: del data["val"]["cov"]
    if "cov" in data["test"]: del data["test"]["cov"]
    import gc; gc.collect()
    print("✅ Purged cov_tt elements from memory for optimization")
    
    # Test different architectures
    architectures = {
        'QuadResNet_LogZ_Small': [32, 32],
        'QuadResNet_LogZ_Medium': [64, 64],
        'QuadResNet_LogZ_Large': [128, 128],
        'QuadResNet_LogZ_Deep': [64, 64, 64],
        'QuadResNet_LogZ_Wide': [128, 64, 128],
        'QuadResNet_LogZ_Max': [128, 128, 128],  # Maximum performance from our earlier work
    }
    
    results = {}
    
    for name, hidden_sizes in architectures.items():
        print(f"\nTraining {name} with hidden sizes: {hidden_sizes}")
        
        model = SimpleQuadraticResNetLogZ(hidden_sizes=hidden_sizes)
        params, losses, trained_model = model.train(eta_data, ground_truth, epochs=300)
        predictions = model.predict(trained_model, params, eta_data)
        
        # Calculate metrics
        mse = float(jnp.mean(jnp.square(predictions - ground_truth)))
        mae = float(jnp.mean(jnp.abs(predictions - ground_truth)))
        
        # Create plots using standardized plotting function
        metrics = plot_training_results(
            trainer=model_wrapper,
            eta_data=eta_data,
            ground_truth=ground_truth,
            predictions=predictions,
            losses=losses,
            config=model_wrapper.config,
            model_name=name,
            output_dir="artifacts/logZ_models/quadratic_resnet_logZ",
            save_plots=True,
            show_plots=False
        )
        
        results[name] = {
            'mse': metrics['mse'], 
            'mae': metrics['mae'], 
            'r2': metrics['r2'],
            'final_loss': metrics['final_loss'],
            'hidden_sizes': hidden_sizes,
            'losses': losses
        }
        
        print(f"  Final MSE: {mse:.6f}, MAE: {mae:.6f}")
    
    # Summary
    print(f"\n{'='*60}")
    print("QUADRATIC RESNET LOGZ MODEL COMPARISON")
    print(f"{'='*60}")
    
    for name, result in results.items():
        print(f"{name:25}: MSE={result['mse']:.6f}, MAE={result['mae']:.6f}, "
              f"Architecture={result['hidden_sizes']}")
    
    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]['mse'])
    print(f"\n🏆 Best Model: {best_model[0]} with MSE={best_model[1]['mse']:.6f}")
    
    # Create model comparison plots
    plot_model_comparison(
        results=results,
        output_dir="artifacts/logZ_models/quadratic_resnet_logZ",
        save_plots=True,
        show_plots=False
    )
    
    # Save results summary
    save_results_summary(
        results=results,
        output_dir="artifacts/logZ_models/quadratic_resnet_logZ"
    )
    
    print("\n✅ Quadratic ResNet LogZ training complete!")


if __name__ == "__main__":
    main()
