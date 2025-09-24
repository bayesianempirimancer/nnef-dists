#!/usr/bin/env python3
"""
Training script template for LogZ (Log Normalizer) neural networks.

This script trains networks that learn the log normalizer A(η) and use automatic
differentiation to compute the expected sufficient statistics E[T(X)] via ∇A(η).

Usage:
    python scripts/training/train_template_logZ.py
"""

import sys
import argparse
from pathlib import Path
import jax.numpy as jnp
from jax import random
# matplotlib import removed - now using standardized plotting

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import standardized plotting functions
sys.path.append(str(Path(__file__).parent.parent))
from scripts.plotting.plot_training_results import plot_training_results, plot_model_comparison, save_results_summary, create_standardized_results

# Import dimension inference utility
from src.utils.data_utils import infer_dimensions



# Data generation function removed - now using easy_3d_gaussian data


# Simple configuration class
class SimpleConfig:
    def __init__(self, hidden_sizes, activation="swish"):
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.use_layer_norm = True

class SimpleTemplateLogZ:
    """Template Log Normalizer using official implementation."""
    
    def __init__(self, hidden_sizes=[64, 64], eta_dim=None):
        self.hidden_sizes = hidden_sizes
        self.eta_dim = eta_dim
        # Create proper FullConfig
        from src.config import FullConfig
        self.config = FullConfig()
        self.config.network.hidden_sizes = hidden_sizes
        self.config.network.activation = "swish"
        self.config.network.use_layer_norm = True
    
    def create_model_and_trainer(self):
        """Create the model using the official implementation."""
        # This is a template - replace with actual model import
        from src.models.mlp_logZ import create_model_and_trainer
        # Create proper network config using inferred dimensions
        from src.config import NetworkConfig, FullConfig, TrainingConfig
        
        # Use inferred dimensions from data
        # For LogZ models, input is eta, output is scalar log normalizer
        # Target is mu_T - expectation of sufficient statistic
        if self.eta_dim is None:
            raise ValueError("eta_dim must be provided. Call infer_dimensions() first.")
        
        input_dim = self.eta_dim
        output_dim = 1  # LogZ models output scalar log normalizer
        
        network_config = NetworkConfig(
            hidden_sizes=self.hidden_sizes,
            activation="swish",
            use_layer_norm=True,
            input_dim=input_dim,
            output_dim=output_dim
        )
        training_config = TrainingConfig(num_epochs=300, learning_rate=1e-3)
        full_config = FullConfig(network=network_config, training=training_config)
        return create_model_and_trainer(full_config)
    
    def train(self, eta_data, ground_truth, epochs=300, learning_rate=1e-3):
        """Train the model and measure training time."""
        trainer = self.create_model_and_trainer()
        
        # Prepare training data
        # For LogZ models, ground_truth should be mu_T (same dimension as eta)
        train_data = {
            'eta': eta_data,
            'mu_T': ground_truth  # This is the expectation of sufficient statistic
        }
        
        # Measure training time
        import time
        start_time = time.time()
        
        # Use the trainer's built-in training method
        best_params, training_history = trainer.train(
            train_data=train_data,
            val_data=None,
            epochs=epochs,
            learning_rate=learning_rate
        )
        
        training_time = time.time() - start_time
        
        return best_params, training_history['train_loss'], training_time
    
    def predict(self, trainer, params, eta_data):
        """Make predictions using the trained model."""
        predictions = trainer.predict(params, eta_data)
        return predictions['stats']  # Extract expected sufficient statistics from the prediction dictionary
    
    def benchmark_inference(self, trainer, params, eta_data, num_runs=50):
        """Benchmark inference time with multiple runs for accuracy."""
        import time
        # Warm-up run to ensure compilation is complete
        _ = trainer.predict(params, eta_data[:1])
        
        # Measure inference time over multiple runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = trainer.predict(params, eta_data)
            times.append(time.time() - start_time)
        
        # Return statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        return {
            'avg_inference_time': avg_time,
            'min_inference_time': min_time,
            'max_inference_time': max_time,
            'inference_per_sample': avg_time / len(eta_data),
            'samples_per_second': len(eta_data) / avg_time
        }

    def evaluate(self, predictions, targets):
        """Evaluate predictions against targets."""
        import jax.numpy as jnp
        
        mse = jnp.mean((predictions - targets) ** 2)
        mae = jnp.mean(jnp.abs(predictions - targets))
        
        return {
            'mse': float(mse),
            'mae': float(mae)
        }

# Plotting function removed - now using standardized plotting from scripts/plot_training_results.py

def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Template LogZ models')
    parser.add_argument('--data_file', type=str, help='Path to data file (default: data/easy_3d_gaussian.pkl)')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    
    args = parser.parse_args()
    
    print("Training Template LogZ Model")
    print("=" * 40)
    
    # Load test data using standardized function
    from src.utils.data_utils import load_standardized_ep_data
    data_file = args.data_file if args.data_file else "data/easy_3d_gaussian.pkl"
    eta_data, ground_truth, metadata = load_standardized_ep_data(data_file)
    
    # Test single large architecture
    architectures = {
        'Template_LogZ_Large': [128, 128, 64],
    }
    
    results = {}
    
    for name, hidden_sizes in architectures.items():
        print(f"\nTraining {name} with hidden sizes: {hidden_sizes}")
        
        # Infer dimensions from metadata or data
        eta_dim = infer_dimensions(eta_data, metadata=metadata)
        model = SimpleTemplateLogZ(hidden_sizes=hidden_sizes, eta_dim=eta_dim)
        
        params, losses, training_time = model.train(eta_data, ground_truth, epochs=300)
        trained_model = model.create_model_and_trainer()
        predictions = model.predict(trained_model, params, eta_data)
        
        # Benchmark inference time
        inference_stats = model.benchmark_inference(trained_model, params, eta_data, num_runs=50)
        
        # Calculate metrics
        mse = float(jnp.mean(jnp.square(predictions - ground_truth)))
        mae = float(jnp.mean(jnp.abs(predictions - ground_truth)))
        
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Avg inference time: {inference_stats['avg_inference_time']:.4f}s ({inference_stats['samples_per_second']:.1f} samples/sec)")
        
        # Create plots using standardized plotting function
        metrics = plot_training_results(
            trainer=model,
            eta_data=eta_data,
            ground_truth=ground_truth,
            predictions=predictions,
            losses=losses,
            config=model.config,
            model_name=name,
            output_dir="artifacts/logZ_models/template_logZ",
            save_plots=True,
            show_plots=False
        )
        
        results[name] = create_standardized_results(
            model_name=name,
            architecture_info={"hidden_sizes": hidden_sizes},
            metrics=metrics,
            losses=losses,
            training_time=training_time,
            inference_stats=inference_stats,
            predictions=predictions,
            ground_truth=ground_truth
        )
        
        print(f"  Final MSE: {mse:.6f}, MAE: {mae:.6f}")
    
    # Summary
    print(f"\n{'='*60}")
    print("TEMPLATE LOGZ MODEL COMPARISON")
    print(f"{'='*60}")
    
    for name, result in results.items():
        print(f"{name:20}: MSE={result['mse']:.6f}, MAE={result['mae']:.6f}, "
              f"Architecture={result['hidden_sizes']}")
    
    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]['mse'])
    print(f"\n🏆 Best Model: {best_model[0]} with MSE={best_model[1]['mse']:.6f}")
    
    # Create model comparison plots
    plot_model_comparison(
        results=results,
        output_dir="artifacts/logZ_models/template_logZ",
        save_plots=True,
        show_plots=False
    )
    
    # Save results summary
    save_results_summary(
        results=results,
        output_dir="artifacts/logZ_models/template_logZ"
    )
    
    print("\n✅ Template LogZ training complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Template LogZ models')
    parser.add_argument('--data_file', type=str, help='Path to data file (default: data/easy_3d_gaussian.pkl)')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    
    args = parser.parse_args()
    main()