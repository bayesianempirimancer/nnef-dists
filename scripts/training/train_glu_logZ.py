#!/usr/bin/env python3
"""
Training script for GLU-based log normalizer neural networks.

This script trains GLU networks that learn the log normalizer A(η)
and use automatic differentiation to compute the mean and covariance of
exponential family distributions.

Usage:
    python scripts/training/train_glu_logZ.py --config data/configs/gaussian_1d.yaml
    python scripts/training/train_glu_logZ.py --config data/configs/multivariate_3d.yaml
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


class SimpleGLULogZ:
    """GLU Log Normalizer using official implementation."""
    
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
        from src.models.glu_logZ import create_model_and_trainer
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
        import time
        start_time = time.time()
        
        trainer = self.create_model_and_trainer()
        
        # Prepare training data
        # For LogZ models, ground_truth should be mu_T (same dimension as eta)
        train_data = {
            'eta': eta_data,
            'mu_T': ground_truth  # This is the expectation of sufficient statistic
        }
        
        # Use the trainer's built-in training method
        # LogZTrainer accepts epochs and learning_rate parameters
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
        # LogZTrainer's predict method returns a dictionary with 'stats' key
        predictions = trainer.predict(params, eta_data)
        return predictions['stats']
    
    def benchmark_inference(self, trainer, params, eta_data, num_runs=100):
        """Benchmark inference time with multiple runs for accuracy."""
        import time
        # Warm-up run to ensure compilation is complete
        _ = trainer.predict(params, eta_data[:1])
        
        # Measure inference time over multiple runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = trainer.predict(params, eta_data)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        per_sample_time = avg_time / len(eta_data)
        samples_per_second = len(eta_data) / avg_time
        
        return {
            'avg_inference_time': avg_time,
            'inference_per_sample': per_sample_time,
            'samples_per_second': samples_per_second
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


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train GLU LogZ models')
    parser.add_argument('--data_file', type=str, help='Path to data file (default: data/easy_3d_gaussian.pkl)')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    
    args = parser.parse_args()
    
    print("Training GLU LogZ Model")
    print("=" * 40)
    
    # Load test data using standardized function
    from src.utils.data_utils import load_standardized_ep_data
    data_file = args.data_file if args.data_file else "data/easy_3d_gaussian.pkl"
    eta_data, ground_truth, metadata = load_standardized_ep_data(data_file)
    
    # Test single deep architecture
    architectures = {
        'GLU_LogZ_Deep': [64, 64, 64, 64, 64, 64, 64, 64],
    }
    
    results = {}
    
    for name, hidden_sizes in architectures.items():
        print(f"\nTraining {name} with hidden sizes: {hidden_sizes}")
        
        # Infer dimensions from metadata or data
        eta_dim = infer_dimensions(eta_data, metadata=metadata)
        model = SimpleGLULogZ(hidden_sizes=hidden_sizes, eta_dim=eta_dim)
        
        trainer = model.create_model_and_trainer()
        
        # Train
        params, losses, training_time = model.train(eta_data, ground_truth, epochs=300)
        
        # Evaluate
        predictions = model.predict(trainer, params, eta_data)
        metrics = model.evaluate(predictions, ground_truth)
        
        # Benchmark inference time
        inference_stats = model.benchmark_inference(trainer, params, eta_data, num_runs=50)
        
        # Create plots using standardized plotting function
        metrics = plot_training_results(
            trainer=model,
            eta_data=eta_data,
            ground_truth=ground_truth,
            predictions=predictions,
            losses=losses,
            config=model.config,
            model_name=name,
            output_dir="artifacts/logZ_models/glu_logZ",
            save_plots=True,
            show_plots=False
        )
        
        results[name] = create_standardized_results(
            model_name=name,
            architecture_info={'hidden_sizes': hidden_sizes},
            metrics=metrics,
            losses=losses,
            training_time=training_time,
            inference_stats=inference_stats,
            predictions=predictions,
            ground_truth=ground_truth
        )
        
        print(f"  Final MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Avg inference time: {inference_stats['avg_inference_time']:.4f}s ({inference_stats['samples_per_second']:.1f} samples/sec)")
    
    # Summary
    print(f"\n{'='*60}")
    print("GLU LOGZ MODEL COMPARISON")
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
        output_dir="artifacts/logZ_models/glu_logZ",
        save_plots=True,
        show_plots=False
    )
    
    # Save results summary
    save_results_summary(
        results=results,
        output_dir="artifacts/logZ_models/glu_logZ"
    )
    
    print("\n✅ GLU LogZ training complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train GLU LogZ models')
    parser.add_argument('--data_file', type=str, help='Path to data file (default: data/easy_3d_gaussian.pkl)')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    
    args = parser.parse_args()
    main()
