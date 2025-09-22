#!/usr/bin/env python3
"""
Training script for Invertible Neural Network ET model.

This script trains, evaluates, and plots results for an Invertible NN ET
on the natural parameter to statistics mapping task using standardized
template-based data loading and dimension-agnostic processing.
"""

import sys
from pathlib import Path
import time
import json
import jax
import jax.numpy as jnp
from jax import random
# matplotlib import removed - now using standardized plotting

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import standardized plotting functions
sys.path.append(str(Path(__file__).parent.parent))
from plot_training_results import plot_training_results, plot_model_comparison, save_results_summary

# Import dimension inference utility and standardized data loading from template
from src.utils.data_utils import infer_dimensions
sys.path.append(str(Path(__file__).parent))
from ET_training_template import load_standard_data

# Simple configuration class
class SimpleConfig:
    def __init__(self, hidden_sizes, activation="swish"):
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.use_layer_norm = False

class SimpleInvertibleNNET:
    """Invertible NN ET using official implementation."""
    
    def __init__(self, hidden_sizes=[64, 64], eta_dim=None):
        self.hidden_sizes = hidden_sizes
        self.eta_dim = eta_dim
        self.config = SimpleConfig(hidden_sizes=hidden_sizes, activation="swish")
    
    def create_model_and_trainer(self):
        """Create the model and trainer using the official implementation."""
        from src.models.invertible_nn_ET import create_model_and_trainer
        # Create proper network config using inferred dimensions
        from src.config import NetworkConfig, FullConfig, TrainingConfig
        
        # For ET models, input and output dimensions should be the same
        if self.eta_dim is None:
            raise ValueError("eta_dim must be provided. Call infer_dimensions() first.")
        
        input_dim = self.eta_dim
        output_dim = self.eta_dim  # ET models predict expected statistics
        
        network_config = NetworkConfig(
            hidden_sizes=self.hidden_sizes,
            activation="swish",
            use_layer_norm=True,
            input_dim=input_dim,
            output_dim=output_dim
        )
        
        training_config = TrainingConfig(
            learning_rate=1e-3,
            num_epochs=300,
            batch_size=32,
            patience=50
        )
        
        full_config = FullConfig(network=network_config, training=training_config)
        return create_model_and_trainer(full_config)
    
    def train(self, train_data, val_data, epochs=300, learning_rate=1e-3):
        """Train the model using pre-split data and measure training time."""
        start_time = time.time()
        
        trainer = self.create_model_and_trainer()
        
        # Train model using the pre-split data (BaseTrainer doesn't take epochs/learning_rate as arguments)
        best_params, training_history = trainer.train(
            train_data=train_data,
            val_data=val_data
        )
        
        training_time = time.time() - start_time
        
        return best_params, training_history['train_loss'], training_time
    
    def predict(self, trainer, params, eta_data):
        """Make predictions."""
        return trainer.predict(params, eta_data)
    
    def benchmark_inference(self, trainer, params, eta_data, num_runs=50):
        """Benchmark inference time with multiple runs for accuracy."""
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
            'samples_per_second': len(eta_data) / avg_time,
            'std_inference_time': jnp.std(jnp.array(times))
        }
    
    def evaluate(self, trainer, params, eta_data, ground_truth):
        """Evaluate model performance."""
        predictions = self.predict(trainer, params, eta_data)
        mse = float(jnp.mean((predictions - ground_truth) ** 2))
        mae = float(jnp.mean(jnp.abs(predictions - ground_truth)))
        
        return {
            "mse": mse,
            "mae": mae,
            "predictions": predictions
        }

def main():
    """Main training and evaluation pipeline."""
    print("Training Invertible NN ET Model")
    print("=" * 40)
    
    # Load test data from specified file
    data_file = Path(args.data_file) if args.data_file else Path("data/easy_3d_gaussian.pkl")
    print(f"Loading test data from {data_file}...")
    
    # Load data using standardized template function (dimension-agnostic)
    data_file_override = args.data_file if args.data_file else None
    train_data, val_data, metadata = load_standard_data(data_file_override)
    
    # Define architecture to test (deep 8-layer network)
    architectures = {
        "Deep": [64, 64, 64, 64, 64, 64, 64, 64]
    }
    
    results = {}
    
    # Test the architecture
    for arch_name, hidden_sizes in architectures.items():
        print(f"\nTraining InvertibleNN_ET_{arch_name} with hidden sizes: {hidden_sizes}")
        
        # Infer dimensions from metadata
        eta_dim = infer_dimensions(train_data['eta'], metadata=metadata)
        model_wrapper = SimpleInvertibleNNET(hidden_sizes=hidden_sizes, eta_dim=eta_dim)
        
        trainer = model_wrapper.create_model_and_trainer()
        
        # Train
        print("🚂 Training...")
        params, losses, training_time = model_wrapper.train(train_data, val_data, epochs=300)
        
        # Evaluate
        print("📊 Evaluating...")
        metrics = model_wrapper.evaluate(trainer, params, train_data['eta'], train_data['y'])
        
        # Benchmark inference
        print("⚡ Benchmarking inference...")
        inference_stats = model_wrapper.benchmark_inference(trainer, params, train_data['eta'])
        
        results[f"InvertibleNN_ET_{arch_name}"] = {
            "hidden_sizes": hidden_sizes,
            "mse": metrics["mse"],
            "mae": metrics["mae"],
            "training_time": training_time,
            "inference_stats": inference_stats,
            "final_train_loss": losses[-1] if losses else float('inf')
        }
        
        predictions = metrics["predictions"]
        
        # Store detailed results
        detailed_results = {
            "test_metrics": metrics,
            "architecture": hidden_sizes,
            "model_name": f"InvertibleNN_ET_{arch_name}",
            "predictions": predictions,
            "ground_truth": train_data['y'],
            "training_time": training_time,
            "inference_stats": inference_stats
        }
        
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Avg inference time: {inference_stats['avg_inference_time']:.4f}s ({inference_stats['samples_per_second']:.1f} samples/sec)")
    
    print(f"\n🏆 Best Model: InvertibleNN_ET_Deep with MSE={results['InvertibleNN_ET_Deep']['mse']:.6f}")
    
    # Create output directory
    output_dir = Path("artifacts/ET_models/invertible_nn_ET")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_file = output_dir / "training_results.json"
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            json_results[key] = {}
            for k, v in value.items():
                if hasattr(v, 'tolist'):
                    json_results[key][k] = v.tolist()
                else:
                    json_results[key][k] = v
        json.dump(json_results, f, indent=2)
    
    print(f"\n✅ Invertible NN ET training complete!")
    print(f"📁 Results saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Invertible NN ET model")
    parser.add_argument("--data_file", type=str, default=None,
                      help="Path to training data file (default: data/easy_3d_gaussian.pkl)")
    
    args = parser.parse_args()
    main()