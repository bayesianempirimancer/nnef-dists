#!/usr/bin/env python3
"""
Training script template for ET (Expected Statistics) neural networks.

This script trains networks that directly predict the expected sufficient statistics
E[T(X)] of exponential family distributions.

Usage:
    python scripts/training/train_{model_name}_ET.py --config configs/gaussian_1d.yaml
    python scripts/training/train_{model_name}_ET.py --config configs/multivariate_3d.yaml
"""

import argparse
import sys
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
# matplotlib import removed - now using standardized plotting

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import standardized plotting functions
sys.path.append(str(Path(__file__).parent.parent))
from plot_training_results import plot_training_results, plot_model_comparison, save_results_summary

from src.config import FullConfig
# Import dimension inference utility and metadata-based data loading
from src.utils.data_utils import infer_dimensions, load_data_with_metadata

# Simple configuration class
class SimpleConfig:
    def __init__(self, hidden_sizes, activation="swish"):
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.use_layer_norm = True

def load_standard_data(data_file_override=None):
    """
    Standard data loading function for ET training scripts.
    Uses metadata-based loading to be completely dimension-agnostic.
    
    Args:
        data_file_override: Optional override for data file path
        
    Returns:
        Tuple of (train_data, val_data, metadata) where train_data and val_data 
        are dicts with 'eta' and 'y' keys ready for trainer.train()
    """
    # Use default data file if none specified
    data_file = data_file_override if data_file_override else "data/easy_3d_gaussian.pkl"
    
    # Load data using metadata (dimension-agnostic)
    data = load_data_with_metadata(data_file)
    
    # Prepare train and validation data in the format expected by BaseTrainer
    train_data = {
        'eta': data["train"]["eta"],
        'y': data["train"]["mu_T"]  # Use 'y' key for BaseTrainer compatibility
    }
    
    val_data = {
        'eta': data["val"]["eta"],
        'y': data["val"]["mu_T"]
    }
    
    metadata = data.get('metadata', {})
    
    print(f"Data shapes:")
    print(f"  Train: eta={train_data['eta'].shape}, y={train_data['y'].shape}")
    print(f"  Val: eta={val_data['eta'].shape}, y={val_data['y'].shape}")
    
    return train_data, val_data, metadata

class SimpleTemplateET:
    """Template ET using official implementation."""
    
    def __init__(self, hidden_sizes=[64, 64], eta_dim=None, model_type="mlp"):
        self.hidden_sizes = hidden_sizes
        self.eta_dim = eta_dim
        self.model_type = model_type
        self.config = SimpleConfig(hidden_sizes=hidden_sizes, activation="swish")
    
    def create_model_and_trainer(self):
        """Create the model and trainer using the official implementation."""
        from src.config import NetworkConfig, FullConfig, TrainingConfig
        
        # Use inferred dimensions from data
        # For ET models, input and output dimensions should be the same
        if self.eta_dim is None:
            raise ValueError("eta_dim must be provided. Call infer_dimensions() first.")
        
        input_dim = self.eta_dim
        output_dim = self.eta_dim  # Same as input for ET models
        
        network_config = NetworkConfig(
            hidden_sizes=self.hidden_sizes,
            activation="swish",
            use_layer_norm=True,
            input_dim=input_dim,
            output_dim=output_dim
        )
        training_config = TrainingConfig(num_epochs=300, learning_rate=1e-2)
        full_config = FullConfig(network=network_config, training=training_config)
        
        # Create trainer based on model type
        if self.model_type == "mlp" or self.model_type == "standard_mlp":
            from src.models.mlp_ET import create_model_and_trainer
            return create_model_and_trainer(full_config)
        elif self.model_type == "glu":
            from src.models.glu_ET import create_model_and_trainer
            return create_model_and_trainer(full_config)
        elif self.model_type == "quadratic_resnet":
            from src.models.quadratic_resnet_ET import create_model_and_trainer
            return create_model_and_trainer(full_config)
        else:
            available_models = ["mlp", "glu", "quadratic_resnet"]
            raise ValueError(f"Unknown model: {self.model_type}. Available models: {available_models}")
    
    def train(self, eta_data, ground_truth, epochs=300, learning_rate=1e-2):
        """Train the model and measure training time."""
        trainer = self.create_model_and_trainer()
        
        # Prepare training data
        # For ET models, ground_truth should be mu_T (same dimension as eta)
        train_data = {
            'eta': eta_data,
            'mu_T': ground_truth
        }
        
        # Measure training time
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
            'inference_per_sample': avg_time / len(eta_data),
            'samples_per_second': len(eta_data) / avg_time
        }
    
    def evaluate(self, predictions, ground_truth):
        """Evaluate predictions."""
        mse = jnp.mean((predictions - ground_truth) ** 2)
        mae = jnp.mean(jnp.abs(predictions - ground_truth))
        return {"mse": float(mse), "mae": float(mae)}

def main():
    """Main training function."""
    print("Training Template ET Model")
    print("=" * 40)
    
    # Load data using standard metadata-based loading (dimension-agnostic)
    eta_data, ground_truth, metadata = load_standard_data()
    
    # Test single deep architecture
    architectures = {
        'Template_ET_Deep': [64, 64, 64, 64, 64, 64, 64, 64],
    }
    
    results = {}
    
    for name, hidden_sizes in architectures.items():
        print(f"\nTraining {name} with hidden sizes: {hidden_sizes}")
        
        # Infer dimensions from metadata
        eta_dim = infer_dimensions(eta_data, metadata=metadata)
        model = SimpleTemplateET(hidden_sizes=hidden_sizes, eta_dim=eta_dim, model_type="mlp")
        
        trainer = model.create_model_and_trainer()
        
        # Train and measure training time
        params, losses, training_time = model.train(eta_data, ground_truth, epochs=300)
        
        # Evaluate
        predictions = model.predict(trainer, params, eta_data)
        metrics = model.evaluate(predictions, ground_truth)
        
        # Benchmark inference time
        inference_stats = model.benchmark_inference(trainer, params, eta_data, num_runs=50)
        
        print(f"  Final MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Avg inference time: {inference_stats['avg_inference_time']:.4f}s ({inference_stats['samples_per_second']:.1f} samples/sec)")
        
        # Create plots using standardized plotting function
        metrics_for_plotting = plot_training_results(
            trainer=model,
            eta_data=eta_data,
            ground_truth=ground_truth,
            predictions=predictions,
            losses=losses,
            config=model.config,
            model_name=name,
            output_dir="artifacts/ET_models/template_ET",
            save_plots=True,
            show_plots=False
        )
        
        results[name] = {
            'mse': metrics['mse'], 
            'mae': metrics['mae'], 
            'final_loss': metrics_for_plotting.get('final_loss', losses[-1] if losses else 0),
            'hidden_sizes': hidden_sizes,
            'losses': losses,
            'training_time': training_time,
            'avg_inference_time': inference_stats['avg_inference_time'],
            'inference_per_sample': inference_stats['inference_per_sample'],
            'samples_per_second': inference_stats['samples_per_second']
        }
        
        print(f"  Final MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}")
    
    # Summary
    print(f"\n{'='*60}")
    print("TEMPLATE ET MODEL RESULTS")
    print(f"{'='*60}")
    
    for model_name, result in results.items():
        print(f"{model_name:<20} : MSE={result['mse']:.6f}, MAE={result['mae']:.6f}, Architecture={result['hidden_sizes']}")
        print(f"{'':<20}   Training: {result['training_time']:.2f}s, Inference: {result['samples_per_second']:.1f} samples/sec")
    
    # Create output directory
    output_dir = Path("artifacts/ET_models/template_ET")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model comparison plots using standardized plotting function
    plot_model_comparison(
        results=results,
        output_dir=str(output_dir),
        save_plots=True,
        show_plots=False
    )
    
    # Save results
    import json
    with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    # Save results summary using standardized function
    save_results_summary(
        results=results,
        output_dir=str(output_dir)
    )
        
    print(f"\n✅ Template ET training complete!")
    print(f"📁 Results saved to {output_dir}")
    

if __name__ == "__main__":
    main()
