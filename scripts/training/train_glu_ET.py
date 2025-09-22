#!/usr/bin/env python3
"""
Training script for GLU ET model.

This script trains, evaluates, and plots results for a GLU ET Network
on the natural parameter to statistics mapping task.
"""

import sys
import json
import pickle
import time
from pathlib import Path
import jax
import jax.numpy as jnp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import standardized plotting functions
sys.path.append(str(Path(__file__).parent.parent))
from plot_training_results import plot_training_results, plot_model_comparison, save_results_summary, create_standardized_results

# Import dimension inference utility and standardized data loading from template
from src.utils.data_utils import infer_dimensions
sys.path.append(str(Path(__file__).parent))
from training_template_ET import SimpleTemplateET
from src.utils.data_utils import load_standardized_ep_data

# Simple configuration class
class SimpleConfig:
    def __init__(self, hidden_sizes, activation="swish"):
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.use_layer_norm = False



class SimpleGLUET(SimpleTemplateET):
    """GLU ET using the new template with minimal GLU-specific adapter."""
    
    def __init__(self, hidden_sizes=[64, 64], eta_dim=None):
        super().__init__(hidden_sizes=hidden_sizes, eta_dim=eta_dim, model_type="glu")
    
    # train() method now inherited from SimpleTemplateET (uses standard mu_T key)
    
    def predict(self, trainer, params, eta_data):
        """Make predictions using GLU-specific model.apply() method."""
        return trainer.model.apply(params, eta_data)
    
    def benchmark_inference(self, trainer, params, eta_data, num_runs=50):
        """Benchmark inference time using GLU-specific model.apply() method."""
        # Warm-up run to ensure compilation is complete
        _ = trainer.model.apply(params, eta_data[:1])
        
        # Measure inference time over multiple runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = trainer.model.apply(params, eta_data)
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


def main():
    """Main training and evaluation pipeline."""
    print("Training Standard GLU ET Model")
    print("=" * 40)
    
    # Load data using standardized template function
    eta_data, ground_truth, metadata = load_standardized_ep_data()
    
    # Define architecture to test (deep 8-layer network)
    architectures = {
        "Deep": [64, 64, 64, 64, 64, 64, 64, 64]
    }
    
    results = {}
    best_mse = float('inf')
    best_model = None
    
    for name, hidden_sizes in architectures.items():
        print(f"\nTraining GLU_ET_{name} with hidden sizes: {hidden_sizes}")
        
        # Infer dimensions from metadata
        eta_dim = infer_dimensions(eta_data, metadata=metadata)
        model_wrapper = SimpleGLUET(hidden_sizes=hidden_sizes, eta_dim=eta_dim)
        
        trainer = model_wrapper.create_model_and_trainer()
        
        # Train and measure training time
        params, losses, training_time = model_wrapper.train(eta_data, ground_truth, epochs=300)
        
        # Evaluate accuracy
        predictions = model_wrapper.predict(trainer, params, eta_data)
        metrics = model_wrapper.evaluate(predictions, ground_truth)
        
        # Benchmark inference time
        inference_stats = model_wrapper.benchmark_inference(trainer, params, eta_data, num_runs=50)
        
        print(f"GLU_ET_{name} - MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Avg inference time: {inference_stats['avg_inference_time']:.4f}s ({inference_stats['samples_per_second']:.1f} samples/sec)")
        
        # Store results using standardized function
        results[f"GLU_ET_{name}"] = create_standardized_results(
            model_name=f"GLU_ET_{name}",
            architecture_info={"hidden_sizes": hidden_sizes},
            metrics=metrics,
            losses=losses,
            training_time=training_time,
            inference_stats=inference_stats,
            predictions=predictions,
            ground_truth=ground_truth
        )
        
        # Track best model
        if metrics['mse'] < best_mse:
            best_mse = metrics['mse']
            best_model = f"GLU_ET_{name}"
    
    print(f"\n🏆 Best Model: {best_model} with MSE={best_mse:.6f}")
    
    # Create output directory
    output_dir = Path("artifacts/ET_models/glu_ET")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model comparison plots using standardized plotting function
    plot_model_comparison(
        results=results,
        output_dir=str(output_dir),
        save_plots=True,
        show_plots=False
    )
    
    # Save results
    with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    # Save results summary using standardized function
    save_results_summary(
        results=results,
        output_dir=str(output_dir)
    )
        
    print(f"\n✅ Standard GLU ET training complete!")
    print(f"📁 Results saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train GLU ET models')
    parser.add_argument('--data_file', type=str, help='Path to data file (default: data/easy_3d_gaussian.pkl)')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    
    args = parser.parse_args()
    main()
