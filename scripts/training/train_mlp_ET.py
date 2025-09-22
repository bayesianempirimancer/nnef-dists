#!/usr/bin/env python3
"""
Training script for Standard MLP model.

This script trains, evaluates, and plots results for a Standard MLP
on the natural parameter to statistics mapping task.
"""

import sys
from pathlib import Path
import time
import json
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
# matplotlib import removed - now using standardized plotting

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

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

class SimpleStandardMLPET(SimpleTemplateET):
    """Standard MLP ET using the new template."""
    
    def __init__(self, hidden_sizes=[64, 64], eta_dim=None):
        super().__init__(hidden_sizes=hidden_sizes, eta_dim=eta_dim, model_type="mlp")
    
    # All methods inherited from SimpleTemplateET
    pass


def main():
    """Main training and evaluation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MLP ET models')
    parser.add_argument('--data_file', type=str, help='Path to data file (default: data/easy_3d_gaussian.pkl)')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    
    args = parser.parse_args()
    
    print("Training Standard MLP ET Model")
    print("=" * 40)
    
    # Load data using standardized template function
    data_file = args.data_file if args.data_file else "data/easy_3d_gaussian.pkl"
    eta_data, ground_truth, metadata = load_standardized_ep_data(data_file)
    
    # Define architectures to test (all variants for comprehensive comparison)
    architectures = {
        "Small": [32, 32],
        "Medium": [64, 64],
        "Large": [128, 128],
        "Deep": [64, 64, 64],
        "Wide": [128, 64, 128],
        "Max": [128, 128, 128]
    }
    
    results = {}
    
    # Test the architecture
    for arch_name, hidden_sizes in architectures.items():
        print(f"\nTraining MLP_ET_{arch_name} with hidden sizes: {hidden_sizes}")
        
        # Infer dimensions from metadata
        eta_dim = infer_dimensions(eta_data, metadata=metadata)
        model_wrapper = SimpleStandardMLPET(hidden_sizes=hidden_sizes, eta_dim=eta_dim)
        
        trainer = model_wrapper.create_model_and_trainer()
        
        # Train and measure training time
        params, losses, training_time = model_wrapper.train(eta_data, ground_truth, epochs=300)
        
        # Evaluate accuracy
        predictions = model_wrapper.predict(trainer, params, eta_data)
        metrics = model_wrapper.evaluate(predictions, ground_truth)
        
        # Benchmark inference time
        inference_stats = model_wrapper.benchmark_inference(trainer, params, eta_data, num_runs=50)
        
        results[f"MLP_ET_{arch_name}"] = create_standardized_results(
            model_name=f"MLP_ET_{arch_name}",
            architecture_info={"hidden_sizes": hidden_sizes},
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
    
    # Print summary
    print("\n" + "=" * 60)
    print("STANDARD MLP ET MODEL RESULTS")
    print("=" * 60)
    
    for model_name, result in results.items():
        print(f"{model_name:<20} : MSE={result['mse']:.6f}, MAE={result['mae']:.6f}, Architecture={result['hidden_sizes']}")
        print(f"{'':<20}   Training: {result['training_time']:.2f}s, Inference: {result['samples_per_second']:.1f} samples/sec")
    
    print(f"\n✅ Model training completed successfully!")
    
    # Create output directory
    output_dir = Path("artifacts/ET_models/mlp_ET")
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
        
    print(f"\n✅ Standard MLP ET training complete!")
    print(f"📁 Results saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MLP ET models')
    parser.add_argument('--data_file', type=str, help='Path to data file (default: data/easy_3d_gaussian.pkl)')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    
    args = parser.parse_args()
    main()
