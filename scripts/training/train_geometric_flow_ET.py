#!conda activate numpyro && python
"""
Training script for Geometric Flow ET model.

This script trains, evaluates, and plots results for a Geometric Flow that take in a 
target value for eta.  Finds a 'nearby' value of eta for which the statistic is known 
analytically, and then evolves a dynamical system to determine the value of the sufficient
statistic at the targeted value.  Thus the network has two bits.  One is a function that 
takes in the target eta_1 and analytically computes eta_0 and mu_0 to get the 
initial conditions for the dynamics.  The second is a neuralzed dynamical system that 
evolves from time t=0 to t=1 evolves the initial condition mu_0 to mu_1.  

Usage:
    python scripts/training/train_geometric_flow_ET.py
"""

import sys
import argparse
from pathlib import Path
import json
import time
import pickle
import jax

# Add the project root to Python path for package imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import data and plotting functions
from src.utils.data_utils import infer_dimensions, load_standardized_ep_data, load_ef_data
from scripts.plotting.plot_training_results import plot_training_results, plot_model_comparison, save_results_summary, create_standardized_results

# Import from training directory
from scripts.training.training_template_ET import ET_Template
from src.models.ET_Net import ETTrainer
from src.models.geometric_flow_net import Geometric_Flow_ET_Network, Geometric_Flow_ET_Trainer
from src.config import NetworkConfig, TrainingConfig, FullConfig, ModelSpecificConfig
from src.ef import ef_factory

class Geometric_Flow_Net(ET_Template):
    """Standard Geometric Flow ET using the new template."""
    
    def __init__(self, hidden_sizes=None, eta_dim=None, ef=None, matrix_rank=None, n_time_steps=10, smoothness_weight=1e-3):
        super().__init__(hidden_sizes=hidden_sizes, eta_dim=eta_dim, model_type="geometric_flow")
        self.matrix_rank = matrix_rank
        self.n_time_steps = n_time_steps
        self.smoothness_weight = smoothness_weight
        self.ef = ef or ef_factory('multivariate_normal', x_shape=(3,))
    
    def create_model_and_trainer(self, num_epochs=20):
        """Get the model and training from the src/models/geometric_flow_net.py file."""
        network_config = NetworkConfig(
            hidden_sizes=self.hidden_sizes,
            activation="swish",
            use_layer_norm=False,
            input_dim=self.eta_dim,
            output_dim=self.eta_dim
        )
        # Use optimal training configuration from template
        training_config = self.create_optimal_training_config(num_epochs)
        
        # Create model-specific config with geometric flow specific parameters
        model_specific_config = ModelSpecificConfig(
            matrix_rank=self.matrix_rank,
            n_time_steps=self.n_time_steps,
            smoothness_weight=self.smoothness_weight
        )
        
        full_config = FullConfig(
            network=network_config, 
            training=training_config,
            model_specific=model_specific_config
        )

        # Create the specialized geometric flow trainer
        trainer = Geometric_Flow_ET_Trainer(
            config=full_config,
            matrix_rank=self.matrix_rank,
            n_time_steps=self.n_time_steps,
            smoothness_weight=self.smoothness_weight
        )
        
        return trainer  # returns fully configured Geometric_Flow_ET_Trainer

    def benchmark_inference(self, trainer, params, eta_data, num_runs=50):
        """Benchmark inference time for geometric flow model."""
        import time
        
        # Warm-up run to ensure compilation is complete
        _ = self.predict(trainer, params, eta_data[:1])
        
        # Measure inference time over multiple runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = self.predict(trainer, params, eta_data)
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
            'inference_per_sample': avg_time / len(eta_data)
        }

    def predict(self, trainer, params, eta_data):
        """Make predictions using geometric flow trainer's predict method."""
        predictions_dict = trainer.predict(params, eta_data)
        return predictions_dict['mu_predicted']  # Extract the actual predictions

## Functions currently inherited from ET_Template class
# train, evaluate


def main():
    """Main training and evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Train Geometric Flow ET models')
    parser.add_argument('--data_file', type=str, help='Path to data file (default: data/easy_3d_gaussian.pkl)')
    parser.add_argument('--save_dir', type=str, default='artifacts/ET_models/geometric_flow_ET', help='Path to results dump directory')    
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--save_params', action='store_true', help='Save model parameters as pickle files')
    parser.add_argument('--ef_type', type=str, default='multivariate_normal', help='Exponential family type (default: multivariate_normal)')
    parser.add_argument('--x_shape', type=int, nargs='+', default=[3], help='Shape of x for exponential family (default: [3])')
    
    args = parser.parse_args()

    # Create exponential family from command line arguments
    ef = ef_factory(args.ef_type, x_shape=tuple(args.x_shape))

    # Load data using standardized template function
    if args.data_file is None:
        raise ValueError("data_file must be provided.")
    train, val, test, metadata = load_ef_data(args.data_file)

    # Create output directory
    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eta_dim = infer_dimensions(train["eta"], metadata=metadata)
        
    # Define architectures to test (all variants for comprehensive comparison)
    architectures = {
        "Small": [32, 32, 32],
        "Medium": [64, 64],
        "Large": [128, 128],
        "Deep": [64, 64, 64],
        "Wide": [128, 64, 128],
        "Max": [128, 128, 128]
    }
    
    print("Training Standard Geometric Flow ET Model")
    print("=" * 40)
    
    results = {}
    
    # Test the architectures
    for arch_name, hidden_sizes in architectures.items():
        print(f"\nTraining Geometric Flow ET {arch_name} with hidden sizes: {hidden_sizes}")
        
        # Create model with geometric flow specific configuration
        model = Geometric_Flow_Net(
            hidden_sizes=hidden_sizes, 
            eta_dim=eta_dim, 
            matrix_rank=None,  # Use default rank
            n_time_steps=10,   # Use default time steps
            smoothness_weight=1e-3  # Use default smoothness weight
        )
        trainer = model.create_model_and_trainer(num_epochs=args.epochs)
        
        # Set up exponential family for geometric flow trainer
        trainer.set_exponential_family(ef)

        # Train and measure training time
        t = time.time()
        params, losses = trainer.train(train['eta'], val['eta'], epochs=args.epochs)
        training_time = time.time() - t

        # Evaluate using inherited methods
        predictions = model.predict(trainer, params, test['eta'])
        metrics = model.evaluate(predictions, test['mu_T'])
        
        # Benchmark inference time
        inference_stats = model.benchmark_inference(trainer, params, val['eta'], num_runs=50)
        
        # Calculate parameter count and total depth
        param_count = sum(p.size for p in jax.tree_util.tree_flatten(params)[0])
        total_depth = len(hidden_sizes)
        
        results[f"{model.model_type}_ET_{arch_name}"] = create_standardized_results(
            model_name=f"{model.model_type}_ET_{arch_name}",
            architecture_info={
                "hidden_sizes": hidden_sizes,
                "total_depth": total_depth,
                "parameter_count": param_count
            },
            metrics=metrics,
            losses=losses['train_loss'],
            training_time=training_time,
            inference_stats=inference_stats,
            predictions=predictions,
            ground_truth=test['mu_T']
        )
                
        print(f"  Final MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Avg inference time: {inference_stats['avg_inference_time']:.4f}s ({inference_stats['samples_per_second']:.1f} samples/sec)")
        
        # Save model artifacts by default
        model_name = f"{model.model_type}_ET_{arch_name}"
        saved_files = model.save_model_artifacts(trainer, params, model_name, args.save_dir)
        
        # Model artifacts are now saved by default via template method
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"ET {model.model_type.upper()} MODEL RESULTS")
    print("=" * 60)
    
    for model_name, result in results.items():
        print(f"{model_name:<30} : MSE={result['mse']:.6f}, MAE={result['mae']:.6f}, Architecture={result['hidden_sizes']}")
        print(f"{'':<30}   Training: {result['training_time']:.2f}s, Inference: {result['samples_per_second']:.1f} samples/sec")
    
    # Find best model
    best_model = min(results.keys(), key=lambda k: results[k]['mse'])
    print(f"\n🏆 Best Model: {best_model} with MSE={results[best_model]['mse']:.6f}")
    
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
    
    print(f"\n✅ {model.model_type.upper()} ET training complete!")
    print(f"📁 Results saved to {output_dir}")
    print("\nKey differences from standard backpropagation:")
    print("- Geometric flow dynamics with PSD constraints")
    print("- Flow from known initial conditions to target statistics")
    print("- Neural ODE-based evolution of sufficient statistics")


if __name__ == "__main__":
    main()
