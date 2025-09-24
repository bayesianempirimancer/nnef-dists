#!conda activate numpyro && python
"""
Training script for NoProp ET model.

This script trains, evaluates, and plots results for a NoProp ET network that uses
either continuous-time diffusion (NoProp-CT) or flow matching (NoProp-FM) approaches.
The network learns to predict expected sufficient statistics (mu_T = E[T(x)|η]) 
from natural parameters (eta) using layer-wise training without backpropagation.

Usage:
    python scripts/training/train_noprop_ct_ET.py --data_file data/easy_3d_gaussian.pkl --epochs 300
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
from src.utils.data_utils import infer_dimensions, load_ef_data
from scripts.plotting.plot_training_results import plot_training_results, plot_model_comparison, save_results_summary, create_standardized_results

# Import from training directory
from scripts.training.training_template_ET import ET_Template
from src.models.noprop_ct_ET import create_model_and_trainer
from src.config import NetworkConfig, TrainingConfig, FullConfig


class NoProp_Net(ET_Template):
    """NoProp ET using the ET template and proper noprop algorithm implementation."""
    
    def __init__(self, hidden_sizes=None, eta_dim=None, num_time_steps=10, 
                 noise_schedule="noprop_ct", max_noise=1.0, loss_type="simple_target"):
        super().__init__(hidden_sizes=hidden_sizes, eta_dim=eta_dim, model_type="noprop")
        self.num_time_steps = num_time_steps
        self.noise_schedule = noise_schedule
        self.max_noise = max_noise
        self.loss_type = loss_type
    
    def create_model_and_trainer(self, num_epochs=20):
        """Create the model and trainer using the true noprop implementation."""
        
        # Use inferred dimensions from data
        if self.eta_dim is None:
            raise ValueError("eta_dim must be provided. Call infer_dimensions() first.")
        
        input_dim = self.eta_dim
        output_dim = self.eta_dim  # Same as input for ET models
        
        network_config = NetworkConfig(
            hidden_sizes=self.hidden_sizes,
            activation="swish",
            use_layer_norm=False,  # No layer norm for noprop
            input_dim=input_dim,
            output_dim=output_dim,
            # NoProp-CT specific parameters
            num_time_steps=self.num_time_steps,
            noise_schedule=self.noise_schedule,
            max_noise=self.max_noise,
            # ResNet configuration
            use_resnet=True,  # Enable ResNet by default
            resnet_skip_every=2  # Skip connection every 2 layers
        )
        
        # Use optimal training configuration from template
        training_config = self.create_optimal_training_config(num_epochs)
        full_config = FullConfig(network=network_config, training=training_config)
        return create_model_and_trainer(full_config, self.loss_type)
    
    def predict(self, trainer, params, eta_data):
        """Make predictions using NoProp-CT trainer's predict method."""
        # Update trainer's mlp_params with the passed params
        trainer.mlp_params = params
        return trainer.predict(eta_data)
    
    def benchmark_inference(self, trainer, params, eta_data, num_runs=50):
        """Benchmark inference time with multiple runs for accuracy."""
        # Update trainer's mlp_params with the passed params
        trainer.mlp_params = params
        
        # Warm-up run to ensure compilation is complete
        _ = trainer.predict(eta_data[:1])
        
        # Measure inference time over multiple runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = trainer.predict(eta_data)
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


def main():
    """Main training and evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Train NoProp-CT ET models')
    parser.add_argument('--data_file', type=str, help='Path to data file (default: data/easy_3d_gaussian.pkl)')
    parser.add_argument('--save_dir', type=str, default='artifacts/ET_models/noprop_ct_ET', help='Path to results dump directory')    
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--save_params', action='store_true', help='Save model parameters as pickle files')
    parser.add_argument('--time_steps', type=int, default=10, help='Number of time steps for diffusion')
    parser.add_argument('--noise_schedule', type=str, default='noprop_ct', choices=['noprop_ct', 'flow_matching'], 
                       help='Noise schedule: noprop_ct (defaults to noprop loss) or flow_matching (defaults to flow_matching loss)')
    parser.add_argument('--variant', type=str, default='ct', choices=['ct', 'fm'], 
                       help='NoProp variant: ct (continuous time) or fm (flow matching) - convenience flag')
    parser.add_argument('--loss_type', type=str, default='auto', choices=['auto', 'simple_target', 'noprop', 'flow_matching'], 
                       help='Loss type: auto (based on noise schedule), simple_target, noprop, or flow_matching')
    
    args = parser.parse_args()
    
    # Determine noise schedule and loss type based on arguments
    noise_schedule = args.noise_schedule
    
    # If variant is specified, use it to set noise schedule (unless explicitly overridden)
    if args.variant == 'fm' and args.noise_schedule == 'noprop_ct':
        noise_schedule = 'flow_matching'
    elif args.variant == 'ct' and args.noise_schedule == 'flow_matching':
        noise_schedule = 'noprop_ct'
    
    # Determine loss type based on noise schedule (unless explicitly overridden)
    if args.loss_type == 'auto':
        if noise_schedule == 'noprop_ct':
            loss_type = 'noprop'  # Standard NoProp loss for continuous time
        elif noise_schedule == 'flow_matching':
            loss_type = 'flow_matching'  # Flow matching loss
    else:
        loss_type = args.loss_type

    print("Training NoProp ET Model")
    print("=" * 40)
    print(f"Noise Schedule: {noise_schedule.upper()}")
    print(f"Loss Type: {loss_type}")
    print(f"Variant: {args.variant.upper()} ({'Continuous Time' if args.variant == 'ct' else 'Flow Matching'})")
    print("=" * 40)
    
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
    
    results = {}
    
    # Test the architectures
    for arch_name, hidden_sizes in architectures.items():
        print(f"\nTraining NoProp ET {arch_name} with hidden sizes: {hidden_sizes}")
        
        # Create model with noprop-specific configuration
        model = NoProp_Net(
            hidden_sizes=hidden_sizes,
            eta_dim=eta_dim,
            num_time_steps=args.time_steps,
            noise_schedule=noise_schedule,
            max_noise=1.0,
            loss_type=loss_type
        )
        
        trainer = model.create_model_and_trainer(num_epochs=args.epochs)
        
        # Train using inherited method from ET_Template
        t = time.time()
        params, losses = trainer.train(train, val, epochs=args.epochs)
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
                "num_time_steps": args.time_steps,
                "noise_schedule": noise_schedule,
                "loss_type": loss_type,
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
        print(f"{model_name:<20} : MSE={result['mse']:.6f}, MAE={result['mae']:.6f}, Architecture={result['hidden_sizes']}")
        print(f"{'':<20}   Training: {result['training_time']:.2f}s, Inference: {result['samples_per_second']:.1f} samples/sec")
    
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
    print("- Each layer trained independently")
    print("- No gradient flow between layers")
    print("- Diffusion-based denoising objectives")
    print("- Time-step specific training targets")


if __name__ == "__main__":
    main()