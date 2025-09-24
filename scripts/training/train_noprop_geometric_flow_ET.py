#!conda activate numpyro && python
"""
Training script for NoProp Geometric Flow ET model.

This script trains, evaluates, and plots results for a NoProp Geometric Flow ET network that uses
continuous-time diffusion (NoProp-CT) or flow matching (NoProp-FM) approaches combined with
geometric flow dynamics. The network learns flow dynamics:
    du/dt = A@A^T@(η_target - η_init)

where A is learned using NoProp continuous-time training protocols.

Usage:
    python scripts/training/train_noprop_geometric_flow_ET.py --data_file data/easy_3d_gaussian.pkl --epochs 300
"""

import sys
import argparse
from pathlib import Path
import json
import time
import pickle
import jax
import jax.numpy as jnp

# Add the project root to Python path for package imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import data and plotting functions
from src.utils.data_utils import infer_dimensions, load_ef_data
from scripts.plotting.plot_training_results import plot_training_results, plot_model_comparison, save_results_summary, create_standardized_results

# Import from training directory
from scripts.training.training_template_ET import ET_Template
from src.models.noprop_geometric_flow_ET import create_model_and_trainer
from src.config import NetworkConfig, TrainingConfig, FullConfig
from src.ef import ef_factory


class NoProp_Geometric_Flow_Net(ET_Template):
    """NoProp Geometric Flow ET using the ET template and geometric flow dynamics."""
    
    def __init__(self, hidden_sizes=None, eta_dim=None, num_time_steps=10, 
                 noise_schedule="flow_matching", matrix_rank=None, smoothness_weight=1e-3, loss_type="flow_matching"):
        super().__init__(hidden_sizes=hidden_sizes, eta_dim=eta_dim, model_type="noprop_geometric_flow")
        self.num_time_steps = num_time_steps
        self.noise_schedule = noise_schedule
        self.matrix_rank = matrix_rank
        self.smoothness_weight = smoothness_weight
        self.loss_type = loss_type
        self.ef = ef_factory('multivariate_normal', x_shape=(3,))
    
    def create_model_and_trainer(self, num_epochs=20):
        """Create the model and trainer using the NoProp geometric flow implementation."""
        
        # Use inferred dimensions from data
        if self.eta_dim is None:
            raise ValueError("eta_dim must be provided. Call infer_dimensions() first.")
        
        input_dim = self.eta_dim
        output_dim = self.eta_dim  # Same as input for ET models
        
        network_config = NetworkConfig(
            hidden_sizes=self.hidden_sizes,
            activation="swish",
            use_layer_norm=False,  # No layer norm for geometric flow
            input_dim=input_dim,
            output_dim=output_dim,
            # NoProp Geometric Flow specific parameters
            num_time_steps=self.num_time_steps,
            noise_schedule=self.noise_schedule,
            matrix_rank=self.matrix_rank,
            smoothness_weight=self.smoothness_weight,
            flow_matching_sigma=0.1
        )
        
        # Use optimal training configuration from template
        training_config = self.create_optimal_training_config(num_epochs)
        full_config = FullConfig(network=network_config, training=training_config)
        return create_model_and_trainer(full_config, self.loss_type)
    
    def predict(self, trainer, params, eta_data):
        """Make predictions using NoProp Geometric Flow trainer's predict method."""
        # For geometric flow, we need eta_init, eta_target, and mu_init
        # We'll use the same eta for both init and target for now
        eta_init = eta_data
        eta_target = eta_data
        
        # Use the known initial condition (zeros representing analytical point)
        mu_init = jnp.zeros_like(eta_data)
        
        # Update trainer's mlp_params with the passed params
        trainer.mlp_params = params
        return trainer.predict(eta_init, eta_target, mu_init)
    
    def benchmark_inference(self, trainer, params, eta_data, num_runs=50):
        """Benchmark inference time using geometric flow model."""
        import time
        
        # Prepare inputs for geometric flow
        eta_init = eta_data
        eta_target = eta_data
        mu_init = jnp.zeros_like(eta_data)
        
        # Warm-up run to ensure compilation is complete
        _ = trainer.predict(eta_init[:1], eta_target[:1], mu_init[:1])
        
        # Measure inference time over multiple runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = trainer.predict(eta_init, eta_target, mu_init)
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        per_sample_time = avg_time / len(eta_data)
        samples_per_second = 1.0 / per_sample_time
        
        return {
            'avg_inference_time': avg_time,
            'inference_per_sample': per_sample_time,
            'samples_per_second': samples_per_second
        }


def main():
    """Main training function for NoProp Geometric Flow ET models."""
    parser = argparse.ArgumentParser(description='Train NoProp Geometric Flow ET models')
    parser.add_argument('--data_file', type=str, help='Path to data file (default: data/easy_3d_gaussian.pkl)')
    parser.add_argument('--save_dir', type=str, default='artifacts/ET_models/noprop_geometric_flow_ET', help='Path to results dump directory')    
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--save_params', action='store_true', help='Save model parameters as pickle files')
    parser.add_argument('--time_steps', type=int, default=10, help='Number of time steps for inference')
    parser.add_argument('--matrix_rank', type=int, default=None, help='Rank of matrix A (default: output_dim)')
    parser.add_argument('--smoothness_weight', type=float, default=1e-3, help='Weight for smoothness penalty')
    parser.add_argument('--noise_schedule', type=str, default='flow_matching', choices=['noprop_ct', 'flow_matching'], 
                       help='Noise schedule: noprop_ct (defaults to geometric_flow loss) or flow_matching (defaults to flow_matching loss)')
    parser.add_argument('--variant', type=str, default='fm', choices=['ct', 'fm'], 
                       help='NoProp variant: ct (continuous time) or fm (flow matching) - convenience flag')
    parser.add_argument('--loss_type', type=str, default='auto', choices=['auto', 'geometric_flow', 'simple_target', 'flow_matching'], 
                       help='Loss type: auto (based on noise schedule), geometric_flow, simple_target, or flow_matching')
    
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
            loss_type = 'geometric_flow'  # Geometric flow loss for continuous time
        elif noise_schedule == 'flow_matching':
            loss_type = 'flow_matching'  # Flow matching loss
    else:
        loss_type = args.loss_type

    print("Training NoProp Geometric Flow ET Model")
    print("=" * 50)
    print(f"Noise Schedule: {noise_schedule.upper()}")
    print(f"Loss Type: {loss_type}")
    print(f"Variant: {args.variant.upper()} ({'Continuous Time' if args.variant == 'ct' else 'Flow Matching'})")
    print("=" * 50)
    
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
        print(f"\nTraining NoProp Geometric Flow ET {arch_name} with hidden sizes: {hidden_sizes}")
        
        # Create model with noprop-specific configuration
        model = NoProp_Geometric_Flow_Net(
            hidden_sizes=hidden_sizes,
            eta_dim=eta_dim,
            num_time_steps=args.time_steps,
            noise_schedule=noise_schedule,
            matrix_rank=args.matrix_rank,
            smoothness_weight=args.smoothness_weight,
            loss_type=loss_type
        )
        
        trainer = model.create_model_and_trainer(num_epochs=args.epochs)
        
        # Prepare training data for geometric flow
        # We need eta_init, eta_target, mu_init, mu_T
        # For geometric flow, we learn the flow from mu_init (known initial condition) to mu_T (target)
        # Use zeros as the known initial condition (representing the analytical point)
        train_geometric_flow = {
            'eta_init': train['eta'],
            'eta_target': train['eta'],  # Same as init for now
            'mu_init': jnp.zeros_like(train['mu_T']),  # Known initial condition (analytical point)
            'mu_T': train['mu_T']  # Target to reach
        }
        
        val_geometric_flow = {
            'eta_init': val['eta'],
            'eta_target': val['eta'],
            'mu_init': jnp.zeros_like(val['mu_T']),  # Known initial condition (analytical point)
            'mu_T': val['mu_T']  # Target to reach
        }
        
        # Train using inherited method from ET_Template
        t = time.time()
        params, losses = trainer.train(train_geometric_flow, val_geometric_flow, epochs=args.epochs)
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
                "matrix_rank": args.matrix_rank,
                "smoothness_weight": args.smoothness_weight,
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
    print("- Each layer trained independently")
    print("- No gradient flow between layers")
    print("- Geometric flow dynamics with PSD constraints")
    print("- Flow matching training protocols (default)")
    print("- Standard feedforward architecture")


if __name__ == "__main__":
    main()
