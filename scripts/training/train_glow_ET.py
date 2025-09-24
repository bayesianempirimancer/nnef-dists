#!conda activate numpyro && python
"""
Training script for Standard Glow ET model.

This script trains, evaluates, and plots results for a Standard Glow that predicts expected 
sufficient statistics (mu_T = E[T(x)|η]) from natural parameters (eta).

Usage:
    python scripts/training/train_glow_ET.py
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
from src.models.glow_net_ET import Glow_ET_Network
from src.config import NetworkConfig, TrainingConfig, FullConfig, ModelSpecificConfig


class Standard_Glow_Net(ET_Template):
    """Standard Glow ET using the new template."""
    
    def __init__(self, hidden_sizes = None, eta_dim = None, num_flow_layers = 10, flow_hidden_size = 64):
        super().__init__(hidden_sizes=hidden_sizes, eta_dim=eta_dim, model_type="glow")
        self.num_flow_layers = num_flow_layers
        self.flow_hidden_size = flow_hidden_size
    
    def create_model_and_trainer(self, num_epochs=20):
        """Get the model and training from the src/models/glow_net_ET.py file."""
        network_config = NetworkConfig(
            hidden_sizes=self.hidden_sizes,
            activation="swish",
            use_layer_norm=True,
            input_dim=self.eta_dim,
            output_dim=self.eta_dim
        )
        # Use optimal training configuration from template
        training_config = self.create_optimal_training_config(num_epochs)
        
        # Create model-specific config for flow parameters
        model_specific_config = ModelSpecificConfig(
            num_flow_layers=self.num_flow_layers,
            flow_hidden_size=self.flow_hidden_size
        )
        
        full_config = FullConfig(
            network=network_config, 
            training=training_config,
            model_specific=model_specific_config
        )

        return ETTrainer(Glow_ET_Network(config=network_config), full_config)  # returns fully configured Glow_ET_Trainer

## Functions currently inherited from ET_Template class
# train, predict, evaluate, benchmark_inference


def main():
    """Main training and evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Train Glow ET models')
    parser.add_argument('--data_file', type=str, help='Path to data file (default: data/easy_3d_gaussian.pkl)')
    parser.add_argument('--save_dir', type=str, default='artifacts/ET_models/glow_ET', help='Path to results dump directory')    
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--save_params', action='store_true', help='Save model parameters as pickle files')
    
    args = parser.parse_args()

    # Load data using standardized template function
    if args.data_file is None:
        raise ValueError("data_file must be provided.")
    train, val, test, metadata = load_ef_data(args.data_file)

    # Create output directory
    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

#    eta_data, ground_truth, metadata = load_standardized_ep_data(args.data_file)
    eta_dim = infer_dimensions(train["eta"], metadata=metadata)
        
    # Define architectures to test (all variants for comprehensive comparison)
    architectures = {
        "Small": ([32, 32], 5, 32),
        "Medium": ([64, 64], 10, 64),
        "Large": ([128, 128], 15, 64),
        "Deep": ([64, 64, 64], 20, 32),
        "Wide": ([128, 64, 128], 10, 64),
        "Max": ([128, 128, 128], 25, 64)
    }
    
    print("Training Standard Glow ET Model")
    print("=" * 40)

    results = {}

    # Test the architecture
    for arch_name, (hidden_sizes, num_flow_layers, flow_hidden_size) in architectures.items():
        print(f"\nTraining ET Glow {arch_name} with hidden sizes: {hidden_sizes}, flow layers: {num_flow_layers}, flow hidden: {flow_hidden_size}")
        
        # Infer dimensions from metadata
        model = Standard_Glow_Net(hidden_sizes=hidden_sizes, eta_dim=eta_dim, num_flow_layers=num_flow_layers, flow_hidden_size=flow_hidden_size)
        trainer = model.create_model_and_trainer(num_epochs=args.epochs)
        
        # Train and measure training time
        t=time.time()
        params, losses = trainer.train(train, val, epochs=args.epochs)
        training_time = time.time() - t

        # Evaluate accuracy
        predictions = model.predict(trainer, params, test['eta'])
        metrics = model.evaluate(predictions, test['mu_T'])
        
        # Benchmark inference time
        inference_stats = model.benchmark_inference(trainer, params, val['eta'], num_runs=50)
        
        # Calculate parameter count and total depth
        param_count = sum(p.size for p in jax.tree_util.tree_flatten(params)[0])
        total_depth = len(hidden_sizes) + num_flow_layers  # Base network + flow layers
        
        print(test.keys())
        results[f"{model.model_type}_ET_{arch_name}"] = create_standardized_results(
            model_name=f"{model.model_type}_ET_{arch_name}",
            architecture_info={
                "hidden_sizes": hidden_sizes,
                "flow_layers": num_flow_layers,
                "flow_hidden_size": flow_hidden_size,
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
    
    print(f"\n✅ Model training completed successfully!")
        
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
        
    print(f"\n✅ Standard {model.model_type.upper()} ET training complete!")
    print(f"📁 Results saved to {output_dir}")


if __name__ == "__main__":
    main()
