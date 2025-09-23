#!conda activate numpyro && python
"""
Training script for Standard MLP ET model.

This script trains, evaluates, and plots results for a Standard MLP that predicts expected 
sufficient statistics (mu_T = E[T(x)|η]) from natural parameters (eta).

Usage:
    python scripts/training/training_template_ET.py
"""

import sys
import argparse
from pathlib import Path
import json
import time
import jax

# Add the project root to Python path for package imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import data and plotting functions
from src.utils.data_utils import infer_dimensions, load_standardized_ep_data, load_ef_data
from scripts.plot_training_results import plot_training_results, plot_model_comparison, save_results_summary, create_standardized_results

# Import from training directory
from scripts.training.training_template_ET import ET_Template
from src.models.ET_Net import ETTrainer
from src.models.mlp_ET import MLP_ET_Network
from src.config import NetworkConfig, TrainingConfig, FullConfig

class Standard_MLP_Net(ET_Template):
    """Standard MLP ET using the new template."""
    
    def __init__(self, hidden_sizes = None, eta_dim = None):
        super().__init__(hidden_sizes=hidden_sizes, eta_dim=eta_dim, model_type="mlp")
    
    def create_model_and_trainer(self, num_epochs=20):
        """Get the model and training from the src/models/mlp_ET.py file."""
        network_config = NetworkConfig(
            hidden_sizes=self.hidden_sizes,
            activation="swish",
            use_layer_norm=True,
            input_dim=self.eta_dim,
            output_dim=self.eta_dim
        )
        training_config = TrainingConfig(num_epochs=num_epochs, learning_rate=1e-2)
        full_config = FullConfig(network=network_config, training=training_config)

        return ETTrainer(MLP_ET_Network(config=network_config),full_config)  # returns fully configured MLP_ET_Trainer

## Functions currently inherited from ET_Template class
# train, predict, evaluate, benchmark_inference


def main():
    """Main training and evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Train MLP ET models')
    parser.add_argument('--data_file', type=str, help='Path to data file (default: data/easy_3d_gaussian.pkl)')
    parser.add_argument('--save_dir', type=str, default='artifacts/ET_models/mlp_ET', help='Path to results dump directory')    
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    
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
        "Small": [32, 32],
        "Medium": [64, 64],
        "Large": [128, 128],
        "Deep": [64, 64, 64],
        "Wide": [128, 64, 128],
        "Max": [128, 128, 128]
    }
    
    print("Training Standard MLP ET Model")
    print("=" * 40)

    results = {}

    # Test the architecture
    for arch_name, hidden_sizes in architectures.items():
        print(f"\nTraining ET MLP {arch_name} with hidden sizes: {hidden_sizes}")
        
        # Infer dimensions from metadata
        model = Standard_MLP_Net(hidden_sizes=hidden_sizes, eta_dim=eta_dim)
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
        total_depth = len(hidden_sizes)
        
        print(test.keys())
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
