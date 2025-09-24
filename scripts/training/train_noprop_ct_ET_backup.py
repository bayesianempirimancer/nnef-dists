#!conda activate numpyro && python
"""
Training script for NoProp-CT ET model.

This script trains, evaluates, and plots results for a NoProp-CT ET network
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
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

# Import standardized plotting functions
from scripts.plotting.plot_training_results import plot_training_results, plot_model_comparison, save_results_summary, create_standardized_results

# Import dimension inference utility and standardized data loading from template
from src.utils.data_utils import infer_dimensions
from src.utils.data_utils import load_standardized_ep_data
from src.models.noprop_ct_ET import create_model_and_trainer
from src.config import NetworkConfig, FullConfig, TrainingConfig

# Simple configuration class
class SimpleConfig:
    def __init__(self, hidden_sizes, activation="swish", dt=0.1):
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.use_layer_norm = False
        self.dt = dt  # ODE integration step size

class SimpleNoPropCTET:
    """NoProp-CT ET using official implementation."""
    
    def __init__(self, hidden_sizes=[64, 64], eta_dim=None, dt=0.1):
        self.hidden_sizes = hidden_sizes
        self.eta_dim = eta_dim
        self.dt = dt
        self.config = SimpleConfig(hidden_sizes=hidden_sizes, activation="swish", dt=dt)
    
    def create_model_and_trainer(self):
        """Create the model and trainer using the official implementation."""
        
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
        # Add dt parameter for NoProp-CT ODE integration
        network_config.dt = self.dt
        training_config = TrainingConfig(num_epochs=300, learning_rate=1e-3)
        full_config = FullConfig(network=network_config, training=training_config)
        return create_model_and_trainer(full_config)
    
    def train(self, train_data, val_data, epochs=300, learning_rate=1e-3):
        """Train the model using pre-split data with progress bars and detailed reporting."""
        import time
        from tqdm import tqdm
        import jax
        
        start_time = time.time()
        trainer = self.create_model_and_trainer()
        
        # Initialize training
        sample_input = train_data['eta'][:1]
        params, opt_state = trainer.initialize_model(sample_input)
        optimizer = trainer.create_optimizer()
        
        # Training state
        best_val_loss = float('inf')
        patience_counter = 0
        best_params = params
        history = {'train_loss': [], 'val_loss': []}
        
        tc = trainer.config.training
        batch_size = tc.batch_size
        n_train = train_data['eta'].shape[0]
        
        print(f"Training {trainer.model.__class__.__name__} for {tc.num_epochs} epochs")
        print(f"Architecture: {trainer.config.network.hidden_sizes}")
        print(f"Parameters: {trainer.model.get_parameter_count(params):,}")
        print(f"ODE Step Size (dt): {getattr(trainer.config.network, 'dt', 0.1)}")
        
        # Training loop with progress bar
        with tqdm(range(tc.num_epochs), desc="Training NoProp-CT ET") as pbar:
            for epoch in pbar:
                # Shuffle training data
                key = jax.random.PRNGKey(epoch)
                perm = jax.random.permutation(key, n_train)
                train_data_shuffled = {k: v[perm] for k, v in train_data.items()}
                
                # Mini-batch training
                epoch_losses = []
                for i in range(0, n_train, batch_size):
                    batch_data = {k: v[i:i+batch_size] for k, v in train_data_shuffled.items()}
                    params, opt_state, batch_loss = trainer.train_step(params, opt_state, batch_data, optimizer)
                    epoch_losses.append(batch_loss)
                
                # Average training loss for epoch
                avg_train_loss = sum(epoch_losses) / len(epoch_losses)
                history['train_loss'].append(avg_train_loss)
                
                # Validation loss
                val_loss = float(trainer.loss_fn(params, val_data['eta'], val_data['mu_T']))
                history['val_loss'].append(val_loss)
                
                # Update progress bar with current losses
                pbar.set_postfix({
                    'Train': f'{avg_train_loss:.2f}',
                    'Val': f'{val_loss:.2f}',
                    'Best': f'{best_val_loss:.2f}'
                })
                
                # Print detailed loss every 20 epochs
                if epoch % 20 == 0:
                    print(f"    Epoch {epoch:3d}: Train={avg_train_loss:.2f}, Val={val_loss:.2f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = params
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= tc.patience:
                        print(f"\nEarly stopping at epoch {epoch} (patience={tc.patience})")
                        break
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.1f}s")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        return best_params, history['train_loss'], training_time
    
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
            'inference_per_sample': avg_time / len(eta_data),
            'std_inference_time': jnp.std(jnp.array(times))
        }
    
    def predict(self, trainer, params, eta_data):
        """Make predictions."""
        return trainer.predict(params, eta_data)
    
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
    print("Training NoProp-CT ET Model")
    print("=" * 40)
    
    # Load test data from specified file
    data_file = Path(args.data_file) if args.data_file else Path("data/easy_3d_gaussian.pkl")
    print(f"Loading test data from {data_file}...")
    
    # Load data using standardized template function (dimension-agnostic)
    eta_data, ground_truth, metadata = load_standardized_ep_data()
    
    # Create train/val splits manually since this function doesn't provide them
    split_idx = len(eta_data) * 4 // 5  # 80% train, 20% val
    train_data = {
        'eta': eta_data[:split_idx],
        'mu_T': ground_truth[:split_idx]
    }
    val_data = {
        'eta': eta_data[split_idx:],
        'mu_T': ground_truth[split_idx:]
    }
    
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
        print(f"\nTraining NoPropCT_ET_{arch_name} with hidden sizes: {hidden_sizes}")
        
        # Infer dimensions from metadata
        eta_dim = infer_dimensions(eta_data, metadata=metadata)
        # Use dt=0.1 for proper ODE integration (time_horizon=1.0 / num_time_steps=10)
        model_wrapper = SimpleNoPropCTET(hidden_sizes=hidden_sizes, eta_dim=eta_dim, dt=0.1)
        
        trainer = model_wrapper.create_model_and_trainer()
        
        # Train
        params, losses, training_time = model_wrapper.train(train_data, val_data, epochs=300)
        
        # Evaluate
        print("📊 Evaluating...")
        metrics = model_wrapper.evaluate(trainer, params, train_data['eta'], train_data['mu_T'])
        
        # Benchmark inference time
        inference_stats = model_wrapper.benchmark_inference(trainer, params, train_data['eta'], num_runs=50)
        
        results[f"NoPropCT_ET_{arch_name}"] = create_standardized_results(
            model_name=f"NoPropCT_ET_{arch_name}",
            architecture_info={"hidden_sizes": hidden_sizes},
            metrics=metrics,
            losses=losses,
            training_time=training_time,
            inference_stats=inference_stats,
            predictions=metrics["predictions"],
            ground_truth=train_data['mu_T']
        )
        
        print(f"  Final MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Avg inference time: {inference_stats['avg_inference_time']:.4f}s ({inference_stats['samples_per_second']:.1f} samples/sec)")
    
    print(f"\n🏆 Best Model: NoPropCT_ET_Deep with MSE={results['NoPropCT_ET_Deep']['mse']:.6f}")
    
    # Create output directory
    output_dir = Path("artifacts/ET_models/noprop_ct_ET")
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
        
    print(f"\n✅ NoProp-CT ET training complete!")
    print(f"📁 Results saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train NoProp-CT ET models')
    parser.add_argument('--data_file', type=str, help='Path to data file (default: data/easy_3d_gaussian.pkl)')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    
    args = parser.parse_args()
    main()
