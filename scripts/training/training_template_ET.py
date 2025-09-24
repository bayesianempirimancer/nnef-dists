#!conda activate numpyro && python
"""
Training script template for ET (Expected Statistics) neural networks.

This script trains networks that directly predict the expected sufficient statistics
E[T(X)] of exponential family distributions.

Usage:
    python scripts/training/training_template_ET.py
"""

import sys
from pathlib import Path
import time
import json
import jax.numpy as jnp

# Add the project root to Python path for package imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import data and plotting functions
from src.utils.data_utils import infer_dimensions, load_standardized_ep_data
from scripts.plotting.plot_training_results import plot_training_results, plot_model_comparison, save_results_summary, create_standardized_results
from src.config import TrainingConfig

# Simple configuration class
class Network_Config:
    def __init__(self, hidden_sizes, activation="swish"):
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.use_layer_norm = True

class ET_Template:
    """Template ET using official implementation."""
    
    def __init__(self, hidden_sizes=None, eta_dim=None, model_type = None):
        if eta_dim is None:
            raise ValueError("eta_dim must be provided. Call infer_dimensions() first.")
        self.eta_dim = eta_dim
        self.hidden_sizes = hidden_sizes
        self.model_type = model_type
    
    def create_optimal_training_config(self, num_epochs=300):
        """Create optimal training configuration based on performance testing."""
        return TrainingConfig(
            num_epochs=num_epochs, 
            learning_rate=5e-4,  # Optimized learning rate for AdamW
            optimizer="adamw",   # AdamW - comparing against RMSprop performance
            weight_decay=1e-4,   # L2 regularization for AdamW
            gradient_clip_norm=0.5,  # Gradient clipping for stability
            
            # Optimized momentum parameters (from testing)
            beta1=0.9,      # First moment decay (standard, optimal)
            beta2=0.999,    # Second moment decay (standard, optimal)
            momentum=0.9,   # SGD momentum (if using SGD)
            nesterov=False, # Nesterov acceleration (disabled for AdamW)
            
            # Advanced learning rate scheduling - OneCycle for faster convergence
            use_lr_schedule=True,
            lr_schedule_type="onecycle",  # OneCycle for faster convergence
            lr_decay_steps=num_epochs,  # Decay over full training
            warmup_steps=max(10, num_epochs // 20) if num_epochs > 20 else 0,  # 5% warmup, but only if enough epochs
            
            # OneCycle parameters for faster convergence
            onecycle_max_lr=1e-2,  # Higher max LR for faster initial learning
            onecycle_pct_start=0.3,  # 30% of training for warmup
            
            # Training optimization
            batch_size=128,  # Larger batch size for better gradient estimates
            patience=50,     # Early stopping patience
            early_stopping=True
        )
    
    def config_model_and_trainer(self, num_epochs):
        raise NotImplementedError("config_model_and_trainer must be implemented in the subclass.")

    def create_model_and_trainer(self, full_config, model_and_trainer):
        # Use inferred dimensions from data
        # For ET models, input and output dimensions should be the same
        return model_and_trainer(full_config)
    
    def train(self, eta_data, mu_T_data, epochs=300, learning_rate=1e-2):
        """Train the model and measure training time."""
        trainer = self.create_model_and_trainer()
        
        # Prepare training data
        # For ET models, ground_truth should be mu_T (same dimension as eta)
        train_data = {
            'eta': eta_data,
            'mu_T': mu_T_data
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
    
    def save_model_artifacts(self, trainer, params, model_name, output_dir):
        """Save model parameters, optimizer state, and training configuration."""
        import pickle
        from pathlib import Path
        
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get optimizer state
        optimizer = trainer.create_optimizer()
        opt_state = optimizer.init(params)
        
        # Save model parameters
        params_file = output_dir / f"{model_name}_params.pkl"
        with open(params_file, 'wb') as f:
            pickle.dump(params, f)
        print(f"  Saved model parameters to: {params_file}")
        
        # Save optimizer state
        opt_state_file = output_dir / f"{model_name}_opt_state.pkl"
        with open(opt_state_file, 'wb') as f:
            pickle.dump(opt_state, f)
        print(f"  Saved optimizer state to: {opt_state_file}")
        
        # Save training configuration
        config_file = output_dir / f"{model_name}_config.pkl"
        with open(config_file, 'wb') as f:
            pickle.dump(trainer.config, f)
        print(f"  Saved training config to: {config_file}")
        
        return {
            'params_file': str(params_file),
            'opt_state_file': str(opt_state_file),
            'config_file': str(config_file)
        }
    
    def predict(self, trainer, params, eta_data):
        """Make predictions."""
        return trainer.predict(params, eta_data)

    
    def evaluate(self, predictions, mu_T_data):
        """Evaluate predictions."""
        mse = jnp.mean((predictions - mu_T_data) ** 2)
        mae = jnp.mean(jnp.abs(predictions - mu_T_data))
        return {"mse": float(mse), "mae": float(mae)}

    def benchmark_inference(self, trainer, params, eta_data, num_runs=50):
        """Benchmark inference time using MLP-specific model.apply() method."""
        import time
        
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
    """Main training function."""
    print("Training Template ET Model")
    print("=" * 40)
    
    # Load data using standardized utility function
    from src.utils.data_utils import load_standardized_ep_data
    eta_data, ground_truth, metadata = load_standardized_ep_data()
    
    # Test single large architecture
    architectures = {
        'Template_ET_Large': [128, 128, 64],
    }
    
    results = {}
    
    for name, hidden_sizes in architectures.items():
        print(f"\nTraining {name} with hidden sizes: {hidden_sizes}")
        
        # Infer dimensions from metadata
        eta_dim = infer_dimensions(eta_data, metadata=metadata)
        model = SimpleTemplateET(hidden_sizes=hidden_sizes, eta_dim=eta_dim)
        
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
        
        # Save model artifacts by default
        model_name = f"{model.model_type}_ET_{name}"
        output_dir = "artifacts/ET_models/template_ET"
        saved_files = model.save_model_artifacts(trainer, params, model_name, output_dir)
        
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
    
    # Summary
    print(f"\n{'='*60}")
    print("TEMPLATE ET MODEL RESULTS")
    print(f"{'='*60}")
    
    for model_name, result in results.items():
        print(f"{model_name:<20} : MSE={result['mse']:.6f}, MAE={result['mae']:.6f}, Architecture={result['hidden_sizes']}")
        print(f"{'':20}   Training: {result['training_time']:.2f}s, Inference: {result['samples_per_second']:.1f} samples/sec")
    
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
