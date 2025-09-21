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
from jax import random
# matplotlib import removed - now using standardized plotting

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import standardized plotting functions
sys.path.append(str(Path(__file__).parent.parent))
from plot_training_results import plot_training_results, plot_model_comparison, save_results_summary

# Simple configuration class
class SimpleConfig:
    def __init__(self, hidden_sizes, activation="swish"):
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.use_layer_norm = False

class SimpleStandardMLPET:
    """Standard MLP ET using official implementation."""
    
    def __init__(self, hidden_sizes=[64, 64]):
        self.hidden_sizes = hidden_sizes
        self.config = SimpleConfig(hidden_sizes=hidden_sizes, activation="swish")
    
    def create_model(self):
        """Create the model using the official implementation."""
        from src.models.mlp_ET import MLPNetwork
        # Create proper network config
        from src.config import NetworkConfig
        network_config = NetworkConfig()
        network_config.hidden_sizes = self.hidden_sizes
        network_config.activation = "swish"
        network_config.use_layer_norm = True
        network_config.output_dim = 12  # 3D Gaussian sufficient statistics
        return MLPNetwork(config=network_config)
    
    def train(self, eta_data, ground_truth, epochs=300, learning_rate=1e-3):
        """Train the model."""
        import optax
        import flax.linen as nn
        import jax
        
        model = self.create_model()
        
        # Initialize
        rng = random.PRNGKey(42)
        params = model.init(rng, eta_data[:1])
        
        # Optimizer
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)
        
        # Loss function
        def loss_fn(params, eta_batch, target_batch):
            predictions = model.apply(params, eta_batch)
            return jnp.mean((predictions - target_batch) ** 2)
        
        # Training loop
        losses = []
        batch_size = 32
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # Shuffle data
            indices = random.permutation(rng + epoch, len(eta_data))
            eta_shuffled = eta_data[indices]
            target_shuffled = ground_truth[indices]
            
            # Mini-batch training
            for i in range(0, len(eta_data), batch_size):
                eta_batch = eta_shuffled[i:i+batch_size]
                target_batch = target_shuffled[i:i+batch_size]
                
                loss, grads = jax.value_and_grad(loss_fn)(params, eta_batch, target_batch)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                epoch_losses.append(loss)
            
            avg_loss = jnp.mean(jnp.array(epoch_losses))
            losses.append(float(avg_loss))
            
            if epoch % 50 == 0 or epoch < 10:
                print(f"  Epoch {epoch}: Loss = {avg_loss:.6f}")
        
        return params, losses
    
    def predict(self, model, params, eta_data):
        """Make predictions."""
        return model.apply(params, eta_data)
    
    def evaluate(self, predictions, ground_truth):
        """Evaluate predictions."""
        mse = jnp.mean((predictions - ground_truth) ** 2)
        mae = jnp.mean(jnp.abs(predictions - ground_truth))
        return {"mse": float(mse), "mae": float(mae)}


def main():
    """Main training and evaluation pipeline."""
    print("Training Standard MLP ET Model")
    print("=" * 40)
    
    # Load test data from easy_3d_gaussian
    print("Loading test data from easy_3d_gaussian...")
    data_file = Path("data/easy_3d_gaussian.pkl")
    
    import pickle
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    eta_data = data["train"]["eta"]
    ground_truth = data["train"]["mean"]
    print(f"Data shapes: eta={eta_data.shape}, ground_truth={ground_truth.shape}")
    
    # Purge cov_tt to save memory
    if "cov" in data["train"]: del data["train"]["cov"]
    if "cov" in data["val"]: del data["val"]["cov"]
    if "cov" in data["test"]: del data["test"]["cov"]
    import gc; gc.collect()
    print("✅ Purged cov_tt elements from memory for optimization")
    
    # Define architectures to test
    architectures = {
        "Small": [32, 32],
        "Medium": [64, 64], 
        "Large": [128, 128],
        "Deep": [64, 64, 64],
        "Wide": [128, 64, 128],
        "Max": [128, 128, 128]
    }
    
    results = {}
    
    # Test each architecture
    for arch_name, hidden_sizes in architectures.items():
        print(f"\nTraining MLP_ET_{arch_name} with hidden sizes: {hidden_sizes}")
        
        model_wrapper = SimpleStandardMLPET(hidden_sizes=hidden_sizes)
        model = model_wrapper.create_model()
        
        # Train
        params, losses = model_wrapper.train(eta_data, ground_truth, epochs=300)
        
        # Evaluate
        predictions = model_wrapper.predict(model, params, eta_data)
        metrics = model_wrapper.evaluate(predictions, ground_truth)
        
        results[f"MLP_ET_{arch_name}"] = {
            "hidden_sizes": hidden_sizes,
            "mse": metrics["mse"],
            "mae": metrics["mae"],
            "losses": losses
        }
        
        print(f"  Final MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("STANDARD MLP ET MODEL COMPARISON")
    print("=" * 60)
    
    best_mse = float('inf')
    best_model = None
    
    for model_name, result in results.items():
        print(f"{model_name:<20} : MSE={result['mse']:.6f}, MAE={result['mae']:.6f}, Architecture={result['hidden_sizes']}")
        if result['mse'] < best_mse:
            best_mse = result['mse']
            best_model = model_name
    
    print(f"\n🏆 Best Model: {best_model} with MSE={best_mse:.6f}")
    
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
    main()
