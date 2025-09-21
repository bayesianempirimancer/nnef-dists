#!/usr/bin/env python3
"""
Training script for GLU ET model.

This script trains, evaluates, and plots results for a GLU ET Network
on the natural parameter to statistics mapping task.
"""

import sys
import json
import pickle
from pathlib import Path
import jax
import jax.numpy as jnp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import standardized plotting functions
sys.path.append(str(Path(__file__).parent.parent))
from plot_training_results import plot_training_results, plot_model_comparison, save_results_summary

from src.config import FullConfig, NetworkConfig
from src.models.glu_ET import GLUNetwork, GLUTrainer


def convert_to_9d(eta_12d, mean_12d):
    """Convert 12D data to 9D format for models expecting upper triangular covariance."""
    # Extract components
    eta1 = eta_12d[:, :3]  # 3D
    eta2_flat = eta_12d[:, 3:12]  # 9D
    mu = mean_12d[:, :3]  # 3D
    mu_muT_plus_Sigma_flat = mean_12d[:, 3:12]  # 9D
    
    # Reshape to 3x3 matrices and extract upper triangular
    eta2_matrices = eta2_flat.reshape(-1, 3, 3)
    mu_muT_plus_Sigma_matrices = mu_muT_plus_Sigma_flat.reshape(-1, 3, 3)
    
    # Extract upper triangular elements (6 elements per matrix)
    upper_indices = jnp.triu_indices(3)
    eta2_upper = eta2_matrices[:, upper_indices[0], upper_indices[1]]  # (N, 6)
    mu_muT_plus_Sigma_upper = mu_muT_plus_Sigma_matrices[:, upper_indices[0], upper_indices[1]]  # (N, 6)
    
    # Combine to 9D: [mu (3D), upper_elements (6D)]
    eta_9d = jnp.concatenate([eta1, eta2_upper], axis=1)
    mean_9d = jnp.concatenate([mu, mu_muT_plus_Sigma_upper], axis=1)
    
    return eta_9d, mean_9d


class SimpleGLUET:
    """GLU ET model wrapper using official implementation."""
    
    def __init__(self, hidden_sizes=[64, 64]):
        self.hidden_sizes = hidden_sizes
        # Create proper FullConfig
        self.config = FullConfig()
        self.config.network.hidden_sizes = hidden_sizes
        self.config.network.activation = "swish"
        self.config.network.use_layer_norm = True
        self.config.network.output_dim = 9  # 3D mean + 6D upper triangular covariance
        self.config.training.learning_rate = 1e-3
        self.config.training.num_epochs = 300
        self.config.training.batch_size = 32
        
    def create_model(self):
        """Create the model using the official implementation."""
        from src.models.glu_ET import GLUNetwork
        # Create proper network config
        network_config = NetworkConfig()
        network_config.hidden_sizes = self.hidden_sizes
        network_config.activation = "swish"
        network_config.use_layer_norm = True
        network_config.output_dim = 9  # 3D mean + 6D upper triangular covariance
        return GLUNetwork(config=network_config)
    
    def train(self, eta_data, ground_truth, epochs=300, learning_rate=1e-3):
        """Train the model."""
        import optax
        import flax.linen as nn
        from jax import random
        
        model = self.create_model()
        
        # Initialize model
        rng = random.PRNGKey(42)
        params = model.init(rng, eta_data[:1])
        
        # Create optimizer
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)
        
        # Training loop
        losses = []
        for epoch in range(epochs):
            # Forward pass
            predictions = model.apply(params, eta_data)
            loss = jnp.mean((predictions - ground_truth) ** 2)
            
            # Backward pass
            loss_grad = jax.grad(lambda p: jnp.mean((model.apply(p, eta_data) - ground_truth) ** 2))
            grads = loss_grad(params)
            
            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            losses.append(float(loss))
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
        return params, losses, model


def main():
    """Main training and evaluation pipeline."""
    print("Training Standard GLU ET Model")
    print("=" * 40)
    
    # Load test data from easy_3d_gaussian
    print("Loading test data from easy_3d_gaussian...")
    data_file = Path("data/easy_3d_gaussian.pkl")
    
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
    
    # Convert 12D data to 9D for models expecting upper triangular covariance
    eta_9d, ground_truth_9d = convert_to_9d(eta_data, ground_truth)
    print(f"Converted to 9D: eta={eta_9d.shape}, ground_truth={ground_truth_9d.shape}")
    
    # Define architectures to test
    architectures = {
        "Small": [32, 32],
        "Medium": [64, 64], 
        "Large": [128, 64],
        "Deep": [64, 64, 32]
    }
    
    results = {}
    best_mse = float('inf')
    best_model = None
    
    for name, hidden_sizes in architectures.items():
        print(f"\nTraining GLU_ET_{name} with hidden sizes: {hidden_sizes}")
        
        # Create and train model
        model_wrapper = SimpleGLUET(hidden_sizes=hidden_sizes)
        params, losses, trained_model = model_wrapper.train(eta_9d, ground_truth_9d, epochs=300)
        
        # Evaluate
        predictions = trained_model.apply(params, eta_9d)
        mse = float(jnp.mean((predictions - ground_truth_9d) ** 2))
        mae = float(jnp.mean(jnp.abs(predictions - ground_truth_9d)))
        
        print(f"GLU_ET_{name} - MSE: {mse:.6f}, MAE: {mae:.6f}")
        
        # Store results
        results[f"GLU_ET_{name}"] = {
            "train_loss": losses,
            "test_metrics": {
                "mse": mse,
                "mae": mae
            },
            "config": model_wrapper.config,
            "model_name": f"GLU_ET_{name}",
            "predictions": predictions,
            "ground_truth": ground_truth_9d
        }
        
        # Track best model
        if mse < best_mse:
            best_mse = mse
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
    main()
