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
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

# Simple data generation function
def generate_simple_test_data(n_samples=400, seed=42):
    """Generate simple 3D Gaussian test data."""
    eta_vectors = []
    expected_stats = []
    
    for i in range(n_samples):
        # Generate random mean and covariance
        mean = random.normal(random.PRNGKey(seed + i), (3,)) * 1.0
        A = random.normal(random.PRNGKey(seed + i + 1000), (3, 3))
        covariance_matrix = A.T @ A + jnp.eye(3) * 0.01
        
        # Convert to natural parameters
        sigma_inv = jnp.linalg.inv(covariance_matrix)
        eta1 = sigma_inv @ mean  # η₁ = Σ⁻¹μ
        eta2_matrix = -0.5 * sigma_inv  # η₂ = -0.5Σ⁻¹
        eta_vector = jnp.concatenate([eta1, eta2_matrix.flatten()])
        eta_vectors.append(eta_vector)
        
        # Expected sufficient statistics
        expected_stat = jnp.concatenate([
            mean,  # μ (3 values)
            (jnp.outer(mean, mean) + covariance_matrix).flatten()  # μμᵀ + Σ (9 values)
        ])
        expected_stats.append(expected_stat)
    
    return jnp.array(eta_vectors), jnp.array(expected_stats)


# Simple configuration class
class SimpleConfig:
    def __init__(self, hidden_sizes, activation="swish"):
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.use_layer_norm = True


class SimpleStandardMLPET:
    """Standard MLP ET using official implementation."""
    
    def __init__(self, hidden_sizes=[64, 64]):
        self.hidden_sizes = hidden_sizes
        self.config = SimpleConfig(hidden_sizes=hidden_sizes, activation="swish")
    
    def create_model(self):
        """Create the model using the official implementation."""
        try:
            from src.models.mlp_ET import MLPNetwork
            return MLPNetwork(config=self.config)
        except ImportError:
            # Fallback to simplified implementation if import fails
            import flax.linen as nn
            
            class StandardMLPETModel(nn.Module):
                hidden_sizes: list
                
                @nn.compact
                def __call__(self, x, training=True):
                    for i, hidden_size in enumerate(self.hidden_sizes):
                        x = nn.Dense(hidden_size, name=f'hidden_{i}')(x)
                        x = nn.swish(x)
                        if i < len(self.hidden_sizes) - 1:
                            x = nn.LayerNorm(name=f'layer_norm_{i}')(x)
                    
                    # Output layer - 12 sufficient statistics
                    x = nn.Dense(12, name='output')(x)
                    return x
            
            return StandardMLPETModel(hidden_sizes=self.hidden_sizes)
    
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
    
    # Generate test data
    print("Generating test data...")
    eta_data, ground_truth = generate_simple_test_data(n_samples=400, seed=42)
    print(f"Data shapes: eta={eta_data.shape}, ground_truth={ground_truth.shape}")
    
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
    output_dir = Path("artifacts/ET_models/standard_mlp_ET")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot training curves
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Training curves
    plt.subplot(2, 2, 1)
    for model_name, result in results.items():
        plt.plot(result['losses'], label=model_name, alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.yscale('log')
    
    # Plot 2: MSE comparison
    plt.subplot(2, 2, 2)
    model_names = list(results.keys())
    mse_values = [results[name]['mse'] for name in model_names]
    plt.bar(range(len(model_names)), mse_values)
    plt.xticks(range(len(model_names)), model_names, rotation=45)
    plt.ylabel('MSE')
    plt.title('MSE Comparison')
    plt.yscale('log')
    
    # Plot 3: MAE comparison
    plt.subplot(2, 2, 3)
    mae_values = [results[name]['mae'] for name in model_names]
    plt.bar(range(len(model_names)), mae_values)
    plt.xticks(range(len(model_names)), model_names, rotation=45)
    plt.ylabel('MAE')
    plt.title('MAE Comparison')
    plt.yscale('log')
    
    # Plot 4: Architecture comparison (parameter count estimation)
    plt.subplot(2, 2, 4)
    param_counts = []
    for model_name, result in results.items():
        # Rough estimation: input(12) * first_hidden + hidden_layers + last_hidden * output(12)
        hidden_sizes = result['hidden_sizes']
        if len(hidden_sizes) == 0:
            param_count = 12 * 12  # Direct mapping
        else:
            param_count = 12 * hidden_sizes[0]  # Input layer
            for i in range(len(hidden_sizes) - 1):
                param_count += hidden_sizes[i] * hidden_sizes[i + 1]  # Hidden layers
            param_count += hidden_sizes[-1] * 12  # Output layer
        param_counts.append(param_count)
    
    plt.bar(range(len(model_names)), param_counts)
    plt.xticks(range(len(model_names)), model_names, rotation=45)
    plt.ylabel('Estimated Parameters')
    plt.title('Parameter Count Comparison')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results
    with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
    print(f"\n✅ Standard MLP ET training complete!")
    print(f"📁 Results saved to {output_dir}")


if __name__ == "__main__":
    main()
