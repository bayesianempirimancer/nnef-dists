#!/usr/bin/env python3
"""
Demo script showing how to use the NoProp-CT implementation for exponential family moment mapping.

This script provides a complete end-to-end example of training and evaluating
the continuous-time NoProp model on exponential family distributions.
"""

import argparse
from pathlib import Path
import time
import sys
import pickle
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ef import GaussianNatural1D, MultivariateNormal
from src.noprop_ct import NoPropCTMomentNet, NoPropCTConfig, train_noprop_ct_moment_net


def load_existing_data(ef_type: str, data_dir: Path = Path("data")):
    """Load existing training data from the data directory."""
    
    # Map ef_type to expected parameter dimensions
    if ef_type == "gaussian_1d":
        expected_dim = 2
    elif ef_type == "multivariate_2d":
        expected_dim = 12  # For 2D multivariate normal
    else:
        raise ValueError(f"Unknown EF type: {ef_type}")
    
    # Find suitable data files
    suitable_files = []
    for data_file in data_dir.glob("*.pkl"):
        try:
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            
            if data["train_eta"].shape[1] == expected_dim:
                suitable_files.append((data_file, data["train_eta"].shape[0]))
        except Exception as e:
            print(f"Warning: Could not read {data_file}: {e}")
            continue
    
    if not suitable_files:
        raise FileNotFoundError(f"No suitable data files found for {ef_type} (expected dim {expected_dim})")
    
    # Choose the largest suitable dataset
    best_file, _ = max(suitable_files, key=lambda x: x[1])
    
    print(f"Loading data from {best_file.name}")
    with open(best_file, 'rb') as f:
        data = pickle.load(f)
    
    # Convert to the format expected by the training function
    train_data = {
        "eta": data["train_eta"],
        "y": data["train_y"]
    }
    
    val_data = {
        "eta": data["val_eta"], 
        "y": data["val_y"]
    }
    
    # Create test data by splitting some validation data
    n_val = val_data["eta"].shape[0]
    n_test = min(n_val // 2, 500)  # Use half of validation data as test, max 500
    
    test_data = {
        "eta": val_data["eta"][:n_test],
        "y": val_data["y"][:n_test]
    }
    
    # Keep remaining validation data
    val_data = {
        "eta": val_data["eta"][n_test:],
        "y": val_data["y"][n_test:]
    }
    
    return train_data, val_data, test_data



def run_demo(ef_type: str = "gaussian_1d", num_epochs: int = 100, data_dir: str = "data"):
    """Run a complete NoProp-CT demonstration."""
    
    print(f"Running NoProp-CT demo with {ef_type}")
    print("=" * 50)
    
    # Create exponential family
    if ef_type == "gaussian_1d":
        ef = GaussianNatural1D()
    elif ef_type == "multivariate_2d":
        ef = MultivariateNormal(x_shape=(2,))
    else:
        raise ValueError(f"Unknown EF type: {ef_type}")
    
    print(f"Exponential family: {ef.__class__.__name__}")
    print(f"Natural parameter dimension: {ef.eta_dim}")
    print(f"x_shape: {ef.x_shape}")
    
    # Load existing data
    print("\\nLoading existing training data...")
    train_data, val_data, test_data = load_existing_data(ef_type, Path(data_dir))
    
    print(f"Training samples: {train_data['eta'].shape[0]}")
    print(f"Validation samples: {val_data['eta'].shape[0]}")
    print(f"Test samples: {test_data['eta'].shape[0]}")
    
    # Configure NoProp-CT
    config = NoPropCTConfig(
        hidden_sizes=(64, 32, 16),
        activation="tanh",  # Changed to tanh for better stability
        noise_scale=0.05,   # Reduced noise for stability
        time_horizon=1.0,
        num_time_steps=10,
        ode_solver="euler",
        learning_rate=1e-3,
        denoising_weight=1.0,
        consistency_weight=0.1,
    )
    
    print(f"\\nNoProp-CT Configuration:")
    print(f"  Hidden sizes: {config.hidden_sizes}")
    print(f"  Time horizon: {config.time_horizon}")
    print(f"  Time steps: {config.num_time_steps}")
    print(f"  Noise scale: {config.noise_scale}")
    print(f"  Learning rate: {config.learning_rate}")
    
    # Train model
    print(f"\\nTraining NoProp-CT model for {num_epochs} epochs...")
    start_time = time.time()
    
    state, history = train_noprop_ct_moment_net(
        ef=ef,
        train_data=train_data,
        val_data=val_data,
        config=config,
        num_epochs=num_epochs,
        batch_size=64,
        seed=42,
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on test data
    print("\\nEvaluating on test data...")
    model = NoPropCTMomentNet(ef=ef, config=config)
    
    test_pred = model.apply(state.params, test_data["eta"], training=False)
    test_mse = float(jnp.mean(jnp.square(test_pred - test_data["y"])))
    test_mae = float(jnp.mean(jnp.abs(test_pred - test_data["y"])))
    
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    
    # Component-wise analysis
    component_mse = jnp.mean(jnp.square(test_pred - test_data["y"]), axis=0)
    print(f"Component-wise MSE: {[f'{x:.6f}' for x in component_mse]}")
    
    # Create visualizations
    print("\\nGenerating visualizations...")
    
    # Training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('NoProp-CT Training Results', fontsize=14)
    
    epochs = range(len(history['train_loss']))
    
    # Training losses
    axes[0, 0].semilogy(epochs, history['train_denoising'], 'b-', label='Denoising', linewidth=2)
    axes[0, 0].semilogy(epochs, history['train_consistency'], 'r-', label='Consistency', linewidth=2)
    axes[0, 0].semilogy(epochs, history['train_loss'], 'g-', label='Total', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation losses
    axes[0, 1].semilogy(epochs, history['val_denoising'], 'b-', label='Denoising', linewidth=2)
    axes[0, 1].semilogy(epochs, history['val_consistency'], 'r-', label='Consistency', linewidth=2)
    axes[0, 1].semilogy(epochs, history['val_loss'], 'g-', label='Total', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Prediction vs truth scatter plot
    axes[1, 0].scatter(test_data["y"][:, 0], test_pred[:, 0], alpha=0.6, s=20)
    min_val = min(jnp.min(test_data["y"][:, 0]), jnp.min(test_pred[:, 0]))
    max_val = max(jnp.max(test_data["y"][:, 0]), jnp.max(test_pred[:, 0]))
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[1, 0].set_xlabel('True Values (Component 1)')
    axes[1, 0].set_ylabel('Predicted Values')
    axes[1, 0].set_title('Predictions vs Truth')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Component-wise MSE
    components = range(len(component_mse))
    axes[1, 1].bar(components, component_mse, alpha=0.7)
    axes[1, 1].set_xlabel('Component')
    axes[1, 1].set_ylabel('MSE')
    axes[1, 1].set_title('Component-wise Test MSE')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save results
    output_dir = Path("artifacts/noprop_ct_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / f"demo_results_{ef_type}.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f"demo_results_{ef_type}.pdf", bbox_inches='tight')
    print(f"NoProp-CT plots saved to {output_dir}/demo_results_{ef_type}.png")
    plt.close()  # Close instead of show to ensure saving
    
    # Save summary
    summary = f"""
    NoProp-CT Demo Results
    ======================
    
    Exponential Family: {ef.__class__.__name__}
    Natural Parameter Dimension: {ef.eta_dim}
    
    Dataset:
    - Training samples: {train_data['eta'].shape[0]}
    - Validation samples: {val_data['eta'].shape[0]}
    - Test samples: {test_data['eta'].shape[0]}
    
    Model Configuration:
    - Hidden sizes: {config.hidden_sizes}
    - Time horizon: {config.time_horizon}
    - Time steps: {config.num_time_steps}
    - Noise scale: {config.noise_scale}
    
    Training:
    - Epochs: {num_epochs}
    - Training time: {training_time:.2f}s
    - Final train loss: {history['train_loss'][-1]:.6f}
    - Final val loss: {history['val_loss'][-1]:.6f}
    
    Test Performance:
    - MSE: {test_mse:.6f}
    - MAE: {test_mae:.6f}
    - Component MSE: {[f'{x:.6f}' for x in component_mse]}
    """
    
    with open(output_dir / f"demo_summary_{ef_type}.txt", 'w') as f:
        f.write(summary)
    
    print(summary)
    print(f"Results saved to {output_dir}")
    
    return state, history, test_mse


def main():
    parser = argparse.ArgumentParser(description='NoProp-CT Demo')
    parser.add_argument('--ef-type', choices=['gaussian_1d', 'multivariate_2d'], 
                       default='gaussian_1d', help='Exponential family type')
    parser.add_argument('--num-epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing training data files')
    
    args = parser.parse_args()
    
    # Run demo
    state, history, test_mse = run_demo(
        ef_type=args.ef_type,
        num_epochs=args.num_epochs,
        data_dir=args.data_dir
    )
    
    print(f"\\nDemo completed successfully!")
    print(f"Final test MSE: {test_mse:.6f}")


if __name__ == "__main__":
    main()
