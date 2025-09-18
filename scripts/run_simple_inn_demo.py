#!/usr/bin/env python3
"""
Simple Invertible Neural Network Demo

A simplified version of the INN that works specifically for 2D problems
(natural parameters and moments both 2D).
"""

import argparse
from pathlib import Path
import time
import sys
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ef import GaussianNatural1D
from src.simple_inn import SimpleInvertibleNet, SimpleINNConfig, train_simple_inn
from scripts.run_noprop_ct_demo import load_existing_data


def run_simple_inn_demo(ef_type: str = "gaussian_1d", num_epochs: int = 50):
    """Run simple INN demonstration."""
    
    print(f"Running Simple Invertible Neural Network demo")
    print("=" * 50)
    
    # Only works for 2D case
    if ef_type != "gaussian_1d":
        raise ValueError("Simple INN only supports gaussian_1d (2D case)")
    
    ef = GaussianNatural1D()
    print(f"Exponential family: {ef.__class__.__name__}")
    print(f"Natural parameter dimension: {ef.eta_dim}")
    
    # Load data
    print("\\nLoading existing training data...")
    train_data, val_data, test_data = load_existing_data(ef_type)
    
    print(f"Training samples: {train_data['eta'].shape[0]}")
    print(f"Validation samples: {val_data['eta'].shape[0]}")
    print(f"Test samples: {test_data['eta'].shape[0]}")
    
    # Configure model
    config = SimpleINNConfig(
        num_layers=4,
        hidden_size=64,
        activation="tanh",
        learning_rate=1e-3,
        clamp_alpha=2.0,
    )
    
    print(f"\\nSimple INN Configuration:")
    print(f"  Layers: {config.num_layers}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Activation: {config.activation}")
    print(f"  Learning rate: {config.learning_rate}")
    
    # Train
    print(f"\\nTraining Simple INN for {num_epochs} epochs...")
    start_time = time.time()
    
    state, history = train_simple_inn(
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
    
    # Evaluate
    print("\\nEvaluating on test data...")
    model = SimpleInvertibleNet(ef=ef, config=config)
    
    # Forward pass: η → μ
    test_pred, test_log_det = model.apply(state.params, test_data["eta"], reverse=False)
    test_mse = float(jnp.mean(jnp.square(test_pred - test_data["y"])))
    test_mae = float(jnp.mean(jnp.abs(test_pred - test_data["y"])))
    
    print(f"Forward MSE (η → μ): {test_mse:.6f}")
    print(f"Forward MAE: {test_mae:.6f}")
    
    # Test invertibility: μ → η
    reconstructed_eta, recon_log_det = model.apply(state.params, test_pred, reverse=True)
    reconstruction_error = float(jnp.mean(jnp.square(reconstructed_eta - test_data["eta"])))
    print(f"Invertibility error: {reconstruction_error:.8f}")
    
    # Log determinant analysis
    print(f"Mean log det (forward): {jnp.mean(test_log_det):.6f}")
    print(f"Log det consistency: {jnp.mean(jnp.abs(test_log_det + recon_log_det)):.8f}")
    
    # Visualizations
    print("\\nGenerating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Simple Invertible Neural Network Results', fontsize=14)
    
    epochs = range(len(history['train_loss']))
    
    # Training curves
    axes[0, 0].semilogy(epochs, history['train_loss'], 'b-', label='Total', linewidth=2)
    axes[0, 0].semilogy(epochs, history['train_mse'], 'r-', label='MSE', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation curves
    axes[0, 1].semilogy(epochs, history['val_loss'], 'b-', label='Total', linewidth=2)
    axes[0, 1].semilogy(epochs, history['val_mse'], 'r-', label='MSE', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Log determinant evolution
    axes[0, 2].plot(epochs, history['train_log_det'], 'g-', label='Train', linewidth=2)
    axes[0, 2].plot(epochs, history['val_log_det'], 'orange', label='Val', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Log Det Regularization')
    axes[0, 2].set_title('Log Determinant Evolution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Forward mapping visualization
    axes[1, 0].scatter(test_data["eta"][:, 0], test_data["eta"][:, 1], 
                      c='blue', label='η (input)', s=20, alpha=0.7)
    axes[1, 0].scatter(test_pred[:, 0], test_pred[:, 1], 
                      c='red', label='μ (predicted)', s=20, alpha=0.7)
    axes[1, 0].set_xlabel('Component 1')
    axes[1, 0].set_ylabel('Component 2')
    axes[1, 0].set_title('Forward Mapping: η → μ')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Invertibility test
    axes[1, 1].scatter(test_data["eta"][:, 0], reconstructed_eta[:, 0], alpha=0.6, s=20, label='η₁')
    axes[1, 1].scatter(test_data["eta"][:, 1], reconstructed_eta[:, 1], alpha=0.6, s=20, label='η₂')
    
    min_val = min(jnp.min(test_data["eta"]), jnp.min(reconstructed_eta))
    max_val = max(jnp.max(test_data["eta"]), jnp.max(reconstructed_eta))
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[1, 1].set_xlabel('True η')
    axes[1, 1].set_ylabel('Reconstructed η')
    axes[1, 1].set_title(f'Invertibility Test\\n(Error: {reconstruction_error:.2e})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Prediction accuracy
    axes[1, 2].scatter(test_data["y"][:, 0], test_pred[:, 0], alpha=0.6, s=20, label='μ₁')
    axes[1, 2].scatter(test_data["y"][:, 1], test_pred[:, 1], alpha=0.6, s=20, label='μ₂')
    
    min_val = min(jnp.min(test_data["y"]), jnp.min(test_pred))
    max_val = max(jnp.max(test_data["y"]), jnp.max(test_pred))
    axes[1, 2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[1, 2].set_xlabel('True Moments')
    axes[1, 2].set_ylabel('Predicted Moments')
    axes[1, 2].set_title(f'Moment Prediction\\n(MSE: {test_mse:.6f})')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save results
    output_dir = Path("artifacts/simple_inn_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "simple_inn_results.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "simple_inn_results.pdf", bbox_inches='tight')
    print(f"Plots saved to {output_dir}/simple_inn_results.png")
    plt.close()  # Close instead of show to ensure saving
    
    # Summary
    summary = f"""
    Simple Invertible Neural Network Results
    ========================================
    
    Model: {config.num_layers} coupling layers + permutations
    Training time: {training_time:.2f}s
    
    Performance:
    - Forward MSE (η → μ): {test_mse:.6f}
    - Forward MAE: {test_mae:.6f}
    - Invertibility error: {reconstruction_error:.8f}
    - Mean log determinant: {jnp.mean(test_log_det):.6f}
    
    The invertibility error should be very small (< 1e-6) for a good invertible model.
    Log determinant indicates volume change: positive = expansion, negative = contraction.
    """
    
    print(summary)
    
    with open(output_dir / "summary.txt", 'w') as f:
        f.write(summary)
    
    return state, history, test_mse


def main():
    parser = argparse.ArgumentParser(description='Simple INN Demo')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of epochs')
    
    args = parser.parse_args()
    
    state, history, test_mse = run_simple_inn_demo(num_epochs=args.num_epochs)
    print(f"\\nDemo completed! Final test MSE: {test_mse:.6f}")


if __name__ == "__main__":
    main()
