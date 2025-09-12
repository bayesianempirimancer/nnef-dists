#!/usr/bin/env python
"""Plotting routines to compare model results with ground truth."""

import sys
from pathlib import Path

# Add parent directory to path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import jax.numpy as jnp
from src.ef import ef_factory
from src.data_utils import load_training_data


def plot_gaussian_1d_results(data_file, history_file, save_dir="artifacts/plots"):
    """Plot results for Gaussian 1D model - simplified version."""
    
    # Load data and history
    print(f"Loading data from {data_file}")
    train_data, val_data, config_hash = load_training_data(data_file)
    
    print(f"Loading training history from {history_file}")
    with open(history_file, "rb") as f:
        history_data = pickle.load(f)
    history = history_data["history"]
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Get validation data
    eta_val = jnp.array(val_data["eta"])
    y_true_val = jnp.array(val_data["y"])
    
    print(f"Validation data shapes: eta {eta_val.shape}, y {y_true_val.shape}")
    
    # For Gaussian 1D: η = [-μ/σ², -1/(2σ²)]
    # So: μ = -η₁/(2η₂), σ² = -1/(2η₂), μ₂ = μ² + σ²
    
    # Calculate theoretical moments from natural parameters
    mu_theory = -eta_val[:, 0] / (2 * eta_val[:, 1])
    sigma2_theory = -1 / (2 * eta_val[:, 1])
    mu2_theory = mu_theory**2 + sigma2_theory
    
    # Calculate MSE for each output
    mse_mu = float(jnp.mean((y_true_val[:, 0] - mu_theory)**2))
    mse_mu2 = float(jnp.mean((y_true_val[:, 1] - mu2_theory)**2))
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: First moment (μ)
    ax1.scatter(y_true_val[:, 0], mu_theory, alpha=0.6, s=20, label=f'Predictions (MSE={mse_mu:.6f})')
    min_val = min(y_true_val[:, 0].min(), mu_theory.min())
    max_val = max(y_true_val[:, 0].max(), mu_theory.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    ax1.set_xlabel('True μ (first moment)')
    ax1.set_ylabel('Predicted μ')
    ax1.set_title('First Moment: True vs Predicted')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Second moment (μ₂)
    ax2.scatter(y_true_val[:, 1], mu2_theory, alpha=0.6, s=20, label=f'Predictions (MSE={mse_mu2:.6f})')
    min_val = min(y_true_val[:, 1].min(), mu2_theory.min())
    max_val = max(y_true_val[:, 1].max(), mu2_theory.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    ax2.set_xlabel('True μ₂ (second moment)')
    ax2.set_ylabel('Predicted μ₂')
    ax2.set_title('Second Moment: True vs Predicted')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"{save_dir}/moments_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved moments comparison to: {plot_path}")
    
    print(f"✅ Plot saved to {save_dir}/")
    print(f"Final training MSE: {history['train_mse'][-1]:.6f}")
    print(f"Final validation MSE: {history['val_mse'][-1]:.6f}")
    print(f"First moment MSE: {mse_mu:.6f}")
    print(f"Second moment MSE: {mse_mu2:.6f}")


def main():
    """Main function to run plotting."""
    data_file = "data/training_data_aee3fc097f906c743d8dae9a130ea1eb.pkl"
    history_file = "artifacts/test_training_history.pkl"
    
    if not Path(data_file).exists():
        print(f"❌ Data file {data_file} not found!")
        return
    
    if not Path(history_file).exists():
        print(f"❌ History file {history_file} not found!")
        return
    
    plot_gaussian_1d_results(data_file, history_file)


if __name__ == "__main__":
    main()
