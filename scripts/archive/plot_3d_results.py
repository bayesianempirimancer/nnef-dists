#!/usr/bin/env python
"""Specialized plotting routines for 3D multivariate Gaussian results."""

import sys
from pathlib import Path

# Add parent directory to path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import jax.numpy as jnp
from src.ef import ef_factory
from src.data_utils import load_training_data


def plot_3d_multivariate_results(data_file, history_file, save_dir="artifacts/plots"):
    """Plot comprehensive results for 3D multivariate Gaussian model."""
    
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
    
    # For 3D multivariate normal:
    # eta has 12 dimensions: [eta1_0, eta1_1, eta1_2, eta2_00, eta2_01, eta2_02, eta2_10, eta2_11, eta2_12, eta2_20, eta2_21, eta2_22]
    # y has 9 dimensions: [mu_0, mu_1, mu_2, sigma_00, sigma_01, sigma_02, sigma_11, sigma_12, sigma_22]
    
    # Extract linear terms (eta1) and quadratic terms (eta2) from natural parameters
    eta1_true = eta_val[:, :3]  # Linear terms: [eta1_0, eta1_1, eta1_2]
    eta2_true = eta_val[:, 3:12].reshape(-1, 3, 3)  # Reshape to 3x3 matrices
    
    # Extract means and covariance from sufficient statistics
    mu_true = y_true_val[:, :3]  # Means: [mu_0, mu_1, mu_2]
    sigma_true = y_true_val[:, 3:9]  # Covariance elements: [sigma_00, sigma_01, sigma_02, sigma_11, sigma_12, sigma_22]
    
    # Convert covariance vector back to matrices for easier comparison
    sigma_matrices_true = jnp.zeros((sigma_true.shape[0], 3, 3))
    for i in range(sigma_true.shape[0]):
        sigma_matrices_true[i, 0, 0] = sigma_true[i, 0]  # sigma_00
        sigma_matrices_true[i, 0, 1] = sigma_matrices_true[i, 1, 0] = sigma_true[i, 1]  # sigma_01
        sigma_matrices_true[i, 0, 2] = sigma_matrices_true[i, 2, 0] = sigma_true[i, 2]  # sigma_02
        sigma_matrices_true[i, 1, 1] = sigma_true[i, 3]  # sigma_11
        sigma_matrices_true[i, 1, 2] = sigma_matrices_true[i, 2, 1] = sigma_true[i, 4]  # sigma_12
        sigma_matrices_true[i, 2, 2] = sigma_true[i, 5]  # sigma_22
    
    # Create comprehensive plots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Training history
    ax1 = plt.subplot(3, 3, 1)
    epochs = range(1, len(history['train_mse']) + 1)
    ax1.plot(epochs, history['train_mse'], label='Train MSE', alpha=0.8, linewidth=2)
    ax1.plot(epochs, history['val_mse'], label='Val MSE', alpha=0.8, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Linear terms comparison (eta1 vs mu)
    ax2 = plt.subplot(3, 3, 2)
    for i in range(3):
        mse = float(jnp.mean((mu_true[:, i] - eta1_true[:, i])**2))
        ax2.scatter(mu_true[:, i], eta1_true[:, i], alpha=0.6, s=15, 
                   label=f'Component {i} (MSE={mse:.4f})')
    
    min_val = min(float(mu_true.min()), float(eta1_true.min()))
    max_val = max(float(mu_true.max()), float(eta1_true.max()))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect prediction')
    ax2.set_xlabel('True Means (μ)')
    ax2.set_ylabel('Linear Terms (η₁)')
    ax2.set_title('Linear Terms: Means vs Natural Parameters')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Covariance diagonal elements
    ax3 = plt.subplot(3, 3, 3)
    diagonal_indices = [0, 3, 5]  # sigma_00, sigma_11, sigma_22
    diagonal_labels = ['σ₀₀', 'σ₁₁', 'σ₂₂']
    for i, (idx, label) in enumerate(zip(diagonal_indices, diagonal_labels)):
        mse = float(jnp.mean((sigma_true[:, idx] - eta2_true[:, i, i])**2))
        ax3.scatter(sigma_true[:, idx], eta2_true[:, i, i], alpha=0.6, s=15, 
                   label=f'{label} (MSE={mse:.4f})')
    
    min_val = min(float(sigma_true[:, diagonal_indices].min()), float(eta2_true[:, :, :].min()))
    max_val = max(float(sigma_true[:, diagonal_indices].max()), float(eta2_true[:, :, :].max()))
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect prediction')
    ax3.set_xlabel('True Covariance Diagonal')
    ax3.set_ylabel('Quadratic Terms Diagonal (η₂)')
    ax3.set_title('Diagonal Elements: Covariance vs Natural Parameters')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Off-diagonal elements (upper triangle)
    ax4 = plt.subplot(3, 3, 4)
    off_diag_indices = [1, 2, 4]  # sigma_01, sigma_02, sigma_12
    off_diag_labels = ['σ₀₁', 'σ₀₂', 'σ₁₂']
    for i, (idx, label) in enumerate(zip(off_diag_indices, off_diag_labels)):
        if i == 0:  # sigma_01 corresponds to eta2[0,1]
            eta2_val = eta2_true[:, 0, 1]
        elif i == 1:  # sigma_02 corresponds to eta2[0,2]
            eta2_val = eta2_true[:, 0, 2]
        else:  # sigma_12 corresponds to eta2[1,2]
            eta2_val = eta2_true[:, 1, 2]
        
        mse = float(jnp.mean((sigma_true[:, idx] - eta2_val)**2))
        ax4.scatter(sigma_true[:, idx], eta2_val, alpha=0.6, s=15, 
                   label=f'{label} (MSE={mse:.4f})')
    
    min_val = min(float(sigma_true[:, off_diag_indices].min()), float(eta2_val.min()))
    max_val = max(float(sigma_true[:, off_diag_indices].max()), float(eta2_val.max()))
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect prediction')
    ax4.set_xlabel('True Covariance Off-diagonal')
    ax4.set_ylabel('Quadratic Terms Off-diagonal (η₂)')
    ax4.set_title('Off-diagonal Elements: Covariance vs Natural Parameters')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Error distribution for linear terms
    ax5 = plt.subplot(3, 3, 5)
    linear_errors = []
    for i in range(3):
        errors = mu_true[:, i] - eta1_true[:, i]
        linear_errors.extend(errors)
    
    ax5.hist(linear_errors, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    ax5.set_xlabel('Prediction Error')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Linear Terms Error Distribution')
    ax5.grid(True, alpha=0.3)
    ax5.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Plot 6: Error distribution for quadratic terms
    ax6 = plt.subplot(3, 3, 6)
    quadratic_errors = []
    for i in range(3):
        for j in range(3):
            if i <= j:  # Only upper triangle
                errors = sigma_matrices_true[:, i, j] - eta2_true[:, i, j]
                quadratic_errors.extend(errors)
    
    ax6.hist(quadratic_errors, bins=50, alpha=0.7, edgecolor='black', color='lightcoral')
    ax6.set_xlabel('Prediction Error')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Quadratic Terms Error Distribution')
    ax6.grid(True, alpha=0.3)
    ax6.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Plot 7: MSE by component
    ax7 = plt.subplot(3, 3, 7)
    component_mse = []
    component_labels = []
    
    # Linear components
    for i in range(3):
        mse = float(jnp.mean((mu_true[:, i] - eta1_true[:, i])**2))
        component_mse.append(mse)
        component_labels.append(f'μ_{i}')
    
    # Quadratic components
    quad_labels = ['σ₀₀', 'σ₀₁', 'σ₀₂', 'σ₁₁', 'σ₁₂', 'σ₂₂']
    for i in range(6):
        if i == 0:  # sigma_00
            true_val = sigma_true[:, 0]
            pred_val = eta2_true[:, 0, 0]
        elif i == 1:  # sigma_01
            true_val = sigma_true[:, 1]
            pred_val = eta2_true[:, 0, 1]
        elif i == 2:  # sigma_02
            true_val = sigma_true[:, 2]
            pred_val = eta2_true[:, 0, 2]
        elif i == 3:  # sigma_11
            true_val = sigma_true[:, 3]
            pred_val = eta2_true[:, 1, 1]
        elif i == 4:  # sigma_12
            true_val = sigma_true[:, 4]
            pred_val = eta2_true[:, 1, 2]
        else:  # sigma_22
            true_val = sigma_true[:, 5]
            pred_val = eta2_true[:, 2, 2]
        
        mse = float(jnp.mean((true_val - pred_val)**2))
        component_mse.append(mse)
        component_labels.append(quad_labels[i])
    
    bars = ax7.bar(range(len(component_mse)), component_mse)
    ax7.set_xlabel('Component')
    ax7.set_ylabel('MSE')
    ax7.set_title('MSE by Component')
    ax7.set_xticks(range(len(component_labels)))
    ax7.set_xticklabels(component_labels, rotation=45, ha='right')
    ax7.grid(True, alpha=0.3)
    
    # Color bars differently for means vs covariances
    for i, bar in enumerate(bars):
        if i < 3:  # Means
            bar.set_color('skyblue')
        else:  # Covariances
            bar.set_color('lightcoral')
    
    # Plot 8: Covariance matrix heatmap (average error)
    ax8 = plt.subplot(3, 3, 8)
    error_matrix = jnp.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i <= j:  # Upper triangle
                if i == 0 and j == 0:
                    true_val = sigma_true[:, 0]
                    pred_val = eta2_true[:, 0, 0]
                elif i == 0 and j == 1:
                    true_val = sigma_true[:, 1]
                    pred_val = eta2_true[:, 0, 1]
                elif i == 0 and j == 2:
                    true_val = sigma_true[:, 2]
                    pred_val = eta2_true[:, 0, 2]
                elif i == 1 and j == 1:
                    true_val = sigma_true[:, 3]
                    pred_val = eta2_true[:, 1, 1]
                elif i == 1 and j == 2:
                    true_val = sigma_true[:, 4]
                    pred_val = eta2_true[:, 1, 2]
                else:  # i == 2 and j == 2
                    true_val = sigma_true[:, 5]
                    pred_val = eta2_true[:, 2, 2]
                
                error_matrix[i, j] = float(jnp.mean((true_val - pred_val)**2))
                error_matrix[j, i] = error_matrix[i, j]  # Make symmetric
    
    im = ax8.imshow(error_matrix, cmap='Reds', aspect='equal')
    ax8.set_title('Covariance Prediction Error Heatmap')
    ax8.set_xlabel('Index j')
    ax8.set_ylabel('Index i')
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            ax8.text(j, i, f'{error_matrix[i, j]:.4f}', 
                    ha='center', va='center', color='white' if error_matrix[i, j] > 0.5 else 'black')
    
    plt.colorbar(im, ax=ax8)
    
    # Plot 9: Overall statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Calculate overall statistics
    total_linear_mse = float(jnp.mean((mu_true - eta1_true)**2))
    total_quadratic_mse = float(jnp.mean((sigma_matrices_true - eta2_true)**2))
    total_mse = float(jnp.mean((y_true_val - eta_val)**2))
    
    stats_text = f"""
Training Statistics:
• Final Train MSE: {history['train_mse'][-1]:.6f}
• Final Val MSE: {history['val_mse'][-1]:.6f}

Component-wise MSE:
• Linear Terms MSE: {total_linear_mse:.6f}
• Quadratic Terms MSE: {total_quadratic_mse:.6f}
• Total MSE: {total_mse:.6f}

Data Summary:
• Train samples: {train_data['eta'].shape[0]}
• Val samples: {val_data['eta'].shape[0]}
• Input dim: {eta_val.shape[1]}
• Output dim: {y_true_val.shape[1]}
    """
    
    ax9.text(0.1, 0.9, stats_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plot_path = f"{save_dir}/3d_multivariate_comprehensive.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved comprehensive 3D multivariate results to: {plot_path}")
    
    # Print summary
    print(f"✅ 3D Multivariate Results Summary:")
    print(f"Final training MSE: {history['train_mse'][-1]:.6f}")
    print(f"Final validation MSE: {history['val_mse'][-1]:.6f}")
    print(f"Linear terms MSE: {total_linear_mse:.6f}")
    print(f"Quadratic terms MSE: {total_quadratic_mse:.6f}")
    print(f"Total MSE: {total_mse:.6f}")


def main():
    """Main function to run plotting."""
    data_file = "data/training_data_57c09a21b6b2ac3f0a9283b67f8bad02.pkl"
    history_file = "artifacts/large_3d_training_history_57c09a21b6b2ac3f0a9283b67f8bad02.pkl"
    
    if not Path(data_file).exists():
        print(f"❌ Data file {data_file} not found!")
        return
    
    if not Path(history_file).exists():
        print(f"❌ History file {history_file} not found!")
        return
    
    plot_3d_multivariate_results(data_file, history_file)


if __name__ == "__main__":
    main()
