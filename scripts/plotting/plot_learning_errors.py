#!/usr/bin/env python3
"""
4-panel learning error analysis plot for any model type.
Automatically infers data file location from training results.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import the model-agnostic loading function using absolute imports
from scripts.load_model_and_data import load_model_and_data, make_predictions


def create_learning_errors_plot(config, results, data, model, params, metadata, save_path=None):
    """Create the 4-panel learning error analysis plot."""
    
    if data is None:
        raise ValueError("Data is required for creating learning error plots. Please provide a data file.")
    
    # Prepare data
    train_eta = np.array(data['train']['eta'])
    train_mu_T = np.array(data['train']['mu_T'])
    val_eta = np.array(data['val']['eta'])
    val_mu_T = np.array(data['val']['mu_T'])
    test_eta = np.array(data['test']['eta'])
    test_mu_T = np.array(data['test']['mu_T'])
    
    # Make predictions
    print("Making predictions on training data...")
    train_pred = make_predictions(model, params, train_eta)
    print("Making predictions on validation data...")
    val_pred = make_predictions(model, params, val_eta)
    print("Making predictions on test data...")
    test_pred = make_predictions(model, params, test_eta)
    
    # Sample subset for plotting (to avoid overcrowding)
    n_train_plot = min(200, len(train_eta))
    n_val_plot = min(100, len(val_eta))
    n_test_plot = min(100, len(test_eta))
    
    train_indices = np.random.choice(len(train_eta), n_train_plot, replace=False)
    val_indices = np.random.choice(len(val_eta), n_val_plot, replace=False)
    test_indices = np.random.choice(len(test_eta), n_test_plot, replace=False)
    
    train_eta_plot = train_eta[train_indices]
    train_mu_T_plot = train_mu_T[train_indices]
    train_pred_plot = train_pred[train_indices]
    
    val_eta_plot = val_eta[val_indices]
    val_mu_T_plot = val_mu_T[val_indices]
    val_pred_plot = val_pred[val_indices]
    
    test_eta_plot = test_eta[test_indices]
    test_mu_T_plot = test_mu_T[test_indices]
    test_pred_plot = test_pred[test_indices]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{config.get("model_name", "Model")} - Learning Error Analysis', fontsize=16, y=0.98)
    
    # Panel 1: Loss history with theoretical minimum
    ax1 = axes[0, 0]
    epochs = range(len(results['train_losses']))
    ax1.plot(epochs, results['train_losses'], 'b-', label='Training Loss', alpha=0.8, linewidth=2)
    ax1.plot(epochs, results['val_losses'], 'r-', label='Validation Loss', alpha=0.8, linewidth=2)
    
    # Add theoretical minimum lines if available
    # First try to get from metadata
    theoretical_min_mse_train = metadata.get('total_expected_MSE_train')
    theoretical_min_mse_val = metadata.get('total_expected_MSE_val')
    
    # If not in metadata, compute from data dictionaries
    if theoretical_min_mse_train is None and 'expected_MSE' in data['train']:
        theoretical_min_mse_train = np.mean(data['train']['expected_MSE'])
    if theoretical_min_mse_val is None and 'expected_MSE' in data['val']:
        theoretical_min_mse_val = np.mean(data['val']['expected_MSE'])
    
    if theoretical_min_mse_train is not None:
        ax1.axhline(y=theoretical_min_mse_train, color='b', linestyle='--', alpha=0.6, 
                   label=f'Theoretical Min Train: {theoretical_min_mse_train:.2e}')
    if theoretical_min_mse_val is not None:
        ax1.axhline(y=theoretical_min_mse_val, color='r', linestyle='--', alpha=0.6,
                   label=f'Theoretical Min Val: {theoretical_min_mse_val:.2e}')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Panel 2: Predicted mu_T vs True mu_T
    ax2 = axes[0, 1]
    
    # For multi-dimensional output, plot first dimension
    if train_mu_T_plot.shape[1] > 1:
        # Plot first dimension
        ax2.scatter(train_mu_T_plot[:, 0], train_pred_plot[:, 0], 
                   alpha=0.6, s=20, c='blue', label='Train (dim 0)')
        ax2.scatter(val_mu_T_plot[:, 0], val_pred_plot[:, 0], 
                   alpha=0.8, s=30, c='red', label='Val (dim 0)', marker='^')
        ax2.scatter(test_mu_T_plot[:, 0], test_pred_plot[:, 0], 
                   alpha=0.8, s=30, c='green', label='Test (dim 0)', marker='s')
        
        # Perfect prediction line
        min_val = min(train_mu_T_plot[:, 0].min(), val_mu_T_plot[:, 0].min(), test_mu_T_plot[:, 0].min())
        max_val = max(train_mu_T_plot[:, 0].max(), val_mu_T_plot[:, 0].max(), test_mu_T_plot[:, 0].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
        
        ax2.set_xlabel('True mu_T (dim 0)')
        ax2.set_ylabel('Predicted mu_T (dim 0)')
    else:
        # Single dimension
        ax2.scatter(train_mu_T_plot.flatten(), train_pred_plot.flatten(), 
                   alpha=0.6, s=20, c='blue', label='Train')
        ax2.scatter(val_mu_T_plot.flatten(), val_pred_plot.flatten(), 
                   alpha=0.8, s=30, c='red', label='Val', marker='^')
        ax2.scatter(test_mu_T_plot.flatten(), test_pred_plot.flatten(), 
                   alpha=0.8, s=30, c='green', label='Test', marker='s')
        
        # Perfect prediction line
        min_val = min(train_mu_T_plot.min(), val_mu_T_plot.min(), test_mu_T_plot.min())
        max_val = max(train_mu_T_plot.max(), val_mu_T_plot.max(), test_mu_T_plot.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
        
        ax2.set_xlabel('True mu_T')
        ax2.set_ylabel('Predicted mu_T')
    
    ax2.set_title('Predicted vs True mu_T')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Error vs True mu_T
    ax3 = axes[1, 0]
    
    # Calculate errors
    train_errors = train_pred_plot - train_mu_T_plot
    val_errors = val_pred_plot - val_mu_T_plot
    test_errors = test_pred_plot - test_mu_T_plot
    
    # For multi-dimensional output, plot first dimension
    if train_errors.shape[1] > 1:
        # Plot first dimension
        ax3.scatter(train_mu_T_plot[:, 0], train_errors[:, 0], 
                   alpha=0.6, s=20, c='blue', label='Train (dim 0)')
        ax3.scatter(val_mu_T_plot[:, 0], val_errors[:, 0], 
                   alpha=0.8, s=30, c='red', label='Val (dim 0)', marker='^')
        ax3.scatter(test_mu_T_plot[:, 0], test_errors[:, 0], 
                   alpha=0.8, s=30, c='green', label='Test (dim 0)', marker='s')
        
        ax3.set_xlabel('True mu_T (dim 0)')
        ax3.set_ylabel('Error (dim 0)')
    else:
        # Single dimension
        ax3.scatter(train_mu_T_plot.flatten(), train_errors.flatten(), 
                   alpha=0.6, s=20, c='blue', label='Train')
        ax3.scatter(val_mu_T_plot.flatten(), val_errors.flatten(), 
                   alpha=0.8, s=30, c='red', label='Val', marker='^')
        ax3.scatter(test_mu_T_plot.flatten(), test_errors.flatten(), 
                   alpha=0.8, s=30, c='green', label='Test', marker='s')
        
        ax3.set_xlabel('True mu_T')
        ax3.set_ylabel('Error')
    
    # Zero line
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.set_title('Error vs True mu_T')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Error vs abs(eta)
    ax4 = axes[1, 1]
    
    # Calculate eta norms and error norms
    train_eta_norms = np.linalg.norm(train_eta_plot, axis=1)
    val_eta_norms = np.linalg.norm(val_eta_plot, axis=1)
    test_eta_norms = np.linalg.norm(test_eta_plot, axis=1)
    
    train_error_norms = np.linalg.norm(train_errors, axis=1)
    val_error_norms = np.linalg.norm(val_errors, axis=1)
    test_error_norms = np.linalg.norm(test_errors, axis=1)
    
    ax4.scatter(train_eta_norms, train_error_norms, 
               alpha=0.6, s=20, c='blue', label='Train')
    ax4.scatter(val_eta_norms, val_error_norms, 
               alpha=0.8, s=30, c='red', label='Val', marker='^')
    ax4.scatter(test_eta_norms, test_error_norms, 
               alpha=0.8, s=30, c='green', label='Test', marker='s')
    
    ax4.set_xlabel('||eta||')
    ax4.set_ylabel('||Error||')
    ax4.set_title('Error vs ||eta||')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # Add summary statistics to the plot
    summary_text = f'Final Train MSE: {results["final_train_loss"]:.6f}\n'
    summary_text += f'Final Val MSE: {results["final_val_loss"]:.6f}\n'
    summary_text += f'Best Val MSE: {results["best_val_loss"]:.6f}\n'
    summary_text += f'Inference: {results.get("inference_time_per_sample", 0)*1000:.2f}ms/sample'
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning error analysis plot saved to: {save_path}")
    
    plt.close()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Create 4-panel learning error analysis plot")
    parser.add_argument("--model-dir", type=str, required=True, 
                       help="Path to model artifacts directory")
    parser.add_argument("--data", type=str, 
                       help="Path to training data pickle file (optional, will be inferred from results if not provided)")
    parser.add_argument("--save", type=str, 
                       help="Output file path (default: model_dir/learning_errors.png)")
    
    args = parser.parse_args()
    
    # Set random seed for reproducible sampling
    np.random.seed(42)
    
    print(f"Loading model from {args.model_dir}")
    
    try:
        # Load everything using the model-agnostic function
        config, results, data, model, params, metadata = load_model_and_data(args.model_dir, args.data)
        
        # Set save path
        if args.save:
            save_path = args.save
        else:
            save_dir = Path(args.model_dir)
            save_path = save_dir / "learning_errors.png"
        
        print("Creating learning error analysis plot...")
        create_learning_errors_plot(config, results, data, model, params, metadata, save_path)
        
        print("✅ Learning error analysis plotting complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
