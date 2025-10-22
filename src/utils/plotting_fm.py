#!/usr/bin/env python3
"""
Specialized plotting functions for Flow Matching and Continuous Time models.

This module provides plotting utilities specifically designed for diffusion-like models
that use the (z, x, t) interface, such as NoProp CT and FM models.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
import jax.numpy as jnp


def make_predictions_fm(model, params, eta_data: np.ndarray, batch_size: int = 100) -> np.ndarray:
    """
    Make predictions using a diffusion model with predict method.
    
    Args:
        model: The diffusion model (CT or FM)
        params: Model parameters
        eta_data: Input data (natural parameters)
        batch_size: Batch size for prediction
        
    Returns:
        Predictions as numpy array
    """
    predictions = []
    
    for i in range(0, len(eta_data), batch_size):
        batch_eta = eta_data[i:i+batch_size]
        batch_eta_jnp = jnp.array(batch_eta)
        
        # Use the model's predict method which handles the full diffusion process
        pred = model.predict(params, batch_eta_jnp, num_steps=50)  # Use 50 steps for reasonable speed
        predictions.append(np.array(pred))
    
    return np.vstack(predictions)


def calculate_performance_metrics_fm(y_true, y_pred):
    """
    Calculate performance metrics for diffusion model predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true))
    
    # R² calculation
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # MAPE calculation (avoid division by zero)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }


def create_learning_plot_fm(
    config: Dict[str, Any],
    results: Dict[str, Any], 
    data: Dict[str, Any],
    model: Any,
    params: Any,
    metadata: Dict[str, Any],
    save_path: Optional[str] = None
) -> None:
    """
    Create a comprehensive learning plot for diffusion models (CT/FM).
    
    Args:
        config: Model configuration
        results: Training results
        data: Training data
        model: Trained diffusion model
        params: Model parameters
        metadata: Data metadata
        save_path: Path to save the plot (optional)
    """
    if data is None:
        raise ValueError("Data is required for creating learning plots")
    
    # Prepare data
    train_eta = np.array(data['train']['eta'])
    train_mu_T = np.array(data['train']['mu_T'])
    val_eta = np.array(data['val']['eta'])
    val_mu_T = np.array(data['val']['mu_T'])
    test_eta = np.array(data['test']['eta'])
    test_mu_T = np.array(data['test']['mu_T'])
    
    # Make predictions using diffusion model interface
    print("Making predictions for training data...")
    train_pred = make_predictions_fm(model, params, train_eta)
    print("Making predictions for validation data...")
    val_pred = make_predictions_fm(model, params, val_eta)
    print("Making predictions for test data...")
    test_pred = make_predictions_fm(model, params, test_eta)
    
    # Calculate MSE for all datasets
    train_mse = np.mean((train_pred - train_mu_T) ** 2)
    val_mse = np.mean((val_pred - val_mu_T) ** 2)
    test_mse = np.mean((test_pred - test_mu_T) ** 2)
    
    # Get theoretical minimum MSE if available
    theoretical_min_mse_train = metadata.get('total_expected_MSE_train')
    theoretical_min_mse_val = metadata.get('total_expected_MSE_val')
    
    if theoretical_min_mse_train is None and 'expected_MSE' in data['train']:
        theoretical_min_mse_train = np.mean(data['train']['expected_MSE'])
    if theoretical_min_mse_val is None and 'expected_MSE' in data['val']:
        theoretical_min_mse_val = np.mean(data['val']['expected_MSE'])
    
    # Compute theoretical minimum MSE from covariance matrices and ESS if not available
    if theoretical_min_mse_train is None and 'cov_TT' in data['train'] and 'ess' in data['train']:
        # Expected MSE is mean(diag(cov_TT))/ess averaged over batch
        train_cov_TT = np.array(data['train']['cov_TT'])
        train_ess = np.array(data['train']['ess'])
        diag_cov = np.diagonal(train_cov_TT, axis1=-1, axis2=-2)  # (n_samples, dim)
        theoretical_min_mse_train = np.mean(np.mean(diag_cov, axis=-1)/train_ess)

    if theoretical_min_mse_val is None and 'cov_TT' in data['val'] and 'ess' in data['val']:
        val_cov_TT = np.array(data['val']['cov_TT'])
        val_ess = np.array(data['val']['ess'])
        diag_cov = np.diagonal(val_cov_TT, axis1=-1, axis2=-2)  # (n_samples, dim)
        theoretical_min_mse_val = np.mean(np.mean(diag_cov, axis=-1)/val_ess)
    
    # Calculate performance metrics
    train_metrics = calculate_performance_metrics_fm(train_mu_T, train_pred)
    val_metrics = calculate_performance_metrics_fm(val_mu_T, val_pred)
    test_metrics = calculate_performance_metrics_fm(test_mu_T, test_pred)
    
    # Sample subset for plotting (to avoid overcrowding)
    n_train_plot = min(300, len(train_eta))
    n_val_plot = min(150, len(val_eta))
    n_test_plot = min(150, len(test_eta))
    
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
    
    # Create figure with 6 panels
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Main title with model info
    model_name = config.get('model_name', 'Diffusion Model').upper()
    fig.suptitle(f'{model_name} - Training Analysis (Diffusion Model)', fontsize=16, y=0.95)
    
    # Panel 1: Loss History (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    epochs = range(len(results['train_losses']))
    
    # Plot losses
    ax1.plot(epochs, results['train_losses'], 'b-', label='Training Loss', alpha=0.8, linewidth=2)
    ax1.plot(epochs, results['val_losses'], 'r-', label='Validation Loss', alpha=0.8, linewidth=2)
    
    # Add theoretical minimum lines if available
    if theoretical_min_mse_train is not None:
        ax1.axhline(y=theoretical_min_mse_train, color='b', linestyle='--', alpha=0.6, 
                   label=f'Theoretical Min Train: {theoretical_min_mse_train:.2e}')
    if theoretical_min_mse_val is not None:
        ax1.axhline(y=theoretical_min_mse_val, color='r', linestyle='--', alpha=0.6,
                   label=f'Theoretical Min Val: {theoretical_min_mse_val:.2e}')
    
    # Highlight best validation epoch
    best_epoch = results['val_losses'].index(min(results['val_losses']))
    ax1.axvline(x=best_epoch, color='orange', linestyle=':', alpha=0.7, 
               label=f'Best Val Epoch: {best_epoch}')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training History')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Panel 2: MSE Bar Plot (top-center)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Prepare data for bar plot
    datasets = ['Train', 'Val', 'Test']
    mse_values = [train_mse, val_mse, test_mse]
    colors = ['blue', 'red', 'green']
    
    bars = ax2.bar(datasets, mse_values, color=colors, alpha=0.7)
    
    # Add theoretical minimum as horizontal lines if available
    if theoretical_min_mse_train is not None:
        ax2.axhline(y=theoretical_min_mse_train, color='blue', linestyle='--', alpha=0.6, 
                   label=f'Min Train: {theoretical_min_mse_train:.2e}')
    if theoretical_min_mse_val is not None:
        ax2.axhline(y=theoretical_min_mse_val, color='red', linestyle='--', alpha=0.6,
                   label=f'Min Val: {theoretical_min_mse_val:.2e}')
    
    # Add value labels on bars
    for bar, value in zip(bars, mse_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2e}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_ylabel('MSE')
    ax2.set_title('Mean Squared Error')
    ax2.set_yscale('log')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Predicted vs True (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    
    # For multi-dimensional output, plot first dimension
    if train_mu_T_plot.shape[1] > 1:
        ax3.scatter(train_mu_T_plot[:, 0], train_pred_plot[:, 0], 
                   alpha=0.4, s=15, c='blue', label='Train')
        ax3.scatter(val_mu_T_plot[:, 0], val_pred_plot[:, 0], 
                   alpha=0.7, s=20, c='red', label='Val', marker='^')
        ax3.scatter(test_mu_T_plot[:, 0], test_pred_plot[:, 0], 
                   alpha=0.7, s=20, c='green', label='Test', marker='D')
        
        # Perfect prediction line
        min_val = min(train_mu_T_plot[:, 0].min(), val_mu_T_plot[:, 0].min(), test_mu_T_plot[:, 0].min())
        max_val = max(train_mu_T_plot[:, 0].max(), val_mu_T_plot[:, 0].max(), test_mu_T_plot[:, 0].max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect')
        
        ax3.set_xlabel('True mu_T (dim 0)')
        ax3.set_ylabel('Predicted mu_T (dim 0)')
    else:
        ax3.scatter(train_mu_T_plot.flatten(), train_pred_plot.flatten(), 
                   alpha=0.4, s=15, c='blue', label='Train')
        ax3.scatter(val_mu_T_plot.flatten(), val_pred_plot.flatten(), 
                   alpha=0.7, s=20, c='red', label='Val', marker='^')
        ax3.scatter(test_mu_T_plot.flatten(), test_pred_plot.flatten(), 
                   alpha=0.7, s=20, c='green', label='Test', marker='D')
        
        # Perfect prediction line
        min_val = min(train_mu_T_plot.min(), val_mu_T_plot.min(), test_mu_T_plot.min())
        max_val = max(train_mu_T_plot.max(), val_mu_T_plot.max(), test_mu_T_plot.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect')
        
        ax3.set_xlabel('True mu_T')
        ax3.set_ylabel('Predicted mu_T')
    
    ax3.set_title('Predicted vs True')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Error Analysis (bottom-left)
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Calculate errors
    train_errors = train_pred_plot - train_mu_T_plot
    val_errors = val_pred_plot - val_mu_T_plot
    test_errors = test_pred_plot - test_mu_T_plot
    
    # For multi-dimensional output, plot first dimension
    if train_errors.shape[1] > 1:
        # Plot first dimension
        ax4.scatter(train_mu_T_plot[:, 0], train_errors[:, 0], 
                   alpha=0.4, s=15, c='blue', label='Train')
        ax4.scatter(val_mu_T_plot[:, 0], val_errors[:, 0], 
                   alpha=0.7, s=20, c='red', label='Val', marker='^')
        ax4.scatter(test_mu_T_plot[:, 0], test_errors[:, 0], 
                   alpha=0.7, s=20, c='green', label='Test', marker='D')
        
        ax4.set_xlabel('True mu_T (dim 0)')
        ax4.set_ylabel('Error (dim 0)')
    else:
        # Single dimension
        ax4.scatter(train_mu_T_plot.flatten(), train_errors.flatten(), 
                   alpha=0.4, s=15, c='blue', label='Train')
        ax4.scatter(val_mu_T_plot.flatten(), val_errors.flatten(), 
                   alpha=0.7, s=20, c='red', label='Val', marker='^')
        ax4.scatter(test_mu_T_plot.flatten(), test_errors.flatten(), 
                   alpha=0.7, s=20, c='green', label='Test', marker='D')
        
        ax4.set_xlabel('True mu_T')
        ax4.set_ylabel('Error')
    
    # Zero line
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.set_title('Error Analysis')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Architecture Info (bottom-center)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    
    # Model architecture info
    arch_text = f"Diffusion Model Architecture:\n"
    arch_text += f"Model Type: {config.get('model_type', 'Unknown')}\n"
    arch_text += f"Z Shape: {config.get('z_shape', 'Unknown')}\n"
    arch_text += f"X Shape: {config.get('x_shape', 'Unknown')}\n"
    arch_text += f"Hidden Dims: {config.get('model_hidden_dims', 'Unknown')}\n"
    arch_text += f"Time Embed: {config.get('time_embed_method', 'Unknown')}\n"
    arch_text += f"Noise Schedule: {config.get('noise_schedule', 'Unknown')}\n"
    if config.get('model_dropout_rate', 0) > 0:
        arch_text += f"Dropout: {config['model_dropout_rate']}\n"
    
    arch_text += f"\nTraining:\n"
    arch_text += f"Epochs: {len(results['train_losses'])}\n"
    param_count = results.get('parameter_count', results.get('param_count', 'N/A'))
    if param_count != 'N/A':
        arch_text += f"Parameters: {param_count:,}\n"
    else:
        arch_text += f"Parameters: N/A\n"
    arch_text += f"Time: {results.get('training_time', 0):.1f}s\n"
    arch_text += f"Inference: {results.get('inference_time_per_sample', 0)*1000:.2f}ms/sample\n"
    
    # Training dynamics
    arch_text += f"\nDynamics:\n"
    arch_text += f"Best Epoch: {best_epoch}\n"
    overfitting_ratio = results['val_losses'][-1] / results['train_losses'][-1] if len(results['train_losses']) > 0 and len(results['val_losses']) > 0 else 'N/A'
    overfitting_str = f"{overfitting_ratio:.3f}" if overfitting_ratio != 'N/A' else 'N/A'
    arch_text += f"Overfitting: {overfitting_str}"
    
    ax5.text(0.05, 0.95, arch_text, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    # Panel 6: Performance Metrics (bottom-right)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    metrics_text = f"Performance Metrics:\n\n"
    metrics_text += f"Training:\n"
    metrics_text += f"  MSE: {train_metrics['mse']:.2e}\n"
    metrics_text += f"  RMSE: {train_metrics['rmse']:.2e}\n"
    metrics_text += f"  MAE: {train_metrics['mae']:.2e}\n"
    metrics_text += f"  R²: {train_metrics['r2']:.4f}\n"
    metrics_text += f"  MAPE: {train_metrics['mape']:.2f}%\n\n"
    
    metrics_text += f"Validation:\n"
    metrics_text += f"  MSE: {val_metrics['mse']:.2e}\n"
    metrics_text += f"  RMSE: {val_metrics['rmse']:.2e}\n"
    metrics_text += f"  MAE: {val_metrics['mae']:.2e}\n"
    metrics_text += f"  R²: {val_metrics['r2']:.4f}\n"
    metrics_text += f"  MAPE: {val_metrics['mape']:.2f}%\n\n"
    
    metrics_text += f"Test:\n"
    metrics_text += f"  MSE: {test_metrics['mse']:.2e}\n"
    metrics_text += f"  RMSE: {test_metrics['rmse']:.2e}\n"
    metrics_text += f"  MAE: {test_metrics['mae']:.2e}\n"
    metrics_text += f"  R²: {test_metrics['r2']:.4f}\n"
    metrics_text += f"  MAPE: {test_metrics['mape']:.2f}%"
    
    ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
    
    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Diffusion model learning analysis plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Example usage
    print("Diffusion Model Plotting Utilities")
    print("This module provides specialized plotting for CT/FM models.")
