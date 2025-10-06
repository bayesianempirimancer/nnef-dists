#!/usr/bin/env python3
"""
Enhanced 6-panel learning error analysis plot for any model type.
Provides comprehensive training analysis with model architecture, performance metrics,
and detailed error analysis.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# import seaborn as sns  # Not needed for this plot

# Import the model-agnostic loading function using relative imports
from ..load_model_and_data import load_model_and_data, make_predictions


def calculate_performance_metrics(y_true, y_pred):
    """Calculate comprehensive performance metrics."""
    metrics = {}
    
    # Flatten arrays for scalar metrics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Basic metrics
    metrics['mse'] = mean_squared_error(y_true_flat, y_pred_flat)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true_flat, y_pred_flat)
    metrics['r2'] = r2_score(y_true_flat, y_pred_flat)
    
    # Additional metrics
    metrics['mape'] = np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100
    metrics['max_error'] = np.max(np.abs(y_true_flat - y_pred_flat))
    
    # Per-dimension metrics for multi-dimensional outputs
    if y_true.shape[1] > 1:
        metrics['per_dim'] = {}
        for i in range(y_true.shape[1]):
            metrics['per_dim'][f'dim_{i}'] = {
                'mse': mean_squared_error(y_true[:, i], y_pred[:, i]),
                'mae': mean_absolute_error(y_true[:, i], y_pred[:, i]),
                'r2': r2_score(y_true[:, i], y_pred[:, i])
            }
    
    return metrics


def analyze_training_dynamics(train_losses, val_losses):
    """Analyze training dynamics and convergence."""
    analysis = {}
    
    # Basic statistics
    analysis['final_train_loss'] = train_losses[-1]
    analysis['final_val_loss'] = val_losses[-1]
    analysis['best_val_loss'] = min(val_losses)
    analysis['best_val_epoch'] = val_losses.index(min(val_losses))
    
    # Convergence analysis
    if len(train_losses) > 10:
        # Early vs late training
        early_train = np.mean(train_losses[:len(train_losses)//3])
        late_train = np.mean(train_losses[-len(train_losses)//3:])
        analysis['convergence_ratio'] = early_train / late_train
        
        # Overfitting indicator
        analysis['overfitting_ratio'] = val_losses[-1] / train_losses[-1]
        
        # Training stability (coefficient of variation in last 20% of training)
        last_20_percent = max(1, len(train_losses) // 5)
        analysis['train_stability'] = np.std(train_losses[-last_20_percent:]) / np.mean(train_losses[-last_20_percent:])
        analysis['val_stability'] = np.std(val_losses[-last_20_percent:]) / np.mean(val_losses[-last_20_percent:])
    
    return analysis


def create_enhanced_learning_plot(config, results, data, model, params, metadata, save_path=None):
    """Create enhanced 6-panel learning error analysis plot."""
    
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
    
    # Calculate MSE for all datasets (using predictions)
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
    
    # Calculate performance metrics
    train_metrics = calculate_performance_metrics(train_mu_T, train_pred)
    val_metrics = calculate_performance_metrics(val_mu_T, val_pred)
    test_metrics = calculate_performance_metrics(test_mu_T, test_pred)
    
    # Analyze training dynamics
    training_analysis = analyze_training_dynamics(results['train_losses'], results['val_losses'])
    
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
    model_name = config.get('model_name', 'Model').upper()
    model_type = config.get('model_type', 'unknown').upper()
    fig.suptitle(f'{model_name} ({model_type}) - Training Analysis', fontsize=16, y=0.95)
    
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
    best_epoch = training_analysis['best_val_epoch']
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
                   alpha=0.7, s=20, c='green', label='Test', marker='s')
        
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
                   alpha=0.7, s=20, c='green', label='Test', marker='s')
        
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
                   alpha=0.7, s=20, c='green', label='Test', marker='s')
        
        ax4.set_xlabel('True mu_T (dim 0)')
        ax4.set_ylabel('Error (dim 0)')
    else:
        # Single dimension
        ax4.scatter(train_mu_T_plot.flatten(), train_errors.flatten(), 
                   alpha=0.4, s=15, c='blue', label='Train')
        ax4.scatter(val_mu_T_plot.flatten(), val_errors.flatten(), 
                   alpha=0.7, s=20, c='red', label='Val', marker='^')
        ax4.scatter(test_mu_T_plot.flatten(), test_errors.flatten(), 
                   alpha=0.7, s=20, c='green', label='Test', marker='s')
        
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
    arch_text = f"Architecture:\n"
    arch_text += f"Input: {config['input_dim']}D\n"
    arch_text += f"Hidden: {' → '.join(map(str, config['hidden_sizes']))}\n"
    arch_text += f"Output: {config['output_dim']}D\n"
    arch_text += f"Activation: {config['activation']}\n"
    if config.get('use_resnet', False):
        arch_text += f"ResNet: {config.get('num_resnet_blocks', 0)} blocks\n"
    if config.get('dropout_rate', 0) > 0:
        arch_text += f"Dropout: {config['dropout_rate']}\n"
    
    arch_text += f"\nTraining:\n"
    arch_text += f"Epochs: {len(results['train_losses'])}\n"
    arch_text += f"Parameters: {results.get('parameter_count', 'N/A'):,}\n"
    arch_text += f"Time: {results.get('training_time', 0):.1f}s\n"
    arch_text += f"Inference: {results.get('inference_time_per_sample', 0)*1000:.2f}ms/sample\n"
    
    # Training dynamics
    arch_text += f"\nDynamics:\n"
    arch_text += f"Best Epoch: {training_analysis['best_val_epoch']}\n"
    overfitting_ratio = training_analysis.get('overfitting_ratio', 'N/A')
    convergence_ratio = training_analysis.get('convergence_ratio', 'N/A')
    overfitting_str = f"{overfitting_ratio:.3f}" if overfitting_ratio != 'N/A' else 'N/A'
    convergence_str = f"{convergence_ratio:.2f}" if convergence_ratio != 'N/A' else 'N/A'
    arch_text += f"Overfitting: {overfitting_str}\n"
    arch_text += f"Convergence: {convergence_str}"
    
    ax5.text(0.05, 0.95, arch_text, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
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
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    # Add overall summary at the bottom
    summary_text = f"Model: {config.get('model_name', 'Unknown')} | "
    summary_text += f"Final Train Loss: {results['final_train_loss']:.2e} | "
    summary_text += f"Final Val Loss: {results['final_val_loss']:.2e} | "
    summary_text += f"Best Val Loss: {results['best_val_loss']:.2e} | "
    summary_text += f"Test R²: {test_metrics['r2']:.4f} | "
    summary_text += f"Training Time: {results.get('training_time', 0):.1f}s"
    
    fig.text(0.5, 0.02, summary_text, fontsize=10, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Enhanced learning error analysis plot saved to: {save_path}")
    
    plt.close()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Create enhanced 6-panel learning error analysis plot")
    parser.add_argument("--model-dir", type=str, required=True, 
                       help="Path to model artifacts directory")
    parser.add_argument("--data", type=str, 
                       help="Path to training data pickle file (optional, will be inferred from results if not provided)")
    parser.add_argument("--save", type=str, 
                       help="Output file path (default: model_dir/learning_errors_enhanced.png)")
    
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
            save_path = save_dir / "learning_errors_enhanced.png"
        
        print("Creating enhanced learning error analysis plot...")
        create_enhanced_learning_plot(config, results, data, model, params, metadata, save_path)
        
        print("✅ Enhanced learning error analysis plotting complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()