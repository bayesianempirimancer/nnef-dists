"""
Standardized plotting functions for training results.

This module provides consistent plotting functionality for all training scripts,
ensuring uniform visualization across different models and experiments.
"""

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pickle


def plot_training_results(
    trainer: Any,
    eta_data: jnp.ndarray,
    ground_truth: jnp.ndarray,
    predictions: jnp.ndarray,
    losses: List[float],
    config: Any,
    model_name: str,
    output_dir: str = "artifacts",
    save_plots: bool = True,
    show_plots: bool = False
) -> Dict[str, float]:
    """
    Create comprehensive plots for training results.
    
    Args:
        trainer: The trained model trainer object
        eta_data: Input natural parameters
        ground_truth: Ground truth target values
        predictions: Model predictions
        losses: Training loss history
        config: Model configuration
        model_name: Name of the model for file naming
        output_dir: Directory to save plots
        save_plots: Whether to save plots to disk
        show_plots: Whether to display plots
        
    Returns:
        Dictionary of performance metrics
    """
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Learning curves
    ax1 = plt.subplot(2, 4, 1)
    plt.plot(losses, 'b-', linewidth=2)
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 2. Predictions vs Ground Truth (scatter)
    ax2 = plt.subplot(2, 4, 2)
    plt.scatter(ground_truth, predictions, alpha=0.6, s=20)
    min_val = min(ground_truth.min(), predictions.min())
    max_val = max(ground_truth.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title('Predictions vs Ground Truth', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 3. Residuals
    ax3 = plt.subplot(2, 4, 3)
    residuals = predictions - ground_truth
    plt.scatter(ground_truth, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Ground Truth')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Ground Truth', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 4. Per-statistic performance
    ax4 = plt.subplot(2, 4, 4)
    if ground_truth.ndim > 1 and ground_truth.shape[1] > 1:
        stat_names = [f'E[T_{i}]' for i in range(ground_truth.shape[1])]
        mse_per_stat = np.mean((predictions - ground_truth) ** 2, axis=0)
        bars = plt.bar(range(len(stat_names)), mse_per_stat)
        plt.xlabel('Statistics')
        plt.ylabel('MSE')
        plt.title('MSE per Statistic', fontsize=14, fontweight='bold')
        plt.xticks(range(len(stat_names)), stat_names, rotation=45)
    else:
        plt.text(0.5, 0.5, 'Single output\n(no per-statistic plot)', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        plt.title('MSE per Statistic', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 5. Model-specific analysis
    ax5 = plt.subplot(2, 4, 5)
    if hasattr(trainer, 'model') and hasattr(trainer, 'params'):
        try:
            # For LogZ models, show log normalizer distribution
            if 'logZ' in model_name.lower():
                log_normalizer_values = trainer.model.apply(trainer.params, eta_data, training=False)
                plt.hist(log_normalizer_values, bins=50, alpha=0.7, edgecolor='black')
                plt.xlabel('Log Normalizer A(η)')
                plt.ylabel('Frequency')
                plt.title('Distribution of Log Normalizer Values', fontsize=14, fontweight='bold')
            else:
                # For ET models, show prediction magnitude distribution
                pred_magnitudes = np.linalg.norm(predictions, axis=1) if predictions.ndim > 1 else np.abs(predictions)
                plt.hist(pred_magnitudes, bins=50, alpha=0.7, edgecolor='black')
                plt.xlabel('Prediction Magnitude')
                plt.ylabel('Frequency')
                plt.title('Distribution of Prediction Magnitudes', fontsize=14, fontweight='bold')
        except Exception as e:
            plt.text(0.5, 0.5, f'Model analysis\nnot available\n({str(e)[:30]}...)', 
                    ha='center', va='center', transform=ax5.transAxes, fontsize=10)
            plt.title('Model Analysis', fontsize=14, fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'Model analysis\nnot available', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        plt.title('Model Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 6. Gradient analysis (if available)
    ax6 = plt.subplot(2, 4, 6)
    if hasattr(trainer, 'compute_predictions') or hasattr(trainer, 'model'):
        try:
            if hasattr(trainer, 'compute_predictions'):
                gradients = trainer.compute_predictions(trainer.params, eta_data)
            else:
                gradients = predictions
            if gradients.ndim > 1:
                gradient_magnitudes = np.linalg.norm(gradients, axis=1)
            else:
                gradient_magnitudes = np.abs(gradients)
            plt.hist(gradient_magnitudes, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Gradient Magnitude')
            plt.ylabel('Frequency')
            plt.title('Distribution of Gradient Magnitudes', fontsize=14, fontweight='bold')
        except Exception as e:
            plt.text(0.5, 0.5, f'Gradient analysis\nnot available\n({str(e)[:30]}...)', 
                    ha='center', va='center', transform=ax6.transAxes, fontsize=10)
            plt.title('Gradient Analysis', fontsize=14, fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'Gradient analysis\nnot available', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        plt.title('Gradient Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 7. Input distribution
    ax7 = plt.subplot(2, 4, 7)
    if eta_data.shape[1] <= 2:
        if eta_data.shape[1] == 1:
            plt.hist(eta_data[:, 0], bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Natural Parameters η')
        else:
            plt.scatter(eta_data[:, 0], eta_data[:, 1], alpha=0.6, s=20)
            plt.xlabel('η₁')
            plt.ylabel('η₂')
        plt.title('Input Distribution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
    else:
        # Show first two components
        plt.scatter(eta_data[:, 0], eta_data[:, 1], alpha=0.6, s=20)
        plt.xlabel('η₁')
        plt.ylabel('η₂')
        plt.title(f'Input Distribution\n(first 2 of {eta_data.shape[1]} dims)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    # 8. Performance metrics
    ax8 = plt.subplot(2, 4, 8)
    mse = np.mean((predictions - ground_truth) ** 2)
    mae = np.mean(np.abs(predictions - ground_truth))
    
    # Calculate R² if we have multiple samples
    if ground_truth.shape[0] > 1:
        ss_res = np.sum((ground_truth - predictions) ** 2)
        ss_tot = np.sum((ground_truth - np.mean(ground_truth, axis=0)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
    else:
        r2 = np.nan
    
    metrics_text = f"""
MSE: {mse:.6f}
MAE: {mae:.6f}
R²: {r2:.6f}
Samples: {len(ground_truth)}
Model: {model_name}
"""
    
    plt.text(0.1, 0.5, metrics_text, transform=ax8.transAxes, 
             fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    plt.axis('off')
    plt.title('Performance Metrics', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plots:
        model_dir = Path(output_dir) / model_name.lower().replace('_', '_')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine file name based on model type
        if 'logz' in model_name.lower():
            family = getattr(config.network, 'exp_family', 'unknown')
            output_path = model_dir / f"{model_name.lower()}_training_results_{family}.png"
        else:
            output_path = model_dir / f"{model_name.lower()}_training_results.png"
            
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Training results saved to: {output_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Return performance metrics
    return {
        'mse': float(mse),
        'mae': float(mae),
        'r2': float(r2),
        'final_loss': float(losses[-1]) if losses else np.nan
    }


def plot_model_comparison(
    results: Dict[str, Dict[str, Any]],
    output_dir: str = "artifacts",
    save_plots: bool = True,
    show_plots: bool = False
) -> None:
    """
    Create comparison plots for multiple models.
    
    Args:
        results: Dictionary of model results with structure {model_name: {metrics, losses, etc.}}
        output_dir: Directory to save plots
        save_plots: Whether to save plots to disk
        show_plots: Whether to display plots
    """
    
    if not results:
        print("No results to plot")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Training curves comparison
    ax1 = axes[0, 0]
    for model_name, result in results.items():
        if 'losses' in result:
            ax1.plot(result['losses'], label=model_name, alpha=0.7, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Curves Comparison')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 2. MSE comparison
    ax2 = axes[0, 1]
    model_names = list(results.keys())
    mse_values = [results[name].get('mse', np.nan) for name in model_names]
    bars = ax2.bar(range(len(model_names)), mse_values)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45)
    ax2.set_ylabel('MSE')
    ax2.set_title('MSE Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Color bars by performance
    for i, (bar, mse) in enumerate(zip(bars, mse_values)):
        if not np.isnan(mse):
            bar.set_color(plt.cm.viridis(1 - (mse - min(mse_values)) / (max(mse_values) - min(mse_values))))
    
    # 3. MAE comparison
    ax3 = axes[1, 0]
    mae_values = [results[name].get('mae', np.nan) for name in model_names]
    bars = ax3.bar(range(len(model_names)), mae_values)
    ax3.set_xticks(range(len(model_names)))
    ax3.set_xticklabels(model_names, rotation=45)
    ax3.set_ylabel('MAE')
    ax3.set_title('MAE Comparison')
    ax3.grid(True, alpha=0.3)
    
    # Color bars by performance
    for i, (bar, mae) in enumerate(zip(bars, mae_values)):
        if not np.isnan(mae):
            bar.set_color(plt.cm.viridis(1 - (mae - min(mae_values)) / (max(mae_values) - min(mae_values))))
    
    # 4. R² comparison
    ax4 = axes[1, 1]
    r2_values = [results[name].get('r2', np.nan) for name in model_names]
    bars = ax4.bar(range(len(model_names)), r2_values)
    ax4.set_xticks(range(len(model_names)))
    ax4.set_xticklabels(model_names, rotation=45)
    ax4.set_ylabel('R²')
    ax4.set_title('R² Comparison')
    ax4.grid(True, alpha=0.3)
    
    # Color bars by performance (higher R² is better)
    for i, (bar, r2) in enumerate(zip(bars, r2_values)):
        if not np.isnan(r2):
            # Normalize R² values for coloring (assume range [-1, 1])
            normalized_r2 = (r2 + 1) / 2  # Map [-1, 1] to [0, 1]
            bar.set_color(plt.cm.viridis(normalized_r2))
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plots:
        output_path = Path(output_dir) / "model_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Model comparison saved to: {output_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def save_results_summary(
    results: Dict[str, Dict[str, Any]],
    output_dir: str = "artifacts"
) -> None:
    """
    Save training results summary to a pickle file.
    
    Args:
        results: Dictionary of model results
        output_dir: Directory to save results
    """
    
    output_path = Path(output_dir) / "training_results_summary.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results summary saved to: {output_path}")


def load_results_summary(
    output_dir: str = "artifacts"
) -> Dict[str, Dict[str, Any]]:
    """
    Load training results summary from a pickle file.
    
    Args:
        output_dir: Directory containing results
        
    Returns:
        Dictionary of model results
    """
    
    input_path = Path(output_dir) / "training_results_summary.pkl"
    
    if not input_path.exists():
        print(f"No results summary found at {input_path}")
        return {}
    
    with open(input_path, 'rb') as f:
        results = pickle.load(f)
    
    return results
