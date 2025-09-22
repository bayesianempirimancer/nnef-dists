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


def create_standardized_results(model_name, architecture_info, metrics, losses, 
                               training_time, inference_stats, predictions, ground_truth):
    """
    Create a standardized results dictionary for all models.
    
    Args:
        model_name: Name of the model (e.g., "MLP_ET_Medium", "MLP_LogZ_Medium")
        architecture_info: Dictionary with architecture details (e.g., {"hidden_sizes": [64, 64]})
        metrics: Dictionary with evaluation metrics (mse, mae, r2, etc.)
        losses: List of training losses
        training_time: Training time in seconds
        inference_stats: Dictionary with inference timing stats
        predictions: Model predictions (will be converted to list)
        ground_truth: Ground truth values (will be converted to list)
    
    Returns:
        Dictionary with standardized results structure
    """
    return {
        **architecture_info,  # Include architecture details (hidden_sizes, etc.)
        "mse": metrics["mse"],
        "mae": metrics["mae"],
        "r2": metrics.get("r2", np.nan),
        "final_loss": metrics.get("final_loss", losses[-1] if losses else np.nan),
        "losses": losses,
        "training_time": training_time,
        "avg_inference_time": inference_stats['avg_inference_time'],
        "inference_per_sample": inference_stats['inference_per_sample'],
        "samples_per_second": inference_stats['samples_per_second'],
        "predictions": np.array(predictions).tolist(),
        "ground_truth": np.array(ground_truth).tolist()
    }


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
    Create comprehensive comparison plots for multiple models.
    
    Args:
        results: Dictionary of model results with structure {model_name: {metrics, losses, predictions, ground_truth, etc.}}
        output_dir: Directory to save plots
        save_plots: Whether to save plots to disk
        show_plots: Whether to display plots
    """
    
    if not results:
        print("No results to plot")
        return
    
    # Create comprehensive comparison plots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Training curves comparison (top left)
    ax1 = plt.subplot(3, 3, 1)
    for model_name, result in results.items():
        if 'losses' in result and result['losses']:
            ax1.plot(result['losses'], label=model_name, alpha=0.7, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Curves Comparison', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 2. MSE comparison (top center)
    ax2 = plt.subplot(3, 3, 2)
    model_names = list(results.keys())
    mse_values = [results[name].get('mse', np.nan) for name in model_names]
    bars = ax2.bar(range(len(model_names)), mse_values)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.set_ylabel('MSE')
    ax2.set_title('MSE Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Color bars by performance (lower MSE is better)
    valid_mse = [mse for mse in mse_values if not np.isnan(mse)]
    if valid_mse:
        mse_range = max(valid_mse) - min(valid_mse)
        for i, (bar, mse) in enumerate(zip(bars, mse_values)):
            if not np.isnan(mse) and mse_range > 0:
                # Lower MSE gets better color (closer to 1 in viridis)
                normalized_mse = 1 - (mse - min(valid_mse)) / mse_range
                bar.set_color(plt.cm.viridis(normalized_mse))
            elif not np.isnan(mse):
                bar.set_color(plt.cm.viridis(0.5))
    
    # 3. MAE comparison (top right)
    ax3 = plt.subplot(3, 3, 3)
    mae_values = [results[name].get('mae', np.nan) for name in model_names]
    bars = ax3.bar(range(len(model_names)), mae_values)
    ax3.set_xticks(range(len(model_names)))
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.set_ylabel('MAE')
    ax3.set_title('MAE Comparison', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Color bars by performance (lower MAE is better)
    valid_mae = [mae for mae in mae_values if not np.isnan(mae)]
    if valid_mae:
        mae_range = max(valid_mae) - min(valid_mae)
        for i, (bar, mae) in enumerate(zip(bars, mae_values)):
            if not np.isnan(mae) and mae_range > 0:
                normalized_mae = 1 - (mae - min(valid_mae)) / mae_range
                bar.set_color(plt.cm.viridis(normalized_mae))
            elif not np.isnan(mae):
                bar.set_color(plt.cm.viridis(0.5))
    
    # 4. R² comparison (middle left)
    ax4 = plt.subplot(3, 3, 4)
    r2_values = [results[name].get('r2', np.nan) for name in model_names]
    bars = ax4.bar(range(len(model_names)), r2_values)
    ax4.set_xticks(range(len(model_names)))
    ax4.set_xticklabels(model_names, rotation=45, ha='right')
    ax4.set_ylabel('R²')
    ax4.set_title('R² Comparison', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Color bars by performance (higher R² is better)
    valid_r2 = [r2 for r2 in r2_values if not np.isnan(r2)]
    if valid_r2:
        r2_range = max(valid_r2) - min(valid_r2)
        for i, (bar, r2) in enumerate(zip(bars, r2_values)):
            if not np.isnan(r2) and r2_range > 0:
                normalized_r2 = (r2 - min(valid_r2)) / r2_range
                bar.set_color(plt.cm.viridis(normalized_r2))
            elif not np.isnan(r2):
                bar.set_color(plt.cm.viridis(0.5))
    
    # 5. Performance summary table (middle center)
    ax5 = plt.subplot(3, 3, 5)
    ax5.axis('off')
    
    # Create performance summary table
    table_data = []
    for name in model_names:
        mse = results[name].get('mse', np.nan)
        mae = results[name].get('mae', np.nan)
        r2 = results[name].get('r2', np.nan)
        table_data.append([
            name,
            f"{mse:.4f}" if not np.isnan(mse) else "N/A",
            f"{mae:.4f}" if not np.isnan(mae) else "N/A",
            f"{r2:.4f}" if not np.isnan(r2) else "N/A"
        ])
    
    table = ax5.table(cellText=table_data,
                     colLabels=['Model', 'MSE', 'MAE', 'R²'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax5.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    # 6. High performing model predictions vs ground truth (middle right)
    ax6 = plt.subplot(3, 3, 6)
    
    # Find best performing model (lowest MSE)
    best_model = None
    best_mse = float('inf')
    for name, result in results.items():
        mse = result.get('mse', np.nan)
        if not np.isnan(mse) and mse < best_mse:
            best_mse = mse
            best_model = name
    
    if best_model and 'predictions' in results[best_model] and 'ground_truth' in results[best_model]:
        predictions = np.array(results[best_model]['predictions'])
        ground_truth = np.array(results[best_model]['ground_truth'])
        
        # Flatten if multi-dimensional
        if predictions.ndim > 1:
            predictions = predictions.flatten()
            ground_truth = ground_truth.flatten()
        
        ax6.scatter(ground_truth, predictions, alpha=0.6, s=20, color='green')
        min_val = min(ground_truth.min(), predictions.min())
        max_val = max(ground_truth.max(), predictions.max())
        ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax6.set_xlabel('Ground Truth')
        ax6.set_ylabel('Predictions')
        ax6.set_title(f'Best Model: {best_model}\nMSE: {best_mse:.4f}', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No prediction data\navailable for best model', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Best Model Predictions', fontsize=12, fontweight='bold')
    
    # 7. Low performing model predictions vs ground truth (bottom left)
    ax7 = plt.subplot(3, 3, 7)
    
    # Find worst performing model (highest MSE)
    worst_model = None
    worst_mse = float('-inf')
    for name, result in results.items():
        mse = result.get('mse', np.nan)
        if not np.isnan(mse) and mse > worst_mse:
            worst_mse = mse
            worst_model = name
    
    if worst_model and 'predictions' in results[worst_model] and 'ground_truth' in results[worst_model]:
        predictions = np.array(results[worst_model]['predictions'])
        ground_truth = np.array(results[worst_model]['ground_truth'])
        
        # Flatten if multi-dimensional
        if predictions.ndim > 1:
            predictions = predictions.flatten()
            ground_truth = ground_truth.flatten()
        
        ax7.scatter(ground_truth, predictions, alpha=0.6, s=20, color='red')
        min_val = min(ground_truth.min(), predictions.min())
        max_val = max(ground_truth.max(), predictions.max())
        ax7.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax7.set_xlabel('Ground Truth')
        ax7.set_ylabel('Predictions')
        ax7.set_title(f'Worst Model: {worst_model}\nMSE: {worst_mse:.4f}', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'No prediction data\navailable for worst model', 
                ha='center', va='center', transform=ax7.transAxes, fontsize=12)
        ax7.set_title('Worst Model Predictions', fontsize=12, fontweight='bold')
    
    # 8. Training time comparison (bottom center)
    ax8 = plt.subplot(3, 3, 8)
    training_times = [results[name].get('training_time', np.nan) for name in model_names]
    bars = ax8.bar(range(len(model_names)), training_times)
    ax8.set_xticks(range(len(model_names)))
    ax8.set_xticklabels(model_names, rotation=45, ha='right')
    ax8.set_ylabel('Training Time (s)')
    ax8.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    # Color bars by training time (shorter is better)
    valid_times = [t for t in training_times if not np.isnan(t)]
    if valid_times:
        time_range = max(valid_times) - min(valid_times)
        for i, (bar, time) in enumerate(zip(bars, training_times)):
            if not np.isnan(time) and time_range > 0:
                normalized_time = 1 - (time - min(valid_times)) / time_range
                bar.set_color(plt.cm.viridis(normalized_time))
            elif not np.isnan(time):
                bar.set_color(plt.cm.viridis(0.5))
    
    # 9. Inference speed comparison (bottom right)
    ax9 = plt.subplot(3, 3, 9)
    inference_speeds = [results[name].get('inference_per_sample', np.nan) for name in model_names]
    bars = ax9.bar(range(len(model_names)), inference_speeds)
    ax9.set_xticks(range(len(model_names)))
    ax9.set_xticklabels(model_names, rotation=45, ha='right')
    ax9.set_ylabel('Inference Time (s/sample)')
    ax9.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
    ax9.grid(True, alpha=0.3)
    
    # Color bars by inference speed (faster is better)
    valid_speeds = [s for s in inference_speeds if not np.isnan(s)]
    if valid_speeds:
        speed_range = max(valid_speeds) - min(valid_speeds)
        for i, (bar, speed) in enumerate(zip(bars, inference_speeds)):
            if not np.isnan(speed) and speed_range > 0:
                normalized_speed = 1 - (speed - min(valid_speeds)) / speed_range
                bar.set_color(plt.cm.viridis(normalized_speed))
            elif not np.isnan(speed):
                bar.set_color(plt.cm.viridis(0.5))
    
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
