"""
Model comparison plotting utilities.

Standardized plotting functions for comparing different neural network models
on natural parameter to statistics mapping tasks.
"""

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
# Set style
try:
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
except ImportError:
    # Fallback if seaborn not available
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'


def plot_training_curves(histories: Dict[str, Dict], save_path: Optional[Path] = None) -> None:
    """Plot training and validation curves for multiple models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for model_name, history in histories.items():
        if 'train_loss' in history:
            epochs = range(len(history['train_loss']))
            ax1.plot(epochs, history['train_loss'], label=f'{model_name} (Train)', alpha=0.7)
        
        if 'val_loss' in history:
            # Validation might be recorded less frequently
            val_epochs = np.arange(0, len(history['train_loss']), 
                                 len(history['train_loss']) // len(history['val_loss']))[:len(history['val_loss'])]
            ax2.plot(val_epochs, history['val_loss'], label=f'{model_name} (Val)', marker='o', alpha=0.7)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_performance_comparison(results: Dict[str, Dict], save_path: Optional[Path] = None) -> None:
    """Plot performance comparison across models."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Filter successful results
    successful_results = {k: v for k, v in results.items() 
                         if v.get('status') != 'failed' and 'metrics' in v}
    
    if not successful_results:
        print("No successful results to plot!")
        return
    
    model_names = list(successful_results.keys())
    
    # Extract metrics
    test_mses = [successful_results[name]['metrics'].get('mse', float('inf')) for name in model_names]
    test_maes = [successful_results[name]['metrics'].get('mae', float('inf')) for name in model_names]
    gt_mses = [successful_results[name]['metrics'].get('ground_truth_mse', float('inf')) for name in model_names]
    training_times = [successful_results[name].get('training_time', 0) for name in model_names]
    
    # 1. MSE Comparison
    bars1 = axes[0, 0].bar(range(len(model_names)), test_mses, alpha=0.7, color='skyblue')
    axes[0, 0].set_xticks(range(len(model_names)))
    axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 0].set_ylabel('MSE (vs Empirical)')
    axes[0, 0].set_title('MSE vs Empirical Data')
    axes[0, 0].set_yscale('log')
    
    # Add value labels
    for bar, value in zip(bars1, test_mses):
        if value != float('inf'):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.0f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Ground Truth MSE Comparison
    bars2 = axes[0, 1].bar(range(len(model_names)), gt_mses, alpha=0.7, color='lightcoral')
    axes[0, 1].set_xticks(range(len(model_names)))
    axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 1].set_ylabel('MSE (vs Ground Truth)')
    axes[0, 1].set_title('MSE vs Analytical Ground Truth')
    axes[0, 1].set_yscale('log')
    
    # Add value labels
    for bar, value in zip(bars2, gt_mses):
        if value != float('inf'):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.0f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Training Time Comparison
    bars3 = axes[1, 0].bar(range(len(model_names)), training_times, alpha=0.7, color='lightgreen')
    axes[1, 0].set_xticks(range(len(model_names)))
    axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Training Time (seconds)')
    axes[1, 0].set_title('Training Time Comparison')
    
    # Add value labels
    for bar, value in zip(bars3, training_times):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.0f}s', ha='center', va='bottom', fontsize=8)
    
    # 4. Performance vs Time Scatter
    valid_indices = [i for i, (gt, time) in enumerate(zip(gt_mses, training_times)) 
                    if gt != float('inf') and time > 0]
    
    if valid_indices:
        valid_times = [training_times[i] for i in valid_indices]
        valid_gt_mses = [gt_mses[i] for i in valid_indices]
        valid_names = [model_names[i] for i in valid_indices]
        
        axes[1, 1].scatter(valid_times, valid_gt_mses, s=100, alpha=0.7)
        for i, name in enumerate(valid_names):
            axes[1, 1].annotate(name, (valid_times[i], valid_gt_mses[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        axes[1, 1].set_xlabel('Training Time (seconds)')
        axes[1, 1].set_ylabel('MSE vs Ground Truth')
        axes[1, 1].set_yscale('log')
        axes[1, 1].set_title('Performance vs Training Time')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_architecture_analysis(results: Dict[str, Dict], save_path: Optional[Path] = None) -> None:
    """Plot analysis of different architectures."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Filter successful results
    successful_results = {k: v for k, v in results.items() 
                         if v.get('status') != 'failed' and 'metrics' in v}
    
    if not successful_results:
        print("No successful results to plot!")
        return
    
    # Extract architecture information
    architectures = []
    performances = []
    param_counts = []
    depths = []
    
    for name, result in successful_results.items():
        arch_info = result.get('architecture_info', {})
        metrics = result['metrics']
        
        architectures.append(name)
        performances.append(metrics.get('ground_truth_mse', metrics.get('mse', float('inf'))))
        param_counts.append(arch_info.get('parameter_count', 0))
        depths.append(arch_info.get('depth', len(arch_info.get('hidden_sizes', []))))
    
    # 1. Performance vs Parameters
    if param_counts and any(p > 0 for p in param_counts):
        axes[0, 0].scatter(param_counts, performances, s=100, alpha=0.7)
        for i, name in enumerate(architectures):
            if param_counts[i] > 0:
                axes[0, 0].annotate(name, (param_counts[i], performances[i]), 
                                   xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        axes[0, 0].set_xlabel('Parameter Count')
        axes[0, 0].set_ylabel('MSE vs Ground Truth')
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_title('Performance vs Model Size')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Performance vs Depth
    if depths:
        axes[0, 1].scatter(depths, performances, s=100, alpha=0.7)
        for i, name in enumerate(architectures):
            axes[0, 1].annotate(name, (depths[i], performances[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        axes[0, 1].set_xlabel('Network Depth (# layers)')
        axes[0, 1].set_ylabel('MSE vs Ground Truth')
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_title('Performance vs Network Depth')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Architecture Summary Table
    axes[1, 0].axis('off')
    
    table_data = []
    for name, result in successful_results.items():
        arch_info = result.get('architecture_info', {})
        metrics = result['metrics']
        
        hidden_sizes = arch_info.get('hidden_sizes', [])
        arch_str = f"{len(hidden_sizes)} layers"
        if hidden_sizes:
            if len(set(hidden_sizes)) == 1:
                arch_str += f" x {hidden_sizes[0]} units"
            else:
                arch_str += f" ({min(hidden_sizes)}-{max(hidden_sizes)} units)"
        
        table_data.append([
            name[:20] + "..." if len(name) > 20 else name,
            arch_str,
            f"{metrics.get('ground_truth_mse', metrics.get('mse', float('inf'))):.0f}",
            f"{result.get('training_time', 0):.0f}s"
        ])
    
    table = axes[1, 0].table(cellText=table_data,
                           colLabels=['Model', 'Architecture', 'GT MSE', 'Time'],
                           cellLoc='center',
                           loc='center',
                           bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    axes[1, 0].set_title('Architecture Summary', pad=20)
    
    # 4. Performance Ranking
    axes[1, 1].axis('off')
    
    # Sort by performance
    sorted_results = sorted(successful_results.items(), 
                          key=lambda x: x[1]['metrics'].get('ground_truth_mse', 
                                                           x[1]['metrics'].get('mse', float('inf'))))
    
    ranking_text = "Performance Ranking:\n\n"
    for i, (name, result) in enumerate(sorted_results[:10], 1):  # Top 10
        metrics = result['metrics']
        mse = metrics.get('ground_truth_mse', metrics.get('mse', float('inf')))
        time = result.get('training_time', 0)
        ranking_text += f"{i:2d}. {name[:25]:<25} {mse:8.0f} ({time:3.0f}s)\n"
    
    axes[1, 1].text(0.05, 0.95, ranking_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].set_title('Performance Ranking', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_prediction_analysis(predictions: Dict[str, jnp.ndarray], 
                           targets: jnp.ndarray,
                           ground_truth: Optional[jnp.ndarray] = None,
                           save_path: Optional[Path] = None) -> None:
    """Plot prediction quality analysis."""
    n_models = len(predictions)
    fig, axes = plt.subplots(2, min(n_models, 3), figsize=(15, 10))
    if n_models == 1:
        axes = axes.reshape(2, 1)
    elif n_models == 2:
        axes = axes.reshape(2, 2)
    
    model_names = list(predictions.keys())
    
    for i, (name, pred) in enumerate(predictions.items()):
        if i >= 3:  # Limit to first 3 models
            break
        
        col = i if n_models > 1 else 0
        
        # Prediction vs Target scatter
        axes[0, col].scatter(targets.flatten(), pred.flatten(), alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(targets.min(), pred.min())
        max_val = max(targets.max(), pred.max())
        axes[0, col].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        axes[0, col].set_xlabel('Target Values')
        axes[0, col].set_ylabel('Predicted Values')
        axes[0, col].set_title(f'{name}: Predictions vs Targets')
        axes[0, col].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = pred.flatten() - targets.flatten()
        axes[1, col].scatter(targets.flatten(), residuals, alpha=0.5, s=20)
        axes[1, col].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        
        axes[1, col].set_xlabel('Target Values')
        axes[1, col].set_ylabel('Residuals')
        axes[1, col].set_title(f'{name}: Residual Plot')
        axes[1, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_comprehensive_report(results: Dict[str, Dict], 
                              save_dir: Path,
                              experiment_name: str = "model_comparison") -> None:
    """Create a comprehensive visual report."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot training curves if available
    histories = {name: result.get('history', {}) for name, result in results.items() 
                if 'history' in result}
    if histories:
        plot_training_curves(histories, save_dir / f"{experiment_name}_training_curves.png")
    
    # Plot performance comparison
    plot_performance_comparison(results, save_dir / f"{experiment_name}_performance.png")
    
    # Plot architecture analysis
    plot_architecture_analysis(results, save_dir / f"{experiment_name}_architecture.png")
    
    # Create summary report
    create_summary_report(results, save_dir / f"{experiment_name}_summary.txt")
    
    print(f"Comprehensive report saved to {save_dir}/")


def create_summary_report(results: Dict[str, Dict], save_path: Path) -> None:
    """Create a text summary report."""
    successful_results = {k: v for k, v in results.items() 
                         if v.get('status') != 'failed' and 'metrics' in v}
    
    with open(save_path, 'w') as f:
        f.write("NEURAL NETWORK MODEL COMPARISON REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total Models Tested: {len(results)}\n")
        f.write(f"Successful Models: {len(successful_results)}\n")
        f.write(f"Failed Models: {len(results) - len(successful_results)}\n\n")
        
        if successful_results:
            # Performance ranking
            sorted_results = sorted(successful_results.items(), 
                                  key=lambda x: x[1]['metrics'].get('ground_truth_mse', 
                                                                   x[1]['metrics'].get('mse', float('inf'))))
            
            f.write("PERFORMANCE RANKING (by Ground Truth MSE):\n")
            f.write("-" * 50 + "\n")
            
            for i, (name, result) in enumerate(sorted_results, 1):
                metrics = result['metrics']
                gt_mse = metrics.get('ground_truth_mse', metrics.get('mse', float('inf')))
                training_time = result.get('training_time', 0)
                
                f.write(f"{i:2d}. {name:<30} {gt_mse:>10.0f} MSE ({training_time:>5.0f}s)\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("DETAILED RESULTS:\n\n")
            
            for name, result in sorted_results:
                f.write(f"{name}:\n")
                metrics = result['metrics']
                f.write(f"  MSE (Empirical):    {metrics.get('mse', 'N/A'):>10.2f}\n")
                f.write(f"  MAE (Empirical):    {metrics.get('mae', 'N/A'):>10.2f}\n")
                f.write(f"  MSE (Ground Truth): {metrics.get('ground_truth_mse', 'N/A'):>10.2f}\n")
                f.write(f"  MAE (Ground Truth): {metrics.get('ground_truth_mae', 'N/A'):>10.2f}\n")
                f.write(f"  Training Time:      {result.get('training_time', 'N/A'):>10.1f}s\n")
                
                if 'architecture_info' in result:
                    arch = result['architecture_info']
                    f.write(f"  Architecture:       {arch.get('hidden_sizes', 'N/A')}\n")
                    f.write(f"  Parameters:         {arch.get('parameter_count', 'N/A'):>10,}\n")
                
                f.write("\n")
        
        # Failed models
        failed_results = {k: v for k, v in results.items() if v.get('status') == 'failed'}
        if failed_results:
            f.write("FAILED MODELS:\n")
            f.write("-" * 20 + "\n")
            for name, result in failed_results.items():
                error = result.get('error', 'Unknown error')
                f.write(f"- {name}: {error}\n")
    
    print(f"Summary report saved to {save_path}")
