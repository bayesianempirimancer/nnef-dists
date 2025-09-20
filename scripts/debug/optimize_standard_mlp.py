#!/usr/bin/env python3
"""
Hyperparameter optimization for Standard MLP.

This script systematically tests different hyperparameters and architectures
for the Standard MLP to find the optimal configuration for deep narrow networks.
"""

import sys
from pathlib import Path
import time
import json
import jax.numpy as jnp
import matplotlib.pyplot as plt
from itertools import product

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import FullConfig
from src.models.standard_mlp import create_model_and_trainer
from src.data_utils import load_3d_gaussian_data, compute_ground_truth_3d_tril


# =============================================================================
# HYPERPARAMETER SEARCH SPACE
# =============================================================================

# Architecture variations (keeping deep narrow philosophy)
ARCHITECTURES = [
    # Very Deep, Very Narrow
    ([24] * 24, "Ultra Deep (24x24)"),
    ([32] * 20, "Very Deep (20x32)"),
    ([48] * 16, "Deep (16x48)"),
    ([64] * 12, "Deep Narrow (12x64)"),
    ([80] * 10, "Medium Deep (10x80)"),
    ([96] * 8, "Medium (8x96)"),
    
    # Varying width patterns
    ([32, 64, 64, 64, 64, 64, 64, 64, 64, 32], "Hourglass Deep"),
    ([96, 80, 64, 64, 64, 64, 64, 64, 80, 96], "Diamond Deep"),
    ([32, 48, 64, 80, 80, 64, 48, 32], "Pyramid"),
]

# Learning rates to test
LEARNING_RATES = [1e-4, 3e-4, 5e-4, 8e-4, 1e-3]

# Activations to test
ACTIVATIONS = ["tanh", "swish", "gelu"]

# Batch sizes to test
BATCH_SIZES = [16, 32, 48, 64]

# Weight decay values
WEIGHT_DECAYS = [0.0, 1e-6, 1e-5, 1e-4]

# =============================================================================


def create_optimization_config(hidden_sizes, learning_rate=5e-4, activation="tanh", 
                              batch_size=32, weight_decay=1e-6):
    """Create configuration for optimization experiment."""
    config = FullConfig()
    
    # Architecture
    config.network.hidden_sizes = hidden_sizes
    config.network.activation = activation
    config.network.use_feature_engineering = True
    config.network.output_dim = 9
    config.network.dropout_rate = 0.0  # Start without dropout
    
    # Training
    config.training.learning_rate = learning_rate
    config.training.num_epochs = 80  # Moderate for optimization
    config.training.batch_size = batch_size
    config.training.patience = 20
    config.training.weight_decay = weight_decay
    config.training.gradient_clip_norm = 1.0
    config.training.use_lr_schedule = True
    config.training.lr_decay_rate = 0.95
    config.training.lr_decay_steps = 500
    
    return config


def test_architecture(arch_config, train_data, val_data, test_data, ground_truth, arch_name):
    """Test a single architecture configuration."""
    
    print(f"\n🔧 Testing {arch_name}")
    print(f"   Architecture: {len(arch_config['hidden_sizes'])} layers")
    if len(set(arch_config['hidden_sizes'])) == 1:
        print(f"   Size: {arch_config['hidden_sizes'][0]} units/layer")
    else:
        print(f"   Size: {min(arch_config['hidden_sizes'])}-{max(arch_config['hidden_sizes'])} units/layer")
    
    start_time = time.time()
    
    try:
        # Create configuration
        config = create_optimization_config(
            arch_config['hidden_sizes'],
            learning_rate=arch_config.get('learning_rate', 5e-4),
            activation=arch_config.get('activation', 'tanh'),
            batch_size=arch_config.get('batch_size', 32),
            weight_decay=arch_config.get('weight_decay', 1e-6)
        )
        
        # Create trainer
        trainer = create_model_and_trainer(config)
        
        # Train
        params, history = trainer.train(train_data, val_data)
        training_time = time.time() - start_time
        
        # Evaluate
        metrics = trainer.evaluate(params, test_data, ground_truth)
        param_count = trainer.model.get_parameter_count(params)
        
        gt_mse = metrics.get('ground_truth_mse', metrics['mse'])
        print(f"   ✅ GT MSE: {gt_mse:.0f}, Params: {param_count:,}, Time: {training_time:.1f}s")
        
        return {
            'name': arch_name,
            'hidden_sizes': arch_config['hidden_sizes'],
            'depth': len(arch_config['hidden_sizes']),
            'width': arch_config['hidden_sizes'][0] if arch_config['hidden_sizes'] else 0,
            'avg_width': sum(arch_config['hidden_sizes']) / len(arch_config['hidden_sizes']),
            'parameter_count': param_count,
            'training_time': training_time,
            'metrics': metrics,
            'history': history,
            'config': config.to_dict(),
            'status': 'success'
        }
        
    except Exception as e:
        training_time = time.time() - start_time
        print(f"   ❌ Failed: {str(e)[:80]}...")
        
        return {
            'name': arch_name,
            'hidden_sizes': arch_config['hidden_sizes'],
            'status': 'failed',
            'error': str(e),
            'training_time': training_time
        }


def test_hyperparameter_grid(base_hidden_sizes, train_data, val_data, test_data, ground_truth):
    """Test hyperparameter grid for a specific architecture."""
    
    print(f"\n🔍 HYPERPARAMETER GRID SEARCH")
    print(f"Base architecture: {len(base_hidden_sizes)} layers x {base_hidden_sizes[0]} units")
    
    # Smaller grid for efficiency
    lr_grid = [3e-4, 5e-4, 1e-3]
    activation_grid = ["tanh", "swish"]
    batch_grid = [32, 48]
    wd_grid = [0.0, 1e-6]
    
    grid_results = []
    
    for lr, activation, batch_size, weight_decay in product(lr_grid, activation_grid, batch_grid, wd_grid):
        config_name = f"LR{lr:.0e}_ACT{activation}_BS{batch_size}_WD{weight_decay:.0e}"
        
        arch_config = {
            'hidden_sizes': base_hidden_sizes,
            'learning_rate': lr,
            'activation': activation,
            'batch_size': batch_size,
            'weight_decay': weight_decay
        }
        
        result = test_architecture(arch_config, train_data, val_data, test_data, ground_truth, config_name)
        grid_results.append(result)
    
    # Find best hyperparameters
    successful_results = [r for r in grid_results if r['status'] == 'success']
    
    if successful_results:
        best_result = min(successful_results, key=lambda x: x['metrics'].get('ground_truth_mse', x['metrics']['mse']))
        
        print(f"\n🏆 BEST HYPERPARAMETERS:")
        best_config = FullConfig.from_dict(best_result['config'])
        print(f"   Learning Rate: {best_config.training.learning_rate}")
        print(f"   Activation: {best_config.network.activation}")
        print(f"   Batch Size: {best_config.training.batch_size}")
        print(f"   Weight Decay: {best_config.training.weight_decay}")
        print(f"   GT MSE: {best_result['metrics'].get('ground_truth_mse', best_result['metrics']['mse']):.0f}")
        
        return best_result, grid_results
    
    return None, grid_results


def create_optimization_plots(architecture_results, grid_results, save_dir):
    """Create plots for optimization analysis."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Filter successful results
    arch_successful = [r for r in architecture_results if r['status'] == 'success']
    grid_successful = [r for r in grid_results if r['status'] == 'success']
    
    if not arch_successful:
        print("No successful architecture results to plot")
        return
    
    # 1. Architecture Performance
    depths = [r['depth'] for r in arch_successful]
    widths = [r['avg_width'] for r in arch_successful]
    performances = [r['metrics'].get('ground_truth_mse', r['metrics']['mse']) for r in arch_successful]
    param_counts = [r['parameter_count'] for r in arch_successful]
    
    scatter = axes[0, 0].scatter(depths, performances, s=100, c=widths, cmap='viridis', alpha=0.7)
    axes[0, 0].set_xlabel('Network Depth (layers)')
    axes[0, 0].set_ylabel('MSE vs Ground Truth')
    axes[0, 0].set_title('Architecture Performance\n(color = avg width)')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 0], label='Avg Width')
    
    # 2. Parameters vs Performance
    axes[0, 1].scatter(param_counts, performances, s=100, alpha=0.7, c='blue')
    for i, result in enumerate(arch_successful[:8]):  # Label top 8
        axes[0, 1].annotate(f"{result['depth']}×{result['width']}", 
                           (param_counts[i], performances[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0, 1].set_xlabel('Parameter Count')
    axes[0, 1].set_ylabel('MSE vs Ground Truth')
    axes[0, 1].set_title('Parameter Efficiency')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Training Time vs Performance
    times = [r['training_time'] for r in arch_successful]
    axes[0, 2].scatter(times, performances, s=100, alpha=0.7, c='green')
    axes[0, 2].set_xlabel('Training Time (seconds)')
    axes[0, 2].set_ylabel('MSE vs Ground Truth')
    axes[0, 2].set_title('Time Efficiency')
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Hyperparameter analysis (if available)
    if grid_successful:
        # Extract hyperparameter info from config
        learning_rates = []
        batch_sizes = []
        grid_performances = []
        
        for result in grid_successful:
            config = FullConfig.from_dict(result['config'])
            learning_rates.append(config.training.learning_rate)
            batch_sizes.append(config.training.batch_size)
            grid_performances.append(result['metrics'].get('ground_truth_mse', result['metrics']['mse']))
        
        scatter2 = axes[1, 0].scatter(learning_rates, grid_performances, s=100, c=batch_sizes, cmap='plasma', alpha=0.7)
        axes[1, 0].set_xlabel('Learning Rate')
        axes[1, 0].set_ylabel('MSE vs Ground Truth')
        axes[1, 0].set_title('Hyperparameter Sensitivity\n(color = batch size)')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[1, 0], label='Batch Size')
    
    # 5. Architecture ranking
    axes[1, 1].axis('off')
    
    # Sort by performance
    sorted_arch = sorted(arch_successful, key=lambda x: x['metrics'].get('ground_truth_mse', x['metrics']['mse']))
    
    ranking_text = "Architecture Ranking:\n\n"
    for i, result in enumerate(sorted_arch[:8], 1):  # Top 8
        gt_mse = result['metrics'].get('ground_truth_mse', result['metrics']['mse'])
        ranking_text += f"{i:2d}. {result['name'][:20]:<20} {gt_mse:>8.0f}\n"
        ranking_text += f"    {result['depth']:2d}×{result['width']:3d} ({result['parameter_count']:>5,} params)\n\n"
    
    axes[1, 1].text(0.05, 0.95, ranking_text, transform=axes[1, 1].transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].set_title('Top Architectures', pad=20)
    
    # 6. Best configuration summary
    axes[1, 2].axis('off')
    
    if arch_successful:
        best_arch = sorted_arch[0]
        best_config = FullConfig.from_dict(best_arch['config'])
        
        summary_text = "OPTIMAL CONFIGURATION:\n\n"
        summary_text += f"Architecture:\n"
        summary_text += f"  Layers: {best_arch['depth']}\n"
        summary_text += f"  Units/layer: {best_arch['width']}\n"
        summary_text += f"  Activation: {best_config.network.activation}\n"
        summary_text += f"  Parameters: {best_arch['parameter_count']:,}\n\n"
        
        summary_text += f"Training:\n"
        summary_text += f"  Learning Rate: {best_config.training.learning_rate}\n"
        summary_text += f"  Batch Size: {best_config.training.batch_size}\n"
        summary_text += f"  Weight Decay: {best_config.training.weight_decay}\n\n"
        
        summary_text += f"Performance:\n"
        summary_text += f"  GT MSE: {best_arch['metrics'].get('ground_truth_mse', best_arch['metrics']['mse']):.0f}\n"
        summary_text += f"  Training Time: {best_arch['training_time']:.1f}s\n"
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_title('Optimal Configuration', pad=20)
    
    plt.tight_layout()
    
    # Save plots
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "standard_mlp_optimization.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Optimization plots saved to {save_dir}/")


def main():
    """Run Standard MLP optimization."""
    
    print("🎯 STANDARD MLP HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print("Systematically optimizing Standard MLP for deep narrow performance")
    
    # Load data
    print("\n📊 Loading data...")
    data_dir = Path("data")
    data, ef = load_3d_gaussian_data(data_dir, format="tril")
    
    # Use reasonable subset for optimization (balance speed vs accuracy)
    n_train = min(1500, data["train_eta"].shape[0])
    n_val = min(300, data["val_eta"].shape[0])
    
    train_data = {
        "eta": data["train_eta"][:n_train],
        "y": data["train_y"][:n_train]
    }
    
    val_data = {
        "eta": data["val_eta"][:n_val//2],
        "y": data["val_y"][:n_val//2]
    }
    
    test_data = {
        "eta": data["val_eta"][n_val//2:n_val],
        "y": data["val_y"][n_val//2:n_val]
    }
    
    print(f"Optimization dataset:")
    print(f"  Training: {train_data['eta'].shape[0]} samples")
    print(f"  Validation: {val_data['eta'].shape[0]} samples")
    print(f"  Test: {test_data['eta'].shape[0]} samples")
    
    # Compute ground truth
    ground_truth = compute_ground_truth_3d_tril(test_data['eta'], ef)
    empirical_mse = float(jnp.mean(jnp.square(test_data['y'] - ground_truth)))
    print(f"MCMC sampling error bound: {empirical_mse:.6f}")
    
    # Phase 1: Test different architectures
    print(f"\n🏗️  PHASE 1: ARCHITECTURE OPTIMIZATION")
    print(f"Testing {len(ARCHITECTURES)} different architectures")
    
    architecture_results = []
    arch_start_time = time.time()
    
    for hidden_sizes, arch_name in ARCHITECTURES:
        arch_config = {'hidden_sizes': hidden_sizes}
        result = test_architecture(arch_config, train_data, val_data, test_data, ground_truth, arch_name)
        architecture_results.append(result)
    
    arch_time = time.time() - arch_start_time
    print(f"\nArchitecture testing completed in {arch_time:.1f}s")
    
    # Find best architecture
    successful_arch = [r for r in architecture_results if r['status'] == 'success']
    
    if not successful_arch:
        print("❌ No architectures succeeded!")
        return
    
    best_arch = min(successful_arch, key=lambda x: x['metrics'].get('ground_truth_mse', x['metrics']['mse']))
    print(f"\n🏆 Best Architecture: {best_arch['name']}")
    print(f"   Performance: {best_arch['metrics'].get('ground_truth_mse', best_arch['metrics']['mse']):.0f} GT MSE")
    print(f"   Structure: {best_arch['depth']} layers x {best_arch['width']} units")
    print(f"   Parameters: {best_arch['parameter_count']:,}")
    
    # Phase 2: Hyperparameter optimization on best architecture
    print(f"\n🔧 PHASE 2: HYPERPARAMETER OPTIMIZATION")
    print(f"Optimizing hyperparameters for best architecture: {best_arch['name']}")
    
    best_result, grid_results = test_hyperparameter_grid(
        best_arch['hidden_sizes'], train_data, val_data, test_data, ground_truth
    )
    
    # Create optimization plots
    debug_dir = Path("artifacts/debug_standard_mlp_optimization")
    create_optimization_plots(architecture_results, grid_results, debug_dir)
    
    # Final results
    if best_result:
        final_best = best_result
        print(f"\n🎯 FINAL OPTIMIZED CONFIGURATION:")
        final_config = FullConfig.from_dict(final_best['config'])
        
        print(f"Architecture:")
        print(f"  Hidden sizes: {final_config.network.hidden_sizes}")
        print(f"  Activation: {final_config.network.activation}")
        print(f"  Parameters: {final_best['parameter_count']:,}")
        
        print(f"Training:")
        print(f"  Learning rate: {final_config.training.learning_rate}")
        print(f"  Batch size: {final_config.training.batch_size}")
        print(f"  Weight decay: {final_config.training.weight_decay}")
        
        print(f"Performance:")
        final_gt_mse = final_best['metrics'].get('ground_truth_mse', final_best['metrics']['mse'])
        print(f"  GT MSE: {final_gt_mse:.0f}")
        print(f"  Training time: {final_best['training_time']:.1f}s")
        print(f"  Improvement over MCMC bound: {empirical_mse / final_gt_mse:.1f}x")
        
        # Save optimal configuration
        optimal_config_file = debug_dir / "optimal_standard_mlp_config.json"
        with open(optimal_config_file, 'w') as f:
            json.dump(final_best['config'], f, indent=2)
        
        print(f"\n💾 Optimal config saved to {optimal_config_file}")
    
    else:
        final_best = best_arch
        print(f"\n🎯 BEST ARCHITECTURE (no hyperparameter improvement):")
        print(f"   {final_best['name']}: {final_best['metrics'].get('ground_truth_mse', final_best['metrics']['mse']):.0f} GT MSE")
    
    # Save comprehensive results
    results_file = debug_dir / "standard_mlp_optimization_results.json"
    
    with open(results_file, 'w') as f:
        def convert_for_json(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        optimization_data = {
            'experiment': 'standard_mlp_optimization',
            'mcmc_error_bound': empirical_mse,
            'architectures_tested': len(ARCHITECTURES),
            'hyperparameter_combinations': len(grid_results),
            'best_architecture': convert_for_json(final_best),
            'all_architecture_results': convert_for_json(architecture_results),
            'hyperparameter_grid_results': convert_for_json(grid_results)
        }
        
        json.dump(optimization_data, f, indent=2)
    
    print(f"\n📁 Complete results saved to {debug_dir}/")
    print(f"\n✅ Standard MLP optimization completed!")
    
    # Recommendations for other models
    print(f"\n💡 INSIGHTS FOR OTHER MODELS:")
    if successful_arch:
        # Analyze what worked best
        top_3 = sorted(successful_arch, key=lambda x: x['metrics'].get('ground_truth_mse', x['metrics']['mse']))[:3]
        
        avg_depth = sum(r['depth'] for r in top_3) / len(top_3)
        avg_width = sum(r['width'] for r in top_3) / len(top_3)
        
        print(f"  🎯 Optimal depth range: ~{avg_depth:.0f} layers")
        print(f"  🎯 Optimal width range: ~{avg_width:.0f} units/layer")
        print(f"  🎯 Best activation: {final_config.network.activation if 'final_config' in locals() else 'tanh'}")
        print(f"  🎯 Apply these insights to other model types!")


if __name__ == "__main__":
    main()
