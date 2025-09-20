#!/usr/bin/env python3
"""
Debug script to systematically test depth vs width hypothesis.

This script trains multiple architectures with controlled parameter counts
to isolate the effect of depth vs width on performance.
"""

import sys
from pathlib import Path
import time
import json
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import FullConfig
from src.models.standard_mlp import create_model_and_trainer
from src.data_utils import load_3d_gaussian_data, compute_ground_truth_3d_tril


def create_controlled_architectures():
    """
    Create architectures with similar parameter counts but different depth/width ratios.
    
    Target: ~50,000 parameters for fair comparison
    """
    architectures = []
    
    # Calculate approximate parameter count: (input_size + 1) * hidden + (hidden + 1) * hidden * (layers-1) + (hidden + 1) * output
    # For input_size ≈ 40 (with feature engineering), output = 9
    
    # Very Deep, Very Narrow
    architectures.append({
        'name': 'Very Deep Narrow (20x32)',
        'hidden_sizes': [32] * 20,
        'description': '20 layers × 32 units ≈ 51k params'
    })
    
    # Deep, Narrow  
    architectures.append({
        'name': 'Deep Narrow (12x64)',
        'hidden_sizes': [64] * 12,
        'description': '12 layers × 64 units ≈ 49k params'
    })
    
    # Medium Depth, Medium Width
    architectures.append({
        'name': 'Medium (8x80)',
        'hidden_sizes': [80] * 8,
        'description': '8 layers × 80 units ≈ 51k params'
    })
    
    # Shallow, Wide
    architectures.append({
        'name': 'Shallow Wide (4x128)',
        'hidden_sizes': [128] * 4,
        'description': '4 layers × 128 units ≈ 50k params'
    })
    
    # Very Shallow, Very Wide
    architectures.append({
        'name': 'Very Shallow Wide (2x256)',
        'hidden_sizes': [256] * 2,
        'description': '2 layers × 256 units ≈ 52k params'
    })
    
    return architectures


def train_architecture(arch_config, train_data, val_data, test_data, ground_truth):
    """Train a single architecture and return results."""
    
    print(f"\n🔄 Testing {arch_config['name']}")
    print(f"   Architecture: {arch_config['description']}")
    
    # Create configuration
    config = FullConfig()
    config.network.hidden_sizes = arch_config['hidden_sizes']
    config.network.activation = "tanh"
    config.network.use_feature_engineering = True
    config.network.output_dim = 9
    
    # Training parameters (consistent across all)
    config.training.learning_rate = 5e-4
    config.training.num_epochs = 60  # Shorter for debugging
    config.training.batch_size = 32
    config.training.patience = 15
    config.training.weight_decay = 1e-6
    
    start_time = time.time()
    
    try:
        # Create trainer
        trainer = create_model_and_trainer(config)
        
        # Train
        params, history = trainer.train(train_data, val_data)
        training_time = time.time() - start_time
        
        # Evaluate
        metrics = trainer.evaluate(params, test_data, ground_truth)
        param_count = trainer.model.get_parameter_count(params)
        
        print(f"   ✅ Completed: {metrics.get('ground_truth_mse', metrics['mse']):.0f} MSE, {param_count:,} params, {training_time:.1f}s")
        
        return {
            'name': arch_config['name'],
            'hidden_sizes': arch_config['hidden_sizes'],
            'depth': len(arch_config['hidden_sizes']),
            'width': arch_config['hidden_sizes'][0],
            'parameter_count': param_count,
            'training_time': training_time,
            'metrics': metrics,
            'history': history,
            'status': 'success'
        }
        
    except Exception as e:
        training_time = time.time() - start_time
        print(f"   ❌ Failed: {str(e)[:50]}...")
        
        return {
            'name': arch_config['name'],
            'hidden_sizes': arch_config['hidden_sizes'],
            'depth': len(arch_config['hidden_sizes']),
            'width': arch_config['hidden_sizes'][0],
            'parameter_count': 0,
            'training_time': training_time,
            'status': 'failed',
            'error': str(e)
        }


def analyze_depth_width_relationship(results):
    """Analyze the relationship between depth, width, and performance."""
    
    successful_results = [r for r in results if r['status'] == 'success']
    
    if len(successful_results) < 2:
        print("Not enough successful results for analysis")
        return
    
    print("\n🔍 DEPTH vs WIDTH ANALYSIS:")
    print("-" * 50)
    
    # Extract data for analysis
    depths = [r['depth'] for r in successful_results]
    widths = [r['width'] for r in successful_results]
    performances = [r['metrics'].get('ground_truth_mse', r['metrics']['mse']) for r in successful_results]
    param_counts = [r['parameter_count'] for r in successful_results]
    times = [r['training_time'] for r in successful_results]
    
    # Sort by performance
    sorted_results = sorted(successful_results, key=lambda x: x['metrics'].get('ground_truth_mse', x['metrics']['mse']))
    
    print("Performance Ranking:")
    for i, result in enumerate(sorted_results, 1):
        perf = result['metrics'].get('ground_truth_mse', result['metrics']['mse'])
        print(f"  {i}. {result['name']:<25} {perf:8.0f} MSE ({result['depth']:2d}×{result['width']:3d}, {result['parameter_count']:5,} params)")
    
    # Correlation analysis
    try:
        from scipy.stats import pearsonr, spearmanr
        
        # Depth vs Performance
        depth_corr, depth_p = pearsonr(depths, performances)
        print(f"\nDepth vs Performance correlation: {depth_corr:.3f} (p={depth_p:.3f})")
        
        # Width vs Performance  
        width_corr, width_p = pearsonr(widths, performances)
        print(f"Width vs Performance correlation: {width_corr:.3f} (p={width_p:.3f})")
        
        # Parameter count vs Performance
        param_corr, param_p = pearsonr(param_counts, performances)
        print(f"Parameters vs Performance correlation: {param_corr:.3f} (p={param_p:.3f})")
        
        # Interpretation
        print(f"\nInterpretation:")
        if depth_corr < -0.3 and depth_p < 0.1:
            print("  ✅ Strong evidence: Deeper networks perform better")
        elif depth_corr > 0.3 and depth_p < 0.1:
            print("  ❌ Strong evidence: Shallower networks perform better")
        else:
            print("  ❓ Weak/no evidence for depth effect")
            
        if width_corr < -0.3 and width_p < 0.1:
            print("  ✅ Strong evidence: Narrower networks perform better")
        elif width_corr > 0.3 and width_p < 0.1:
            print("  ❌ Strong evidence: Wider networks perform better")
        else:
            print("  ❓ Weak/no evidence for width effect")
    
    except ImportError:
        print("(scipy not available for correlation analysis)")
    
    return {
        'depths': depths,
        'widths': widths, 
        'performances': performances,
        'param_counts': param_counts,
        'sorted_results': sorted_results
    }


def create_debug_plots(results, analysis_data, save_dir):
    """Create detailed debug plots."""
    
    successful_results = [r for r in results if r['status'] == 'success']
    
    if not successful_results:
        print("No successful results to plot")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    depths = analysis_data['depths']
    widths = analysis_data['widths']
    performances = analysis_data['performances']
    param_counts = analysis_data['param_counts']
    
    # 1. Depth vs Performance
    axes[0, 0].scatter(depths, performances, s=100, alpha=0.7, c='blue')
    for i, result in enumerate(successful_results):
        axes[0, 0].annotate(f"{result['width']}", (depths[i], performances[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0, 0].set_xlabel('Network Depth (layers)')
    axes[0, 0].set_ylabel('MSE vs Ground Truth')
    axes[0, 0].set_title('Performance vs Depth')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 2. Width vs Performance
    axes[0, 1].scatter(widths, performances, s=100, alpha=0.7, c='red')
    for i, result in enumerate(successful_results):
        axes[0, 1].annotate(f"{result['depth']}", (widths[i], performances[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0, 1].set_xlabel('Network Width (units/layer)')
    axes[0, 1].set_ylabel('MSE vs Ground Truth')
    axes[0, 1].set_title('Performance vs Width')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # 3. Parameters vs Performance
    axes[0, 2].scatter(param_counts, performances, s=100, alpha=0.7, c='green')
    for i, result in enumerate(successful_results):
        axes[0, 2].annotate(f"{result['depth']}×{result['width']}", 
                           (param_counts[i], performances[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0, 2].set_xlabel('Parameter Count')
    axes[0, 2].set_ylabel('MSE vs Ground Truth')
    axes[0, 2].set_title('Performance vs Parameters')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xscale('log')
    axes[0, 2].set_yscale('log')
    
    # 4. Depth vs Width colored by performance
    scatter = axes[1, 0].scatter(depths, widths, s=200, c=performances, 
                                cmap='viridis_r', alpha=0.8)
    for i, result in enumerate(successful_results):
        axes[1, 0].annotate(result['name'].split('(')[0].strip(), 
                           (depths[i], widths[i]), ha='center', va='center', fontsize=8)
    axes[1, 0].set_xlabel('Network Depth (layers)')
    axes[1, 0].set_ylabel('Network Width (units/layer)')
    axes[1, 0].set_title('Architecture Space (color = performance)')
    plt.colorbar(scatter, ax=axes[1, 0], label='MSE (lower is better)')
    
    # 5. Training time vs Performance
    times = [r['training_time'] for r in successful_results]
    axes[1, 1].scatter(times, performances, s=100, alpha=0.7)
    for i, result in enumerate(successful_results):
        axes[1, 1].annotate(f"{result['depth']}×{result['width']}", 
                           (times[i], performances[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1, 1].set_xlabel('Training Time (seconds)')
    axes[1, 1].set_ylabel('MSE vs Ground Truth')
    axes[1, 1].set_title('Efficiency: Performance vs Time')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    # 6. Summary table
    axes[1, 2].axis('off')
    
    table_data = []
    for result in analysis_data['sorted_results'][:5]:  # Top 5
        table_data.append([
            f"{result['depth']}×{result['width']}",
            f"{result['parameter_count']:,}",
            f"{result['metrics'].get('ground_truth_mse', result['metrics']['mse']):.0f}",
            f"{result['training_time']:.0f}s"
        ])
    
    table = axes[1, 2].table(cellText=table_data,
                           colLabels=['Arch', 'Params', 'GT MSE', 'Time'],
                           cellLoc='center',
                           loc='center',
                           bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    axes[1, 2].set_title('Top 5 Architectures', pad=20)
    
    plt.tight_layout()
    
    # Save plots
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "depth_vs_width_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Debug plots saved to {save_dir}/")


def main():
    """Debug depth vs width hypothesis with controlled experiments."""
    
    print("🔬 DEBUGGING DEPTH vs WIDTH HYPOTHESIS")
    print("=" * 60)
    print("Controlled experiment with similar parameter counts")
    
    # Load data
    print("\n📊 Loading data...")
    data_dir = Path("data")
    data, ef = load_3d_gaussian_data(data_dir, format="tril")
    
    # Use subset for faster debugging
    n_train = min(1000, data["train_eta"].shape[0])
    n_val = min(200, data["val_eta"].shape[0])
    
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
    
    print(f"Debug dataset sizes:")
    print(f"  Training: {train_data['eta'].shape[0]} samples")
    print(f"  Validation: {val_data['eta'].shape[0]} samples")
    print(f"  Test: {test_data['eta'].shape[0]} samples")
    
    # Compute ground truth
    ground_truth = compute_ground_truth_3d_tril(test_data['eta'], ef)
    empirical_mse = float(jnp.mean(jnp.square(test_data['y'] - ground_truth)))
    print(f"MCMC sampling error bound: {empirical_mse:.6f}")
    
    # Create controlled architectures
    architectures = create_controlled_architectures()
    print(f"\nTesting {len(architectures)} architectures with similar parameter counts")
    
    # Train all architectures
    results = []
    total_start_time = time.time()
    
    for arch_config in architectures:
        result = train_architecture(arch_config, train_data, val_data, test_data, ground_truth)
        results.append(result)
    
    total_time = time.time() - total_start_time
    print(f"\n🏁 All architectures tested in {total_time:.1f}s")
    
    # Analyze results
    analysis_data = analyze_depth_width_relationship(results)
    
    # Create debug plots
    debug_dir = Path("artifacts/debug_depth_width")
    create_debug_plots(results, analysis_data, debug_dir)
    
    # Save detailed results
    results_file = debug_dir / "depth_width_debug.json"
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
        
        debug_data = {
            'experiment': 'depth_vs_width_controlled',
            'total_time': total_time,
            'mcmc_error_bound': empirical_mse,
            'architectures_tested': len(architectures),
            'successful_architectures': len([r for r in results if r['status'] == 'success']),
            'results': convert_for_json(results),
            'analysis': convert_for_json(analysis_data) if analysis_data else None
        }
        
        json.dump(debug_data, f, indent=2)
    
    print(f"\n💾 Debug results saved to {debug_dir}/")
    
    # Final hypothesis test
    successful_results = [r for r in results if r['status'] == 'success']
    if len(successful_results) >= 3:
        # Find best deep vs best shallow
        deep_results = [r for r in successful_results if r['depth'] >= 10]
        shallow_results = [r for r in successful_results if r['depth'] <= 4]
        
        if deep_results and shallow_results:
            best_deep = min(deep_results, key=lambda x: x['metrics'].get('ground_truth_mse', x['metrics']['mse']))
            best_shallow = min(shallow_results, key=lambda x: x['metrics'].get('ground_truth_mse', x['metrics']['mse']))
            
            deep_mse = best_deep['metrics'].get('ground_truth_mse', best_deep['metrics']['mse'])
            shallow_mse = best_shallow['metrics'].get('ground_truth_mse', best_shallow['metrics']['mse'])
            
            print(f"\n🧪 HYPOTHESIS TEST RESULTS:")
            print(f"  Best Deep Network:    {best_deep['name']} → {deep_mse:.0f} MSE")
            print(f"  Best Shallow Network: {best_shallow['name']} → {shallow_mse:.0f} MSE")
            
            if deep_mse < shallow_mse:
                improvement = shallow_mse / deep_mse
                print(f"  ✅ DEEP WINS by {improvement:.2f}x!")
            else:
                degradation = deep_mse / shallow_mse
                print(f"  ❌ SHALLOW WINS by {degradation:.2f}x")
    
    print(f"\n✅ Depth vs width debugging completed!")


if __name__ == "__main__":
    main()
