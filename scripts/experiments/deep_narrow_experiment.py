#!/usr/bin/env python3
"""
Deep Narrow Networks Experiment.

This experiment compares different deep narrow architectures on the 3D Gaussian
natural parameter to statistics mapping task using the standardized framework.
"""

import sys
from pathlib import Path
import time
import json
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_deep_narrow_config, get_ultra_deep_config, FullConfig
from src.models.standard_mlp import create_model_and_trainer as create_mlp
from src.models.deep_flow import create_model_and_trainer as create_flow
from src.data_utils import load_3d_gaussian_data, compute_ground_truth_3d_tril
from plotting.model_comparison import create_comprehensive_report
import jax.numpy as jnp


def create_configs() -> Dict[str, FullConfig]:
    """Create configurations for different deep narrow architectures."""
    
    configs = {}
    
    # Deep Narrow MLP (12 layers x 64 units)
    config1 = get_deep_narrow_config()
    config1.experiment.experiment_name = "deep_narrow_12x64"
    configs["Deep Narrow MLP (12x64)"] = config1
    
    # Ultra Deep MLP (20 layers x 32 units)  
    config2 = get_ultra_deep_config()
    config2.experiment.experiment_name = "ultra_deep_20x32"
    configs["Ultra Deep MLP (20x32)"] = config2
    
    # Very Deep MLP (16 layers x 48 units)
    config3 = FullConfig()
    config3.network.hidden_sizes = [48] * 16
    config3.network.activation = "tanh"
    config3.network.use_feature_engineering = True
    config3.training.learning_rate = 3e-4
    config3.training.batch_size = 24
    config3.training.num_epochs = 120
    config3.training.patience = 25
    config3.experiment.experiment_name = "very_deep_16x48"
    configs["Very Deep MLP (16x48)"] = config3
    
    # Deep Flow Network (30 layers x 64 units)
    config4 = FullConfig()
    config4.network.hidden_sizes = [64] * 4  # Base network layers
    config4.network.use_feature_engineering = True
    config4.model_specific.num_flow_layers = 30
    config4.model_specific.flow_hidden_size = 64
    config4.training.learning_rate = 1e-3
    config4.training.batch_size = 32
    config4.training.num_epochs = 80
    config4.experiment.experiment_name = "deep_flow_30x64"
    configs["Deep Flow Network (30x64)"] = config4
    
    # Narrow Deep MLP (24 layers x 24 units)
    config5 = FullConfig()
    config5.network.hidden_sizes = [24] * 24
    config5.network.activation = "tanh"
    config5.network.use_feature_engineering = True
    config5.training.learning_rate = 1e-4
    config5.training.batch_size = 16
    config5.training.num_epochs = 150
    config5.training.patience = 30
    config5.training.weight_decay = 1e-6
    config5.training.gradient_clip_norm = 0.3
    config5.experiment.experiment_name = "narrow_deep_24x24"
    configs["Narrow Deep MLP (24x24)"] = config5
    
    return configs


def train_model(name: str, config: FullConfig, train_data: Dict, val_data: Dict, 
                test_data: Dict, ground_truth: jnp.ndarray) -> Dict[str, Any]:
    """Train a single model and return results."""
    
    print(f"\n🔄 Training {name}")
    print(f"  Architecture: {config.network.hidden_sizes}")
    print(f"  Parameters: ~{sum(config.network.hidden_sizes) * 50:,} (estimated)")
    
    start_time = time.time()
    
    try:
        # Create trainer
        if "Flow" in name:
            trainer = create_flow(config)
        else:
            trainer = create_mlp(config)
        
        # Train model
        best_params, history = trainer.train(train_data, val_data)
        training_time = time.time() - start_time
        
        # Evaluate
        metrics = trainer.evaluate(best_params, test_data, ground_truth)
        param_count = trainer.model.get_parameter_count(best_params)
        
        print(f"  ✅ Completed in {training_time:.1f}s")
        print(f"  📊 GT MSE: {metrics.get('ground_truth_mse', metrics['mse']):.0f}")
        print(f"  🔢 Parameters: {param_count:,}")
        
        return {
            'status': 'success',
            'metrics': metrics,
            'history': history,
            'training_time': training_time,
            'architecture_info': {
                'hidden_sizes': config.network.hidden_sizes,
                'parameter_count': param_count,
                'depth': len(config.network.hidden_sizes),
                'activation': config.network.activation
            },
            'config': config.to_dict()
        }
        
    except Exception as e:
        training_time = time.time() - start_time
        print(f"  ❌ Failed after {training_time:.1f}s: {str(e)[:100]}...")
        
        return {
            'status': 'failed',
            'error': str(e),
            'training_time': training_time,
            'config': config.to_dict()
        }


def main():
    """Run deep narrow networks experiment."""
    
    print("🚀 DEEP NARROW NETWORKS EXPERIMENT")
    print("=" * 60)
    print("Comparing deep narrow architectures for natural parameter mapping")
    print("Hypothesis: Deep narrow networks > Wide shallow networks")
    
    # Load data
    print("\n📊 Loading 3D Gaussian data...")
    data_dir = Path("data")
    data, ef = load_3d_gaussian_data(data_dir, format="tril")
    
    # Create splits
    train_data = {"eta": data["train_eta"], "y": data["train_y"]}
    val_data = {"eta": data["val_eta"], "y": data["val_y"]}
    
    # Create test split
    n_val = val_data["eta"].shape[0]
    n_test = min(n_val // 2, 500)
    
    test_data = {
        "eta": val_data["eta"][:n_test],
        "y": val_data["y"][:n_test]
    }
    
    val_data = {
        "eta": val_data["eta"][n_test:],
        "y": val_data["y"][n_test:]
    }
    
    print(f"Training: {train_data['eta'].shape[0]} samples")
    print(f"Validation: {val_data['eta'].shape[0]} samples")
    print(f"Test: {test_data['eta'].shape[0]} samples")
    
    # Compute ground truth
    print("\n🎯 Computing ground truth...")
    ground_truth = compute_ground_truth_3d_tril(test_data['eta'], ef)
    empirical_mse = float(jnp.mean(jnp.square(test_data['y'] - ground_truth)))
    print(f"MCMC sampling error bound: {empirical_mse:.6f}")
    
    # Create configurations
    print("\n🏗️  Creating model configurations...")
    configs = create_configs()
    print(f"Testing {len(configs)} different architectures")
    
    # Train all models
    results = {}
    total_start_time = time.time()
    
    for name, config in configs.items():
        result = train_model(name, config, train_data, val_data, test_data, ground_truth)
        results[name] = result
    
    total_time = time.time() - total_start_time
    print(f"\n🏁 All models completed in {total_time:.1f}s")
    
    # Analyze results
    successful_results = {k: v for k, v in results.items() if v['status'] == 'success'}
    failed_results = {k: v for k, v in results.items() if v['status'] == 'failed'}
    
    print(f"\n📈 EXPERIMENT RESULTS:")
    print(f"  Successful models: {len(successful_results)}")
    print(f"  Failed models: {len(failed_results)}")
    
    if successful_results:
        # Sort by ground truth performance
        sorted_results = sorted(successful_results.items(), 
                              key=lambda x: x[1]['metrics'].get('ground_truth_mse', 
                                                               x[1]['metrics']['mse']))
        
        print(f"\n🏆 PERFORMANCE RANKING:")
        for i, (name, result) in enumerate(sorted_results, 1):
            metrics = result['metrics']
            arch_info = result['architecture_info']
            gt_mse = metrics.get('ground_truth_mse', metrics['mse'])
            
            print(f"  {i:2d}. {name:<25} {gt_mse:8.0f} MSE ({result['training_time']:4.0f}s)")
            print(f"      {len(arch_info['hidden_sizes'])} layers x {arch_info['hidden_sizes'][0] if arch_info['hidden_sizes'] else 0} units, {arch_info['parameter_count']:,} params")
        
        # Architecture insights
        print(f"\n🔍 ARCHITECTURE INSIGHTS:")
        best_name, best_result = sorted_results[0]
        best_arch = best_result['architecture_info']
        
        print(f"  🥇 Best Architecture: {best_name}")
        print(f"     Depth: {best_arch['depth']} layers")
        print(f"     Width: {best_arch['hidden_sizes'][0] if best_arch['hidden_sizes'] else 0} units/layer")
        print(f"     Parameters: {best_arch['parameter_count']:,}")
        print(f"     GT MSE: {best_result['metrics'].get('ground_truth_mse', best_result['metrics']['mse']):.0f}")
        print(f"     Training time: {best_result['training_time']:.1f}s")
        
        # Depth vs Performance analysis
        depths = [r['architecture_info']['depth'] for r in successful_results.values()]
        performances = [r['metrics'].get('ground_truth_mse', r['metrics']['mse']) 
                       for r in successful_results.values()]
        
        if len(depths) > 1:
            from scipy.stats import pearsonr
            try:
                correlation, p_value = pearsonr(depths, performances)
                print(f"\n📊 DEPTH vs PERFORMANCE:")
                print(f"     Correlation: {correlation:.3f} (p={p_value:.3f})")
                if correlation < -0.3 and p_value < 0.1:
                    print(f"     ✅ Evidence supports deep narrow hypothesis!")
                elif correlation > 0.3 and p_value < 0.1:
                    print(f"     ❌ Evidence against deep narrow hypothesis")
                else:
                    print(f"     ❓ Inconclusive evidence")
            except ImportError:
                print(f"     (scipy not available for correlation analysis)")
    
    # Save results and create report
    output_dir = Path("artifacts/deep_narrow_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_file = output_dir / "experiment_results.json"
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
        
        experiment_data = {
            'experiment_name': 'deep_narrow_networks',
            'total_time': total_time,
            'mcmc_error_bound': empirical_mse,
            'results': convert_for_json(results)
        }
        
        json.dump(experiment_data, f, indent=2)
    
    # Create comprehensive report
    if successful_results:
        create_comprehensive_report(results, output_dir, "deep_narrow_experiment")
    
    print(f"\n📁 Results saved to {output_dir}/")
    print(f"\n✅ Deep narrow networks experiment completed!")
    
    # Final summary
    if successful_results:
        best_gt_mse = sorted_results[0][1]['metrics'].get('ground_truth_mse', 
                                                          sorted_results[0][1]['metrics']['mse'])
        improvement = empirical_mse / best_gt_mse if best_gt_mse > 0 else float('inf')
        
        print(f"\n🎯 FINAL SUMMARY:")
        print(f"  MCMC Error Bound: {empirical_mse:.6f}")
        print(f"  Best Model MSE: {best_gt_mse:.0f}")
        print(f"  Improvement Factor: {improvement:.1f}x beyond MCMC limit")


if __name__ == "__main__":
    main()
