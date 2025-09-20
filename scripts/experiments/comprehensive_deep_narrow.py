#!/usr/bin/env python3
"""
Comprehensive Deep Narrow Networks Experiment.

This experiment systematically compares different deep narrow architectures
to test the hypothesis that deep narrow networks outperform wide shallow ones
for natural parameter to statistics mapping.
"""

import sys
from pathlib import Path
import time
import json
from typing import Dict, Any
import jax.numpy as jnp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import FullConfig
from src.models.standard_mlp import create_model_and_trainer as create_mlp
from src.models.deep_flow import create_model_and_trainer as create_flow
from src.models.quadratic_resnet import create_model_and_trainer as create_quadratic
from src.data_utils import load_3d_gaussian_data, compute_ground_truth_3d_tril
from plotting.model_comparison import create_comprehensive_report


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

EXPERIMENT_NAME = "comprehensive_deep_narrow"
OUTPUT_DIR = "artifacts/comprehensive_deep_narrow"

# Test different depth vs width combinations
ARCHITECTURES_TO_TEST = [
    # Format: (name, layers, units_per_layer, special_config)
    ("Deep Narrow MLP", 12, 64, {}),
    ("Ultra Deep MLP", 20, 32, {"learning_rate": 3e-4, "batch_size": 16}),
    ("Very Deep MLP", 16, 48, {"learning_rate": 4e-4, "batch_size": 24}),
    ("Narrow Deep MLP", 24, 24, {"learning_rate": 1e-4, "batch_size": 12, "patience": 40}),
    ("Wide Shallow MLP", 3, 512, {"learning_rate": 1e-3, "batch_size": 64}),  # Control
    ("Medium MLP", 6, 128, {"learning_rate": 8e-4, "batch_size": 48}),  # Control
]

# Flow networks to test
FLOW_ARCHITECTURES = [
    ("Deep Flow (30x64)", 30, 64, {}),
    ("Ultra Deep Flow (50x32)", 50, 32, {"learning_rate": 5e-4, "batch_size": 16}),
]

# Quadratic ResNets to test
QUADRATIC_ARCHITECTURES = [
    ("Quadratic ResNet (10x96)", 10, 96, {}),
    ("Deep Quadratic (15x64)", 15, 64, {"learning_rate": 6e-4}),
]

# =============================================================================


def create_mlp_config(name: str, layers: int, units: int, special_config: Dict) -> FullConfig:
    """Create configuration for MLP variants."""
    config = FullConfig()
    
    # Architecture
    config.network.hidden_sizes = [units] * layers
    config.network.activation = "tanh"
    config.network.use_feature_engineering = True
    config.network.output_dim = 9
    
    # Training (with special overrides)
    config.training.learning_rate = special_config.get("learning_rate", 5e-4)
    config.training.batch_size = special_config.get("batch_size", 32)
    config.training.num_epochs = 120
    config.training.patience = special_config.get("patience", 25)
    config.training.weight_decay = 1e-6 if layers > 15 else 0.0
    config.training.gradient_clip_norm = 0.5 if layers > 15 else 1.0
    
    # Experiment
    config.experiment.experiment_name = name.lower().replace(" ", "_")
    config.experiment.output_dir = f"{OUTPUT_DIR}/{config.experiment.experiment_name}"
    
    return config


def create_flow_config(name: str, flow_layers: int, flow_units: int, special_config: Dict) -> FullConfig:
    """Create configuration for Flow Network variants."""
    config = FullConfig()
    
    # Base network
    config.network.hidden_sizes = [flow_units] * 4
    config.network.activation = "tanh"
    config.network.use_feature_engineering = True
    config.network.output_dim = 9
    
    # Flow-specific
    config.model_specific.num_flow_layers = flow_layers
    config.model_specific.flow_hidden_size = flow_units
    config.model_specific.num_timesteps = 100
    config.model_specific.beta_start = 1e-4
    config.model_specific.beta_end = 2e-2
    
    # Training
    config.training.learning_rate = special_config.get("learning_rate", 1e-3)
    config.training.batch_size = special_config.get("batch_size", 32)
    config.training.num_epochs = 80
    config.training.patience = 20
    
    # Experiment
    config.experiment.experiment_name = name.lower().replace(" ", "_")
    config.experiment.output_dir = f"{OUTPUT_DIR}/{config.experiment.experiment_name}"
    
    return config


def create_quadratic_config(name: str, layers: int, units: int, special_config: Dict) -> FullConfig:
    """Create configuration for Quadratic ResNet variants."""
    config = FullConfig()
    
    # Architecture
    config.network.hidden_sizes = [units] * layers
    config.network.activation = "tanh"
    config.network.use_feature_engineering = True
    config.network.residual_connections = True
    config.network.output_dim = 9
    
    # Quadratic-specific
    config.model_specific.use_quadratic_terms = True
    config.model_specific.quadratic_mixing = "adaptive"
    
    # Training
    config.training.learning_rate = special_config.get("learning_rate", 8e-4)
    config.training.batch_size = special_config.get("batch_size", 32)
    config.training.num_epochs = 100
    config.training.patience = 20
    config.training.weight_decay = 1e-5
    
    # Experiment
    config.experiment.experiment_name = name.lower().replace(" ", "_")
    config.experiment.output_dir = f"{OUTPUT_DIR}/{config.experiment.experiment_name}"
    
    return config


def train_single_model(name: str, config: FullConfig, model_type: str,
                      train_data: Dict, val_data: Dict, test_data: Dict, 
                      ground_truth: jnp.ndarray) -> Dict[str, Any]:
    """Train a single model and return results."""
    
    print(f"\n🔄 Training {name}")
    
    # Architecture info
    if model_type == "flow":
        total_layers = config.model_specific.num_flow_layers + len(config.network.hidden_sizes)
        arch_desc = f"{config.model_specific.num_flow_layers} flow + {len(config.network.hidden_sizes)} base layers"
    else:
        total_layers = len(config.network.hidden_sizes)
        arch_desc = f"{total_layers} layers x {config.network.hidden_sizes[0]} units"
    
    print(f"  Architecture: {arch_desc}")
    
    start_time = time.time()
    
    try:
        # Create trainer based on model type
        if model_type == "mlp":
            trainer = create_mlp(config)
        elif model_type == "flow":
            trainer = create_flow(config)
        elif model_type == "quadratic":
            trainer = create_quadratic(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
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
                'depth': total_layers,
                'width': config.network.hidden_sizes[0] if config.network.hidden_sizes else 0,
                'model_type': model_type,
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
    """Run comprehensive deep narrow networks experiment."""
    
    print("🚀 COMPREHENSIVE DEEP NARROW NETWORKS EXPERIMENT")
    print("=" * 70)
    print("Systematically testing deep narrow vs wide shallow architectures")
    print("Hypothesis: Deep narrow networks > Wide shallow networks")
    print("Task: Natural parameter to statistics mapping (3D Gaussian)")
    
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
    
    # Create all configurations
    print(f"\n🏗️  Creating model configurations...")
    
    all_experiments = []
    
    # MLP variants
    for name, layers, units, special in ARCHITECTURES_TO_TEST:
        config = create_mlp_config(name, layers, units, special)
        all_experiments.append((name, config, "mlp"))
    
    # Flow variants
    for name, flow_layers, flow_units, special in FLOW_ARCHITECTURES:
        config = create_flow_config(name, flow_layers, flow_units, special)
        all_experiments.append((name, config, "flow"))
    
    # Quadratic variants
    for name, layers, units, special in QUADRATIC_ARCHITECTURES:
        config = create_quadratic_config(name, layers, units, special)
        all_experiments.append((name, config, "quadratic"))
    
    print(f"Testing {len(all_experiments)} different architectures")
    
    # Train all models
    results = {}
    total_start_time = time.time()
    
    for name, config, model_type in all_experiments:
        result = train_single_model(name, config, model_type, train_data, val_data, test_data, ground_truth)
        results[name] = result
    
    total_time = time.time() - total_start_time
    print(f"\n🏁 All models completed in {total_time:.1f}s")
    
    # Analyze results
    successful_results = {k: v for k, v in results.items() if v['status'] == 'success'}
    failed_results = {k: v for k, v in results.items() if v['status'] == 'failed'}
    
    print(f"\n📈 EXPERIMENT RESULTS:")
    print(f"  Successful models: {len(successful_results)}")
    print(f"  Failed models: {len(failed_results)}")
    
    if failed_results:
        print(f"\n❌ Failed models:")
        for name, result in failed_results.items():
            print(f"  - {name}: {result.get('error', 'Unknown error')[:50]}...")
    
    if successful_results:
        # Sort by ground truth performance
        sorted_results = sorted(successful_results.items(), 
                              key=lambda x: x[1]['metrics'].get('ground_truth_mse', 
                                                               x[1]['metrics']['mse']))
        
        print(f"\n🏆 PERFORMANCE RANKING (by Ground Truth MSE):")
        print(f"{'Rank':<4} {'Model':<25} {'GT MSE':<10} {'Time':<8} {'Architecture'}")
        print("-" * 80)
        
        for i, (name, result) in enumerate(sorted_results, 1):
            metrics = result['metrics']
            arch_info = result['architecture_info']
            gt_mse = metrics.get('ground_truth_mse', metrics['mse'])
            
            depth = arch_info.get('depth', 0)
            width = arch_info.get('width', 0)
            params = arch_info.get('parameter_count', 0)
            
            print(f"{i:2d}.  {name:<25} {gt_mse:>8.0f} {result['training_time']:>6.0f}s  {depth}x{width} ({params:,} params)")
        
        # Deep vs Wide analysis
        print(f"\n🔍 DEEP vs WIDE ANALYSIS:")
        
        # Categorize models
        deep_models = []
        wide_models = []
        
        for name, result in successful_results.items():
            arch_info = result['architecture_info']
            depth = arch_info.get('depth', 0)
            width = arch_info.get('width', 0)
            gt_mse = result['metrics'].get('ground_truth_mse', result['metrics']['mse'])
            
            # Classify as deep (>10 layers) vs wide (>200 units/layer)
            if depth > 10 and width <= 100:
                deep_models.append((name, gt_mse, depth, width))
            elif depth <= 6 and width > 200:
                wide_models.append((name, gt_mse, depth, width))
        
        if deep_models and wide_models:
            best_deep = min(deep_models, key=lambda x: x[1])
            best_wide = min(wide_models, key=lambda x: x[1])
            
            print(f"  🏆 Best Deep Model: {best_deep[0]} ({best_deep[2]}x{best_deep[3]}) - {best_deep[1]:.0f} MSE")
            print(f"  🏆 Best Wide Model: {best_wide[0]} ({best_wide[2]}x{best_wide[3]}) - {best_wide[1]:.0f} MSE")
            
            if best_deep[1] < best_wide[1]:
                improvement = best_wide[1] / best_deep[1]
                print(f"  ✅ Deep narrow wins by {improvement:.2f}x!")
            else:
                degradation = best_deep[1] / best_wide[1]
                print(f"  ❌ Wide shallow wins by {degradation:.2f}x")
        
        # Parameter efficiency analysis
        print(f"\n📊 PARAMETER EFFICIENCY:")
        param_performance = []
        for name, result in successful_results.items():
            arch_info = result['architecture_info']
            gt_mse = result['metrics'].get('ground_truth_mse', result['metrics']['mse'])
            params = arch_info.get('parameter_count', 1)
            efficiency = 1e6 / (gt_mse * params)  # Higher is better
            param_performance.append((name, efficiency, params, gt_mse))
        
        param_performance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  Most efficient models (performance per parameter):")
        for i, (name, efficiency, params, mse) in enumerate(param_performance[:5], 1):
            print(f"    {i}. {name:<25} {efficiency:>8.2f} eff ({params:>6,} params, {mse:>6.0f} MSE)")
        
        # Architecture insights
        print(f"\n🎯 ARCHITECTURE INSIGHTS:")
        best_name, best_result = sorted_results[0]
        best_arch = best_result['architecture_info']
        
        print(f"  🥇 Overall Best: {best_name}")
        print(f"     Depth: {best_arch.get('depth', 0)} layers")
        print(f"     Width: {best_arch.get('width', 0)} units/layer")
        print(f"     Parameters: {best_arch.get('parameter_count', 0):,}")
        print(f"     GT MSE: {best_result['metrics'].get('ground_truth_mse', best_result['metrics']['mse']):.0f}")
        print(f"     Training time: {best_result['training_time']:.1f}s")
        
        # Save comprehensive results
        output_dir = Path(OUTPUT_DIR)
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
                'experiment_name': EXPERIMENT_NAME,
                'total_time': total_time,
                'mcmc_error_bound': empirical_mse,
                'hypothesis': 'Deep narrow networks outperform wide shallow networks',
                'results': convert_for_json(results),
                'analysis': {
                    'best_deep': best_deep if deep_models else None,
                    'best_wide': best_wide if wide_models else None,
                    'parameter_efficiency_ranking': param_performance[:10]
                }
            }
            
            json.dump(experiment_data, f, indent=2)
        
        # Create comprehensive report
        create_comprehensive_report(results, output_dir, EXPERIMENT_NAME)
        
        print(f"\n📁 Results saved to {output_dir}/")
        
        # Final hypothesis test
        print(f"\n🧪 HYPOTHESIS TEST:")
        print(f"  Hypothesis: Deep narrow networks > Wide shallow networks")
        
        if deep_models and wide_models:
            avg_deep_mse = sum(model[1] for model in deep_models) / len(deep_models)
            avg_wide_mse = sum(model[1] for model in wide_models) / len(wide_models)
            
            print(f"  Average Deep MSE: {avg_deep_mse:.0f}")
            print(f"  Average Wide MSE: {avg_wide_mse:.0f}")
            
            if avg_deep_mse < avg_wide_mse:
                print(f"  ✅ HYPOTHESIS SUPPORTED: Deep narrow wins by {avg_wide_mse/avg_deep_mse:.2f}x")
            else:
                print(f"  ❌ HYPOTHESIS REJECTED: Wide shallow wins by {avg_deep_mse/avg_wide_mse:.2f}x")
        else:
            print(f"  ❓ INCONCLUSIVE: Need both deep and wide models to compare")
    
    else:
        print("\n❌ No models completed successfully!")
    
    print(f"\n✅ Comprehensive deep narrow experiment completed!")


if __name__ == "__main__":
    main()
