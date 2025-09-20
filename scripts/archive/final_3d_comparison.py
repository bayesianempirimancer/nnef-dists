#!/usr/bin/env python3
"""
Final comprehensive comparison on 3D Gaussian dataset.

This script tests the most important approaches:
1. Standard MLP (baseline)
2. NoProp-CT (continuous-time)
3. Diffusion Model
4. Quadratic ResNet
5. Deep Flow Network (NEW!)

All models use the same feature engineering for fair comparison.
Ground truth computed using: Σ^{-1} = -2η₂, μ = Ση₁
"""

import sys
from pathlib import Path
import pickle
import jax
import jax.numpy as jnp
from jax import random
import optax
import matplotlib.pyplot as plt
import numpy as np
import time
import json
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our models
from src.ef import MultivariateNormal
from src.model import nat2statMLP
from src.noprop_ct import NoPropCTMomentNet, NoPropCTConfig, create_noprop_ct_train_state, train_noprop_ct_moment_net
from src.diffusion_moments import DiffusionMomentNet, DiffusionConfig, create_diffusion_train_state, train_diffusion_moment_net
from src.deep_flow_network import DeepFlowNetwork, create_flow_train_state, train_flow_network


def load_largest_3d_dataset(data_dir: Path):
    """Load the largest 3D Gaussian dataset available."""
    
    suitable_files = []
    for data_file in data_dir.glob("*.pkl"):
        try:
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            
            if data["train_eta"].shape[1] == 12:  # 3D Gaussian
                suitable_files.append((data_file, data["train_eta"].shape[0]))
        except Exception:
            continue
    
    if not suitable_files:
        raise FileNotFoundError("No 3D Gaussian datasets found!")
    
    best_file, n_samples = max(suitable_files, key=lambda x: x[1])
    ef = MultivariateNormal(x_shape=(3,))
    
    print(f"Loading 3D Gaussian data from {best_file.name}")
    print(f"Dataset size: {n_samples} training samples")
    
    with open(best_file, 'rb') as f:
        data = pickle.load(f)
    
    return data, ef


def compute_ground_truth_3d(eta: jnp.ndarray) -> jnp.ndarray:
    """
    Compute ground truth statistics for 3D Gaussian using analytical formulas.
    
    For multivariate Gaussian: Σ^{-1} = -2η₂, μ = Ση₁
    Natural parameters: η₁ (3D), η₂ (3x3 matrix, flattened to 9D)
    Sufficient statistics: x (3D), xx^T (3x3 matrix, flattened to 9D)
    Expected sufficient statistics: μ (3D), Σ + μμ^T (3x3 matrix, flattened to 9D)
    """
    batch_size = eta.shape[0]
    
    # Extract η₁ (first 3 components) and η₂ (last 9 components)
    eta1 = eta[:, :3]  # Shape: (batch_size, 3)
    eta2 = eta[:, 3:].reshape(batch_size, 3, 3)  # Shape: (batch_size, 3, 3)
    
    # Compute Σ^{-1} = -2η₂
    Sigma_inv = -2.0 * eta2  # Shape: (batch_size, 3, 3)
    
    # Compute Σ = (Σ^{-1})^{-1}
    Sigma = jnp.linalg.inv(Sigma_inv)  # Shape: (batch_size, 3, 3)
    
    # Compute μ = Ση₁
    mu = jnp.einsum('bij,bj->bi', Sigma, eta1)  # Shape: (batch_size, 3)
    
    # Expected sufficient statistics: [μ, Σ + μμ^T]
    mu_muT = jnp.einsum('bi,bj->bij', mu, mu)  # Shape: (batch_size, 3, 3)
    expected_xxT = Sigma + mu_muT  # Shape: (batch_size, 3, 3)
    
    # Flatten to match our data format: [μ₁, μ₂, μ₃, Σ₁₁, Σ₁₂, Σ₁₃, Σ₂₁, Σ₂₂, Σ₂₃, Σ₃₁, Σ₃₂, Σ₃₃]
    expected_stats = jnp.concatenate([
        mu,  # μ₁, μ₂, μ₃
        expected_xxT[:, 0, 0:1],  # Σ₁₁
        expected_xxT[:, 0, 1:2],  # Σ₁₂
        expected_xxT[:, 0, 2:3],  # Σ₁₃
        expected_xxT[:, 1, 0:1],  # Σ₂₁
        expected_xxT[:, 1, 1:2],  # Σ₂₂
        expected_xxT[:, 1, 2:3],  # Σ₂₃
        expected_xxT[:, 2, 0:1],  # Σ₃₁
        expected_xxT[:, 2, 1:2],  # Σ₃₂
        expected_xxT[:, 2, 2:3],  # Σ₃₃
    ], axis=1)  # Shape: (batch_size, 12)
    
    return expected_stats


def train_model_with_retry(model_func, model_name: str, train_data: Dict, val_data: Dict, 
                          test_data: Dict, config: Dict, max_retries: int = 2) -> Dict:
    """Train a model with retry logic for robustness."""
    
    for attempt in range(max_retries + 1):
        try:
            print(f"\n🔄 Training {model_name} (attempt {attempt + 1}/{max_retries + 1})")
            start_time = time.time()
            
            result = model_func(train_data, val_data, test_data, config)
            result['training_time'] = time.time() - start_time
            result['model_name'] = model_name
            
            print(f"✅ {model_name} completed successfully in {result['training_time']:.1f}s")
            return result
            
        except Exception as e:
            print(f"❌ {model_name} failed on attempt {attempt + 1}: {str(e)[:100]}...")
            if attempt < max_retries:
                print(f"   Retrying with different random seed...")
                time.sleep(1)  # Brief pause before retry
            else:
                print(f"   Max retries reached. Skipping {model_name}.")
                return {
                    'model_name': model_name,
                    'test_mse': float('inf'),
                    'test_mae': float('inf'),
                    'ground_truth_mse': float('inf'),
                    'ground_truth_mae': float('inf'),
                    'training_time': 0.0,
                    'status': 'failed',
                    'error': str(e)
                }


def train_standard_mlp(train_data, val_data, test_data, config):
    """Train standard MLP with feature engineering."""
    
    # Create model with feature engineering
    model = nat2statMLP(
        hidden_sizes=[256, 256, 256],
        activation='tanh',
        use_feature_engineering=True,
        output_dim=12
    )
    
    # Initialize
    rng = random.PRNGKey(42)
    eta_sample = train_data['eta'][:1]
    params = model.init(rng, eta_sample)
    
    # Train
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=config['learning_rate'])
    )
    opt_state = optimizer.init(params)
    
    history = model.train_with_optimizer(
        params, opt_state, optimizer, train_data, val_data,
        num_epochs=config['num_epochs'], batch_size=config['batch_size'],
        patience=config['patience']
    )
    
    # Evaluate
    test_pred = model.apply(params, test_data['eta'])
    test_mse = float(jnp.mean(jnp.square(test_pred - test_data['y'])))
    test_mae = float(jnp.mean(jnp.abs(test_pred - test_data['y'])))
    
    # Ground truth comparison
    ground_truth = compute_ground_truth_3d(test_data['eta'])
    gt_mse = float(jnp.mean(jnp.square(test_pred - ground_truth)))
    gt_mae = float(jnp.mean(jnp.abs(test_pred - ground_truth)))
    
    return {
        'test_mse': test_mse,
        'test_mae': test_mae,
        'ground_truth_mse': gt_mse,
        'ground_truth_mae': gt_mae,
        'history': history
    }


def train_noprop_ct(train_data, val_data, test_data, config):
    """Train NoProp-CT model."""
    
    noprop_config = NoPropCTConfig(
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_time_steps=config['num_time_steps'],
        learning_rate=config['learning_rate']
    )
    
    model = NoPropCTMomentNet(noprop_config)
    params, optimizer, opt_state = create_noprop_ct_train_state(model, random.PRNGKey(42), test_data['eta'][:1])
    
    history = train_noprop_ct_moment_net(
        model, params, optimizer, opt_state, train_data, val_data, noprop_config
    )
    
    # Evaluate
    test_pred = model.apply(params, test_data['eta'])
    test_mse = float(jnp.mean(jnp.square(test_pred - test_data['y'])))
    test_mae = float(jnp.mean(jnp.abs(test_pred - test_data['y'])))
    
    # Ground truth comparison
    ground_truth = compute_ground_truth_3d(test_data['eta'])
    gt_mse = float(jnp.mean(jnp.square(test_pred - ground_truth)))
    gt_mae = float(jnp.mean(jnp.abs(test_pred - ground_truth)))
    
    return {
        'test_mse': test_mse,
        'test_mae': test_mae,
        'ground_truth_mse': gt_mse,
        'ground_truth_mae': gt_mae,
        'history': history
    }


def train_diffusion(train_data, val_data, test_data, config):
    """Train diffusion model."""
    
    diffusion_config = DiffusionConfig(
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_timesteps=config['num_timesteps'],
        learning_rate=config['learning_rate']
    )
    
    model = DiffusionMomentNet(diffusion_config)
    params, optimizer, opt_state = create_diffusion_train_state(model, random.PRNGKey(43), test_data['eta'][:1])
    
    history = train_diffusion_moment_net(
        model, params, optimizer, opt_state, train_data, val_data, diffusion_config
    )
    
    # Evaluate
    test_pred = model.apply(params, test_data['eta'])
    test_mse = float(jnp.mean(jnp.square(test_pred - test_data['y'])))
    test_mae = float(jnp.mean(jnp.abs(test_pred - test_data['y'])))
    
    # Ground truth comparison
    ground_truth = compute_ground_truth_3d(test_data['eta'])
    gt_mse = float(jnp.mean(jnp.square(test_pred - ground_truth)))
    gt_mae = float(jnp.mean(jnp.abs(test_pred - ground_truth)))
    
    return {
        'test_mse': test_mse,
        'test_mae': test_mae,
        'ground_truth_mse': gt_mse,
        'ground_truth_mae': gt_mae,
        'history': history
    }




def train_deep_flow(train_data, val_data, test_data, config):
    """Train deep flow network."""
    
    model = DeepFlowNetwork(
        num_layers=20,
        hidden_size=config['hidden_size'],
        output_dim=12,
        activation="tanh",
        dropout_rate=0.0,  # Disabled for stability
        use_feature_engineering=True
    )
    
    params, optimizer, opt_state = create_flow_train_state(model, random.PRNGKey(49), test_data['eta'][:1], config['learning_rate'])
    
    flow_config = {
        'num_epochs': config['num_epochs'],
        'batch_size': config['batch_size'],
        'num_timesteps': 100,
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'patience': config['patience']
    }
    
    params, history = train_flow_network(
        model, params, optimizer, opt_state, train_data, val_data, flow_config
    )
    
    # Evaluate
    test_pred = model.apply(params, test_data['eta'], training=False)
    test_mse = float(jnp.mean(jnp.square(test_pred - test_data['y'])))
    test_mae = float(jnp.mean(jnp.abs(test_pred - test_data['y'])))
    
    # Ground truth comparison
    ground_truth = compute_ground_truth_3d(test_data['eta'])
    gt_mse = float(jnp.mean(jnp.square(test_pred - ground_truth)))
    gt_mae = float(jnp.mean(jnp.abs(test_pred - ground_truth)))
    
    return {
        'test_mse': test_mse,
        'test_mae': test_mae,
        'ground_truth_mse': gt_mse,
        'ground_truth_mae': gt_mae,
        'history': history
    }


def create_comparison_plots(results: Dict[str, Dict], output_dir: Path):
    """Create comprehensive comparison plots."""
    
    # Filter out failed models
    successful_results = {k: v for k, v in results.items() if v.get('status') != 'failed'}
    
    if not successful_results:
        print("No successful results to plot!")
        return
    
    # Create main comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Extract data for plotting
    model_names = list(successful_results.keys())
    test_mses = [successful_results[name]['test_mse'] for name in model_names]
    gt_mses = [successful_results[name]['ground_truth_mse'] for name in model_names]
    training_times = [successful_results[name]['training_time'] for name in model_names]
    
    # 1. MSE Comparison (Test vs Empirical)
    bars1 = axes[0, 0].bar(range(len(model_names)), test_mses, color='skyblue', alpha=0.7)
    axes[0, 0].set_xticks(range(len(model_names)))
    axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 0].set_ylabel('MSE (vs Empirical)')
    axes[0, 0].set_title('MSE vs Empirical Data')
    axes[0, 0].set_yscale('log')
    
    # Add value labels
    for bar, value in zip(bars1, test_mses):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 2. MSE Comparison (vs Ground Truth)
    bars2 = axes[0, 1].bar(range(len(model_names)), gt_mses, color='lightcoral', alpha=0.7)
    axes[0, 1].set_xticks(range(len(model_names)))
    axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 1].set_ylabel('MSE (vs Ground Truth)')
    axes[0, 1].set_title('MSE vs Analytical Ground Truth')
    axes[0, 1].set_yscale('log')
    
    # Add value labels
    for bar, value in zip(bars2, gt_mses):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Training Time Comparison
    bars3 = axes[0, 2].bar(range(len(model_names)), training_times, color='lightgreen', alpha=0.7)
    axes[0, 2].set_xticks(range(len(model_names)))
    axes[0, 2].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 2].set_ylabel('Training Time (seconds)')
    axes[0, 2].set_title('Training Time Comparison')
    
    # Add value labels
    for bar, value in zip(bars3, training_times):
        height = bar.get_height()
        axes[0, 2].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.0f}s', ha='center', va='bottom', fontsize=8)
    
    # 4. Performance vs Time Scatter
    axes[1, 0].scatter(training_times, gt_mses, s=100, alpha=0.7)
    for i, name in enumerate(model_names):
        axes[1, 0].annotate(name, (training_times[i], gt_mses[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1, 0].set_xlabel('Training Time (seconds)')
    axes[1, 0].set_ylabel('MSE vs Ground Truth')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('Performance vs Training Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Empirical vs Ground Truth Comparison
    empirical_mses = []
    ground_truth_mses = []
    for name in model_names:
        empirical_mses.append(successful_results[name]['test_mse'])
        ground_truth_mses.append(successful_results[name]['ground_truth_mse'])
    
    axes[1, 1].scatter(empirical_mses, ground_truth_mses, s=100, alpha=0.7)
    axes[1, 1].plot([min(empirical_mses), max(empirical_mses)], 
                    [min(empirical_mses), max(empirical_mses)], 'r--', alpha=0.8, label='Perfect Match')
    axes[1, 1].set_xlabel('MSE vs Empirical Data')
    axes[1, 1].set_ylabel('MSE vs Ground Truth')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('Empirical vs Ground Truth Performance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Summary Table
    axes[1, 2].axis('off')
    
    # Create summary table
    summary_data = []
    for name in model_names:
        result = successful_results[name]
        summary_data.append([
            name,
            f"{result['test_mse']:.1f}",
            f"{result['ground_truth_mse']:.1f}",
            f"{result['training_time']:.0f}s"
        ])
    
    table = axes[1, 2].table(cellText=summary_data,
                           colLabels=['Model', 'MSE (Emp)', 'MSE (GT)', 'Time'],
                           cellLoc='center',
                           loc='center',
                           bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    axes[1, 2].set_title('Summary Results', pad=20)
    
    plt.tight_layout()
    
    # Save plots
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "final_3d_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "final_3d_comparison.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparison plots saved to {output_dir}/")


def main():
    """Run final comparison on 3D Gaussian dataset."""
    
    print("🚀 FINAL 3D GAUSSIAN COMPARISON")
    print("=" * 80)
    print("Testing key neural network approaches with ground truth comparison")
    print("Ground truth: Σ^{-1} = -2η₂, μ = Ση₁")
    
    # Load data
    data_dir = Path("data")
    data, ef = load_largest_3d_dataset(data_dir)
    
    # Create train/val/test splits
    train_data = {
        "eta": data["train_eta"],
        "y": data["train_y"]
    }
    
    val_data = {
        "eta": data["val_eta"], 
        "y": data["val_y"]
    }
    
    # Create test split from validation data
    n_val = val_data["eta"].shape[0]
    n_test = min(n_val // 2, 500)
    
    test_data = {
        "eta": val_data["eta"][:n_test],
        "y": val_data["y"][:n_test]
    }
    
    # Keep remaining as validation
    val_data = {
        "eta": val_data["eta"][n_test:],
        "y": val_data["y"][n_test:]
    }
    
    print(f"\nDataset sizes:")
    print(f"  Training: {train_data['eta'].shape[0]} samples")
    print(f"  Validation: {val_data['eta'].shape[0]} samples")
    print(f"  Test: {test_data['eta'].shape[0]} samples")
    print(f"  Input dimension: {train_data['eta'].shape[1]}")
    print(f"  Output dimension: {train_data['y'].shape[1]}")
    
    # Compute ground truth for test set
    ground_truth = compute_ground_truth_3d(test_data['eta'])
    empirical_mse = float(jnp.mean(jnp.square(test_data['y'] - ground_truth)))
    print(f"\nGround truth analysis:")
    print(f"  MSE between empirical and analytical: {empirical_mse:.6f}")
    print(f"  This represents the MCMC sampling error bound")
    
    # Training configuration
    config = {
        'num_epochs': 60,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'hidden_size': 256,
        'num_layers': 6,
        'num_time_steps': 100,
        'patience': 15
    }
    
    print(f"\nTraining configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Define models to test
    models_to_test = {
        'Standard MLP': train_standard_mlp,
        'NoProp-CT': train_noprop_ct,
        'Diffusion': train_diffusion,
        'Deep Flow Network': train_deep_flow
    }
    
    # Train all models
    results = {}
    total_start_time = time.time()
    
    for model_name, train_func in models_to_test.items():
        result = train_model_with_retry(train_func, model_name, train_data, val_data, test_data, config)
        results[model_name] = result
        
        # Print intermediate results
        if result.get('status') != 'failed':
            print(f"  📊 {model_name}: MSE={result['test_mse']:.1f}, GT_MSE={result['ground_truth_mse']:.1f}")
    
    total_time = time.time() - total_start_time
    print(f"\n🏁 All models completed in {total_time:.1f}s")
    
    # Create comparison plots
    output_dir = Path("artifacts/final_3d_comparison")
    create_comparison_plots(results, output_dir)
    
    # Find best models
    successful_results = {k: v for k, v in results.items() if v.get('status') != 'failed'}
    
    if successful_results:
        best_empirical = min(successful_results.items(), key=lambda x: x[1]['test_mse'])
        best_ground_truth = min(successful_results.items(), key=lambda x: x[1]['ground_truth_mse'])
        
        print(f"\n🏆 FINAL RESULTS")
        print(f"{'='*50}")
        print(f"Best vs Empirical Data: {best_empirical[0]} (MSE: {best_empirical[1]['test_mse']:.1f})")
        print(f"Best vs Ground Truth: {best_ground_truth[0]} (MSE: {best_ground_truth[1]['ground_truth_mse']:.1f})")
        
        print(f"\n📈 All Results (vs Ground Truth):")
        sorted_results = sorted(successful_results.items(), key=lambda x: x[1]['ground_truth_mse'])
        for i, (name, result) in enumerate(sorted_results, 1):
            print(f"  {i:2d}. {name:20s}: {result['ground_truth_mse']:8.1f} (time: {result['training_time']:5.0f}s)")
        
        # Save detailed results
        results_file = output_dir / "detailed_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to {results_file}")
        
        # Analysis
        print(f"\n🔍 ANALYSIS:")
        print(f"  • MCMC sampling error bound: {empirical_mse:.6f}")
        print(f"  • Best model achieves: {best_ground_truth[1]['ground_truth_mse']:.1f} MSE")
        print(f"  • Deep Flow Network performance: {successful_results.get('Deep Flow Network', {}).get('ground_truth_mse', 'N/A')}")
    
    else:
        print("\n❌ No models completed successfully!")
    
    print(f"\n✅ Final 3D comparison completed!")
    print(f"📁 Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
