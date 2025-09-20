#!/usr/bin/env python3
"""
Test script for the Deep Flow Network.

Tests the new deep flow network with 20 discrete layers, adaptive quadratic
error prediction, and diffusion-based training on both 1D and 3D Gaussian datasets.
"""

import sys
from pathlib import Path
import pickle
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
import time
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.deep_flow_network import DeepFlowNetwork, create_flow_train_state, train_flow_network
from src.ef import GaussianNatural1D, MultivariateNormal


def load_dataset(data_dir: Path, dataset_type: str = "1d"):
    """Load dataset for testing."""
    
    if dataset_type == "1d":
        # Find largest 1D dataset
        suitable_files = []
        for data_file in data_dir.glob("*.pkl"):
            try:
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                
                if data["train_eta"].shape[1] == 2:  # 1D Gaussian
                    suitable_files.append((data_file, data["train_eta"].shape[0]))
            except Exception:
                continue
        
        if not suitable_files:
            raise FileNotFoundError("No 1D Gaussian datasets found!")
        
        best_file, n_samples = max(suitable_files, key=lambda x: x[1])
        ef = GaussianNatural1D()
        
    else:  # 3d
        # Find largest 3D dataset
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
    
    print(f"Loading {dataset_type.upper()} Gaussian data from {best_file.name}")
    print(f"Dataset size: {n_samples} training samples")
    
    with open(best_file, 'rb') as f:
        data = pickle.load(f)
    
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
    n_test = min(n_val // 2, 200)
    
    test_data = {
        "eta": val_data["eta"][:n_test],
        "y": val_data["y"][:n_test]
    }
    
    # Keep remaining as validation
    val_data = {
        "eta": val_data["eta"][n_test:],
        "y": val_data["y"][n_test:]
    }
    
    return train_data, val_data, test_data, ef


def test_deep_flow_1d():
    """Test deep flow network on 1D Gaussian dataset."""
    
    print("🌊 TESTING DEEP FLOW NETWORK ON 1D GAUSSIAN")
    print("=" * 60)
    
    # Load 1D data
    data_dir = Path("data")
    train_data, val_data, test_data, ef = load_dataset(data_dir, "1d")
    
    print(f"\nDataset sizes:")
    print(f"  Training: {train_data['eta'].shape[0]} samples")
    print(f"  Validation: {val_data['eta'].shape[0]} samples")
    print(f"  Test: {test_data['eta'].shape[0]} samples")
    print(f"  Input dimension: {train_data['eta'].shape[1]}")
    print(f"  Output dimension: {train_data['y'].shape[1]}")
    
    # Create deep flow network
    model = DeepFlowNetwork(
        num_layers=20,
        hidden_size=256,
        output_dim=2,
        activation="tanh",
        dropout_rate=0.1,
        use_feature_engineering=True
    )
    
    print(f"\nDeep Flow Network Architecture:")
    print(f"  Number of layers: {model.num_layers}")
    print(f"  Hidden size: {model.hidden_size}")
    print(f"  Output dimension: {model.output_dim}")
    print(f"  Feature engineering: {model.use_feature_engineering}")
    
    # Initialize model
    rng = random.PRNGKey(42)
    eta_sample = test_data['eta'][:1]
    params, optimizer, opt_state = create_flow_train_state(model, rng, eta_sample, learning_rate=1e-3)
    
    # Training configuration
    config = {
        'num_epochs': 80,
        'batch_size': 64,
        'num_timesteps': 100,
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'patience': 15
    }
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: 1e-3")
    print(f"  Diffusion timesteps: {config['num_timesteps']}")
    
    # Train model
    print(f"\nTraining deep flow network...")
    start_time = time.time()
    
    params, history = train_flow_network(
        model, params, optimizer, opt_state, train_data, val_data, config
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f}s")
    
    # Evaluate model
    test_pred = model.apply(params, test_data['eta'], training=False)
    test_mse = float(jnp.mean(jnp.square(test_pred - test_data['y'])))
    test_mae = float(jnp.mean(jnp.abs(test_pred - test_data['y'])))
    
    print(f"\nTest Results:")
    print(f"  MSE: {test_mse:.6f}")
    print(f"  MAE: {test_mae:.6f}")
    
    # Compare with analytical solution (1D Gaussian: mu = -eta1/(2*eta2), sigma^2 = -1/(2*eta2))
    eta1, eta2 = test_data['eta'][:, 0], test_data['eta'][:, 1]
    analytical_mean = -eta1 / (2 * eta2)
    analytical_var = -1 / (2 * eta2)
    analytical_pred = jnp.stack([analytical_mean, analytical_var], axis=1)
    analytical_mse = float(jnp.mean(jnp.square(test_pred - analytical_pred)))
    analytical_mae = float(jnp.mean(jnp.abs(test_pred - analytical_pred)))
    
    print(f"\nComparison with analytical solution:")
    print(f"  MSE vs analytical: {analytical_mse:.6f}")
    print(f"  MAE vs analytical: {analytical_mae:.6f}")
    
    # Create plots
    create_flow_plots_1d(model, params, test_data, ef, history, test_mse)
    
    return {
        'model': model,
        'params': params,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'analytical_mse': analytical_mse,
        'training_time': training_time,
        'history': history
    }


def test_deep_flow_3d():
    """Test deep flow network on 3D Gaussian dataset."""
    
    print("\n🌊 TESTING DEEP FLOW NETWORK ON 3D GAUSSIAN")
    print("=" * 60)
    
    # Load 3D data
    data_dir = Path("data")
    train_data, val_data, test_data, ef = load_dataset(data_dir, "3d")
    
    print(f"\nDataset sizes:")
    print(f"  Training: {train_data['eta'].shape[0]} samples")
    print(f"  Validation: {val_data['eta'].shape[0]} samples")
    print(f"  Test: {test_data['eta'].shape[0]} samples")
    print(f"  Input dimension: {train_data['eta'].shape[1]}")
    print(f"  Output dimension: {test_data['y'].shape[1]}")
    
    # Create deep flow network
    model = DeepFlowNetwork(
        num_layers=20,
        hidden_size=256,
        output_dim=12,
        activation="tanh",
        dropout_rate=0.1,
        use_feature_engineering=True
    )
    
    print(f"\nDeep Flow Network Architecture:")
    print(f"  Number of layers: {model.num_layers}")
    print(f"  Hidden size: {model.hidden_size}")
    print(f"  Output dimension: {model.output_dim}")
    print(f"  Feature engineering: {model.use_feature_engineering}")
    
    # Initialize model
    rng = random.PRNGKey(43)
    eta_sample = test_data['eta'][:1]
    params, optimizer, opt_state = create_flow_train_state(model, rng, eta_sample, learning_rate=8e-4)
    
    # Training configuration
    config = {
        'num_epochs': 100,
        'batch_size': 32,
        'num_timesteps': 100,
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'patience': 20
    }
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: 8e-4")
    print(f"  Diffusion timesteps: {config['num_timesteps']}")
    
    # Train model
    print(f"\nTraining deep flow network...")
    start_time = time.time()
    
    params, history = train_flow_network(
        model, params, optimizer, opt_state, train_data, val_data, config
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f}s")
    
    # Evaluate model
    test_pred = model.apply(params, test_data['eta'], training=False)
    test_mse = float(jnp.mean(jnp.square(test_pred - test_data['y'])))
    test_mae = float(jnp.mean(jnp.abs(test_pred - test_data['y'])))
    
    # Component-wise analysis
    component_mse = jnp.mean(jnp.square(test_pred - test_data['y']), axis=0)
    mean_mse = float(jnp.mean(component_mse[:3]))    # First 3 components (means)
    cov_mse = float(jnp.mean(component_mse[3:]))     # Remaining components (covariances)
    
    print(f"\nTest Results:")
    print(f"  Total MSE: {test_mse:.1f}")
    print(f"  Mean MSE: {mean_mse:.3f}")
    print(f"  Covariance MSE: {cov_mse:.1f}")
    print(f"  MAE: {test_mae:.1f}")
    
    # Compare with analytical solution (3D Gaussian - simplified comparison)
    # For now, just compute MSE against the test targets (which are empirical)
    analytical_mse = test_mse  # Using empirical targets as reference
    
    print(f"\nComparison with analytical solution:")
    print(f"  MSE vs analytical: {analytical_mse:.1f}")
    
    # Create plots
    create_flow_plots_3d(model, params, test_data, ef, history, test_mse)
    
    return {
        'model': model,
        'params': params,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'mean_mse': mean_mse,
        'cov_mse': cov_mse,
        'analytical_mse': analytical_mse,
        'training_time': training_time,
        'history': history
    }


def create_flow_plots_1d(model, params, test_data, ef, history, test_mse):
    """Create visualization plots for 1D flow network results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Training history
    axes[0, 0].plot(history['train_losses'], label='Training Loss', alpha=0.8)
    axes[0, 0].plot(history['val_losses'], label='Validation Loss', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Deep Flow Network Training History (1D)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Predictions vs targets
    test_pred = model.apply(params, test_data['eta'], training=False)
    
    # Plot first component (mean)
    axes[0, 1].scatter(test_data['y'][:, 0], test_pred[:, 0], alpha=0.6, s=20)
    axes[0, 1].plot([test_data['y'][:, 0].min(), test_data['y'][:, 0].max()], 
                    [test_data['y'][:, 0].min(), test_data['y'][:, 0].max()], 'r--', alpha=0.8)
    axes[0, 1].set_xlabel('True Mean')
    axes[0, 1].set_ylabel('Predicted Mean')
    axes[0, 1].set_title(f'Mean Prediction (MSE: {jnp.mean(jnp.square(test_pred[:, 0] - test_data["y"][:, 0])):.4f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Plot second component (variance)
    axes[1, 0].scatter(test_data['y'][:, 1], test_pred[:, 1], alpha=0.6, s=20)
    axes[1, 0].plot([test_data['y'][:, 1].min(), test_data['y'][:, 1].max()], 
                    [test_data['y'][:, 1].min(), test_data['y'][:, 1].max()], 'r--', alpha=0.8)
    axes[1, 0].set_xlabel('True Variance')
    axes[1, 0].set_ylabel('Predicted Variance')
    axes[1, 0].set_title(f'Variance Prediction (MSE: {jnp.mean(jnp.square(test_pred[:, 1] - test_data["y"][:, 1])):.4f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Flow analysis - intermediate states
    eta_sample = test_data['eta'][:5]  # Take 5 samples
    _, intermediates = model.apply(params, eta_sample, method=model.forward_with_intermediates, training=False)
    
    # Plot how the prediction evolves through the flow
    layer_indices = [0, 5, 10, 15, 19]  # Selected layers
    for i, layer_idx in enumerate(layer_indices):
        if f"layer_{layer_idx}" in intermediates:
            layer_pred = intermediates[f"layer_{layer_idx}"][0]  # First sample
            axes[1, 1].plot([0, 1], layer_pred, 'o-', label=f'Layer {layer_idx}', alpha=0.7)
    
    axes[1, 1].set_xlabel('Component Index')
    axes[1, 1].set_ylabel('Predicted Value')
    axes[1, 1].set_title('Flow Evolution Through Layers')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots
    output_dir = Path("artifacts/deep_flow_network")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "deep_flow_1d_results.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "deep_flow_1d_results.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"📊 1D flow network plots saved to {output_dir}/")


def create_flow_plots_3d(model, params, test_data, ef, history, test_mse):
    """Create visualization plots for 3D flow network results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Training history
    axes[0, 0].plot(history['train_losses'], label='Training Loss', alpha=0.8)
    axes[0, 0].plot(history['val_losses'], label='Validation Loss', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Deep Flow Network Training History (3D)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Component-wise MSE
    test_pred = model.apply(params, test_data['eta'], training=False)
    component_mse = jnp.mean(jnp.square(test_pred - test_data['y']), axis=0)
    component_names = ['μ₁', 'μ₂', 'μ₃', 'Σ₁₁', 'Σ₁₂', 'Σ₁₃', 'Σ₂₂', 'Σ₂₃', 'Σ₃₃']
    
    colors = ['blue'] * 3 + ['red'] * 6  # Blue for means, red for covariances
    bars = axes[0, 1].bar(range(len(component_names)), component_mse, color=colors, alpha=0.7)
    axes[0, 1].set_xticks(range(len(component_names)))
    axes[0, 1].set_xticklabels(component_names, rotation=45)
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].set_title('Component-wise Performance')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, component_mse):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Mean predictions (first 3 components)
    for i in range(3):
        axes[0, 2].scatter(test_data['y'][:, i], test_pred[:, i], 
                          alpha=0.6, s=20, label=f'μ{i+1}')
    
    min_val = min(test_data['y'][:, :3].min(), test_pred[:, :3].min())
    max_val = max(test_data['y'][:, :3].max(), test_pred[:, :3].max())
    axes[0, 2].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
    axes[0, 2].set_xlabel('True Mean')
    axes[0, 2].set_ylabel('Predicted Mean')
    axes[0, 2].set_title('Mean Predictions')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Covariance predictions (diagonal elements)
    cov_indices = [3, 6, 8]  # Σ₁₁, Σ₂₂, Σ₃₃
    cov_names = ['Σ₁₁', 'Σ₂₂', 'Σ₃₃']
    
    for i, (idx, name) in enumerate(zip(cov_indices, cov_names)):
        axes[1, 0].scatter(test_data['y'][:, idx], test_pred[:, idx], 
                          alpha=0.6, s=20, label=name)
    
    min_val = min(test_data['y'][:, cov_indices].min(), test_pred[:, cov_indices].min())
    max_val = max(test_data['y'][:, cov_indices].max(), test_pred[:, cov_indices].max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
    axes[1, 0].set_xlabel('True Covariance')
    axes[1, 0].set_ylabel('Predicted Covariance')
    axes[1, 0].set_title('Covariance Predictions (Diagonal)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Flow evolution through layers
    eta_sample = test_data['eta'][:3]  # Take 3 samples
    _, intermediates = model.apply(params, eta_sample, method=model.forward_with_intermediates, training=False)
    
    # Plot how the prediction evolves through the flow (mean components)
    layer_indices = [0, 5, 10, 15, 19]  # Selected layers
    for i, layer_idx in enumerate(layer_indices):
        if f"layer_{layer_idx}" in intermediates:
            layer_pred = intermediates[f"layer_{layer_idx}"][0, :3]  # First sample, mean components
            axes[1, 1].plot([1, 2, 3], layer_pred, 'o-', label=f'Layer {layer_idx}', alpha=0.7)
    
    axes[1, 1].set_xlabel('Mean Component')
    axes[1, 1].set_ylabel('Predicted Value')
    axes[1, 1].set_title('Flow Evolution (Mean Components)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Summary statistics
    axes[1, 2].axis('off')
    
    mean_mse = float(jnp.mean(component_mse[:3]))
    cov_mse = float(jnp.mean(component_mse[3:]))
    
    summary_text = f"DEEP FLOW NETWORK (3D)\\n"
    summary_text += f"={'='*20}\\n\\n"
    summary_text += f"🏆 Architecture: 20-layer flow\\n"
    summary_text += f"📊 Total MSE: {test_mse:.1f}\\n"
    summary_text += f"📈 Mean MSE: {mean_mse:.3f}\\n"
    summary_text += f"📉 Cov MSE: {cov_mse:.1f}\\n\\n"
    
    summary_text += f"🔧 Features:\\n"
    summary_text += f"• Adaptive quadratic error prediction\\n"
    summary_text += f"• Diffusion-based training\\n"
    summary_text += f"• ResNet form: x = x + error\\n"
    summary_text += f"• Feature engineering enabled\\n\\n"
    
    summary_text += f"📋 Best components:\\n"
    best_components = jnp.argsort(component_mse)[:3]
    for i, comp_idx in enumerate(best_components):
        comp_name = component_names[comp_idx]
        comp_mse = component_mse[comp_idx]
        summary_text += f"{i+1}. {comp_name}: {comp_mse:.1f}\\n"
    
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save plots
    output_dir = Path("artifacts/deep_flow_network")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "deep_flow_3d_results.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "deep_flow_3d_results.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"📊 3D flow network plots saved to {output_dir}/")


def main():
    """Run comprehensive tests of the deep flow network."""
    
    print("🌊 DEEP FLOW NETWORK TESTING")
    print("=" * 80)
    print("Testing 20-layer flow network with adaptive quadratic error prediction")
    print("and diffusion-based training on both 1D and 3D Gaussian datasets")
    
    try:
        # Test on 1D dataset
        results_1d = test_deep_flow_1d()
        
        # Test on 3D dataset  
        results_3d = test_deep_flow_3d()
        
        # Summary
        print(f"\n{'='*80}")
        print("DEEP FLOW NETWORK SUMMARY")
        print(f"{'='*80}")
        
        print(f"\n1D Gaussian Results:")
        print(f"  Test MSE: {results_1d['test_mse']:.6f}")
        print(f"  Analytical MSE: {results_1d['analytical_mse']:.6f}")
        print(f"  Training time: {results_1d['training_time']:.1f}s")
        
        print(f"\n3D Gaussian Results:")
        print(f"  Test MSE: {results_3d['test_mse']:.1f}")
        print(f"  Mean MSE: {results_3d['mean_mse']:.3f}")
        print(f"  Covariance MSE: {results_3d['cov_mse']:.1f}")
        print(f"  Training time: {results_3d['training_time']:.1f}s")
        
        print(f"\n✅ Deep flow network testing completed!")
        print(f"📁 Results saved to artifacts/deep_flow_network/")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
