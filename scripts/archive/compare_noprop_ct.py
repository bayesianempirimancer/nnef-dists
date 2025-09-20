#!/usr/bin/env python3
"""
Comparison script between standard MLP and NoProp-CT approaches for moment mapping.

This script trains both models on the same data and compares their performance,
training dynamics, and computational efficiency.
"""

import argparse
import pickle
import time
from pathlib import Path
from typing import Dict, Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import yaml

from src.ef import ef_factory
from src.model import nat2statMLP
from src.train import train_moment_net
from src.noprop_ct import NoPropCTMomentNet, NoPropCTConfig, train_noprop_ct_moment_net
from src.data_utils import load_data


def plot_comparison_results(standard_history: Dict, noprop_history: Dict, 
                           save_path: Path, config: Dict[str, Any]):
    """Create comprehensive comparison plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Standard MLP vs NoProp-CT Comparison', fontsize=16)
    
    epochs_std = range(len(standard_history['train_mse']))
    epochs_noprop = range(len(noprop_history['train_loss']))
    
    # Training loss comparison
    axes[0, 0].semilogy(epochs_std, standard_history['train_mse'], 'b-', label='Standard MLP', linewidth=2)
    axes[0, 0].semilogy(epochs_noprop, noprop_history['train_denoising'], 'r-', label='NoProp-CT', linewidth=2)
    axes[0, 0].set_title('Training Loss (MSE)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation loss comparison
    axes[0, 1].semilogy(epochs_std, standard_history['val_mse'], 'b-', label='Standard MLP', linewidth=2)
    axes[0, 1].semilogy(epochs_noprop, noprop_history['val_denoising'], 'r-', label='NoProp-CT', linewidth=2)
    axes[0, 1].set_title('Validation Loss (MSE)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # NoProp-CT specific losses
    axes[0, 2].semilogy(epochs_noprop, noprop_history['train_denoising'], 'g-', label='Denoising', linewidth=2)
    axes[0, 2].semilogy(epochs_noprop, noprop_history['train_consistency'], 'orange', label='Consistency', linewidth=2)
    axes[0, 2].semilogy(epochs_noprop, noprop_history['train_loss'], 'r-', label='Total', linewidth=2)
    axes[0, 2].set_title('NoProp-CT Loss Components')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Final loss comparison (bar plot)
    methods = ['Standard MLP', 'NoProp-CT']
    final_train = [standard_history['train_mse'][-1], noprop_history['train_denoising'][-1]]
    final_val = [standard_history['val_mse'][-1], noprop_history['val_denoising'][-1]]
    
    x = jnp.arange(len(methods))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, final_train, width, label='Train', alpha=0.8)
    axes[1, 0].bar(x + width/2, final_val, width, label='Validation', alpha=0.8)
    axes[1, 0].set_title('Final Loss Comparison')
    axes[1, 0].set_ylabel('MSE Loss')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(methods)
    axes[1, 0].legend()
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning curves (smoothed)
    window = max(1, len(epochs_std) // 20)  # Smooth over ~5% of epochs
    
    def smooth(data, window):
        if window <= 1:
            return data
        return jnp.convolve(data, jnp.ones(window)/window, mode='valid')
    
    smooth_std_train = smooth(standard_history['train_mse'], window)
    smooth_noprop_train = smooth(noprop_history['train_denoising'], window)
    smooth_epochs_std = range(len(smooth_std_train))
    smooth_epochs_noprop = range(len(smooth_noprop_train))
    
    axes[1, 1].plot(smooth_epochs_std, smooth_std_train, 'b-', label='Standard MLP', linewidth=2)
    axes[1, 1].plot(smooth_epochs_noprop, smooth_noprop_train, 'r-', label='NoProp-CT', linewidth=2)
    axes[1, 1].set_title('Smoothed Learning Curves')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Training Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    # Configuration summary
    axes[1, 2].axis('off')
    config_text = f"""
    Configuration Summary:
    
    Dataset: {config.get('ef_type', 'Unknown')}
    Training samples: {config.get('num_train', 'Unknown')}
    
    Standard MLP:
    Hidden sizes: {config.get('hidden_sizes', 'Unknown')}
    Learning rate: {config.get('learning_rate', 'Unknown')}
    
    NoProp-CT:
    Time horizon: {config.get('noprop_time_horizon', 'Unknown')}
    Time steps: {config.get('noprop_time_steps', 'Unknown')}
    Noise scale: {config.get('noprop_noise_scale', 'Unknown')}
    """
    axes[1, 2].text(0.1, 0.9, config_text, transform=axes[1, 2].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path / 'comparison_results.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_path / 'comparison_results.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to {save_path}")


def evaluate_models(standard_state, noprop_state, test_data: Dict, ef, config: Dict):
    """Evaluate both models on test data and compute detailed metrics."""
    
    # Standard MLP evaluation
    standard_model = nat2statMLP(ef, 
                                hidden_sizes=config['hidden_sizes'], 
                                activation=config['activation'],
                                output_dim=ef.eta_dim)
    
    std_pred = standard_model.apply({"params": standard_state.params}, test_data["eta"])
    std_mse = float(jnp.mean(jnp.square(std_pred - test_data["y"])))
    std_mae = float(jnp.mean(jnp.abs(std_pred - test_data["y"])))
    
    # NoProp-CT evaluation
    noprop_model = NoPropCTMomentNet(ef=ef, config=noprop_state.config)
    noprop_pred = noprop_model.apply(noprop_state.params, test_data["eta"], training=False)
    noprop_mse = float(jnp.mean(jnp.square(noprop_pred - test_data["y"])))
    noprop_mae = float(jnp.mean(jnp.abs(noprop_pred - test_data["y"])))
    
    # Component-wise analysis
    std_component_mse = jnp.mean(jnp.square(std_pred - test_data["y"]), axis=0)
    noprop_component_mse = jnp.mean(jnp.square(noprop_pred - test_data["y"]), axis=0)
    
    results = {
        'standard_mlp': {
            'mse': std_mse,
            'mae': std_mae,
            'component_mse': [float(x) for x in std_component_mse],
        },
        'noprop_ct': {
            'mse': noprop_mse,
            'mae': noprop_mae,
            'component_mse': [float(x) for x in noprop_component_mse],
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare Standard MLP vs NoProp-CT')
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    parser.add_argument('--data-path', type=str, required=True, help='Path to training data')
    parser.add_argument('--output-dir', type=str, default='artifacts/comparison', 
                       help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # NoProp-CT specific arguments
    parser.add_argument('--noprop-time-horizon', type=float, default=1.0, 
                       help='Time horizon for NoProp-CT')
    parser.add_argument('--noprop-time-steps', type=int, default=10, 
                       help='Number of time steps for ODE integration')
    parser.add_argument('--noprop-noise-scale', type=float, default=0.1, 
                       help='Noise scale for initial conditions')
    parser.add_argument('--noprop-lr', type=float, default=1e-3, 
                       help='Learning rate for NoProp-CT')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with command line args
    config.update({
        'noprop_time_horizon': args.noprop_time_horizon,
        'noprop_time_steps': args.noprop_time_steps,
        'noprop_noise_scale': args.noprop_noise_scale,
        'noprop_lr': args.noprop_lr,
    })
    
    # Load data
    print("Loading data...")
    train_data, val_data, test_data = load_data(args.data_path)
    
    # Create exponential family
    ef = ef_factory(config["ef_type"], config.get("ef_params", {}))
    
    print(f"Training on {ef.__class__.__name__} with {train_data['eta'].shape[0]} samples")
    print(f"Natural parameter dimension: {ef.eta_dim}")
    
    # Train Standard MLP
    print("\\n" + "="*50)
    print("Training Standard MLP...")
    print("="*50)
    
    start_time = time.time()
    standard_state, standard_history = train_moment_net(
        ef=ef,
        train_data=train_data,
        val_data=val_data,
        hidden_sizes=config["hidden_sizes"],
        activation=config["activation"],
        learning_rate=config["learning_rate"],
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    standard_time = time.time() - start_time
    
    print(f"Standard MLP training completed in {standard_time:.2f} seconds")
    print(f"Final train MSE: {standard_history['train_mse'][-1]:.6f}")
    print(f"Final val MSE: {standard_history['val_mse'][-1]:.6f}")
    
    # Train NoProp-CT
    print("\\n" + "="*50)
    print("Training NoProp-CT...")
    print("="*50)
    
    noprop_config = NoPropCTConfig(
        hidden_sizes=config["hidden_sizes"],
        activation=config["activation"],
        time_horizon=args.noprop_time_horizon,
        num_time_steps=args.noprop_time_steps,
        noise_scale=args.noprop_noise_scale,
        learning_rate=args.noprop_lr,
    )
    
    start_time = time.time()
    noprop_state, noprop_history = train_noprop_ct_moment_net(
        ef=ef,
        train_data=train_data,
        val_data=val_data,
        config=noprop_config,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    noprop_time = time.time() - start_time
    
    print(f"NoProp-CT training completed in {noprop_time:.2f} seconds")
    print(f"Final train denoising loss: {noprop_history['train_denoising'][-1]:.6f}")
    print(f"Final val denoising loss: {noprop_history['val_denoising'][-1]:.6f}")
    
    # Evaluate on test data
    print("\\n" + "="*50)
    print("Evaluating models...")
    print("="*50)
    
    test_results = evaluate_models(standard_state, noprop_state, test_data, ef, config)
    
    print("Test Results:")
    print(f"Standard MLP - MSE: {test_results['standard_mlp']['mse']:.6f}, "
          f"MAE: {test_results['standard_mlp']['mae']:.6f}")
    print(f"NoProp-CT    - MSE: {test_results['noprop_ct']['mse']:.6f}, "
          f"MAE: {test_results['noprop_ct']['mae']:.6f}")
    
    # Create comparison plots
    print("\\nGenerating comparison plots...")
    plot_comparison_results(standard_history, noprop_history, output_dir, config)
    
    # Save results
    results = {
        'config': config,
        'training_time': {
            'standard_mlp': standard_time,
            'noprop_ct': noprop_time,
        },
        'training_history': {
            'standard_mlp': standard_history,
            'noprop_ct': noprop_history,
        },
        'test_results': test_results,
        'model_states': {
            'standard_mlp': standard_state,
            'noprop_ct': noprop_state,
        }
    }
    
    with open(output_dir / 'comparison_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary
    summary = f"""
    Comparison Summary
    ==================
    
    Dataset: {config.get('ef_type', 'Unknown')}
    Training samples: {train_data['eta'].shape[0]}
    Validation samples: {val_data['eta'].shape[0]}
    Test samples: {test_data['eta'].shape[0]}
    
    Training Time:
    - Standard MLP: {standard_time:.2f}s
    - NoProp-CT: {noprop_time:.2f}s
    - Speedup: {standard_time/noprop_time:.2f}x
    
    Final Test Performance:
    - Standard MLP MSE: {test_results['standard_mlp']['mse']:.6f}
    - NoProp-CT MSE: {test_results['noprop_ct']['mse']:.6f}
    - Improvement: {(test_results['standard_mlp']['mse'] - test_results['noprop_ct']['mse'])/test_results['standard_mlp']['mse']*100:.2f}%
    
    Component-wise MSE:
    Standard MLP: {test_results['standard_mlp']['component_mse']}
    NoProp-CT:    {test_results['noprop_ct']['component_mse']}
    """
    
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write(summary)
    
    print(summary)
    print(f"\\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
