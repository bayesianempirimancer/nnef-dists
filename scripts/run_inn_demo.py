#!/usr/bin/env python3
"""
Invertible Neural Network (INN) Demo for Exponential Family Moment Mapping

This script demonstrates the GLOW-inspired invertible neural network approach 
for learning bijective mappings between natural parameters and expected sufficient statistics.
"""

import argparse
from pathlib import Path
import time
import sys
import pickle
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ef import GaussianNatural1D, MultivariateNormal
from src.invertible_moments import InvertibleMomentNet, INNConfig, train_inn_moment_net
from scripts.run_noprop_ct_demo import load_existing_data  # Reuse data loading


def run_inn_demo(ef_type: str = "gaussian_1d", num_epochs: int = 100, data_dir: str = "data"):
    """Run a complete INN demonstration."""
    
    print(f"Running Invertible Neural Network demo with {ef_type}")
    print("=" * 60)
    
    # Create exponential family
    if ef_type == "gaussian_1d":
        ef = GaussianNatural1D()
    elif ef_type == "multivariate_2d":
        ef = MultivariateNormal(x_shape=(2,))
    else:
        raise ValueError(f"Unknown EF type: {ef_type}")
    
    print(f"Exponential family: {ef.__class__.__name__}")
    print(f"Natural parameter dimension: {ef.eta_dim}")
    print(f"x_shape: {ef.x_shape}")
    
    # Load existing data
    print("\\nLoading existing training data...")
    train_data, val_data, test_data = load_existing_data(ef_type, Path(data_dir))
    
    print(f"Training samples: {train_data['eta'].shape[0]}")
    print(f"Validation samples: {val_data['eta'].shape[0]}")
    print(f"Test samples: {test_data['eta'].shape[0]}")
    
    # Configure INN
    config = INNConfig(
        num_flow_layers=6,
        hidden_sizes=(64, 64, 32),
        activation="relu",
        use_batch_norm=False,
        coupling_type="affine",
        clamp_alpha=2.0,
        use_invertible_conv=True,
        conv_lu_decomposition=True,
        learning_rate=1e-3,
        weight_decay=1e-5,
        gradient_clip_norm=1.0,
        reconstruction_weight=1.0,
        regularization_weight=1e-3,
    )
    
    print(f"\\nINN Configuration:")
    print(f"  Flow layers: {config.num_flow_layers}")
    print(f"  Hidden sizes: {config.hidden_sizes}")
    print(f"  Coupling type: {config.coupling_type}")
    print(f"  Invertible conv: {config.use_invertible_conv}")
    print(f"  Learning rate: {config.learning_rate}")
    
    # Train model
    print(f"\\nTraining INN for {num_epochs} epochs...")
    start_time = time.time()
    
    state, history = train_inn_moment_net(
        ef=ef,
        train_data=train_data,
        val_data=val_data,
        config=config,
        num_epochs=num_epochs,
        batch_size=64,
        seed=42,
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on test data
    print("\\nEvaluating on test data...")
    model = InvertibleMomentNet(ef=ef, config=config)
    
    # Forward pass: predict moments
    test_pred, test_log_det = model.apply(state.params, test_data["eta"], reverse=False)
    test_mse = float(jnp.mean(jnp.square(test_pred - test_data["y"])))
    test_mae = float(jnp.mean(jnp.abs(test_pred - test_data["y"])))
    
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(f"Mean log det: {jnp.mean(test_log_det):.6f}")
    
    # Test invertibility
    print("\\nTesting invertibility...")
    reconstructed_eta, recon_log_det = model.apply(state.params, test_pred, reverse=True)
    reconstruction_error = float(jnp.mean(jnp.square(reconstructed_eta - test_data["eta"])))
    print(f"η reconstruction error: {reconstruction_error:.8f}")
    print(f"Log det consistency: {jnp.mean(jnp.abs(test_log_det + recon_log_det)):.8f}")
    
    # Component-wise analysis
    component_mse = jnp.mean(jnp.square(test_pred - test_data["y"]), axis=0)
    print(f"Component-wise MSE: {[f'{x:.6f}' for x in component_mse]}")
    
    # Create visualizations
    print("\\nGenerating visualizations...")
    
    # Training curves
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Invertible Neural Network Training Results', fontsize=14)
    
    epochs = range(len(history['train_loss']))
    
    # Training losses
    axes[0, 0].semilogy(epochs, history['train_loss'], 'b-', label='Total Loss', linewidth=2)
    axes[0, 0].semilogy(epochs, history['train_reconstruction'], 'r-', label='Reconstruction', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation losses
    axes[0, 1].semilogy(epochs, history['val_loss'], 'b-', label='Total Loss', linewidth=2)
    axes[0, 1].semilogy(epochs, history['val_reconstruction'], 'r-', label='Reconstruction', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Log determinant evolution
    axes[0, 2].plot(epochs, history['train_log_det'], 'g-', label='Train', linewidth=2)
    axes[0, 2].plot(epochs, history['val_log_det'], 'orange', label='Validation', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Log Det Loss')
    axes[0, 2].set_title('Log Determinant Evolution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Prediction vs truth scatter plot
    axes[1, 0].scatter(test_data["y"][:, 0], test_pred[:, 0], alpha=0.6, s=20, label='Component 1')
    if test_data["y"].shape[1] > 1:
        axes[1, 0].scatter(test_data["y"][:, 1], test_pred[:, 1], alpha=0.6, s=20, label='Component 2')
    
    min_val = min(jnp.min(test_data["y"]), jnp.min(test_pred))
    max_val = max(jnp.max(test_data["y"]), jnp.max(test_pred))
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[1, 0].set_xlabel('True Moments')
    axes[1, 0].set_ylabel('Predicted Moments')
    axes[1, 0].set_title('Forward Mapping: η → μ')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Invertibility test visualization
    axes[1, 1].scatter(test_data["eta"][:, 0], reconstructed_eta[:, 0], alpha=0.6, s=20, label='Component 1')
    if test_data["eta"].shape[1] > 1:
        axes[1, 1].scatter(test_data["eta"][:, 1], reconstructed_eta[:, 1], alpha=0.6, s=20, label='Component 2')
    
    min_val = min(jnp.min(test_data["eta"]), jnp.min(reconstructed_eta))
    max_val = max(jnp.max(test_data["eta"]), jnp.max(reconstructed_eta))
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[1, 1].set_xlabel('True η')
    axes[1, 1].set_ylabel('Reconstructed η')
    axes[1, 1].set_title('Invertibility Test: μ → η')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Component-wise MSE
    components = range(len(component_mse))
    axes[1, 2].bar(components, component_mse, alpha=0.7, color='skyblue')
    axes[1, 2].set_xlabel('Component')
    axes[1, 2].set_ylabel('MSE')
    axes[1, 2].set_title('Component-wise Test MSE')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save results
    output_dir = Path("artifacts/inn_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / f"demo_results_{ef_type}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save model state and history
    results = {
        'model_state': state,
        'training_history': history,
        'config': config,
        'test_results': {
            'mse': test_mse,
            'mae': test_mae,
            'component_mse': [float(x) for x in component_mse],
            'reconstruction_error': reconstruction_error,
            'log_det_mean': float(jnp.mean(test_log_det)),
        }
    }
    
    with open(output_dir / f"inn_results_{ef_type}.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary
    summary = f"""
    Invertible Neural Network Demo Results
    ======================================
    
    Exponential Family: {ef.__class__.__name__}
    Natural Parameter Dimension: {ef.eta_dim}
    
    Dataset:
    - Training samples: {train_data['eta'].shape[0]}
    - Validation samples: {val_data['eta'].shape[0]}
    - Test samples: {test_data['eta'].shape[0]}
    
    Model Configuration:
    - Flow layers: {config.num_flow_layers}
    - Hidden sizes: {config.hidden_sizes}
    - Coupling type: {config.coupling_type}
    - Invertible conv: {config.use_invertible_conv}
    
    Training:
    - Epochs: {num_epochs}
    - Training time: {training_time:.2f}s
    - Final train loss: {history['train_loss'][-1]:.6f}
    - Final val loss: {history['val_loss'][-1]:.6f}
    
    Test Performance:
    - Forward MSE (η → μ): {test_mse:.6f}
    - Forward MAE: {test_mae:.6f}
    - Invertibility error (μ → η → μ): {reconstruction_error:.8f}
    - Mean log determinant: {jnp.mean(test_log_det):.6f}
    - Component MSE: {[f'{x:.6f}' for x in component_mse]}
    
    Invertibility Analysis:
    - The network should be perfectly invertible (reconstruction error ≈ 0)
    - Log determinant measures the volume change of the transformation
    - Positive log det means expansion, negative means contraction
    """
    
    with open(output_dir / f"demo_summary_{ef_type}.txt", 'w') as f:
        f.write(summary)
    
    print(summary)
    print(f"Results saved to {output_dir}")
    
    return state, history, test_mse


def visualize_bijective_mapping(ef_type: str = "gaussian_1d", data_dir: str = "data"):
    """Visualize the bijective mapping properties of the INN."""
    
    print("Visualizing bijective mapping...")
    
    # Load data
    if ef_type == "gaussian_1d":
        ef = GaussianNatural1D()
    else:
        ef = MultivariateNormal(x_shape=(2,))
    
    train_data, val_data, test_data = load_existing_data(ef_type, Path(data_dir))
    
    # Quick training for visualization
    config = INNConfig(
        num_flow_layers=4,
        hidden_sizes=(32, 32),
        learning_rate=1e-3,
    )
    
    print("Training small model for visualization...")
    rng = random.PRNGKey(42)
    model = InvertibleMomentNet(ef=ef, config=config)
    
    from src.invertible_moments import create_inn_train_state
    state = create_inn_train_state(rng, model, config)
    
    # Quick training (just a few epochs)
    for epoch in range(10):
        # Simple training step
        perm = random.permutation(rng, jnp.arange(100))  # Use subset
        batch_eta = train_data["eta"][perm[:32]]
        batch_y = train_data["y"][perm[:32]]
        
        # Compute loss
        loss_dict = model.apply(state.params, method=model.log_likelihood, 
                               eta=batch_eta, moments=batch_y)
        print(f"Epoch {epoch}: Loss = {loss_dict['total_loss']:.4f}")
        
        rng, _ = random.split(rng)
    
    # Visualize the mapping
    test_subset = test_data["eta"][:50]  # Use subset for clarity
    test_moments_subset = test_data["y"][:50]
    
    # Forward mapping
    pred_moments, log_det_forward = model.apply(state.params, test_subset, reverse=False)
    
    # Reverse mapping
    pred_eta, log_det_reverse = model.apply(state.params, pred_moments, reverse=True)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('INN Bijective Mapping Visualization', fontsize=14)
    
    # Forward mapping: η → μ
    axes[0, 0].scatter(test_subset[:, 0], test_subset[:, 1], 
                      c='blue', label='η (input)', s=30, alpha=0.7)
    axes[0, 0].scatter(pred_moments[:, 0], pred_moments[:, 1], 
                      c='red', label='μ (predicted)', s=30, alpha=0.7)
    axes[0, 0].set_xlabel('Component 1')
    axes[0, 0].set_ylabel('Component 2')
    axes[0, 0].set_title('Forward Mapping: η → μ')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reverse mapping: μ → η
    axes[0, 1].scatter(pred_moments[:, 0], pred_moments[:, 1], 
                      c='red', label='μ (input)', s=30, alpha=0.7)
    axes[0, 1].scatter(pred_eta[:, 0], pred_eta[:, 1], 
                      c='green', label='η (reconstructed)', s=30, alpha=0.7)
    axes[0, 1].set_xlabel('Component 1')
    axes[0, 1].set_ylabel('Component 2')
    axes[0, 1].set_title('Reverse Mapping: μ → η')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Invertibility check
    reconstruction_error = jnp.mean(jnp.square(pred_eta - test_subset), axis=0)
    axes[1, 0].scatter(test_subset[:, 0], pred_eta[:, 0], alpha=0.6, s=20, label='Component 1')
    axes[1, 0].scatter(test_subset[:, 1], pred_eta[:, 1], alpha=0.6, s=20, label='Component 2')
    
    min_val = min(jnp.min(test_subset), jnp.min(pred_eta))
    max_val = max(jnp.max(test_subset), jnp.max(pred_eta))
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[1, 0].set_xlabel('Original η')
    axes[1, 0].set_ylabel('Reconstructed η')
    axes[1, 0].set_title(f'Invertibility Test\\n(MSE: {jnp.mean(reconstruction_error):.2e})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Log determinant analysis
    axes[1, 1].hist(log_det_forward, bins=20, alpha=0.7, label='Forward', color='blue')
    axes[1, 1].hist(-log_det_reverse, bins=20, alpha=0.7, label='Reverse (neg)', color='red')
    axes[1, 1].set_xlabel('Log Determinant')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Log Determinant Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = Path("artifacts/inn_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "bijective_mapping_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Bijective mapping visualization saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Invertible Neural Network Demo')
    parser.add_argument('--ef-type', choices=['gaussian_1d', 'multivariate_2d'], 
                       default='gaussian_1d', help='Exponential family type')
    parser.add_argument('--num-epochs', type=int, default=50, 
                       help='Number of training epochs')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing training data files')
    parser.add_argument('--visualize-mapping', action='store_true',
                       help='Create visualization of bijective mapping')
    
    args = parser.parse_args()
    
    if args.visualize_mapping:
        visualize_bijective_mapping(args.ef_type, args.data_dir)
    else:
        # Run demo
        state, history, test_mse = run_inn_demo(
            ef_type=args.ef_type,
            num_epochs=args.num_epochs,
            data_dir=args.data_dir
        )
        
        print(f"\\nDemo completed successfully!")
        print(f"Final test MSE: {test_mse:.6f}")


if __name__ == "__main__":
    main()
