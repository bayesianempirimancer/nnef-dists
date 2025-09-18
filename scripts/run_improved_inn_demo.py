#!/usr/bin/env python3
"""
Improved Invertible Neural Network Demo

Tests the enhanced INN with deeper architecture, ActNorm, and geometric preprocessing.
"""

import argparse
from pathlib import Path
import time
import sys
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ef import GaussianNatural1D
from src.improved_inn import ImprovedInvertibleNet, ImprovedINNConfig, train_improved_inn
from scripts.run_noprop_ct_demo import load_existing_data


def run_improved_inn_demo(ef_type: str = "gaussian_1d", num_epochs: int = 50):
    """Run improved INN demonstration."""
    
    print(f"Running Improved Invertible Neural Network demo")
    print("=" * 60)
    
    if ef_type != "gaussian_1d":
        raise ValueError("Improved INN currently only supports gaussian_1d (2D case)")
    
    ef = GaussianNatural1D()
    print(f"Exponential family: {ef.__class__.__name__}")
    print(f"Natural parameter dimension: {ef.eta_dim}")
    
    # Load data
    print("\\nLoading existing training data...")
    train_data, val_data, test_data = load_existing_data(ef_type)
    
    print(f"Training samples: {train_data['eta'].shape[0]}")
    print(f"Validation samples: {val_data['eta'].shape[0]}")
    print(f"Test samples: {test_data['eta'].shape[0]}")
    
    # Configure optimal deeper model with geometric preprocessing
    config = ImprovedINNConfig(
        num_layers=8,  # Sweet spot for depth
        hidden_size=96,  # Good capacity
        num_hidden_layers=3,  # Deep coupling networks with ResNet
        activation="tanh",  # Stable for deep networks
        learning_rate=8e-4,  # Slightly lower for stability with preprocessing
        clamp_alpha=2.0,  # Reasonable clamping
        log_det_weight=0.01,  # Balanced regularization
        invertibility_weight=0.3,  # Stronger invertibility focus
        weight_decay=1e-5,
        use_geometric_preprocessing=True,  # Enable with improved stability
        preprocessing_epsilon=1e-6,
    )
    
    print(f"\\nImproved INN Configuration:")
    print(f"  Layers: {config.num_layers}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Coupling depth: {config.num_hidden_layers}")
    print(f"  Activation: {config.activation}")
    print(f"  Geometric preprocessing: {config.use_geometric_preprocessing}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Log det weight: {config.log_det_weight}")
    print(f"  Invertibility weight: {config.invertibility_weight}")
    
    # Train
    print(f"\\nTraining Improved INN for {num_epochs} epochs...")
    start_time = time.time()
    
    state, history = train_improved_inn(
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
    
    # Evaluate
    print("\\nEvaluating on test data...")
    model = ImprovedInvertibleNet(ef=ef, config=config)
    
    # Forward pass: η → μ
    test_pred, test_log_det = model.apply(state.params, test_data["eta"], reverse=False)
    test_mse = float(jnp.mean(jnp.square(test_pred - test_data["y"])))
    test_mae = float(jnp.mean(jnp.abs(test_pred - test_data["y"])))
    
    print(f"Forward MSE (η → μ): {test_mse:.6f}")
    print(f"Forward MAE: {test_mae:.6f}")
    
    # Test invertibility: μ → η
    reconstructed_eta, recon_log_det = model.apply(state.params, test_pred, reverse=True)
    reconstruction_error = float(jnp.mean(jnp.square(reconstructed_eta - test_data["eta"])))
    print(f"Invertibility error: {reconstruction_error:.8f}")
    
    # Log determinant analysis
    print(f"Mean log det (forward): {jnp.mean(test_log_det):.6f}")
    print(f"Log det consistency: {jnp.mean(jnp.abs(test_log_det + recon_log_det)):.8f}")
    
    # Component-wise analysis
    component_mse = jnp.mean(jnp.square(test_pred - test_data["y"]), axis=0)
    print(f"Component-wise MSE: {[f'{x:.6f}' for x in component_mse]}")
    
    # Visualizations
    print("\\nGenerating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Improved Invertible Neural Network Results', fontsize=16)
    
    epochs = range(len(history['train_loss']))
    
    # Training curves with all components
    axes[0, 0].semilogy(epochs, history['train_loss'], 'b-', label='Total', linewidth=2)
    axes[0, 0].semilogy(epochs, history['train_mse'], 'r-', label='MSE', linewidth=2)
    axes[0, 0].semilogy(epochs, history['train_invertibility'], 'g-', label='Invertibility', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Components')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation curves
    axes[0, 1].semilogy(epochs, history['val_loss'], 'b-', label='Total', linewidth=2)
    axes[0, 1].semilogy(epochs, history['val_mse'], 'r-', label='MSE', linewidth=2)
    axes[0, 1].semilogy(epochs, history['val_invertibility'], 'g-', label='Invertibility', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss Components')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Log determinant evolution
    axes[0, 2].plot(epochs, history['train_log_det'], 'purple', label='Train', linewidth=2)
    axes[0, 2].plot(epochs, history['val_log_det'], 'orange', label='Val', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Log Det Regularization')
    axes[0, 2].set_title('Log Determinant Evolution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Forward mapping visualization
    axes[1, 0].scatter(test_data["eta"][:, 0], test_data["eta"][:, 1], 
                      c='blue', label='η (input)', s=20, alpha=0.7)
    axes[1, 0].scatter(test_pred[:, 0], test_pred[:, 1], 
                      c='red', label='μ (predicted)', s=20, alpha=0.7)
    axes[1, 0].set_xlabel('Component 1')
    axes[1, 0].set_ylabel('Component 2')
    axes[1, 0].set_title('Forward Mapping: η → μ')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Invertibility test
    axes[1, 1].scatter(test_data["eta"][:, 0], reconstructed_eta[:, 0], alpha=0.6, s=20, label='η₁')
    axes[1, 1].scatter(test_data["eta"][:, 1], reconstructed_eta[:, 1], alpha=0.6, s=20, label='η₂')
    
    min_val = min(jnp.min(test_data["eta"]), jnp.min(reconstructed_eta))
    max_val = max(jnp.max(test_data["eta"]), jnp.max(reconstructed_eta))
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[1, 1].set_xlabel('True η')
    axes[1, 1].set_ylabel('Reconstructed η')
    axes[1, 1].set_title(f'Invertibility Test\\n(Error: {reconstruction_error:.2e})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Prediction accuracy
    axes[1, 2].scatter(test_data["y"][:, 0], test_pred[:, 0], alpha=0.6, s=20, label='μ₁')
    axes[1, 2].scatter(test_data["y"][:, 1], test_pred[:, 1], alpha=0.6, s=20, label='μ₂')
    
    min_val = min(jnp.min(test_data["y"]), jnp.min(test_pred))
    max_val = max(jnp.max(test_data["y"]), jnp.max(test_pred))
    axes[1, 2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[1, 2].set_xlabel('True Moments')
    axes[1, 2].set_ylabel('Predicted Moments')
    axes[1, 2].set_title(f'Moment Prediction\\n(MSE: {test_mse:.6f})')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save results
    output_dir = Path("artifacts/improved_inn_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "improved_inn_results.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "improved_inn_results.pdf", bbox_inches='tight')
    print(f"Plots saved to {output_dir}/improved_inn_results.png")
    plt.close()
    
    # Summary
    summary = f"""
    Improved Invertible Neural Network Results
    ==========================================
    
    Architecture Improvements:
    - Layers: {config.num_layers} (vs 4 in simple version)
    - Hidden size: {config.hidden_size} (vs 64 in simple version)
    - Coupling depth: {config.num_hidden_layers} layers
    - ActNorm: Included for better conditioning
    - Geometric preprocessing: {config.use_geometric_preprocessing}
    
    Training Configuration:
    - Learning rate: {config.learning_rate}
    - Log det weight: {config.log_det_weight}
    - Invertibility weight: {config.invertibility_weight}
    
    Performance:
    - Training time: {training_time:.2f}s
    - Forward MSE (η → μ): {test_mse:.6f}
    - Forward MAE: {test_mae:.6f}
    - Invertibility error: {reconstruction_error:.8f}
    - Mean log determinant: {jnp.mean(test_log_det):.6f}
    - Component MSE: {[f'{x:.6f}' for x in component_mse]}
    
    Improvements vs Simple INN:
    - Much deeper architecture (8 vs 4 layers)
    - Residual connections in coupling networks
    - ActNorm for better conditioning
    - Geometric preprocessing for natural parameters
    - Explicit invertibility loss term
    - Better regularization
    
    The invertibility error should be much smaller than the simple version.
    """
    
    print(summary)
    
    with open(output_dir / "summary.txt", 'w') as f:
        f.write(summary)
    
    return state, history, test_mse, reconstruction_error


def main():
    parser = argparse.ArgumentParser(description='Improved INN Demo')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--compare-with-simple', action='store_true', 
                       help='Also run simple INN for comparison')
    
    args = parser.parse_args()
    
    # Run improved INN
    print("🚀 Running Improved INN...")
    state, history, test_mse, invertibility_error = run_improved_inn_demo(num_epochs=args.num_epochs)
    
    print(f"\\n✅ Improved INN completed!")
    print(f"📊 Test MSE: {test_mse:.6f}")
    print(f"🔄 Invertibility error: {invertibility_error:.8f}")
    
    # Optional comparison with simple version
    if args.compare_with_simple:
        print("\\n" + "="*60)
        print("🔄 Running Simple INN for comparison...")
        
        from src.simple_inn import SimpleInvertibleNet, SimpleINNConfig, train_simple_inn
        
        simple_config = SimpleINNConfig(num_layers=4, hidden_size=64)
        ef = GaussianNatural1D()
        train_data, val_data, test_data = load_existing_data("gaussian_1d")
        
        simple_state, simple_history = train_simple_inn(
            ef=ef, train_data=train_data, val_data=val_data,
            config=simple_config, num_epochs=args.num_epochs,
            batch_size=64, seed=42
        )
        
        simple_model = SimpleInvertibleNet(ef=ef, config=simple_config)
        simple_pred, _ = simple_model.apply(simple_state.params, test_data["eta"], reverse=False)
        simple_mse = float(jnp.mean(jnp.square(simple_pred - test_data["y"])))
        
        simple_recon, _ = simple_model.apply(simple_state.params, simple_pred, reverse=True)
        simple_invertibility = float(jnp.mean(jnp.square(simple_recon - test_data["eta"])))
        
        print(f"\\n📈 Comparison Results:")
        print(f"  Simple INN   - MSE: {simple_mse:.6f}, Invertibility: {simple_invertibility:.2e}")
        print(f"  Improved INN - MSE: {test_mse:.6f}, Invertibility: {invertibility_error:.2e}")
        print(f"  MSE improvement: {(simple_mse - test_mse)/simple_mse*100:.1f}%")
        print(f"  Invertibility improvement: {simple_invertibility/invertibility_error:.1f}x better")


if __name__ == "__main__":
    main()
