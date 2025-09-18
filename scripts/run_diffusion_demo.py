#!/usr/bin/env python3
"""
Diffusion Model Demo for Exponential Family Moment Mapping

This script demonstrates the diffusion model approach for learning the mapping
from natural parameters to expected sufficient statistics.
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
from src.diffusion_moments import DiffusionMomentNet, DiffusionConfig, train_diffusion_moment_net
from scripts.run_noprop_ct_demo import load_existing_data  # Reuse data loading


def run_diffusion_demo(ef_type: str = "gaussian_1d", num_epochs: int = 100, data_dir: str = "data"):
    """Run a complete diffusion model demonstration."""
    
    print(f"Running Diffusion Model demo with {ef_type}")
    print("=" * 50)
    
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
    
    # Configure Diffusion Model
    config = DiffusionConfig(
        hidden_sizes=(128, 128, 64),
        activation="swish",
        use_time_embedding=True,
        time_embed_dim=64,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        schedule_type="linear",
        learning_rate=1e-3,
        prediction_type="epsilon",  # Predict noise
        loss_type="mse",
        num_inference_steps=50,
    )
    
    print(f"\\nDiffusion Model Configuration:")
    print(f"  Hidden sizes: {config.hidden_sizes}")
    print(f"  Timesteps: {config.num_timesteps}")
    print(f"  Schedule: {config.schedule_type}")
    print(f"  Prediction type: {config.prediction_type}")
    print(f"  Learning rate: {config.learning_rate}")
    
    # Train model
    print(f"\\nTraining Diffusion Model for {num_epochs} epochs...")
    start_time = time.time()
    
    state, history = train_diffusion_moment_net(
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
    model = DiffusionMomentNet(ef=ef, config=config)
    
    # Sample from the model
    test_rng = random.PRNGKey(123)
    test_pred = model.apply(state.params, test_data["eta"], num_inference_steps=50, rng=test_rng)
    
    test_mse = float(jnp.mean(jnp.square(test_pred - test_data["y"])))
    test_mae = float(jnp.mean(jnp.abs(test_pred - test_data["y"])))
    
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    
    # Component-wise analysis
    component_mse = jnp.mean(jnp.square(test_pred - test_data["y"]), axis=0)
    print(f"Component-wise MSE: {[f'{x:.6f}' for x in component_mse]}")
    
    # Create visualizations
    print("\\nGenerating visualizations...")
    
    # Training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Diffusion Model Training Results', fontsize=14)
    
    epochs = range(len(history['train_loss']))
    
    # Training losses
    axes[0, 0].semilogy(epochs, history['train_loss'], 'b-', label='Total Loss', linewidth=2)
    axes[0, 0].semilogy(epochs, history['train_mse'], 'r-', label='MSE', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation losses
    axes[0, 1].semilogy(epochs, history['val_loss'], 'b-', label='Total Loss', linewidth=2)
    axes[0, 1].semilogy(epochs, history['val_mse'], 'r-', label='MSE', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Prediction vs truth scatter plot
    axes[1, 0].scatter(test_data["y"][:, 0], test_pred[:, 0], alpha=0.6, s=20, label='Component 1')
    if test_data["y"].shape[1] > 1:
        axes[1, 0].scatter(test_data["y"][:, 1], test_pred[:, 1], alpha=0.6, s=20, label='Component 2')
    
    min_val = min(jnp.min(test_data["y"]), jnp.min(test_pred))
    max_val = max(jnp.max(test_data["y"]), jnp.max(test_pred))
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[1, 0].set_xlabel('True Values')
    axes[1, 0].set_ylabel('Predicted Values')
    axes[1, 0].set_title('Predictions vs Truth')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Component-wise MSE
    components = range(len(component_mse))
    axes[1, 1].bar(components, component_mse, alpha=0.7)
    axes[1, 1].set_xlabel('Component')
    axes[1, 1].set_ylabel('MSE')
    axes[1, 1].set_title('Component-wise Test MSE')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save results
    output_dir = Path("artifacts/diffusion_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / f"demo_results_{ef_type}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save model state and history for comparison
    results = {
        'model_state': state,
        'training_history': history,
        'config': config,
        'test_results': {
            'mse': test_mse,
            'mae': test_mae,
            'component_mse': [float(x) for x in component_mse],
        }
    }
    
    with open(output_dir / f"diffusion_results_{ef_type}.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary
    summary = f"""
    Diffusion Model Demo Results
    ============================
    
    Exponential Family: {ef.__class__.__name__}
    Natural Parameter Dimension: {ef.eta_dim}
    
    Dataset:
    - Training samples: {train_data['eta'].shape[0]}
    - Validation samples: {val_data['eta'].shape[0]}
    - Test samples: {test_data['eta'].shape[0]}
    
    Model Configuration:
    - Hidden sizes: {config.hidden_sizes}
    - Timesteps: {config.num_timesteps}
    - Schedule: {config.schedule_type}
    - Prediction type: {config.prediction_type}
    
    Training:
    - Epochs: {num_epochs}
    - Training time: {training_time:.2f}s
    - Final train loss: {history['train_loss'][-1]:.6f}
    - Final val loss: {history['val_loss'][-1]:.6f}
    
    Test Performance:
    - MSE: {test_mse:.6f}
    - MAE: {test_mae:.6f}
    - Component MSE: {[f'{x:.6f}' for x in component_mse]}
    """
    
    with open(output_dir / f"demo_summary_{ef_type}.txt", 'w') as f:
        f.write(summary)
    
    print(summary)
    print(f"Results saved to {output_dir}")
    
    return state, history, test_mse


def visualize_diffusion_process(ef_type: str = "gaussian_1d", data_dir: str = "data"):
    """Visualize the diffusion forward and reverse processes."""
    
    print("Visualizing diffusion process...")
    
    # Load a small amount of test data
    if ef_type == "gaussian_1d":
        ef = GaussianNatural1D()
    else:
        ef = MultivariateNormal(x_shape=(2,))
    
    train_data, val_data, test_data = load_existing_data(ef_type, Path(data_dir))
    
    # Take a few samples for visualization
    sample_moments = test_data["y"][:5]  # First 5 samples
    sample_eta = test_data["eta"][:5]
    
    config = DiffusionConfig(num_timesteps=1000, beta_start=1e-4, beta_end=0.02)
    model = DiffusionMomentNet(ef=ef, config=config)
    
    # Initialize model (we'll use random parameters for visualization)
    rng = random.PRNGKey(42)
    dummy_eta = jnp.zeros((1, ef.eta_dim))
    dummy_moments = jnp.zeros((1, ef.eta_dim))
    dummy_t = jnp.zeros((1,), dtype=jnp.int32)
    params = model.init(rng, dummy_moments, dummy_t, dummy_eta, training=False)
    
    # Visualize forward process (adding noise)
    timesteps_to_show = [0, 100, 250, 500, 750, 999]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Diffusion Forward Process: Adding Noise to Moments', fontsize=14)
    
    for i, t in enumerate(timesteps_to_show):
        ax = axes[i // 3, i % 3]
        
        # Add noise at timestep t
        t_batch = jnp.full((sample_moments.shape[0],), t)
        noise = random.normal(rng, sample_moments.shape)
        noisy_moments = model.apply(params, method=model.q_sample, 
                                   x_0=sample_moments, t=t_batch, noise=noise)
        
        # Plot original vs noisy
        ax.scatter(sample_moments[:, 0], sample_moments[:, 1], 
                  c='blue', label='Original', s=50, alpha=0.7)
        ax.scatter(noisy_moments[:, 0], noisy_moments[:, 1], 
                  c='red', label=f'Noisy (t={t})', s=50, alpha=0.7)
        
        ax.set_xlabel('Moment 1')
        ax.set_ylabel('Moment 2')
        ax.set_title(f'Timestep {t}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = Path("artifacts/diffusion_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "diffusion_process_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Diffusion process visualization saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Diffusion Model Demo')
    parser.add_argument('--ef-type', choices=['gaussian_1d', 'multivariate_2d'], 
                       default='gaussian_1d', help='Exponential family type')
    parser.add_argument('--num-epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing training data files')
    parser.add_argument('--visualize-process', action='store_true',
                       help='Create visualization of diffusion process')
    
    args = parser.parse_args()
    
    if args.visualize_process:
        visualize_diffusion_process(args.ef_type, args.data_dir)
    else:
        # Run demo
        state, history, test_mse = run_diffusion_demo(
            ef_type=args.ef_type,
            num_epochs=args.num_epochs,
            data_dir=args.data_dir
        )
        
        print(f"\\nDemo completed successfully!")
        print(f"Final test MSE: {test_mse:.6f}")


if __name__ == "__main__":
    main()
