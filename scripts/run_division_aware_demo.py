#!/usr/bin/env python3
"""
Demo script for the division-aware MLP that can learn division operations.

This addresses the architectural bias against division operations in standard neural networks.
"""

import argparse
from pathlib import Path
import time
import sys
import pickle
import jax
import jax.numpy as jnp
from jax import random
import optax
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ef import GaussianNatural1D
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from division_aware_mlp import DivisionAwareMomentNet, create_division_aware_train_state
from scripts.run_noprop_ct_demo import load_existing_data


def train_division_aware_net(
    ef: GaussianNatural1D,
    train_data: dict,
    val_data: dict,
    config: dict,
    num_epochs: int,
    batch_size: int,
    seed: int,
) -> tuple:
    """Train the division-aware network."""
    
    rng = random.PRNGKey(seed)
    
    # Create model
    model, params = create_division_aware_train_state(ef, config, rng)
    
    # Create optimizer
    optimizer = optax.adam(config['learning_rate'])
    opt_state = optimizer.init(params)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training epoch
        train_loss = 0.0
        num_batches = 0
        
        # Shuffle training data
        rng, shuffle_key = random.split(rng)
        n_train = train_data['eta'].shape[0]
        indices = random.permutation(shuffle_key, n_train)
        
        for i in range(0, n_train, batch_size):
            batch_indices = indices[i:i+batch_size]
            eta_batch = train_data['eta'][batch_indices]
            y_batch = train_data['y'][batch_indices]
            
            def loss_fn(params):
                pred = model.apply(params, eta_batch)
                return jnp.mean(jnp.square(pred - y_batch))
            
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            train_loss += float(loss)
            num_batches += 1
        
        train_loss /= num_batches
        
        # Validation
        val_pred = model.apply(params, val_data['eta'])
        val_loss = float(jnp.mean(jnp.square(val_pred - val_data['y'])))
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    return params, {'train_losses': train_losses, 'val_losses': val_losses}


def run_demo():
    """Run the division-aware MLP demo."""
    
    print("🚀 Running Division-Aware MLP...")
    print("Running Division-Aware MLP demo")
    print("=" * 60)
    
    # Load data
    ef = GaussianNatural1D()
    ef_type = "gaussian_1d"
    
    print(f"Exponential family: {ef.__class__.__name__}")
    print(f"Natural parameter dimension: {ef.eta_dim}")
    print()
    
    train_data, val_data, test_data = load_existing_data(ef_type)
    
    print(f"Training samples: {train_data['eta'].shape[0]}")
    print(f"Validation samples: {val_data['eta'].shape[0]}")
    print(f"Test samples: {test_data['eta'].shape[0]}")
    
    # Configure division-aware model
    config = {
        'hidden_sizes': (64, 32),
        'activation': 'tanh',
        'use_division_layers': True,
        'use_reciprocal_layers': True,
        'learning_rate': 1e-3,
    }
    
    print(f"\nDivision-Aware MLP Configuration:")
    print(f"  Hidden sizes: {config['hidden_sizes']}")
    print(f"  Activation: {config['activation']}")
    print(f"  Division layers: {config['use_division_layers']}")
    print(f"  Reciprocal layers: {config['use_reciprocal_layers']}")
    print(f"  Learning rate: {config['learning_rate']}")
    
    # Train
    start_time = time.time()
    print(f"\nTraining Division-Aware MLP for 50 epochs...")
    
    params, history = train_division_aware_net(
        ef=ef,
        train_data=train_data,
        val_data=val_data,
        config=config,
        num_epochs=50,
        batch_size=64,
        seed=42,
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate
    print("\nEvaluating on test data...")
    model = DivisionAwareMomentNet(ef=ef, **config)
    
    test_pred = model.apply(params, test_data['eta'])
    test_mse = float(jnp.mean(jnp.square(test_pred - test_data['y'])))
    test_mae = float(jnp.mean(jnp.abs(test_pred - test_data['y'])))
    
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    
    # Component-wise analysis
    component_mse = jnp.mean(jnp.square(test_pred - test_data['y']), axis=0)
    print(f"Component-wise MSE: {component_mse}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training history
    axes[0, 0].plot(history['train_losses'], label='Train')
    axes[0, 0].plot(history['val_losses'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].set_title('Training History')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Predictions vs targets
    axes[0, 1].scatter(test_data['y'][:, 0], test_pred[:, 0], alpha=0.6, s=20)
    axes[0, 1].plot([test_data['y'][:, 0].min(), test_data['y'][:, 0].max()], 
                   [test_data['y'][:, 0].min(), test_data['y'][:, 0].max()], 'r--')
    axes[0, 1].set_xlabel('True E[x]')
    axes[0, 1].set_ylabel('Predicted E[x]')
    axes[0, 1].set_title('E[x] Predictions')
    axes[0, 1].grid(True)
    
    axes[1, 0].scatter(test_data['y'][:, 1], test_pred[:, 1], alpha=0.6, s=20)
    axes[1, 0].plot([test_data['y'][:, 1].min(), test_data['y'][:, 1].max()], 
                   [test_data['y'][:, 1].min(), test_data['y'][:, 1].max()], 'r--')
    axes[1, 0].set_xlabel('True E[x²]')
    axes[1, 0].set_ylabel('Predicted E[x²]')
    axes[1, 0].set_title('E[x²] Predictions')
    axes[1, 0].grid(True)
    
    # Error analysis
    errors = jnp.abs(test_pred - test_data['y'])
    axes[1, 1].hist(errors[:, 0], bins=30, alpha=0.7, label='E[x] errors')
    axes[1, 1].hist(errors[:, 1], bins=30, alpha=0.7, label='E[x²] errors')
    axes[1, 1].set_xlabel('Absolute Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save plots
    output_dir = Path("artifacts/division_aware_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "division_aware_results.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "division_aware_results.pdf", bbox_inches='tight')
    plt.close()
    
    # Save summary
    summary_file = output_dir / "demo_summary_gaussian_1d.txt"
    with open(summary_file, 'w') as f:
        f.write("Division-Aware MLP Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Architecture: Division-aware MLP\n")
        f.write(f"Hidden sizes: {config['hidden_sizes']}\n")
        f.write(f"Activation: {config['activation']}\n")
        f.write(f"Division layers: {config['use_division_layers']}\n")
        f.write(f"Reciprocal layers: {config['use_reciprocal_layers']}\n\n")
        f.write(f"Training time: {training_time:.2f}s\n")
        f.write(f"Test MSE: {test_mse:.6f}\n")
        f.write(f"Test MAE: {test_mae:.6f}\n")
        f.write(f"Component MSE: {component_mse}\n\n")
        f.write("Key insight: Explicit division operations should help\n")
        f.write("learn functions like 1/eta2 and eta1/eta2.\n")
    
    print(f"\n    Division-Aware MLP Results")
    print(f"    ==========================")
    print(f"    ")
    print(f"    Architecture: Division-aware MLP")
    print(f"    Hidden sizes: {config['hidden_sizes']}")
    print(f"    Activation: {config['activation']}")
    print(f"    Division layers: {config['use_division_layers']}")
    print(f"    Reciprocal layers: {config['use_reciprocal_layers']}")
    print(f"    ")
    print(f"    Performance:")
    print(f"    - Training time: {training_time:.2f}s")
    print(f"    - Test MSE: {test_mse:.6f}")
    print(f"    - Test MAE: {test_mae:.6f}")
    print(f"    - Component MSE: {component_mse}")
    print(f"    ")
    print(f"    Key insight: Explicit division operations should help")
    print(f"    learn functions like 1/eta2 and eta1/eta2.")
    print(f"    ")
    
    print(f"\n✅ Division-Aware MLP completed!")
    print(f"📊 Test MSE: {test_mse:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Division-Aware MLP demo")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of training epochs")
    args = parser.parse_args()
    
    run_demo()
