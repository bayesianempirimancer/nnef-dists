#!/usr/bin/env python3
"""
Generate learning curves for CT model training.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pickle
import matplotlib.pyplot as plt
import numpy as np

from src.models.noprop.ct import Config, NoPropCT
from src.models.noprop.train_ct import CTTrainingConfig, NoPropCTTrainer


def train_with_logging(data_path: str, epochs: int = 50, batch_size: int = 256):
    """Train CT model with logging for learning curves."""
    
    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    eta = data['eta']
    mu_T = data['mu_T']
    cov_TT = data['cov_TT']
    ess = data['ess']
    
    print(f"Data shapes: eta={eta.shape}, mu_T={mu_T.shape}")
    
    # Create training config
    config = CTTrainingConfig(
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=1e-3,
        dropout_epochs=int(epochs * 0.9)  # 90% of epochs with dropout
    )
    
    # Create trainer
    trainer = NoPropCTTrainer(config, eta.shape[1], mu_T.shape[1])
    
    # Initialize model
    key = jr.PRNGKey(42)
    params = trainer.ct_model.init(key, mu_T[:1], eta[:1], jnp.array([0.5]))
    
    print(f"Model initialized with {sum(x.size for x in jax.tree.leaves(params)):,} parameters")
    
    # Create optimizer
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)
    
    # Training with logging
    train_losses = []
    val_losses = []
    
    # Split data for validation
    n_train = int(0.8 * len(eta))
    train_eta = eta[:n_train]
    train_mu_T = mu_T[:n_train]
    val_eta = eta[n_train:]
    val_mu_T = mu_T[n_train:]
    
    print(f"Training set: {len(train_eta)}, Validation set: {len(val_eta)}")
    
    # Create loss function
    def simple_loss_fn(params, x, target, key, use_dropout=True):
        batch_size = x.shape[0]
        t = jr.uniform(key, (batch_size,), minval=0.0, maxval=1.0)
        model_output = trainer.ct_model.apply(params, target, x, t, training=use_dropout)
        mse_loss = jnp.mean((model_output - target) ** 2)
        return mse_loss
    
    print(f"Starting training for {config.num_epochs} epochs...")
    
    for epoch in range(config.num_epochs):
        # Determine if dropout should be active
        use_dropout = epoch < config.dropout_epochs
        
        # Training step
        batch_indices = jr.choice(key, len(train_eta), (config.batch_size,), replace=False)
        batch_eta = train_eta[batch_indices]
        batch_mu_T = train_mu_T[batch_indices]
        
        key, subkey = jr.split(key)
        train_loss, grads = jax.value_and_grad(simple_loss_fn)(params, batch_eta, batch_mu_T, subkey, use_dropout)
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        # Validation step
        val_batch_indices = jr.choice(key, len(val_eta), (min(config.batch_size, len(val_eta)),), replace=False)
        val_batch_eta = val_eta[val_batch_indices]
        val_batch_mu_T = val_mu_T[val_batch_indices]
        
        key, val_subkey = jr.split(key)
        val_loss = simple_loss_fn(params, val_batch_eta, val_batch_mu_T, val_subkey, use_dropout=False)
        
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        
        if epoch % 10 == 0:
            dropout_status = "with dropout" if use_dropout else "no dropout"
            print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f} ({dropout_status})")
    
    return train_losses, val_losses, config


def plot_learning_curves(train_losses, val_losses, config, save_path=None):
    """Plot learning curves."""
    
    plt.figure(figsize=(12, 8))
    
    # Plot losses
    epochs = range(len(train_losses))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', alpha=0.8)
    plt.plot(epochs, val_losses, label='Validation Loss', alpha=0.8)
    plt.axvline(x=config.dropout_epochs, color='red', linestyle='--', alpha=0.7, label='Dropout Off')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot losses (log scale)
    plt.subplot(2, 2, 2)
    plt.semilogy(epochs, train_losses, label='Training Loss', alpha=0.8)
    plt.semilogy(epochs, val_losses, label='Validation Loss', alpha=0.8)
    plt.axvline(x=config.dropout_epochs, color='red', linestyle='--', alpha=0.7, label='Dropout Off')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Learning Curves (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss difference
    plt.subplot(2, 2, 3)
    loss_diff = np.array(val_losses) - np.array(train_losses)
    plt.plot(epochs, loss_diff, label='Val - Train Loss', color='green', alpha=0.8)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.axvline(x=config.dropout_epochs, color='red', linestyle='--', alpha=0.7, label='Dropout Off')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Difference')
    plt.title('Overfitting Indicator')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot smoothed curves
    plt.subplot(2, 2, 4)
    window = min(10, len(train_losses) // 5)
    if window > 1:
        train_smooth = np.convolve(train_losses, np.ones(window)/window, mode='valid')
        val_smooth = np.convolve(val_losses, np.ones(window)/window, mode='valid')
        smooth_epochs = range(window-1, len(train_losses))
        plt.plot(smooth_epochs, train_smooth, label=f'Training Loss (smoothed {window})', alpha=0.8)
        plt.plot(smooth_epochs, val_smooth, label=f'Validation Loss (smoothed {window})', alpha=0.8)
    else:
        plt.plot(epochs, train_losses, label='Training Loss', alpha=0.8)
        plt.plot(epochs, val_losses, label='Validation Loss', alpha=0.8)
    
    plt.axvline(x=config.dropout_epochs, color='red', linestyle='--', alpha=0.7, label='Dropout Off')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Smoothed Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curves saved to {save_path}")
    
    plt.show()


def main():
    """Main function to generate learning curves."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate learning curves for CT model")
    parser.add_argument("--data", type=str, default="data/multivariate_normal_tril_data_10000.pkl", help="Path to training data")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for learning curves")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--save-path", type=str, default="artifacts/ct_learning_curves.png", help="Path to save plot")
    
    args = parser.parse_args()
    
    # Train with logging
    train_losses, val_losses, config = train_with_logging(args.data, args.epochs, args.batch_size)
    
    # Plot learning curves
    plot_learning_curves(train_losses, val_losses, config, args.save_path)
    
    print(f"\nTraining completed!")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {val_losses[-1]:.6f}")
    print(f"Best validation loss: {min(val_losses):.6f} at epoch {val_losses.index(min(val_losses))}")


if __name__ == "__main__":
    main()
