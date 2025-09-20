#!/usr/bin/env python3
"""
Training script template for ET (Expected Statistics) neural networks.

This script trains networks that directly predict the expected sufficient statistics
E[T(X)] of exponential family distributions.

Usage:
    python scripts/models/train_{model_name}_ET.py --config configs/gaussian_1d.yaml
    python scripts/models/train_{model_name}_ET.py --config configs/multivariate_3d.yaml
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from config import load_config, FullConfig
from data_utils import generate_exponential_family_data
from ef import MultivariateNormal

def plot_training_results(trainer, eta_data, ground_truth, predictions, losses, config, model_name):
    """Create comprehensive plots for training results."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Learning curves
    ax1 = plt.subplot(2, 4, 1)
    plt.plot(losses, 'b-', linewidth=2)
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # 2. Predictions vs Ground Truth (scatter)
    ax2 = plt.subplot(2, 4, 2)
    plt.scatter(ground_truth, predictions, alpha=0.6, s=20)
    min_val = min(ground_truth.min(), predictions.min())
    max_val = max(ground_truth.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title('Predictions vs Ground Truth', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 3. Residuals
    ax3 = plt.subplot(2, 4, 3)
    residuals = predictions - ground_truth
    plt.scatter(ground_truth, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Ground Truth')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Ground Truth', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 4. Per-statistic performance
    ax4 = plt.subplot(2, 4, 4)
    stat_names = [f'E[T_{i}]' for i in range(ground_truth.shape[1])]
    mse_per_stat = np.mean((predictions - ground_truth) ** 2, axis=0)
    bars = plt.bar(range(len(stat_names)), mse_per_stat)
    plt.xlabel('Statistics')
    plt.ylabel('MSE')
    plt.title('MSE per Statistic', fontsize=14, fontweight='bold')
    plt.xticks(range(len(stat_names)), stat_names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 5. Output distribution
    ax5 = plt.subplot(2, 4, 5)
    if predictions.shape[1] <= 2:
        if predictions.shape[1] == 1:
            plt.hist(predictions[:, 0], bins=50, alpha=0.7, edgecolor='black', label='Predictions')
            plt.hist(ground_truth[:, 0], bins=50, alpha=0.7, edgecolor='black', label='Ground Truth')
            plt.xlabel('E[T(X)]')
            plt.legend()
        else:
            plt.scatter(predictions[:, 0], predictions[:, 1], alpha=0.6, s=20, label='Predictions')
            plt.scatter(ground_truth[:, 0], ground_truth[:, 1], alpha=0.6, s=20, label='Ground Truth')
            plt.xlabel('E[T₁(X)]')
            plt.ylabel('E[T₂(X)]')
            plt.legend()
        plt.title('Output Distribution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, f'{predictions.shape[1]}D output\n(too high for 2D plot)', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        plt.title('Output Distribution', fontsize=14, fontweight='bold')
    
    # 6. Prediction magnitudes
    ax6 = plt.subplot(2, 4, 6)
    pred_magnitudes = np.linalg.norm(predictions, axis=1)
    gt_magnitudes = np.linalg.norm(ground_truth, axis=1)
    plt.hist(pred_magnitudes, bins=50, alpha=0.7, edgecolor='black', label='Predictions')
    plt.hist(gt_magnitudes, bins=50, alpha=0.7, edgecolor='black', label='Ground Truth')
    plt.xlabel('Prediction Magnitude ||E[T(X)]||')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Magnitudes', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Input distribution
    ax7 = plt.subplot(2, 4, 7)
    if eta_data.shape[1] <= 2:
        if eta_data.shape[1] == 1:
            plt.hist(eta_data[:, 0], bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Natural Parameters η')
        else:
            plt.scatter(eta_data[:, 0], eta_data[:, 1], alpha=0.6, s=20)
            plt.xlabel('η₁')
            plt.ylabel('η₂')
        plt.title('Input Distribution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, f'{eta_data.shape[1]}D input\n(too high for 2D plot)', 
                ha='center', va='center', transform=ax7.transAxes, fontsize=12)
        plt.title('Input Distribution', fontsize=14, fontweight='bold')
    
    # 8. Performance metrics
    ax8 = plt.subplot(2, 4, 8)
    mse = np.mean((predictions - ground_truth) ** 2)
    mae = np.mean(np.abs(predictions - ground_truth))
    r2 = 1 - np.sum((ground_truth - predictions) ** 2) / np.sum((ground_truth - np.mean(ground_truth)) ** 2)
    
    metrics_text = f"""
    MSE: {mse:.6f}
    MAE: {mae:.6f}
    R²: {r2:.4f}
    
    Final Loss: {losses[-1]:.6f}
    Parameters: {sum(x.size for x in jax.tree.leaves(trainer.params)):,}
    """
    
    plt.text(0.1, 0.5, metrics_text, transform=ax8.transAxes, 
             fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    plt.axis('off')
    plt.title('Performance Metrics', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_path = f"{model_name}_ET_training_results_{config.network.exp_family}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Training results saved to: {output_path}")
    
    return fig

def create_trainer(config, model_class, model_name):
    """Create trainer instance based on model class."""
    if model_name == "standard_mlp":
        from models.standard_mlp_ET import StandardMLPTrainer
        return StandardMLPTrainer(config)
    elif model_name == "deep_flow":
        from models.deep_flow_ET import DeepFlowTrainer
        return DeepFlowTrainer(config)
    elif model_name == "glu":
        from models.glu_ET import GLUTrainer
        return GLUTrainer(config)
    elif model_name == "invertible_nn":
        from models.invertible_nn_ET import InvertibleNNTrainer
        return InvertibleNNTrainer(config)
    elif model_name == "noprop_ct":
        from models.noprop_ct_ET import NoPropCTTrainer
        return NoPropCTTrainer(config)
    elif model_name == "quadratic_resnet":
        from models.quadratic_resnet_ET import QuadraticResNetTrainer
        return QuadraticResNetTrainer(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def main():
    parser = argparse.ArgumentParser(description='Train ET Neural Network')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (standard_mlp, deep_flow, glu, invertible_nn, noprop_ct, quadratic_resnet)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    print(f"Model: {args.model}")
    print(f"Exponential family: {config.network.exp_family}")
    print(f"Hidden sizes: {config.network.hidden_sizes}")
    print(f"Activation: {config.network.activation}")
    
    # Set random seed
    rng = random.PRNGKey(args.seed)
    
    # Generate data
    print("\nGenerating training data...")
    eta_data, ground_truth = generate_exponential_family_data(
        exp_family=config.network.exp_family,
        n_samples=config.training.n_samples,
        seed=args.seed
    )
    
    print(f"Data shapes: eta={eta_data.shape}, ground_truth={ground_truth.shape}")
    
    # Create exponential family instance
    ef = MultivariateNormal(dim=config.network.exp_family.split('_')[-1])
    
    # Create trainer
    print(f"\nCreating {args.model} ET trainer...")
    trainer = create_trainer(config, None, args.model)
    
    # Initialize parameters
    rng, init_rng = random.split(rng)
    trainer.params = trainer.model.init(init_rng, eta_data[:1])
    
    print(f"Model parameters: {sum(x.size for x in jax.tree.leaves(trainer.params)):,}")
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    losses = []
    
    for epoch in range(args.epochs):
        # Shuffle data
        rng, shuffle_rng = random.split(rng)
        indices = random.permutation(shuffle_rng, len(eta_data))
        eta_batch = eta_data[indices]
        target_batch = ground_truth[indices]
        
        # Train epoch
        trainer.params, trainer.opt_state, loss = trainer.train_epoch(
            trainer.params, trainer.opt_state, eta_batch, target_batch, ef
        )
        
        losses.append(float(loss))
        
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:3d}: Loss = {loss:.6f}")
    
    # Compute final predictions
    print("\nComputing final predictions...")
    predictions = trainer.compute_predictions(trainer.params, eta_data)
    
    # Performance metrics
    mse = np.mean((predictions - ground_truth) ** 2)
    mae = np.mean(np.abs(predictions - ground_truth))
    r2 = 1 - np.sum((ground_truth - predictions) ** 2) / np.sum((ground_truth - np.mean(ground_truth)) ** 2)
    
    print(f"\nFinal Performance:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²:  {r2:.4f}")
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating plots...")
        plot_training_results(trainer, eta_data, ground_truth, predictions, losses, config, args.model)
        plt.show()
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
