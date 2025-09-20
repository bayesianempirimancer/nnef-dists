#!/usr/bin/env python3
"""
Visualization tools for NoProp-CT training dynamics and trajectories.

This script provides detailed visualizations of how the continuous-time NoProp
model evolves during training and inference.
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from jax import random

from src.ef import ef_factory
from src.noprop_ct import NoPropCTMomentNet, NeuralODESolver


def plot_ode_trajectory(model: NoPropCTMomentNet, params: Dict, eta: jnp.ndarray, 
                       save_path: Path, num_time_points: int = 50):
    """Plot the ODE trajectory for a single input."""
    
    # Create time points
    t_eval = jnp.linspace(0, model.config.time_horizon, num_time_points)
    
    # Initial condition (with noise)
    rng = random.PRNGKey(42)
    noise = random.normal(rng, eta.shape) * model.config.noise_scale
    initial_state = eta + noise
    
    # Define vector field function
    def vector_field_fn(state, eta_batch, t):
        return model.vector_field.apply(params['params'], state, eta_batch, t)
    
    # Compute trajectory
    trajectory = []
    state = initial_state
    
    for i, t in enumerate(t_eval):
        trajectory.append(state.copy())
        if i < len(t_eval) - 1:
            dt = t_eval[i+1] - t_eval[i]
            state = NeuralODESolver.euler_step(vector_field_fn, state, eta, t, dt)
    
    trajectory = jnp.stack(trajectory)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('NoProp-CT ODE Trajectory Analysis', fontsize=14)
    
    # Plot trajectory in state space
    if eta.shape[-1] >= 2:
        axes[0, 0].plot(trajectory[:, 0, 0], trajectory[:, 0, 1], 'b-', linewidth=2, alpha=0.8)
        axes[0, 0].scatter(initial_state[0, 0], initial_state[0, 1], c='red', s=100, 
                          marker='o', label='Initial (noisy)', zorder=5)
        axes[0, 0].scatter(eta[0, 0], eta[0, 1], c='green', s=100, 
                          marker='*', label='Target', zorder=5)
        axes[0, 0].scatter(trajectory[-1, 0, 0], trajectory[-1, 0, 1], c='blue', s=100, 
                          marker='s', label='Final', zorder=5)
        axes[0, 0].set_xlabel('η₁')
        axes[0, 0].set_ylabel('η₂')
        axes[0, 0].set_title('Trajectory in Parameter Space')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].plot(t_eval, trajectory[:, 0, 0], 'b-', linewidth=2)
        axes[0, 0].axhline(y=eta[0, 0], color='green', linestyle='--', label='Target')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('η')
        axes[0, 0].set_title('Trajectory vs Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot distance to target over time
    distances = jnp.linalg.norm(trajectory - eta[None, :, :], axis=-1)
    axes[0, 1].plot(t_eval, distances[:, 0], 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Distance to Target')
    axes[0, 1].set_title('Convergence Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Plot vector field magnitude
    vector_fields = []
    for i, t in enumerate(t_eval[:-1]):
        vf = vector_field_fn(trajectory[i], eta, t)
        vector_fields.append(jnp.linalg.norm(vf, axis=-1))
    
    vector_fields = jnp.stack(vector_fields)
    axes[1, 0].plot(t_eval[:-1], vector_fields[:, 0], 'purple', linewidth=2)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Vector Field Magnitude')
    axes[1, 0].set_title('Dynamics Strength Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot component-wise evolution
    for dim in range(min(3, eta.shape[-1])):
        axes[1, 1].plot(t_eval, trajectory[:, 0, dim], 
                       label=f'η_{dim+1}', linewidth=2)
        axes[1, 1].axhline(y=eta[0, dim], color=plt.gca().lines[-1].get_color(), 
                          linestyle='--', alpha=0.5)
    
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Parameter Value')
    axes[1, 1].set_title('Component Evolution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'ode_trajectory.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_vector_field_2d(model: NoPropCTMomentNet, params: Dict, eta_range: Tuple[float, float],
                        save_path: Path, time_points: List[float] = [0.0, 0.5, 1.0]):
    """Plot 2D vector field at different time points."""
    
    if model.ef.eta_dim != 2:
        print("Vector field visualization only available for 2D problems")
        return
    
    # Create grid
    n_grid = 20
    eta1_vals = jnp.linspace(eta_range[0], eta_range[1], n_grid)
    eta2_vals = jnp.linspace(eta_range[0], eta_range[1], n_grid)
    eta1_grid, eta2_grid = jnp.meshgrid(eta1_vals, eta2_vals)
    
    eta_grid = jnp.stack([eta1_grid.flatten(), eta2_grid.flatten()], axis=1)
    
    fig, axes = plt.subplots(1, len(time_points), figsize=(5*len(time_points), 5))
    if len(time_points) == 1:
        axes = [axes]
    
    for i, t in enumerate(time_points):
        # Compute vector field
        def vector_field_fn(state, eta_batch, time):
            return model.vector_field.apply(params['params'], state, eta_batch, time)
        
        vectors = vector_field_fn(eta_grid, eta_grid, t)
        
        # Reshape for plotting
        u = vectors[:, 0].reshape(n_grid, n_grid)
        v = vectors[:, 1].reshape(n_grid, n_grid)
        
        # Plot vector field
        axes[i].quiver(eta1_grid, eta2_grid, u, v, alpha=0.7)
        axes[i].set_xlabel('η₁')
        axes[i].set_ylabel('η₂')
        axes[i].set_title(f'Vector Field at t={t:.1f}')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path / 'vector_field_2d.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_trajectory_animation(model: NoPropCTMomentNet, params: Dict, eta_samples: jnp.ndarray,
                               save_path: Path, num_time_points: int = 100):
    """Create an animation of multiple ODE trajectories."""
    
    if eta_samples.shape[-1] != 2:
        print("Animation only available for 2D problems")
        return
    
    # Create time points
    t_eval = jnp.linspace(0, model.config.time_horizon, num_time_points)
    
    # Compute trajectories for multiple samples
    trajectories = []
    rng = random.PRNGKey(42)
    
    def vector_field_fn(state, eta_batch, t):
        return model.vector_field.apply(params['params'], state, eta_batch, t)
    
    for eta in eta_samples[:5]:  # Limit to 5 trajectories for clarity
        # Initial condition with noise
        noise = random.normal(rng, eta.shape) * model.config.noise_scale
        initial_state = eta + noise
        
        trajectory = []
        state = initial_state[None, :]  # Add batch dimension
        
        for i, t in enumerate(t_eval):
            trajectory.append(state[0].copy())
            if i < len(t_eval) - 1:
                dt = t_eval[i+1] - t_eval[i]
                state = NeuralODESolver.euler_step(vector_field_fn, state, eta[None, :], t, dt)
        
        trajectories.append(jnp.stack(trajectory))
        rng, _ = random.split(rng)
    
    trajectories = jnp.stack(trajectories)
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def animate(frame):
        ax.clear()
        
        # Plot partial trajectories up to current frame
        for i, traj in enumerate(trajectories):
            ax.plot(traj[:frame+1, 0], traj[:frame+1, 1], 
                   alpha=0.7, linewidth=2, label=f'Trajectory {i+1}')
            
            # Mark current position
            if frame < len(traj):
                ax.scatter(traj[frame, 0], traj[frame, 1], s=100, zorder=5)
        
        # Mark targets
        for i, eta in enumerate(eta_samples[:5]):
            ax.scatter(eta[0], eta[1], marker='*', s=200, c='red', 
                      alpha=0.8, zorder=5)
        
        ax.set_xlabel('η₁')
        ax.set_ylabel('η₂')
        ax.set_title(f'NoProp-CT Trajectories (t={t_eval[frame]:.3f})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set consistent axis limits
        all_points = jnp.concatenate([trajectories.reshape(-1, 2), eta_samples[:5]])
        margin = 0.1 * (jnp.max(all_points) - jnp.min(all_points))
        ax.set_xlim(jnp.min(all_points[:, 0]) - margin, jnp.max(all_points[:, 0]) + margin)
        ax.set_ylim(jnp.min(all_points[:, 1]) - margin, jnp.max(all_points[:, 1]) + margin)
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=num_time_points, interval=100, repeat=True)
    anim.save(save_path / 'trajectory_animation.gif', writer='pillow', fps=10)
    plt.close()
    
    print(f"Animation saved to {save_path / 'trajectory_animation.gif'}")


def analyze_training_dynamics(history: Dict, save_path: Path):
    """Analyze and visualize training dynamics specific to NoProp-CT."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('NoProp-CT Training Dynamics Analysis', fontsize=14)
    
    epochs = range(len(history['train_loss']))
    
    # Loss components over time
    axes[0, 0].semilogy(epochs, history['train_denoising'], 'b-', label='Denoising', linewidth=2)
    axes[0, 0].semilogy(epochs, history['train_consistency'], 'r-', label='Consistency', linewidth=2)
    axes[0, 0].semilogy(epochs, history['train_loss'], 'g-', label='Total', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Components')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation performance
    axes[0, 1].semilogy(epochs, history['val_denoising'], 'b-', label='Denoising', linewidth=2)
    axes[0, 1].semilogy(epochs, history['val_consistency'], 'r-', label='Consistency', linewidth=2)
    axes[0, 1].semilogy(epochs, history['val_loss'], 'g-', label='Total', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss Components')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss ratio analysis
    denoising_ratio = jnp.array(history['train_denoising']) / jnp.array(history['train_loss'])
    consistency_ratio = jnp.array(history['train_consistency']) / jnp.array(history['train_loss'])
    
    axes[1, 0].plot(epochs, denoising_ratio, 'b-', label='Denoising', linewidth=2)
    axes[1, 0].plot(epochs, consistency_ratio, 'r-', label='Consistency', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss Ratio')
    axes[1, 0].set_title('Loss Component Ratios')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Training stability (loss variance)
    window_size = max(1, len(epochs) // 20)
    if window_size > 1:
        def rolling_std(data, window):
            return jnp.array([jnp.std(data[max(0, i-window):i+1]) 
                             for i in range(len(data))])
        
        train_stability = rolling_std(history['train_loss'], window_size)
        val_stability = rolling_std(history['val_loss'], window_size)
        
        axes[1, 1].plot(epochs, train_stability, 'b-', label='Train', linewidth=2)
        axes[1, 1].plot(epochs, val_stability, 'r-', label='Validation', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Std (Rolling)')
        axes[1, 1].set_title('Training Stability')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data\\nfor stability analysis', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path / 'training_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize NoProp-CT dynamics')
    parser.add_argument('--results', type=str, required=True, 
                       help='Path to comparison results pickle file')
    parser.add_argument('--output-dir', type=str, default='artifacts/noprop_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--create-animation', action='store_true',
                       help='Create trajectory animation (slower)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    with open(args.results, 'rb') as f:
        results = pickle.load(f)
    
    noprop_state = results['model_states']['noprop_ct']
    noprop_history = results['training_history']['noprop_ct']
    config = results['config']
    
    # Create exponential family
    ef = ef_factory(config["ef_type"], config.get("ef_params", {}))
    
    # Create model
    from src.noprop_ct import NoPropCTConfig
    noprop_config = NoPropCTConfig(
        hidden_sizes=config["hidden_sizes"],
        activation=config["activation"],
    )
    model = NoPropCTMomentNet(ef=ef, config=noprop_config)
    
    print("Generating NoProp-CT visualizations...")
    
    # Analyze training dynamics
    print("- Training dynamics analysis...")
    analyze_training_dynamics(noprop_history, output_dir)
    
    # Create sample data for trajectory analysis
    rng = random.PRNGKey(42)
    if ef.eta_dim == 1:
        eta_samples = random.uniform(rng, (10, 1), minval=-1, maxval=1)
    elif ef.eta_dim == 2:
        eta1 = random.uniform(rng, (10, 1), minval=-1, maxval=1)
        eta2 = random.uniform(rng, (10, 1), minval=-2, maxval=-0.1)
        eta_samples = jnp.concatenate([eta1, eta2], axis=1)
    else:
        eta_samples = random.normal(rng, (10, ef.eta_dim)) * 0.5
    
    # Plot ODE trajectories
    print("- ODE trajectory analysis...")
    plot_ode_trajectory(model, noprop_state.params, eta_samples[:1], output_dir)
    
    # 2D vector field visualization
    if ef.eta_dim == 2:
        print("- Vector field visualization...")
        plot_vector_field_2d(model, noprop_state.params, (-2, 2), output_dir)
        
        if args.create_animation:
            print("- Creating trajectory animation...")
            create_trajectory_animation(model, noprop_state.params, eta_samples, output_dir)
    
    print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
