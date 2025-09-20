#!/usr/bin/env python3
"""
Training script for Convex Neural Network-based log normalizer.

This script trains Input Convex Neural Networks (ICNNs) that learn the log normalizer A(η)
while maintaining convexity properties essential for exponential family distributions.

Usage:
    python scripts/models/train_convex_nn_logZ.py
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import optax
import flax.linen as nn

# Add project root to path  
sys.path.append(str(Path(__file__).parent.parent.parent))

# Simple data generation function
def generate_simple_test_data(n_samples=400, seed=42):
    """Generate simple 2D Gaussian test data with better scaling."""
    eta_vectors = []
    expected_stats = []
    
    for i in range(n_samples):
        # Generate smaller, better conditioned data
        mean = random.normal(random.PRNGKey(seed + i), (2,)) * 0.5  # Smaller means
        A = random.normal(random.PRNGKey(seed + i + 1000), (2, 2)) * 0.3  # Smaller variance
        covariance_matrix = A.T @ A + jnp.eye(2) * 0.1  # Better conditioning
        
        # Convert to natural parameters
        sigma_inv = jnp.linalg.inv(covariance_matrix)
        eta1 = sigma_inv @ mean  # η₁ = Σ⁻¹μ  
        eta2_matrix = -0.5 * sigma_inv  # η₂ = -0.5Σ⁻¹
        eta_vector = jnp.concatenate([eta1, eta2_matrix.flatten()])
        eta_vectors.append(eta_vector)
        
        # Expected sufficient statistics - should match the gradient dimensions
        # For 2D Gaussian: gradient of A(η) w.r.t. [η1_1, η1_2, η2_11, η2_12, η2_21, η2_22]
        # gives us [μ_1, μ_2, x1*x1, x1*x2, x2*x1, x2*x2] expectations
        expected_stat = jnp.concatenate([
            mean,  # μ (2 values)
            (jnp.outer(mean, mean) + covariance_matrix).flatten()  # μμᵀ + Σ (4 values)
        ])
        expected_stats.append(expected_stat)
    
    eta_array = jnp.array(eta_vectors)
    stats_array = jnp.array(expected_stats)
    
    # Normalize the data for better training
    eta_mean = jnp.mean(eta_array, axis=0)
    eta_std = jnp.std(eta_array, axis=0) + 1e-8
    eta_normalized = (eta_array - eta_mean) / eta_std
    
    stats_mean = jnp.mean(stats_array, axis=0)
    stats_std = jnp.std(stats_array, axis=0) + 1e-8
    stats_normalized = (stats_array - stats_mean) / stats_std
    
    print(f"Data ranges - eta: [{eta_normalized.min():.3f}, {eta_normalized.max():.3f}], stats: [{stats_normalized.min():.3f}, {stats_normalized.max():.3f}]")
    
    return eta_normalized, stats_normalized


class ConvexLogZModel(nn.Module):
    """Convex Neural Network for log normalizer."""
    hidden_sizes: list
    
    @nn.compact
    def __call__(self, x, training=True):
        original_input = x
        
        # First layer (no previous z)
        if len(self.hidden_sizes) > 0:
            W_x = self.param('W_x_0', nn.initializers.xavier_uniform(), 
                           (x.shape[-1], self.hidden_sizes[0]))
            b = self.param('b_0', nn.initializers.zeros, (self.hidden_sizes[0],))
            z = nn.relu(jnp.dot(x, W_x) + b)
            
            # Additional hidden layers with convexity constraints
            for i, hidden_size in enumerate(self.hidden_sizes[1:], 1):
                # Non-negative weights from previous layer (maintains convexity)
                W_z = self.param(f'W_z_{i}', nn.initializers.uniform(scale=0.1),
                               (z.shape[-1], hidden_size))
                W_z_nonneg = nn.softplus(W_z)  # Ensure non-negative
                
                # Skip connection from input (unrestricted weights)
                W_x = self.param(f'W_x_{i}', nn.initializers.xavier_uniform(),
                               (original_input.shape[-1], hidden_size))
                b = self.param(f'b_{i}', nn.initializers.zeros, (hidden_size,))
                
                z = nn.relu(jnp.dot(z, W_z_nonneg) + jnp.dot(original_input, W_x) + b)
        else:
            z = x
        
        # Final output layer (scalar log normalizer)
        if len(self.hidden_sizes) > 0:
            W_final_z = self.param('W_final_z', nn.initializers.uniform(scale=0.1),
                                 (z.shape[-1], 1))
            W_final_z_nonneg = nn.softplus(W_final_z)  # Non-negative for convexity
            W_final_x = self.param('W_final_x', nn.initializers.xavier_uniform(),
                                 (original_input.shape[-1], 1))
            b_final = self.param('b_final', nn.initializers.zeros, (1,))
            
            output = (jnp.dot(z, W_final_z_nonneg) + 
                    jnp.dot(original_input, W_final_x) + b_final)
        else:
            W_direct = self.param('W_direct', nn.initializers.xavier_uniform(),
                                (original_input.shape[-1], 1))
            b_direct = self.param('b_direct', nn.initializers.zeros, (1,))
            output = jnp.dot(original_input, W_direct) + b_direct
        
        return jnp.squeeze(output, axis=-1)


class SimpleConvexLogZ:
    """Convex Neural Network Log Normalizer."""
    
    def __init__(self, hidden_sizes=[32, 16]):
        self.hidden_sizes = hidden_sizes
    
    def create_model(self):
        """Create the convex neural network model."""
        return ConvexLogZModel(hidden_sizes=self.hidden_sizes)
    
    def train(self, eta_data, ground_truth, epochs=200, learning_rate=1e-3):
        """Train the convex model."""
        model = self.create_model()
        
        # Initialize parameters
        rng = random.PRNGKey(42)
        params = model.init(rng, eta_data[:1])
        
        print(f"Model parameters: {sum(x.size for x in jax.tree.leaves(params)):,}")
        
        # Optimizer with learning rate schedule
        schedule = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=100,
            decay_rate=0.95
        )
        optimizer = optax.adamw(learning_rate=schedule, weight_decay=1e-5)
        opt_state = optimizer.init(params)
        
        # Loss function using gradient of log normalizer
        def loss_fn(params, eta_batch, target_batch):
            def single_log_normalizer(eta_single):
                return model.apply(params, eta_single[None, :], training=False)[0]
            
            grad_fn = jax.grad(single_log_normalizer)
            network_mean = jax.vmap(grad_fn)(eta_batch)
            mse_loss = jnp.mean(jnp.square(network_mean - target_batch))
            
            # Minimal regularization
            log_norm = model.apply(params, eta_batch, training=True)
            l2_loss = jnp.mean(jnp.square(log_norm))
            return mse_loss + 1e-8 * l2_loss
        
        # Training loop
        losses = []
        print(f"Training Convex Neural Network for {epochs} epochs")
        print(f"Architecture: {self.hidden_sizes}")
        
        for epoch in range(epochs):
            loss, grads = jax.value_and_grad(loss_fn)(params, eta_data, ground_truth)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            losses.append(float(loss))
            
            if epoch % 25 == 0 or epoch < 10:
                print(f"  Epoch {epoch}: Loss = {loss:.6f}")
        
        return params, losses, model
    
    def predict(self, model, params, eta_data):
        """Make predictions using the trained model."""
        def single_log_normalizer(eta_single):
            return model.apply(params, eta_single[None, :], training=False)[0]
        
        grad_fn = jax.grad(single_log_normalizer)
        predictions = jax.vmap(grad_fn)(eta_data)
        return predictions


def main():
    """Main training function."""
    print("Training Convex Neural Network LogZ Model")
    print("=" * 50)
    
    # Generate test data
    print("Generating test data...")
    eta_data, ground_truth = generate_simple_test_data(n_samples=1000, seed=42)
    print(f"Data shape: eta={eta_data.shape}, stats={ground_truth.shape}")
    
    # Create and train model
    print("\nCreating Convex Neural Network model...")
    trainer = SimpleConvexLogZ(hidden_sizes=[256, 128, 64, 32])
    
    print("\nTraining model...")
    params, losses, model = trainer.train(eta_data, ground_truth, epochs=600, learning_rate=2e-3)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = trainer.predict(model, params, eta_data)
    
    # Compute final metrics
    mse = float(jnp.mean(jnp.square(predictions - ground_truth)))
    mae = float(jnp.mean(jnp.abs(predictions - ground_truth)))
    
    print(f"\nFinal Results:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Final Loss: {losses[-1]:.6f}")
    
    print("\nKey advantages of Convex Neural Networks:")
    print("- Guaranteed convexity of log normalizer A(η)")
    print("- Positive semi-definite Hessian (valid covariance)")
    print("- Stable gradients due to convex properties")
    print("- Non-negative weights ensure convexity constraints")
    
    return params, losses, predictions


if __name__ == "__main__":
    main()
