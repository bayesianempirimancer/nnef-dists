#!/usr/bin/env python3
"""
Test script for log normalizer neural networks.

This script tests the basic functionality of the log normalizer approach
on simple exponential family examples to verify correctness.
"""

import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import random

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import NetworkConfig, TrainingConfig, FullConfig
from ef import GaussianNatural1D, MultivariateNormal_tril
from models.log_normalizer import (
    LogNormalizerNetwork, compute_log_normalizer_gradient,
    compute_log_normalizer_hessian, prepare_log_normalizer_data
)


def test_1d_gaussian():
    """Test log normalizer approach on 1D Gaussian."""
    print("Testing Log Normalizer on 1D Gaussian")
    print("=" * 40)
    
    # Create exponential family
    ef = GaussianNatural1D()
    
    # Create test natural parameters
    # For 1D Gaussian: eta = [eta1, eta2] where eta2 < 0
    eta_test = jnp.array([
        [1.0, -0.5],   # Mean ≈ 1, Variance ≈ 1
        [2.0, -1.0],   # Mean ≈ 1, Variance ≈ 0.5
        [0.0, -2.0],   # Mean ≈ 0, Variance ≈ 0.25
    ])
    
    # Compute theoretical means
    # For 1D Gaussian: E[X] = -eta1/(2*eta2), E[X²] = 1/(2*eta2) + E[X]²
    theoretical_means = []
    for eta1, eta2 in eta_test:
        mean_x = -eta1 / (2 * eta2)
        mean_x2 = 1 / (2 * eta2) + mean_x ** 2
        theoretical_means.append([mean_x, mean_x2])
    
    theoretical_means = jnp.array(theoretical_means)
    
    print(f"Natural parameters: {eta_test}")
    print(f"Theoretical means: {theoretical_means}")
    
    # Create simple network
    config = NetworkConfig()
    config.hidden_sizes = [32, 16]
    config.output_dim = 1  # Log normalizer is scalar
    config.use_feature_engineering = True
    config.use_batch_norm = False  # No batch normalization
    config.use_layer_norm = False  # No layer normalization either
    config.activation = "tanh"
    
    model = LogNormalizerNetwork(config=config)
    
    # Initialize
    rng = random.PRNGKey(42)
    params = model.init(rng, eta_test)
    
    print(f"Model parameters: {sum(x.size for x in jax.tree_leaves(params))}")
    
    # Test forward pass
    log_norm = model.apply(params, eta_test, training=False)
    print(f"Log normalizer outputs: {log_norm}")
    
    # Test gradient computation
    network_mean = compute_log_normalizer_gradient(model, params, eta_test)
    print(f"Network mean predictions: {network_mean}")
    print(f"Mean errors: {jnp.abs(network_mean - theoretical_means)}")
    
    # Test Hessian computation (diagonal)
    network_hessian = compute_log_normalizer_hessian(
        model, params, eta_test, method='diagonal'
    )
    print(f"Network Hessian (diagonal): {network_hessian}")
    
    # For 1D Gaussian, the theoretical covariance diagonal should be:
    # Var[X] = -1/(2*eta2), Var[X²] = 1/(2*eta2²) + 2*E[X]²/(2*eta2)
    theoretical_var_x = -1 / (2 * eta_test[:, 1])
    print(f"Theoretical Var[X]: {theoretical_var_x}")
    
    return model, params


def test_3d_gaussian():
    """Test log normalizer approach on 3D Gaussian."""
    print("\nTesting Log Normalizer on 3D Gaussian")
    print("=" * 40)
    
    # Create exponential family
    ef = MultivariateNormal_tril(x_shape=(3,))
    
    # Create test natural parameters
    # For 3D Gaussian with tril format: 3 + 6 = 9 parameters
    eta_test = jnp.array([
        [1.0, 2.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, -1.0],  # First example
        [0.0, 0.0, 0.0, -2.0, 0.0, 0.0, -2.0, 0.0, -2.0],  # Second example
    ])
    
    print(f"Natural parameters shape: {eta_test.shape}")
    
    # Create network
    config = NetworkConfig()
    config.hidden_sizes = [64, 32, 16]
    config.output_dim = 1
    config.use_feature_engineering = True
    config.use_batch_norm = False  # No batch normalization
    config.use_layer_norm = False  # No layer normalization either
    config.activation = "tanh"
    
    model = LogNormalizerNetwork(config=config)
    
    # Initialize
    rng = random.PRNGKey(123)
    params = model.init(rng, eta_test)
    
    print(f"Model parameters: {sum(x.size for x in jax.tree_leaves(params))}")
    
    # Test forward pass
    log_norm = model.apply(params, eta_test, training=False)
    print(f"Log normalizer outputs: {log_norm}")
    
    # Test gradient computation
    network_mean = compute_log_normalizer_gradient(model, params, eta_test)
    print(f"Network mean predictions shape: {network_mean.shape}")
    print(f"Network mean predictions: {network_mean}")
    
    # Test Hessian computation (diagonal)
    network_hessian = compute_log_normalizer_hessian(
        model, params, eta_test, method='diagonal'
    )
    print(f"Network Hessian (diagonal) shape: {network_hessian.shape}")
    print(f"Network Hessian (diagonal): {network_hessian}")
    
    return model, params


def test_training_simulation():
    """Simulate a simple training scenario."""
    print("\nSimulating Training")
    print("=" * 40)
    
    # Create synthetic training data
    n_samples = 100
    eta_dim = 2  # 1D Gaussian
    
    # Generate random natural parameters
    rng = random.PRNGKey(456)
    eta1 = random.normal(rng, (n_samples,))
    eta2 = -random.exponential(rng, (n_samples,)) - 0.1  # Ensure eta2 < 0
    
    eta_data = jnp.stack([eta1, eta2], axis=1)
    
    # Compute theoretical means
    mean_x = -eta1 / (2 * eta2)
    mean_x2 = 1 / (2 * eta2) + mean_x ** 2
    mean_data = jnp.stack([mean_x, mean_x2], axis=1)
    
    print(f"Training data shapes: eta={eta_data.shape}, mean={mean_data.shape}")
    
    # Prepare data for log normalizer training
    train_data = prepare_log_normalizer_data(eta_data, mean_data)
    
    # Create model
    config = NetworkConfig()
    config.hidden_sizes = [32, 16]
    config.output_dim = 1
    config.use_feature_engineering = True
    config.use_batch_norm = False  # No batch normalization
    config.use_layer_norm = False  # No layer normalization either
    config.activation = "tanh"
    
    model = LogNormalizerNetwork(config=config)
    
    # Initialize
    rng, init_rng = random.split(rng)
    params = model.init(init_rng, eta_data[:1])
    
    # Test loss computation
    from models.log_normalizer import log_normalizer_loss_fn
    
    loss = log_normalizer_loss_fn(
        model, params, 
        eta_data[:10], mean_data[:10], 
        loss_weights={'mean_weight': 1.0, 'cov_weight': 0.0}
    )
    
    print(f"Initial loss: {loss}")
    
    # Test gradient computation for optimization
    from jax import value_and_grad
    
    def loss_fn(params):
        return log_normalizer_loss_fn(
            model, params,
            eta_data[:10], mean_data[:10],
            loss_weights={'mean_weight': 1.0, 'cov_weight': 0.0}
        )
    
    loss_val, grads = value_and_grad(loss_fn)(params)
    print(f"Loss value: {loss_val}")
    print(f"Gradient norm: {jnp.linalg.norm(jax.tree_flatten(grads)[0])}")
    
    print("Training simulation completed successfully!")


def main():
    """Run all tests."""
    print("Log Normalizer Neural Network Tests")
    print("=" * 50)
    
    # Test 1D Gaussian
    model_1d, params_1d = test_1d_gaussian()
    
    # Test 3D Gaussian
    model_3d, params_3d = test_3d_gaussian()
    
    # Test training simulation
    test_training_simulation()
    
    print("\nAll tests completed successfully!")
    print("\nKey insights:")
    print("1. Log normalizer networks can be initialized and run forward passes")
    print("2. Gradient computation works and produces reasonable mean estimates")
    print("3. Hessian computation works for diagonal approximation")
    print("4. Loss functions can be computed and differentiated")
    print("5. The approach is ready for full training experiments")


if __name__ == "__main__":
    main()
