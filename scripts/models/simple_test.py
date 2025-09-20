#!/usr/bin/env python3
"""
Simple test script for model validation.

This script tests a few key models to ensure they work correctly.
"""

import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

def generate_simple_test_data(n_samples=200, seed=42):
    """Generate simple 3D Gaussian test data."""
    rng = random.PRNGKey(seed)
    eta_vectors = []
    expected_stats = []
    
    for i in range(n_samples):
        # Generate random mean and covariance
        mean = random.normal(random.PRNGKey(seed + i), (3,)) * 1.0
        A = random.normal(random.PRNGKey(seed + i + 1000), (3, 3))
        covariance_matrix = A.T @ A + jnp.eye(3) * 0.01
        
        # Convert to natural parameters
        sigma_inv = jnp.linalg.inv(covariance_matrix)
        eta1 = sigma_inv @ mean  # η₁ = Σ⁻¹μ
        eta2_matrix = -0.5 * sigma_inv  # η₂ = -0.5Σ⁻¹
        eta_vector = jnp.concatenate([eta1, eta2_matrix.flatten()])
        eta_vectors.append(eta_vector)
        
        # Expected sufficient statistics
        expected_stat = jnp.concatenate([
            mean,  # μ (3 values)
            (jnp.outer(mean, mean) + covariance_matrix).flatten()  # μμᵀ + Σ (9 values)
        ])
        expected_stats.append(expected_stat)
    
    return jnp.array(eta_vectors), jnp.array(expected_stats)

def test_mlp_logZ():
    """Test MLP logZ model."""
    print("Testing MLP LogZ model...")
    
    try:
        # Generate test data
        eta_data, ground_truth = generate_simple_test_data(n_samples=200, seed=42)
        print(f"Data shapes: eta={eta_data.shape}, ground_truth={ground_truth.shape}")
        
        # Test if we can import the model
        from models.mlp_logZ import MLPLogNormalizerNetwork, MLPLogNormalizerTrainer
        
        # Create a simple config
        class SimpleConfig:
            def __init__(self):
                self.network = type('obj', (object,), {
                    'hidden_sizes': [64, 64],
                    'activation': 'swish',
                    'exp_family': 'multivariate_normal_3d'
                })()
                self.training = type('obj', (object,), {
                    'learning_rate': 1e-3,
                    'batch_size': 32,
                    'n_samples': 200
                })()
        
        config = SimpleConfig()
        
        # Create model
        model = MLPLogNormalizerNetwork(config=config.network)
        
        # Initialize parameters
        rng = random.PRNGKey(42)
        params = model.init(rng, eta_data[:1])
        
        param_count = sum(x.size for x in jax.tree.leaves(params))
        print(f"Model parameters: {param_count:,}")
        
        # Test forward pass
        output = model.apply(params, eta_data[:5], training=False)
        print(f"Forward pass output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Check for numerical issues
        if jnp.any(jnp.isnan(output)) or jnp.any(jnp.isinf(output)):
            print("❌ Model produced NaN or Inf values!")
            return False
        
        print("✅ MLP LogZ model working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing MLP LogZ: {e}")
        return False

def test_standard_mlp_ET():
    """Test Standard MLP ET model."""
    print("\nTesting Standard MLP ET model...")
    
    try:
        # Generate test data
        eta_data, ground_truth = generate_simple_test_data(n_samples=200, seed=42)
        
        # Test if we can import the model
        from models.standard_mlp_ET import StandardMLPNetwork, StandardMLPTrainer
        
        # Create a simple config
        class SimpleConfig:
            def __init__(self):
                self.network = type('obj', (object,), {
                    'hidden_sizes': [64, 64],
                    'activation': 'swish',
                    'exp_family': 'multivariate_normal_3d'
                })()
                self.training = type('obj', (object,), {
                    'learning_rate': 1e-3,
                    'batch_size': 32,
                    'n_samples': 200
                })()
        
        config = SimpleConfig()
        
        # Create model
        model = StandardMLPNetwork(config=config.network)
        
        # Initialize parameters
        rng = random.PRNGKey(42)
        params = model.init(rng, eta_data[:1])
        
        param_count = sum(x.size for x in jax.tree.leaves(params))
        print(f"Model parameters: {param_count:,}")
        
        # Test forward pass
        output = model.apply(params, eta_data[:5], training=False)
        print(f"Forward pass output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Check for numerical issues
        if jnp.any(jnp.isnan(output)) or jnp.any(jnp.isinf(output)):
            print("❌ Model produced NaN or Inf values!")
            return False
        
        print("✅ Standard MLP ET model working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Standard MLP ET: {e}")
        return False

def main():
    print("SIMPLE MODEL TESTING")
    print("="*50)
    
    results = {}
    
    # Test key models
    results['mlp_logZ'] = test_mlp_logZ()
    results['standard_mlp_ET'] = test_standard_mlp_ET()
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    successful = sum(results.values())
    total = len(results)
    
    for model, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{model:20}: {status}")
    
    print(f"\nOverall: {successful}/{total} models working")
    
    if successful == total:
        print("🎉 All tested models are working correctly!")
    else:
        print("⚠️  Some models need attention.")

if __name__ == "__main__":
    main()
