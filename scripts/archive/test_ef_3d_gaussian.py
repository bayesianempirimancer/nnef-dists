#!/usr/bin/env python3
"""
Comprehensive test suite for ef.py focused on 3D Gaussian case.

This script tests the MultivariateNormal class and ExponentialFamily base functionality
with 3D Gaussian distributions, verifying:
- Correct sufficient statistics computation
- Flatten/unflatten operations
- Log-unnormalized density computation
- Expected statistics computation
- Consistency with analytical solutions
"""

import sys
from pathlib import Path
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ef import MultivariateNormal, ef_factory


def test_multivariate_normal_initialization():
    """Test MultivariateNormal class initialization for 3D case."""
    print("🧪 Testing MultivariateNormal initialization...")
    
    # Test 3D case
    ef_3d = MultivariateNormal(x_shape=(3,))
    
    # Check basic properties
    assert ef_3d.x_shape == (3,), f"Expected x_shape (3,), got {ef_3d.x_shape}"
    assert ef_3d.x_dim == 3, f"Expected x_dim 3, got {ef_3d.x_dim}"
    assert ef_3d.eta_dim == 12, f"Expected eta_dim 12, got {ef_3d.eta_dim}"  # 3 + 9 = 12
    
    # Check stat_specs
    stat_specs = ef_3d.stat_specs
    expected_specs = {"x": (3,), "xxT": (3, 3)}
    assert stat_specs == expected_specs, f"Expected {expected_specs}, got {stat_specs}"
    
    # Check stat_names
    stat_names = ef_3d.stat_names()
    expected_names = ["x", "xxT"]
    assert stat_names == expected_names, f"Expected {expected_names}, got {stat_names}"
    
    print("  ✅ Initialization tests passed")
    return ef_3d


def test_stat_computation():
    """Test sufficient statistics computation for 3D samples."""
    print("🧪 Testing sufficient statistics computation...")
    
    ef_3d = MultivariateNormal(x_shape=(3,))
    
    # Test single sample
    rng = random.PRNGKey(42)
    x_single = random.normal(rng, (3,))  # Shape: (3,)
    
    stats_single = ef_3d._compute_stats(x_single)
    
    # Check shapes
    assert stats_single["x"].shape == (3,), f"Expected x shape (3,), got {stats_single['x'].shape}"
    assert stats_single["xxT"].shape == (3, 3), f"Expected xxT shape (3,3), got {stats_single['xxT'].shape}"
    
    # Check values
    expected_x = x_single
    expected_xxT = jnp.outer(x_single, x_single)
    
    assert jnp.allclose(stats_single["x"], expected_x), "x statistic incorrect"
    assert jnp.allclose(stats_single["xxT"], expected_xxT), "xxT statistic incorrect"
    
    # Test batch of samples
    x_batch = random.normal(rng, (100, 3))  # Shape: (100, 3)
    stats_batch = ef_3d._compute_stats(x_batch)
    
    # Check batch shapes
    assert stats_batch["x"].shape == (100, 3), f"Expected batch x shape (100,3), got {stats_batch['x'].shape}"
    assert stats_batch["xxT"].shape == (100, 3, 3), f"Expected batch xxT shape (100,3,3), got {stats_batch['xxT'].shape}"
    
    # Check batch values for first sample
    assert jnp.allclose(stats_batch["x"][0], x_batch[0]), "Batch x statistic incorrect"
    expected_xxT_0 = jnp.outer(x_batch[0], x_batch[0])
    assert jnp.allclose(stats_batch["xxT"][0], expected_xxT_0), "Batch xxT statistic incorrect"
    
    print("  ✅ Sufficient statistics computation tests passed")
    return ef_3d, x_batch, stats_batch


def test_flatten_unflatten_operations():
    """Test flatten and unflatten operations for 3D parameters."""
    print("🧪 Testing flatten/unflatten operations...")
    
    ef_3d = MultivariateNormal(x_shape=(3,))
    
    # Create test statistics dict
    rng = random.PRNGKey(123)
    test_stats = {
        "x": random.normal(rng, (3,)),
        "xxT": random.normal(random.split(rng)[1], (3, 3))
    }
    
    # Test flatten
    flattened = ef_3d.flatten_stats_or_eta(test_stats)
    
    # Check shape
    expected_flat_dim = 3 + 9  # x: 3 elements, xxT: 9 elements
    assert flattened.shape == (12,), f"Expected flattened shape (12,), got {flattened.shape}"
    
    # Test unflatten
    unflattened = ef_3d.unflatten_stats_or_eta(flattened)
    
    # Check shapes
    assert unflattened["x"].shape == (3,), f"Expected unflattened x shape (3,), got {unflattened['x'].shape}"
    assert unflattened["xxT"].shape == (3, 3), f"Expected unflattened xxT shape (3,3), got {unflattened['xxT'].shape}"
    
    # Check values (round-trip consistency)
    assert jnp.allclose(unflattened["x"], test_stats["x"]), "x round-trip failed"
    assert jnp.allclose(unflattened["xxT"], test_stats["xxT"]), "xxT round-trip failed"
    
    # Test batch flatten/unflatten
    batch_stats = {
        "x": random.normal(rng, (50, 3)),
        "xxT": random.normal(random.split(rng)[0], (50, 3, 3))
    }
    
    batch_flattened = ef_3d.flatten_stats_or_eta(batch_stats)
    assert batch_flattened.shape == (50, 12), f"Expected batch flattened shape (50,12), got {batch_flattened.shape}"
    
    batch_unflattened = ef_3d.unflatten_stats_or_eta(batch_flattened)
    assert jnp.allclose(batch_unflattened["x"], batch_stats["x"]), "Batch x round-trip failed"
    assert jnp.allclose(batch_unflattened["xxT"], batch_stats["xxT"]), "Batch xxT round-trip failed"
    
    print("  ✅ Flatten/unflatten tests passed")
    return ef_3d


def test_log_unnormalized_computation():
    """Test log-unnormalized density computation."""
    print("🧪 Testing log-unnormalized density computation...")
    
    ef_3d = MultivariateNormal(x_shape=(3,))
    
    # Create test data
    rng = random.PRNGKey(456)
    x = random.normal(rng, (3,))
    
    # Test with dict eta
    eta_dict = {
        "x": random.normal(random.split(rng)[0], (3,)),
        "xxT": random.normal(random.split(rng)[1], (3, 3))
    }
    
    log_p_dict = ef_3d.log_unnormalized(x, eta_dict)
    
    # Manual computation for verification
    stats = ef_3d._compute_stats(x)
    expected_log_p = (jnp.sum(stats["x"] * eta_dict["x"]) + 
                     jnp.sum(stats["xxT"] * eta_dict["xxT"]))
    
    assert jnp.allclose(log_p_dict, expected_log_p), "Dict eta log-unnormalized computation incorrect"
    
    # Test with flattened eta
    eta_flat = ef_3d.flatten_stats_or_eta(eta_dict)
    log_p_flat = ef_3d.log_unnormalized(x, eta_flat)
    
    assert jnp.allclose(log_p_dict, log_p_flat), "Dict and flat eta should give same result"
    
    # Test batch computation
    x_batch = random.normal(rng, (20, 3))
    eta_batch = random.normal(rng, (20, 12))
    
    log_p_batch = ef_3d.log_unnormalized(x_batch, eta_batch)
    assert log_p_batch.shape == (20,), f"Expected batch log_p shape (20,), got {log_p_batch.shape}"
    
    print("  ✅ Log-unnormalized computation tests passed")
    return ef_3d


def test_expected_statistics():
    """Test expected statistics computation."""
    print("🧪 Testing expected statistics computation...")
    
    ef_3d = MultivariateNormal(x_shape=(3,))
    
    # Create batch of samples
    rng = random.PRNGKey(789)
    n_samples = 1000
    x_samples = random.normal(rng, (n_samples, 3))
    
    # Compute expected statistics
    expected_stats = ef_3d.compute_expected_stats(x_samples)
    
    # Check shapes
    assert expected_stats["x"].shape == (3,), f"Expected mean x shape (3,), got {expected_stats['x'].shape}"
    assert expected_stats["xxT"].shape == (3, 3), f"Expected mean xxT shape (3,3), got {expected_stats['xxT'].shape}"
    
    # Manual verification
    manual_mean_x = jnp.mean(x_samples, axis=0)
    manual_mean_xxT = jnp.mean(x_samples[:, :, None] * x_samples[:, None, :], axis=0)
    
    assert jnp.allclose(expected_stats["x"], manual_mean_x, atol=1e-6), "Expected x statistic incorrect"
    assert jnp.allclose(expected_stats["xxT"], manual_mean_xxT, atol=1e-6), "Expected xxT statistic incorrect"
    
    # Test flattened version
    expected_stats_flat = ef_3d.compute_expected_stats(x_samples, flatten=True)
    assert expected_stats_flat.shape == (12,), f"Expected flattened shape (12,), got {expected_stats_flat.shape}"
    
    # Verify consistency
    expected_stats_unflat = ef_3d.unflatten_stats_or_eta(expected_stats_flat)
    assert jnp.allclose(expected_stats_unflat["x"], expected_stats["x"]), "Flattened expected stats inconsistent"
    assert jnp.allclose(expected_stats_unflat["xxT"], expected_stats["xxT"]), "Flattened expected stats inconsistent"
    
    print("  ✅ Expected statistics tests passed")
    return ef_3d


def test_logdensity_function():
    """Test logdensity function creation and usage."""
    print("🧪 Testing logdensity function...")
    
    ef_3d = MultivariateNormal(x_shape=(3,))
    
    # Create test eta
    rng = random.PRNGKey(101112)
    eta_dict = {
        "x": random.normal(rng, (3,)),
        "xxT": random.normal(random.split(rng)[0], (3, 3))
    }
    
    # Create logdensity function
    logdensity_fn = ef_3d.make_logdensity_fn(eta_dict)
    
    # Test with flattened x
    x = random.normal(random.split(rng)[1], (3,))
    x_flat = x.flatten()
    
    log_p_fn = logdensity_fn(x_flat)
    log_p_direct = ef_3d.log_unnormalized(x, eta_dict)
    
    assert jnp.allclose(log_p_fn, log_p_direct), "Logdensity function inconsistent with direct computation"
    
    # Test default initial position
    init_pos = ef_3d.default_initial_position()
    assert init_pos.shape == (3,), f"Expected init pos shape (3,), got {init_pos.shape}"
    assert jnp.allclose(init_pos, jnp.zeros(3)), "Default initial position should be zeros"
    
    print("  ✅ Logdensity function tests passed")
    return ef_3d


def test_ef_factory():
    """Test exponential family factory function."""
    print("🧪 Testing ef_factory...")
    
    # Test multivariate normal creation
    ef_2d = ef_factory("multivariate_normal", x_shape=(2,))
    assert isinstance(ef_2d, MultivariateNormal), "Factory should return MultivariateNormal"
    assert ef_2d.x_shape == (2,), f"Expected x_shape (2,), got {ef_2d.x_shape}"
    
    ef_3d = ef_factory("mv_normal", x_shape=[3])  # Test with list input
    assert isinstance(ef_3d, MultivariateNormal), "Factory should return MultivariateNormal"
    assert ef_3d.x_shape == (3,), f"Expected x_shape (3,), got {ef_3d.x_shape}"
    
    # Test default 2D
    ef_default = ef_factory("multivariate_normal")
    assert ef_default.x_shape == (2,), f"Expected default x_shape (2,), got {ef_default.x_shape}"
    
    print("  ✅ Factory function tests passed")


def test_analytical_consistency():
    """Test consistency with analytical solutions for known cases."""
    print("🧪 Testing analytical consistency...")
    
    ef_3d = MultivariateNormal(x_shape=(3,))
    
    # Generate samples from a known 3D Gaussian
    rng = random.PRNGKey(131415)
    
    # Known mean and covariance
    true_mean = jnp.array([1.0, -0.5, 2.0])
    true_cov = jnp.array([[2.0, 0.5, 0.0],
                          [0.5, 1.5, -0.3],
                          [0.0, -0.3, 1.0]])
    
    # Generate samples
    n_samples = 5000
    samples = random.multivariate_normal(rng, true_mean, true_cov, (n_samples,))
    
    # Compute empirical expected statistics
    empirical_stats = ef_3d.compute_expected_stats(samples)
    
    # Analytical expected statistics
    analytical_mean = true_mean
    analytical_second_moment = true_cov + jnp.outer(true_mean, true_mean)
    
    # Compare (with tolerance for sampling error)
    mean_error = jnp.max(jnp.abs(empirical_stats["x"] - analytical_mean))
    second_moment_error = jnp.max(jnp.abs(empirical_stats["xxT"] - analytical_second_moment))
    
    print(f"  Mean error: {mean_error:.4f}")
    print(f"  Second moment error: {second_moment_error:.4f}")
    
    # These should be small for large sample sizes
    assert mean_error < 0.1, f"Mean error too large: {mean_error}"
    assert second_moment_error < 0.2, f"Second moment error too large: {second_moment_error}"
    
    print("  ✅ Analytical consistency tests passed")
    return ef_3d


def test_edge_cases():
    """Test edge cases and error handling."""
    print("🧪 Testing edge cases...")
    
    ef_3d = MultivariateNormal(x_shape=(3,))
    
    # Test with zero samples
    try:
        empty_samples = jnp.zeros((0, 3))
        empty_stats = ef_3d.compute_expected_stats(empty_samples)
        print("  ⚠️  Empty sample handling needs improvement")
    except:
        print("  ✅ Empty samples properly rejected")
    
    # Test with wrong shapes
    try:
        wrong_shape = jnp.zeros((10, 2))  # Wrong last dimension
        ef_3d._compute_stats(wrong_shape)
        print("  ⚠️  Shape validation needs improvement")
    except:
        print("  ✅ Wrong shapes properly rejected")
    
    # Test with NaN/Inf values
    nan_sample = jnp.array([1.0, jnp.nan, 2.0])
    nan_stats = ef_3d._compute_stats(nan_sample)
    assert jnp.isnan(nan_stats["x"][1]), "NaN should propagate in statistics"
    
    print("  ✅ Edge case tests completed")


def run_comprehensive_3d_test():
    """Run all 3D Gaussian tests for ef.py."""
    print("🧊 COMPREHENSIVE 3D GAUSSIAN TESTS FOR ef.py")
    print("=" * 60)
    
    try:
        # Run all test functions
        ef_3d = test_multivariate_normal_initialization()
        test_stat_computation()
        test_flatten_unflatten_operations()
        test_log_unnormalized_computation()
        test_expected_statistics()
        test_logdensity_function()
        test_ef_factory()
        test_analytical_consistency()
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("🎉 ALL 3D GAUSSIAN TESTS PASSED!")
        print("=" * 60)
        
        # Performance benchmark
        print("\n🚀 Performance benchmark...")
        rng = random.PRNGKey(999)
        
        # Large batch computation
        large_batch = random.normal(rng, (10000, 3))
        
        import time
        start_time = time.time()
        stats = ef_3d.compute_stats(large_batch, flatten=False)
        compute_time = time.time() - start_time
        
        start_time = time.time()
        expected_stats = ef_3d.compute_expected_stats(large_batch)
        expected_time = time.time() - start_time
        
        print(f"  Compute stats (10k samples): {compute_time:.4f}s")
        print(f"  Expected stats (10k samples): {expected_time:.4f}s")
        
        # Memory usage check
        eta_batch = random.normal(rng, (1000, 12))
        x_batch = random.normal(rng, (1000, 3))
        
        start_time = time.time()
        log_p_batch = ef_3d.log_unnormalized(x_batch, eta_batch)
        batch_time = time.time() - start_time
        
        print(f"  Log-unnormalized (1k batch): {batch_time:.4f}s")
        print(f"  Result shape: {log_p_batch.shape}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Starting comprehensive 3D Gaussian tests for ef.py...")
    print()
    
    success = run_comprehensive_3d_test()
    
    if success:
        print("\n✅ ef.py is ready for 3D Gaussian applications!")
    else:
        print("\n❌ ef.py needs fixes before 3D Gaussian use!")
        sys.exit(1)
