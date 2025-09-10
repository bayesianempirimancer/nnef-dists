#!/usr/bin/env python
"""Simple test script to verify compatibility without full training."""

import sys
from pathlib import Path

# Add src to path so we can import without installing
sys.path.insert(0, str(Path(__file__).parent / "src"))

import jax.numpy as jnp
from nnef_dist.ef import ef_factory, GaussianNatural1D, MultivariateNormal
from nnef_dist.sampling import run_hmc
from nnef_dist.train import build_dataset

def test_basic_functionality():
    print("Testing basic EF functionality...")
    
    # Test GaussianNatural1D
    gauss = GaussianNatural1D()
    print(f"✓ GaussianNatural1D: x_shape={gauss.x_shape}, eta_dim={gauss.eta_dim}")
    
    # Test MultivariateNormal
    mv = MultivariateNormal(x_shape=(2,))
    print(f"✓ MultivariateNormal: x_shape={mv.x_shape}, eta_dim={mv.eta_dim}")
    
    # Test factory
    ef1 = ef_factory('gaussian_1d')
    ef2 = ef_factory('mv_normal', x_shape=(3,))
    print(f"✓ Factory: {ef1.__class__.__name__}, {ef2.__class__.__name__}")
    
    return True

def test_sampling():
    print("\nTesting sampling...")
    
    ef = GaussianNatural1D()
    eta = jnp.array([1.0, -0.5])
    
    # Test logdensity function
    logp_fn = ef.make_logdensity_fn(eta)
    print(f"✓ make_logdensity_fn: {callable(logp_fn)}")
    
    # Test HMC sampling (very small)
    samples = run_hmc(
        logp_fn,
        num_samples=5,
        num_warmup=3,
        step_size=0.1,
        num_integration_steps=3,
        initial_position=jnp.zeros((ef.x_dim,)),
        seed=0
    )
    print(f"✓ HMC sampling: samples shape {samples.shape}")
    
    return True

def test_dataset_building():
    print("\nTesting dataset building...")
    
    ef = GaussianNatural1D()
    
    # Very small dataset
    train_data, val_data = build_dataset(
        ef=ef,
        train_points=4,
        val_points=2,
        eta_ranges=((-1.0, 1.0), (-1.0, -0.1)),
        sampler_cfg={
            'num_samples': 10,
            'num_warmup': 5,
            'step_size': 0.1,
            'num_integration_steps': 5
        },
        seed=0
    )
    
    print(f"✓ Dataset: train {train_data['eta'].shape}, val {val_data['eta'].shape}")
    return True

if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_sampling()
        test_dataset_building()
        print("\n🎉 All tests passed! The refactored code is working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
