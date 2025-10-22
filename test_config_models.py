#!/usr/bin/env python3
"""
Test script for NoProp CT and FM models with config pattern.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn

# Import the new models
from src.models.noprop.ct_new import Config as CTConfig, NoPropCT
from src.models.noprop.fm_new import Config as FMConfig, NoPropFM

def test_ct_model():
    """Test the NoProp CT model with config."""
    print("Testing NoProp CT model with config...")
    
    # Create config
    config = CTConfig(
        input_dim=9,
        output_dim=9,
        num_timesteps=10
    )
    
    # Create CT model
    ct_model = NoPropCT(config=config)
    
    # Test initialization
    key = jr.PRNGKey(42)
    eta_sample = jnp.ones((2, 9))
    
    params = ct_model.init(key, eta_sample)
    print(f"CT model initialized with {sum(x.size for x in jax.tree_leaves(params)):,} parameters")
    
    # Test forward pass
    predictions, internal_loss = ct_model.apply(params, eta_sample)
    print(f"CT predictions shape: {predictions.shape}")
    print(f"CT internal loss: {internal_loss}")
    
    print("✅ CT model test completed successfully!")

def test_fm_model():
    """Test the NoProp FM model with config."""
    print("\nTesting NoProp FM model with config...")
    
    # Create config
    config = FMConfig(
        input_dim=9,
        output_dim=9,
        num_timesteps=10
    )
    
    # Create FM model
    fm_model = NoPropFM(config=config)
    
    # Test initialization
    key = jr.PRNGKey(42)
    eta_sample = jnp.ones((2, 9))
    
    params = fm_model.init(key, eta_sample)
    print(f"FM model initialized with {sum(x.size for x in jax.tree_leaves(params)):,} parameters")
    
    # Test forward pass
    predictions, internal_loss = fm_model.apply(params, eta_sample)
    print(f"FM predictions shape: {predictions.shape}")
    print(f"FM internal loss: {internal_loss}")
    
    print("✅ FM model test completed successfully!")

if __name__ == "__main__":
    test_ct_model()
    test_fm_model()
    print("\n🎉 All config model tests completed successfully!")
