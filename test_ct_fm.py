#!/usr/bin/env python3
"""
Test script for NoProp CT and FM models.
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

# Import the models
from src.models.noprop.ct import NoPropCT
from src.models.noprop.fm import NoPropFM

def test_ct_model():
    """Test the NoProp CT model."""
    print("Testing NoProp CT model...")
    
    # Create a simple MLP model
    class SimpleMLP(nn.Module):
        hidden_dim: int = 64
        output_dim: int = 9
        
        @nn.compact
        def __call__(self, x, t, training=True):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.swish(x)
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.swish(x)
            x = nn.Dense(self.output_dim)(x)
            return x
    
    # Create CT model
    mlp = SimpleMLP()
    ct_model = NoPropCT(
        z_shape=(9,),
        x_shape=(9,),
        model=mlp,
        num_timesteps=10
    )
    
    # Test initialization
    key = jr.PRNGKey(42)
    z_sample = jnp.ones((2, 9))
    x_sample = jnp.ones((2, 9))
    t_sample = jnp.array([0.5, 0.7])
    
    params = ct_model.init(key, z_sample, x_sample, t_sample)
    print(f"CT model initialized with {sum(x.size for x in jax.tree_leaves(params)):,} parameters")
    
    # Test forward pass
    output = ct_model.apply(params, z_sample, x_sample, t_sample)
    print(f"CT output shape: {output.shape}")
    
    print("✅ CT model test completed successfully!")

def test_fm_model():
    """Test the NoProp FM model."""
    print("\nTesting NoProp FM model...")
    
    # Create a simple MLP model
    class SimpleMLP(nn.Module):
        hidden_dim: int = 64
        output_dim: int = 9
        
        @nn.compact
        def __call__(self, x, t, training=True):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.swish(x)
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.swish(x)
            x = nn.Dense(self.output_dim)(x)
            return x
    
    # Create FM model
    mlp = SimpleMLP()
    
    fm_model = NoPropFM(
        z_shape=(9,),
        x_shape=(9,),
        model=mlp,
        noise_schedule=None,
        num_timesteps=10
    )
    
    # Test initialization
    key = jr.PRNGKey(42)
    z_sample = jnp.ones((2, 9))
    x_sample = jnp.ones((2, 9))
    t_sample = jnp.array([0.5, 0.7])
    
    params = fm_model.init(key, z_sample, x_sample, t_sample)
    print(f"FM model initialized with {sum(x.size for x in jax.tree_leaves(params)):,} parameters")
    
    # Test forward pass
    output = fm_model.apply(params, z_sample, x_sample, t_sample)
    print(f"FM output shape: {output.shape}")
    
    print("✅ FM model test completed successfully!")

if __name__ == "__main__":
    test_ct_model()
    test_fm_model()
    print("\n🎉 All tests completed successfully!")
