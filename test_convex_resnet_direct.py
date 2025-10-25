#!/usr/bin/env python3
"""
Test script for the ConvexConditionalResnet directly.
"""

import sys
from pathlib import Path
import jax
import jax.numpy as jnp

# Add the project root to the path for imports
sys.path.append(str(Path(__file__).parent))

from src.models.flow_models.crn import ConvexConditionalResnet, Config


def test_convex_resnet_direct():
    """Test the ConvexConditionalResnet directly."""
    print("Testing ConvexConditionalResnet directly...")
    
    # Set up test data
    batch_size = 3
    z_dim = 2
    x_dim = 2
    
    z = jnp.ones((batch_size, z_dim))
    x = jnp.ones((batch_size, x_dim))
    t = jnp.ones((batch_size,))
    
    # Create config
    config = Config()
    
    # Test 1: Direct instantiation
    print("\n--- Test 1: Direct instantiation ---")
    try:
        resnet = ConvexConditionalResnet(**config.config_dict)
        print("✓ ConvexConditionalResnet created successfully")
    except Exception as e:
        print(f"✗ Failed to create ConvexConditionalResnet: {e}")
        return False
    
    # Test 2: Forward pass
    print("\n--- Test 2: Forward pass ---")
    try:
        rng = jax.random.PRNGKey(42)
        params = resnet.init(rng, z, x, t, training=True)
        output = resnet.apply(params, z, x, t, training=True, rngs={'dropout': rng})
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        print(f"  Expected shape: {z.shape}, Got: {output.shape}")
        
        if output.shape == z.shape:
            print("✓ Output shape matches input shape")
        else:
            print(f"✗ Output shape mismatch: expected {z.shape}, got {output.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    # Test 3: Broadcasting
    print("\n--- Test 3: Broadcasting test ---")
    try:
        # Single sample
        z_single = jnp.ones((z_dim,))
        x_single = jnp.ones((x_dim,))
        t_single = 0.5
        
        output_single = resnet.apply(params, z_single, x_single, t_single, training=True, rngs={'dropout': rng})
        print(f"✓ Single sample forward pass successful, output shape: {output_single.shape}")
        
        # Scalar t
        output_scalar_t = resnet.apply(params, z, x, 0.5, training=True, rngs={'dropout': rng})
        print(f"✓ Scalar t forward pass successful, output shape: {output_scalar_t.shape}")
        
    except Exception as e:
        print(f"✗ Broadcasting test failed: {e}")
        return False
    
    print("\n✅ All tests passed! ConvexConditionalResnet is working correctly.")
    return True


if __name__ == "__main__":
    success = test_convex_resnet_direct()
    if not success:
        sys.exit(1)

