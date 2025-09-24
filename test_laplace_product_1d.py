#!/usr/bin/env python
"""
Test script for 1D LaplaceProduct distribution.
Compares generated samples with true probability density computed via numerical integration.
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy import integrate
from jax import random

# Add src to path for imports
sys.path.append('src')
from ef import LaplaceProduct
from generate_data import run_hmc

def compute_normalization_constant_1d(eta_xm1, eta_xp1, x_range=(-10, 10), n_points=1000):
    """
    Compute the normalization constant for 1D LaplaceProduct via numerical integration.
    
    Args:
        eta_xm1: Natural parameter for -|x-1| term
        eta_xp1: Natural parameter for -|x+1| term  
        x_range: Range for numerical integration
        n_points: Number of points for integration
        
    Returns:
        Normalization constant Z
    """
    x_vals = np.linspace(x_range[0], x_range[1], n_points)
    
    # Compute unnormalized log probability
    log_prob_vals = eta_xm1 * (-np.abs(x_vals - 1)) + eta_xp1 * (-np.abs(x_vals + 1))
    
    # Convert to probability and integrate
    prob_vals = np.exp(log_prob_vals)
    Z = integrate.trapezoid(prob_vals, x_vals)
    
    return Z

def true_density_1d(x, eta_xm1, eta_xp1, Z):
    """
    Compute true probability density for 1D LaplaceProduct.
    
    Args:
        x: Point(s) to evaluate density at
        eta_xm1: Natural parameter for -|x-1| term
        eta_xp1: Natural parameter for -|x+1| term
        Z: Normalization constant
        
    Returns:
        Probability density at x
    """
    log_prob = eta_xm1 * (-np.abs(x - 1)) + eta_xp1 * (-np.abs(x + 1))
    return np.exp(log_prob) / Z

def generate_samples_1d(eta_xm1, eta_xp1, num_samples=1000, key=None):
    """
    Generate samples from 1D LaplaceProduct using HMC.
    
    Args:
        eta_xm1: Natural parameter for -|x-1| term
        eta_xp1: Natural parameter for -|x+1| term
        num_samples: Number of samples to generate
        key: JAX random key
        
    Returns:
        Array of samples
    """
    if key is None:
        key = random.PRNGKey(42)
    
    # Create LaplaceProduct distribution
    ef = LaplaceProduct(x_shape=(1,))
    
    # Create eta dictionary
    eta = {"xm1": jnp.array([eta_xm1]), "xp1": jnp.array([eta_xp1])}
    
    # Create log density function
    logp = ef.make_logdensity_fn(eta)
    
    # Generate samples using HMC
    samples = run_hmc(
        logp,
        num_samples=num_samples,
        num_warmup=500,
        step_size=0.1,
        num_integration_steps=10,
        initial_position=jnp.array([0.0]),
        seed=key,
    )
    
    return samples.flatten()

def test_laplace_product_1d():
    """Test 1D LaplaceProduct distribution with various eta parameters."""
    
    # Test cases: (eta_xm1, eta_xp1, description)
    test_cases = [
        (1.0, 1.0, "Equal parameters"),
        (2.0, 1.0, "Asymmetric: stronger at x=1"),
        (1.0, 2.0, "Asymmetric: stronger at x=-1"),
        (0.5, 0.5, "Weak parameters"),
        (3.0, 1.5, "Strong parameters"),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (eta_xm1, eta_xp1, description) in enumerate(test_cases):
        print(f"\nTesting case {i+1}: {description}")
        print(f"eta_xm1 = {eta_xm1}, eta_xp1 = {eta_xp1}")
        
        # Compute normalization constant
        Z = compute_normalization_constant_1d(eta_xm1, eta_xp1)
        print(f"Normalization constant Z = {Z:.6f}")
        
        # Generate samples
        key = random.PRNGKey(42 + i)
        samples = generate_samples_1d(eta_xm1, eta_xp1, num_samples=1000, key=key)
        print(f"Generated {len(samples)} samples")
        print(f"Sample range: [{samples.min():.3f}, {samples.max():.3f}]")
        print(f"Sample mean: {samples.mean():.3f}, std: {samples.std():.3f}")
        
        # Create histogram
        ax = axes[i]
        x_range = (-4, 4)
        x_vals = np.linspace(x_range[0], x_range[1], 200)
        true_dens = true_density_1d(x_vals, eta_xm1, eta_xp1, Z)
        
        # Plot true density
        ax.plot(x_vals, true_dens, 'r-', linewidth=2, label='True density')
        
        # Plot histogram of samples
        ax.hist(samples, bins=50, density=True, alpha=0.7, 
                label='Generated samples', color='blue', edgecolor='black')
        
        ax.set_xlim(x_range)
        ax.set_xlabel('x')
        ax.set_ylabel('Density')
        ax.set_title(f'{description}\nη₁={eta_xm1}, η₂={eta_xp1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Compute and print some statistics
        sample_mean = samples.mean()
        sample_var = samples.var()
        print(f"Sample statistics: mean={sample_mean:.3f}, var={sample_var:.3f}")
        
        # Check if samples are reasonable (not all NaN or infinite)
        if np.any(np.isnan(samples)) or np.any(np.isinf(samples)):
            print("WARNING: Generated samples contain NaN or infinite values!")
        else:
            print("✓ Samples are finite")
    
    # Remove unused subplot
    if len(test_cases) < len(axes):
        axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig('laplace_product_1d_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Test completed! Plot saved as 'laplace_product_1d_test.png'")

def test_normalization_accuracy():
    """Test the accuracy of our normalization constant computation."""
    print("\n" + "="*50)
    print("TESTING NORMALIZATION ACCURACY")
    print("="*50)
    
    # Test with known case: if eta_xm1 = eta_xp1 = 1, we should get a symmetric distribution
    eta_xm1, eta_xp1 = 1.0, 1.0
    
    # Compute normalization constant
    Z = compute_normalization_constant_1d(eta_xm1, eta_xp1)
    print(f"Computed normalization constant: {Z:.8f}")
    
    # For symmetric case, we can compute analytically
    # The distribution is proportional to exp(-|x-1| - |x+1|)
    # This has peaks at x = -1 and x = 1, with exponential decay
    
    # Test integration over different ranges
    ranges = [(-3, 3), (-4, 4), (-5, 5), (-6, 6)]
    for x_range in ranges:
        Z_test = compute_normalization_constant_1d(eta_xm1, eta_xp1, x_range, n_points=2000)
        print(f"Integration over {x_range}: Z = {Z_test:.8f}")
    
    print("✓ Normalization test completed")

if __name__ == "__main__":
    print("Testing 1D LaplaceProduct Distribution")
    print("="*50)
    
    # Test normalization accuracy first
    test_normalization_accuracy()
    
    # Run main test
    test_laplace_product_1d()
