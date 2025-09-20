#!/usr/bin/env python3
"""
Example demonstrating different methods for computing normalization factors
from samples of unnormalized probability distributions.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

from ef import MultivariateNormal, GaussianNatural1D
from normalization import (
    importance_sampling_normalization,
    self_normalized_importance_sampling,
    gaussian_proposal_normalization,
    bridge_sampling_normalization,
)


def example_1d_gaussian():
    """Example with 1D Gaussian where we know the true normalization factor."""
    print("=== 1D Gaussian Example ===")
    
    # Create 1D Gaussian in natural parameterization
    # log p*(x) = eta1 * x + eta2 * x^2
    # For eta = [μ/σ², -1/(2σ²)], this is N(μ, σ²)
    mu, sigma = 2.0, 1.5
    eta1 = mu / (sigma**2)
    eta2 = -1.0 / (2 * sigma**2)
    
    print(f"True parameters: μ={mu}, σ={sigma}")
    print(f"Natural parameters: η₁={eta1:.4f}, η₂={eta2:.4f}")
    
    # True normalization factor for this Gaussian
    true_Z = jnp.sqrt(2 * jnp.pi * sigma**2)
    print(f"True normalization factor: {true_Z:.4f}")
    
    # Create exponential family distribution
    ef_dist = GaussianNatural1D()
    eta_dict = {"x": jnp.array([eta1]), "x2": jnp.array([eta2])}
    log_unnormalized_fn = lambda x: ef_dist.log_unnormalized(x.reshape(-1, 1), eta_dict).squeeze()
    
    # Method 1: Importance sampling with standard normal proposal
    key = random.PRNGKey(42)
    proposal_samples = random.normal(key, shape=(5000,))
    
    def log_standard_normal(x):
        return -0.5 * x**2 - 0.5 * jnp.log(2 * jnp.pi)
    
    Z_est1, se1 = importance_sampling_normalization(
        proposal_samples.reshape(-1, 1),
        log_unnormalized_fn,
        lambda x: log_standard_normal(x.squeeze())
    )
    
    print(f"Importance sampling: {Z_est1:.4f} ± {se1:.4f}")
    print(f"Relative error: {abs(Z_est1 - true_Z) / true_Z * 100:.2f}%")
    
    # Method 2: Bridge sampling
    key1, key2 = random.split(key)
    target_samples = random.normal(key1, shape=(2000,)) * sigma + mu  # Samples from true distribution
    proposal_samples = random.normal(key2, shape=(2000,))  # Samples from standard normal
    
    Z_est2, iterations = bridge_sampling_normalization(
        target_samples.reshape(-1, 1),
        proposal_samples.reshape(-1, 1),
        log_unnormalized_fn,
        lambda x: log_standard_normal(x.squeeze())
    )
    
    print(f"Bridge sampling: {Z_est2:.4f} (converged in {iterations} iterations)")
    print(f"Relative error: {abs(Z_est2 - true_Z) / true_Z * 100:.2f}%")


def example_2d_gaussian():
    """Example with 2D Gaussian."""
    print("\n=== 2D Gaussian Example ===")
    
    # Create 2D Gaussian with known parameters
    true_mean = jnp.array([1.0, -0.5])
    true_cov = jnp.array([[2.0, 0.5], [0.5, 1.0]])
    
    # Convert to natural parameters: η₁ = Σ⁻¹μ, η₂ = -½Σ⁻¹
    true_prec = jnp.linalg.inv(true_cov)
    eta_x = true_prec @ true_mean
    eta_xxT = -0.5 * true_prec
    
    print(f"True mean: {true_mean}")
    print(f"True covariance:\n{true_cov}")
    
    # True normalization factor
    true_Z = jnp.sqrt((2 * jnp.pi)**2 * jnp.linalg.det(true_cov))
    print(f"True normalization factor: {true_Z:.4f}")
    
    # Create exponential family distribution
    ef_dist = MultivariateNormal(x_shape=(2,))
    eta_dict = {"x": eta_x, "xxT": eta_xxT}
    
    def log_unnormalized_fn(x):
        # Handle both single samples and batches
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return jnp.array([ef_dist.log_unnormalized(xi, eta_dict) for xi in x])
    
    # Method 1: Importance sampling with isotropic Gaussian proposal
    key = random.PRNGKey(123)
    proposal_samples = random.multivariate_normal(key, jnp.zeros(2), jnp.eye(2), shape=(3000,))
    
    Z_est, se = gaussian_proposal_normalization(
        proposal_samples,
        log_unnormalized_fn
    )
    
    print(f"Gaussian proposal IS: {Z_est:.4f} ± {se:.4f}")
    print(f"Relative error: {abs(Z_est - true_Z) / true_Z * 100:.2f}%")


def demonstrate_self_normalization():
    """Demonstrate self-normalized importance sampling."""
    print("\n=== Self-Normalized Importance Sampling ===")
    
    # Create a simple 1D distribution
    ef_dist = GaussianNatural1D()
    eta_dict = {"x": jnp.array([1.0]), "x2": jnp.array([-0.5])}
    log_unnormalized_fn = lambda x: ef_dist.log_unnormalized(x.reshape(-1, 1), eta_dict).squeeze()
    
    # Generate some samples (in practice, these would come from MCMC or other sampling)
    key = random.PRNGKey(456)
    samples = random.normal(key, shape=(1000,))
    
    # Get normalized probabilities
    normalized_probs = self_normalized_importance_sampling(
        samples.reshape(-1, 1),
        log_unnormalized_fn
    )
    
    print(f"Sample range: [{samples.min():.2f}, {samples.max():.2f}]")
    print(f"Normalized probability range: [{normalized_probs.min():.6f}, {normalized_probs.max():.6f}]")
    print(f"Sum of normalized probabilities: {normalized_probs.sum():.6f}")
    print("(Should be close to 1.0 for large N)")


if __name__ == "__main__":
    example_1d_gaussian()
    example_2d_gaussian() 
    demonstrate_self_normalization()
    
    print("\n" + "="*50)
    print("Summary of Methods:")
    print("1. Importance Sampling: Use when you have a good proposal distribution")
    print("2. Bridge Sampling: Use when you have samples from both target and proposal")
    print("3. Self-Normalized IS: Use when you only have samples from target")
    print("4. Annealed IS: Use for complex, multimodal distributions")
    print("="*50)
