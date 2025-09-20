"""
Methods for computing normalization factors from samples of unnormalized distributions.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple, Union
import jax
import jax.numpy as jnp
from jax import random
import jax.scipy.stats as stats

Array = jax.Array


def importance_sampling_normalization(
    samples: Array,
    log_unnormalized_fn: Callable[[Array], Array],
    log_proposal_fn: Callable[[Array], Array],
) -> Tuple[float, float]:
    """
    Estimate normalization factor using importance sampling.
    
    Args:
        samples: Samples from proposal distribution q, shape (N, d)
        log_unnormalized_fn: Function computing log p*(x)
        log_proposal_fn: Function computing log q(x)
    
    Returns:
        (Z_estimate, standard_error)
    """
    log_p_star = log_unnormalized_fn(samples)
    log_q = log_proposal_fn(samples)
    
    # Compute importance weights in log space for numerical stability
    log_weights = log_p_star - log_q
    
    # Use log-sum-exp trick for numerical stability
    max_log_weight = jnp.max(log_weights)
    weights = jnp.exp(log_weights - max_log_weight)
    
    # Estimate: Z = (1/N) * exp(max_log_weight) * sum(weights)
    Z_estimate = jnp.mean(weights) * jnp.exp(max_log_weight)
    
    # Compute standard error
    var_weights = jnp.var(weights)
    se = jnp.sqrt(var_weights / len(samples)) * jnp.exp(max_log_weight)
    
    return float(Z_estimate), float(se)


def self_normalized_importance_sampling(
    samples: Array,
    log_unnormalized_fn: Callable[[Array], Array],
) -> Array:
    """
    Self-normalized importance sampling when you only have samples from p*.
    
    This doesn't give you Z directly, but gives you normalized probabilities.
    
    Args:
        samples: Samples from the unnormalized distribution, shape (N, d)
        log_unnormalized_fn: Function computing log p*(x)
    
    Returns:
        Normalized probability estimates for each sample
    """
    log_p_star = log_unnormalized_fn(samples)
    
    # Self-normalize using log-sum-exp
    log_Z_approx = jax.scipy.special.logsumexp(log_p_star) - jnp.log(len(samples))
    log_probs = log_p_star - log_Z_approx
    
    return jnp.exp(log_probs)


def annealed_importance_sampling(
    log_unnormalized_fn: Callable[[Array], Array],
    log_proposal_fn: Callable[[Array], Array],
    proposal_sampler: Callable[[Array, int], Array],
    num_samples: int,
    num_temps: int = 10,
    key: Optional[Array] = None,
) -> Tuple[float, float]:
    """
    Annealed Importance Sampling for complex distributions.
    
    Uses geometric bridging: p_β(x) ∝ q(x)^(1-β) * p*(x)^β
    
    Args:
        log_unnormalized_fn: Function computing log p*(x)
        log_proposal_fn: Function computing log q(x) 
        proposal_sampler: Function to sample from q(x)
        num_samples: Number of samples to use
        num_temps: Number of temperature levels
        key: JAX random key
    
    Returns:
        (Z_estimate, standard_error)
    """
    if key is None:
        key = random.PRNGKey(0)
    
    # Temperature schedule: β = 0 (proposal) to β = 1 (target)
    betas = jnp.linspace(0.0, 1.0, num_temps)
    
    # Sample from proposal
    samples = proposal_sampler(key, num_samples)
    
    # Initialize log weights
    log_weights = jnp.zeros(num_samples)
    
    # Sequential importance sampling across temperatures
    for i in range(1, num_temps):
        beta_prev, beta_curr = betas[i-1], betas[i]
        
        # Compute log weight increments
        log_p_star = log_unnormalized_fn(samples)
        log_q = log_proposal_fn(samples)
        
        # Weight increment: (p*/q)^(β_curr - β_prev)
        log_weight_increment = (beta_curr - beta_prev) * (log_p_star - log_q)
        log_weights += log_weight_increment
        
        # Optional: Resample to avoid degeneracy (not implemented here)
    
    # Final estimate
    max_log_weight = jnp.max(log_weights)
    weights = jnp.exp(log_weights - max_log_weight)
    
    Z_estimate = jnp.mean(weights) * jnp.exp(max_log_weight)
    se = jnp.sqrt(jnp.var(weights) / num_samples) * jnp.exp(max_log_weight)
    
    return float(Z_estimate), float(se)


def bridge_sampling_normalization(
    samples_p: Array,
    samples_q: Array,
    log_unnormalized_fn: Callable[[Array], Array],
    log_proposal_fn: Callable[[Array], Array],
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[float, int]:
    """
    Bridge sampling estimator for normalization constant.
    
    Args:
        samples_p: Samples from unnormalized target p*, shape (N₁, d)
        samples_q: Samples from proposal q, shape (N₂, d)
        log_unnormalized_fn: Function computing log p*(x)
        log_proposal_fn: Function computing log q(x)
        max_iter: Maximum iterations for iterative solution
        tol: Convergence tolerance
    
    Returns:
        (Z_estimate, num_iterations)
    """
    N1, N2 = len(samples_p), len(samples_q)
    
    # Evaluate log densities
    log_p_at_p = log_unnormalized_fn(samples_p)
    log_q_at_p = log_proposal_fn(samples_p)
    log_p_at_q = log_unnormalized_fn(samples_q)
    log_q_at_q = log_proposal_fn(samples_q)
    
    # Initialize estimate
    log_r = 0.0  # log(Z_p / Z_q), where Z_q is known (often = 1)
    
    for iteration in range(max_iter):
        log_r_old = log_r
        
        # Bridge function weights
        log_w_p = log_q_at_p - jnp.logaddexp(log_p_at_p - log_r, log_q_at_p)
        log_w_q = log_p_at_q - log_r - jnp.logaddexp(log_p_at_q - log_r, log_q_at_q)
        
        # Update estimate
        log_r = (jax.scipy.special.logsumexp(log_w_q) - jnp.log(N2)) - \
                (jax.scipy.special.logsumexp(log_w_p) - jnp.log(N1))
        
        # Check convergence
        if jnp.abs(log_r - log_r_old) < tol:
            break
    
    return float(jnp.exp(log_r)), iteration + 1


def gaussian_proposal_normalization(
    samples: Array,
    log_unnormalized_fn: Callable[[Array], Array],
    proposal_mean: Optional[Array] = None,
    proposal_cov: Optional[Array] = None,
) -> Tuple[float, float]:
    """
    Convenience function for importance sampling with Gaussian proposal.
    
    Args:
        samples: Samples from Gaussian proposal, shape (N, d)
        log_unnormalized_fn: Function computing log p*(x)
        proposal_mean: Mean of Gaussian proposal (default: sample mean)
        proposal_cov: Covariance of Gaussian proposal (default: sample covariance)
    
    Returns:
        (Z_estimate, standard_error)
    """
    if proposal_mean is None:
        proposal_mean = jnp.mean(samples, axis=0)
    if proposal_cov is None:
        proposal_cov = jnp.cov(samples.T)
    
    def log_gaussian_proposal(x: Array) -> Array:
        return stats.multivariate_normal.logpdf(x, proposal_mean, proposal_cov)
    
    return importance_sampling_normalization(
        samples, log_unnormalized_fn, log_gaussian_proposal
    )


# Example usage with your exponential family framework
def ef_normalization_example():
    """Example of computing normalization for an exponential family distribution."""
    from ef import MultivariateNormal
    
    # Create a multivariate normal distribution
    ef_dist = MultivariateNormal(x_shape=(2,))
    
    # Natural parameters (must ensure integrability)
    eta = {"x": jnp.array([1.0, -0.5]), "xxT": -0.5 * jnp.eye(2)}
    
    # Create log unnormalized density function
    log_unnormalized_fn = ef_dist.make_logdensity_fn(eta)
    
    # Generate samples from a Gaussian proposal
    key = random.PRNGKey(42)
    samples = random.multivariate_normal(key, jnp.zeros(2), jnp.eye(2), shape=(1000,))
    
    # Estimate normalization factor
    Z_est, se = gaussian_proposal_normalization(
        samples, 
        lambda x: log_unnormalized_fn(x.flatten()),  # Handle flattening
    )
    
    print(f"Estimated normalization factor: {Z_est:.4f} ± {se:.4f}")
    return Z_est, se


if __name__ == "__main__":
    ef_normalization_example()
