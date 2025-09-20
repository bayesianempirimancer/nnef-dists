"""
Exact analytical covariance computation for exponential families.

This module provides fast analytical computation of Σ_TT(η) for known
exponential families, avoiding expensive Hessian computations.

For multivariate Gaussian with sufficient statistics T(x) = [x, x⊗x],
we compute the exact covariance matrix Cov[T_i, T_j] using moment formulas.
"""

import jax.numpy as jnp
from typing import Dict, Any
from ..ef import ExponentialFamily, MultivariateNormal


def compute_exact_covariance_matrix(mu: jnp.ndarray, Sigma: jnp.ndarray) -> jnp.ndarray:
    """
    Compute exact covariance matrix for multivariate Gaussian sufficient statistics.
    
    For X ~ N(μ, Σ), T = [X, X⊗X], compute Cov[T_i, T_j] exactly using
    multivariate Gaussian moment formulas.
    
    Args:
        mu: Mean vector [d,]
        Sigma: Covariance matrix [d, d]
        
    Returns:
        Covariance matrix of T [d + d², d + d²]
    """
    d = mu.shape[0]
    T_dim = d + d * d  # Linear + quadratic terms
    
    cov_matrix = jnp.zeros((T_dim, T_dim))
    
    # Helper function to get T index
    def get_T_index(component_type, i, j=None):
        """Get index in T vector for component."""
        if component_type == 'linear':
            return i
        elif component_type == 'quadratic':
            return d + i * d + j
        else:
            raise ValueError("component_type must be 'linear' or 'quadratic'")
    
    # Block 1: Cov[X_i, X_j] = Σ_ij
    for i in range(d):
        for j in range(d):
            idx_i = get_T_index('linear', i)
            idx_j = get_T_index('linear', j)
            cov_matrix = cov_matrix.at[idx_i, idx_j].set(Sigma[i, j])
    
    # Block 2: Cov[X_i, X_j * X_k]
    # Using: Cov[X_i, X_j * X_k] = μ_j * Σ_ik + μ_k * Σ_ij
    for i in range(d):
        for j in range(d):
            for k in range(d):
                idx_i = get_T_index('linear', i)
                idx_jk = get_T_index('quadratic', j, k)
                
                cov_val = mu[j] * Sigma[i, k] + mu[k] * Sigma[i, j]
                
                cov_matrix = cov_matrix.at[idx_i, idx_jk].set(cov_val)
                cov_matrix = cov_matrix.at[idx_jk, idx_i].set(cov_val)  # Symmetry
    
    # Block 3: Cov[X_i * X_j, X_k * X_l]
    # Using exact multivariate Gaussian moment formula:
    # Cov[X_i * X_j, X_k * X_l] = 
    #   Σ_ik * Σ_jl + Σ_il * Σ_jk + 
    #   μ_i * μ_k * Σ_jl + μ_i * μ_l * Σ_jk + μ_j * μ_k * Σ_il + μ_j * μ_l * Σ_ik
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    idx_ij = get_T_index('quadratic', i, j)
                    idx_kl = get_T_index('quadratic', k, l)
                    
                    cov_val = (
                        Sigma[i, k] * Sigma[j, l] + Sigma[i, l] * Sigma[j, k] +
                        mu[i] * mu[k] * Sigma[j, l] + mu[i] * mu[l] * Sigma[j, k] +
                        mu[j] * mu[k] * Sigma[i, l] + mu[j] * mu[l] * Sigma[i, k]
                    )
                    
                    cov_matrix = cov_matrix.at[idx_ij, idx_kl].set(cov_val)
    
    return cov_matrix


def compute_covariance_for_eta(eta_dict: Dict[str, jnp.ndarray], 
                              ef: MultivariateNormal) -> jnp.ndarray:
    """
    Compute exact covariance matrix from natural parameters.
    
    Args:
        eta_dict: Natural parameters {'x': η₁, 'xxT': η₂}
        ef: Multivariate normal exponential family instance
        
    Returns:
        Covariance matrix Σ_TT(η)
    """
    eta1 = eta_dict['x']  # [d,]
    eta2 = eta_dict['xxT']  # [d, d]
    
    # Convert to Gaussian parameters
    Sigma = -0.5 * jnp.linalg.inv(eta2)
    mu = jnp.linalg.solve(eta2, -0.5 * eta1)
    
    # Compute exact covariance
    return compute_exact_covariance_matrix(mu, Sigma)


def verify_covariance_properties(cov_matrix: jnp.ndarray, 
                                tolerance: float = 1e-10) -> Dict[str, bool]:
    """
    Verify mathematical properties of the covariance matrix.
    
    Args:
        cov_matrix: Covariance matrix to verify
        tolerance: Numerical tolerance for checks
        
    Returns:
        Dictionary with verification results
    """
    # Check positive semidefiniteness
    eigenvals = jnp.linalg.eigvals(cov_matrix)
    min_eigenval = jnp.min(jnp.real(eigenvals))  # Real part for numerical stability
    is_psd = min_eigenval >= -tolerance
    
    # Check symmetry
    symmetry_error = jnp.max(jnp.abs(cov_matrix - cov_matrix.T))
    is_symmetric = symmetry_error < tolerance
    
    # Check for NaN/Inf
    is_finite = jnp.all(jnp.isfinite(cov_matrix))
    
    return {
        'positive_semidefinite': is_psd,
        'symmetric': is_symmetric,
        'finite': is_finite,
        'min_eigenvalue': float(min_eigenval),
        'symmetry_error': float(symmetry_error),
        'condition_number': float(jnp.linalg.cond(cov_matrix))
    }


def supports_exact_covariance(ef: ExponentialFamily) -> bool:
    """Check if exact covariance computation is available for this exponential family."""
    return isinstance(ef, MultivariateNormal)
