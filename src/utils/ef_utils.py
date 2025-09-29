"""
Exponential Family Utilities.

This module provides utilities for working with exponential family distributions,
including exact covariance computation and data format conversion.
"""

import argparse
import pickle
import numpy as np
import jax.numpy as jnp
import sys
from pathlib import Path
from typing import Dict, Any

from ..ef import ExponentialFamily, MultivariateNormal, MultivariateNormal_tril


# =============================================================================
# Exact Covariance Computation for Multivariate Normal T(x) = [X, X⊗X].  
# Note that this computes the covariance of T(X) for a given value of eta.
# =============================================================================

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
    d = mu.shape[-1]
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


# =============================================================================
# Data Format Conversion
# =============================================================================

def get_tril_projection_matrix(n: int):
    """Get the matrix that converts from full format to triangular format.
    
    Args:
        n: Dimension of the multivariate normal
        
    Returns:
        conversion_matrix: Matrix of shape (n + n*(n+1)//2, n + n*n)
    """
    # Create the triangular exponential family to get indices
    mvn_tril = MultivariateNormal_tril(x_shape=(n,))
    tril_indices = mvn_tril.tril_indices
    
    dim_tril = n + n * (n + 1) // 2
    dim_full = n + n * n

    P = jnp.zeros((dim_tril, dim_full))
    for i in range(dim_tril):
        P = P.at[i, tril_indices[i]].set(1.0)
    return P


def convert_data_to_tril(input_file: Path, output_file: Path):
    """Convert data file from full format to triangular format.
    
    Args:
        input_file: Path to input data file with eta, mu_T, cov_TT
        output_file: Path to output data file with eta_tril, mu_T_tril, cov_TT_tril
    """
    print(f"Converting {input_file} to triangular format...")
    
    # Load the data
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    dim = data['val']['eta'].shape[-1]
    dim = (-1 + jnp.sqrt(1 + 4*dim))/2
    dim = int(jnp.round(dim))

    projection_matrix = get_tril_projection_matrix(dim)
    tril_dim = projection_matrix.shape[-2]
    print(f"Conversion matrix shape: {projection_matrix.shape}")

    print(f"Input data structure:")
    for split in ['train', 'val', 'test']:
        if split in data:
            print(f"  {split}:")
            for key, value in data[split].items():
                if value is not None:
                    print(f"    {key}: {value.shape}")

    # Convert each split
    tril_data = {}
    
    for split in ['train', 'val', 'test']:
        if split not in data:
            continue
            
        print(f"\nConverting {split} split...")
        
        # Convert eta and mu_T using the conversion matrix
        eta_full = jnp.array(data[split]['eta'])
        mu_T_full = jnp.array(data[split]['mu_T'])
        
        # Apply conversion: [n_samples, tril_dim] = [n_samples, full_dim] @ [full_dim, tril_dim].T
        eta_tril = eta_full @ projection_matrix.T
        mu_T_tril = mu_T_full @ projection_matrix.T
        
        # Convert cov_TT using the conversion matrix
        cov_TT_full = jnp.array(data[split]['cov_TT'])  # Shape: [n_samples, full_dim, full_dim]
        
        cov_TT_tril = projection_matrix @ cov_TT_full @ projection_matrix.mT
        
        tril_data[split] = {
            'eta': np.array(eta_tril),
            'mu_T': np.array(mu_T_tril),
            'cov_TT': np.array(cov_TT_tril)
        }
        
        print(f"  eta: {eta_full.shape} -> {eta_tril.shape}")
        print(f"  mu_T: {mu_T_full.shape} -> {mu_T_tril.shape}")
        print(f"  cov_TT: {cov_TT_full.shape} -> {cov_TT_tril.shape}")
    
    # Copy metadata and update dimensions
    if 'metadata' in data:
        metadata = data['metadata'].copy()
        metadata['eta_dim'] = tril_dim
        metadata['mu_T_dim'] = metadata['eta_dim']
        metadata['cov_TT_shape'] = [metadata['eta_dim'], metadata['eta_dim']]
        metadata['exponential_family'] = metadata['exponential_family'].replace('multivariate_normal', 'multivariate_normal_tril')
        
        # Update ef_distribution_name to indicate triangular format
        if 'ef_distribution_name' in metadata:
            metadata['ef_distribution_name'] = 'multivariate_normal_tril'
        
        # x_shape and x_dim remain the same since we're still modeling the same x
        # Only the parameterization (eta) changes from full to triangular
        
        tril_data['metadata'] = metadata
    
    # Save converted data
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(tril_data, f)
    
    print(f"\n✅ Conversion complete!")
    print(f"📁 Output saved to: {output_file}")
    
    # Verify the output
    print(f"\nVerifying output data structure:")
    for split in ['train', 'val', 'test']:
        if split in tril_data:
            print(f"  {split}:")
            for key, value in tril_data[split].items():
                if value is not None:
                    print(f"    {key}: {value.shape}")
    
    print(f"\nDimension verification:")
    eta_dim = tril_data['train']['eta'].shape[-1]
    mu_T_dim = tril_data['train']['mu_T'].shape[-1]
    cov_TT_dims = tril_data['train']['cov_TT'].shape[-2:]
    
    print(f"eta.shape[-1] = {eta_dim}")
    print(f"mu_T.shape[-1] = {mu_T_dim}")
    print(f"cov_TT.shape[-2:] = {cov_TT_dims}")
    print(f"✅ eta.shape[-1] == mu_T.shape[-1]: {eta_dim == mu_T_dim}")
    print(f"✅ cov_TT.shape[-2:] == ({eta_dim}, {eta_dim}): {cov_TT_dims == (eta_dim, eta_dim)}")
    
    # Verify with exponential family
    ef_tril = MultivariateNormal_tril(x_shape=(dim,))
    expected_dim = ef_tril.eta_dim
    print(f"✅ Expected eta_dim from EF: {expected_dim}")
    print(f"✅ Matches expected dimension: {eta_dim == expected_dim}")


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Command line interface for data conversion."""
    parser = argparse.ArgumentParser(description='Convert full multivariate normal data to triangular format')
    parser.add_argument('input_file', type=str, help='Input data file path')
    parser.add_argument('--output', type=str, help='Output data file path (default: input_file with _tril suffix)')
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if args.output:
        output_file = Path(args.output)
    else:
        # Add _tril suffix before .pkl extension
        output_file = input_file.parent / f"{input_file.stem}_tril{input_file.suffix}"
    
    convert_data_to_tril(input_file, output_file)


if __name__ == "__main__":
    main()
