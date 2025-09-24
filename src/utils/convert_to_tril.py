#!/usr/bin/env python3
"""
Convert full multivariate normal data to triangular format.

This script converts data files from the full 12D format (eta, mu_T, cov_TT) 
to the minimal 9D triangular format (eta_tril, mu_T_tril, cov_TT_tril) for 3D Gaussian.

The triangular format uses only the lower triangular part of the covariance matrix,
reducing the parameter count from N + N*N = 3 + 9 = 12 to N + N*(N+1)/2 = 3 + 6 = 9.
"""

import argparse
import pickle
import numpy as np
import jax.numpy as jnp
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.ef import MultivariateNormal_tril

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
        n_dim: Dimension of the multivariate normal (default: 3)
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

def main():
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
