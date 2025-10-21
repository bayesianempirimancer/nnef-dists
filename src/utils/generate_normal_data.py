#!/usr/bin/env python3
"""
Generate a dataset for model comparison.

"""

import os
import sys
import pickle
import numpy as np
import jax.numpy as jnp
from jax import random
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.expfam.ef import ef_factory

def identify_extreme_points(eta: jnp.ndarray, mu_T: jnp.ndarray, num_extreme_per_dim: int = 2) -> jnp.ndarray:
    """
    Identify the most extreme points for each dimension of eta and mu_T.
    
    Args:
        eta: Natural parameters array of shape (N, eta_dim)
        mu_T: Expectation parameters array of shape (N, mu_T_dim)
        num_extreme_per_dim: Number of extreme points to identify per dimension (default: 2 for min/max)
        
    Returns:
        Boolean array of shape (N,) indicating which points are extreme
    """
    N, eta_dim = eta.shape
    _, mu_T_dim = mu_T.shape
    
    # Track extreme points
    extreme_mask = jnp.zeros(N, dtype=bool)
    
    # Find extreme points for eta dimensions
    for i in range(eta_dim):
        # Find indices of min and max values for this dimension
        min_idx = jnp.argmin(eta[:, i])
        max_idx = jnp.argmax(eta[:, i])
        extreme_mask = extreme_mask.at[min_idx].set(True)
        extreme_mask = extreme_mask.at[max_idx].set(True)
    
    # Find extreme points for mu_T dimensions
    for i in range(mu_T_dim):
        # Find indices of min and max values for this dimension
        min_idx = jnp.argmin(mu_T[:, i])
        max_idx = jnp.argmax(mu_T[:, i])
        extreme_mask = extreme_mask.at[min_idx].set(True)
        extreme_mask = extreme_mask.at[max_idx].set(True)
    
    return extreme_mask

def generate_gaussian_data(dim: int = 3, n_train: int = 1000, n_val: int = 250, 
                                         n_test: int = 250, seed: int = 42, difficulty = 'Easy', compute_cov_tt=False, ensure_extreme_in_train=True, tril_format=False):
    """
    Generate easy {dim}D Gaussian data using vectorized batch operations.
    
    Creates three types of challenging cases:
    1. Wide eigenvalue spectrum
    2. Block correlation structure  
    3. Low SNR (dominant eigenvector aligned with mean)
    
    Args:
        n_train: Number of training samples
        n_val: Number of validation samples  
        n_test: Number of test samples
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with train/val/test splits
    """
    print(f"Generating {difficulty} {dim}D Gaussian dataset (vectorized)...")
    print(f"  Training: {n_train} samples")
    print(f"  Validation: {n_val} samples")
    print(f"  Test: {n_test} samples")
    print(f"  Random seed: {seed}")
    
    if difficulty == 'Easy':
        eval_range = (0.2, 2.0)
        corr_range = (-0.2, 0.2)
        snr_min = 1.0
    elif difficulty == 'Medium':
        eval_range = (0.05, 50.0)
        corr_range = (-0.4, 0.4)
        snr_min = 0.5
    elif difficulty == 'Hard':
        eval_range = (0.01, 100.0)
        corr_range = (-0.1, 0.1)
        snr_min = 0.1

    # Create exponential family (always use regular format for generation, convert to tril later if needed)
    ef = ef_factory("multivariate_normal", x_shape=(dim,))
    
    total_samples = n_train + n_val + n_test
    rng = random.PRNGKey(seed)
    
    # Split samples into three equal batches for different case types
    n_per_type = total_samples // 3
    n_remainder = total_samples % 3
    
    
    # Batch 1: Wide eigenvalue spectrum cases
    print("  Generating eigenvalue spectrum cases...")
    rng, subkey = random.split(rng)
    batch1_eta, batch1_mu = generate_wide_eigenvalue_batch(subkey, n_per_type, dim, eval_range=eval_range, ef=ef)
    eta_all = batch1_eta
    mu_all = batch1_mu
    
    # Batch 2: Block correlation structure cases
    print("  Generating block correlation cases...")
    rng, subkey = random.split(rng)
    batch2_eta, batch2_mu = generate_block_correlation_batch(subkey, n_per_type, dim, corr_range=corr_range, ef=ef)
    eta_all = jnp.concatenate([eta_all, batch2_eta], 0)
    mu_all = jnp.concatenate([mu_all, batch2_mu], 0)
    
    # Batch 3: Low SNR cases (dominant eigenvector aligned with mean)
    print("  Generating low SNR cases...")
    rng, subkey = random.split(rng)
    batch3_eta, batch3_mu = generate_low_snr_batch(subkey, n_per_type + n_remainder, dim, snr_min=snr_min, ef=ef)
    eta_all = jnp.concatenate([eta_all, batch3_eta], 0)
    mu_all = jnp.concatenate([mu_all, batch3_mu], 0)
        
    # Randomly permute to mix case types
    rng, subkey = random.split(rng)
    perm = random.permutation(subkey, total_samples)
    eta_all = eta_all[perm]
    mu_all = mu_all[perm]
    
    print(f"  Generated {total_samples} samples successfully!")
    
    if ensure_extreme_in_train:
        # Identify extreme points
        extreme_mask = identify_extreme_points(eta_all, mu_all)
        extreme_indices = jnp.where(extreme_mask)[0]
        non_extreme_indices = jnp.where(~extreme_mask)[0]
        
        print(f"  Identified {len(extreme_indices)} extreme points out of {total_samples} total points")
        
        # Shuffle non-extreme points
        rng, subkey = random.split(rng)
        shuffled_indices = random.permutation(subkey, non_extreme_indices)
        
        # Allocate points: extremes go to training, rest are split between train/val/test
        num_extreme = len(extreme_indices)
        remaining_train = n_train - num_extreme
        remaining_val = n_val
        remaining_test = n_test
        
        # Take extreme points for training
        eta_train_extreme = eta_all[extreme_indices]
        mu_train_extreme = mu_all[extreme_indices]
        
        # Take remaining points for training, validation, and test
        eta_train_regular = eta_all[shuffled_indices[:remaining_train]]
        mu_train_regular = mu_all[shuffled_indices[:remaining_train]]
        
        eta_val = eta_all[shuffled_indices[remaining_train:remaining_train + remaining_val]]
        mu_val = mu_all[shuffled_indices[remaining_train:remaining_train + remaining_val]]
        
        eta_test = eta_all[shuffled_indices[remaining_train + remaining_val:remaining_train + remaining_val + remaining_test]]
        mu_test = mu_all[shuffled_indices[remaining_train + remaining_val:remaining_train + remaining_val + remaining_test]]
        
        # Combine extreme and regular training data
        eta_train = jnp.concatenate([eta_train_extreme, eta_train_regular], axis=0)
        mu_train = jnp.concatenate([mu_train_extreme, mu_train_regular], axis=0)
        
        print(f"  Training set: {len(eta_train)} points ({num_extreme} extreme + {remaining_train} regular)")
        print(f"  Validation set: {len(eta_val)} points")
        print(f"  Test set: {len(eta_test)} points")
    else:
        # Original logic without extreme value preservation
        eta_train = eta_all[:n_train]
        mu_train = mu_all[:n_train]
        
        eta_val = eta_all[n_train:n_train+n_val]
        mu_val = mu_all[n_train:n_train+n_val]
        
        eta_test = eta_all[n_train+n_val:]
        mu_test = mu_all[n_train+n_val:]
    
    # Compute covariance matrices for LogZ training
    print("  Computing covariance matrices...")
    from src.utils.exact_covariance import compute_exact_covariance_matrix
    
    def compute_cov_batch(eta_batch):
        """Compute covariance matrices for a batch of eta parameters."""
        # Unflatten eta to get structured format
        cov_batch = []
        n = eta_batch.shape[0]
        for i in range(n):
            eta_dict = ef.unflatten_stats_or_eta(eta_batch[i])
            mu_i = -0.5 * jnp.linalg.solve(eta_dict['xxT'], eta_dict['x'])
            Sigma_i = -0.5 * jnp.linalg.inv(eta_dict['xxT'])
            # cov_TT should be the covariance matrix of the 12D sufficient statistics T = [X, X⊗X]
            cov_i = compute_exact_covariance_matrix(mu_i, Sigma_i)
            cov_batch.append(cov_i)
        return jnp.array(cov_batch)
    
    if compute_cov_tt:
        cov_train = compute_cov_batch(eta_train)
        cov_val = compute_cov_batch(eta_val) 
        cov_test = compute_cov_batch(eta_test)
    else:
        cov_train = None
        cov_val = None
        cov_test = None
    
    if tril_format:
        # Convert to tril format using the updated ef.py method
        mvn_tril = ef_factory("multivariate_normal_tril", x_shape=(dim,))
        
        # Convert eta and mu arrays to tril format
        eta_train = mvn_tril.standard_nat_to_LTnat(eta_train)
        eta_val = mvn_tril.standard_nat_to_LTnat(eta_val)
        eta_test = mvn_tril.standard_nat_to_LTnat(eta_test)
        mu_train = mvn_tril.standard_nat_to_LTnat(mu_train)
        mu_val = mvn_tril.standard_nat_to_LTnat(mu_val)
        mu_test = mvn_tril.standard_nat_to_LTnat(mu_test)
        
        if compute_cov_tt:
            # For tril format, we need to extract the tril indices from covariance matrices
            indices = mvn_tril.tril_indices
            # Reshape cov matrices to (batch, n*n) and extract tril indices
            cov_train_flat = cov_train.reshape(cov_train.shape[0], -1)
            cov_val_flat = cov_val.reshape(cov_val.shape[0], -1)
            cov_test_flat = cov_test.reshape(cov_test.shape[0], -1)
            
            cov_train = cov_train_flat[:, indices]
            cov_val = cov_val_flat[:, indices]
            cov_test = cov_test_flat[:, indices]

    data = {
        'train': {'eta': eta_train, 'mu_T': mu_train, 'cov_TT': cov_train},
        'val': {'eta': eta_val, 'mu_T': mu_val, 'cov_TT': cov_val},
        'test': {'eta': eta_test, 'mu_T': mu_test, 'cov_TT': cov_test},
        'metadata': {
            'n_train': n_train,
            'n_val': n_val,
            'n_test': n_test,
            'seed': seed,
            'eta_dim': eta_train.shape[1] if isinstance(eta_train, jnp.ndarray) else len(eta_train['x'][0]) + len(eta_train['xxT_tril'][0]),
            'mu_T_dim': mu_train.shape[1] if isinstance(mu_train, jnp.ndarray) else len(mu_train['x'][0]) + len(mu_train['xxT_tril'][0]),
            'exponential_family': f'multivariate_normal{"_tril" if tril_format else ""}_{dim}d_{difficulty.lower()}',
            'ef_distribution_name': 'multivariate_normal_tril' if tril_format else 'multivariate_normal',  # Name for ef_factory
            'x_shape': [dim],  # Shape of the data samples x
            'x_dim': dim,  # Product of x_shape dimensions
            'difficulty': difficulty
        }
    }
    
    return data


def generate_wide_eigenvalue_batch(rng, n_samples: int, d: int, eval_range: tuple, ef):
    """Generate batch with wide eigenvalue spectrum."""
    
    # Generate means
    rng, subkey = random.split(rng)
    mu1_batch = random.normal(subkey, (n_samples, d)) * 2.0
        
    # Generate eigenvalues (wide spectrum)
    rng, subkey = random.split(rng)
    eigenvals_batch = -jnp.exp(random.uniform(subkey, (n_samples, d), minval=eval_range[0], maxval=eval_range[1]))
    
    # Generate random orthogonal matrices
    rng, subkey = random.split(rng)
    random_matrices = random.normal(subkey, (n_samples, d, d))
    
    # QR decomposition for each matrix to get orthogonal eigenvectors
    eigenvecs_batch = jnp.linalg.qr(random_matrices)[0]
    
    # Construct eta2 matrices: V * diag(eigenvals) * V^T (vectorized)
    # Create diagonal matrices for each sample
    diag_batch = jnp.zeros((n_samples, d, d))
    diag_batch = diag_batch.at[jnp.arange(n_samples)[:, None], jnp.arange(d), jnp.arange(d)].set(eigenvals_batch)
    
    # Vectorized matrix multiplication: V * diag * V^T
    eta2_batch = jnp.matmul(jnp.matmul(eigenvecs_batch, diag_batch), jnp.transpose(eigenvecs_batch, (0, 2, 1)))
    
    return process_batch(mu1_batch, eta2_batch, ef)


def generate_block_correlation_batch(rng, n_samples: int, d: int, corr_range: tuple, ef):
    """Generate batch with block correlation structure."""
    
    # Generate means
    rng, subkey = random.split(rng)
    mu1_batch = random.normal(subkey, (n_samples, d)) * 4.0
        
    # Generate block correlation matrices
    rng, subkey = random.split(rng)
    A_batch = random.normal(subkey, (n_samples, d, d))
    
    # Apply block structure
    block_size = d // 2
    rng, subkey = random.split(rng)
    block_strengths = random.uniform(subkey, (n_samples,), minval=corr_range[0], maxval=corr_range[1])
    
    # Vectorized block multiplication
    A_batch = A_batch.at[:, :block_size, :block_size].multiply(block_strengths[:, None, None])
    A_batch = A_batch.at[:, block_size:, block_size:].multiply(block_strengths[:, None, None])
    
    # Create eta2 matrices
    rng, subkey = random.split(rng)
    regularization = random.uniform(subkey, (n_samples,), minval=0.1, maxval=1.0)
    eta2_batch = -jnp.matmul(jnp.transpose(A_batch, (0, 2, 1)), A_batch) - regularization[:, None, None] * jnp.eye(d)
    
    return process_batch(mu1_batch, eta2_batch, ef)


def generate_low_snr_batch(rng, n_samples: int, d: int, snr_min: float, ef):
    """Generate batch with low SNR (dominant eigenvector aligned with mean)."""
    
    # Generate means
    rng, subkey = random.split(rng)
    mu1_batch = random.normal(subkey, (n_samples, d)) * 2.0
    
    # Generate base covariance matrices with wide eigenvalue spectrum
    rng, subkey = random.split(rng)
    eigenvals_batch = jnp.exp(random.uniform(subkey, (n_samples, d), minval=0.1, maxval=2.0))
    
    # Generate random orthogonal matrices
    rng, subkey = random.split(rng)
    random_matrices = random.normal(subkey, (n_samples, d, d))
    eigenvecs_batch = jnp.linalg.qr(random_matrices)[0]
    
    # Create diagonal matrices
    diag_batch = jnp.zeros((n_samples, d, d))
    diag_batch = diag_batch.at[jnp.arange(n_samples)[:, None], jnp.arange(d), jnp.arange(d)].set(eigenvals_batch)
    
    # Base covariance: V * diag * V^T
    Sigma_base = jnp.matmul(jnp.matmul(eigenvecs_batch, diag_batch), jnp.transpose(eigenvecs_batch, (0, 2, 1)))
    
    # Add low SNR component: align dominant eigenvector with mean direction
    rng, subkey = random.split(rng)
    snr_factors = random.uniform(subkey, (n_samples, 1, 1), minval=1.0, maxval=snr_min)
    mu_outer = jnp.expand_dims(mu1_batch, -1) @ jnp.expand_dims(mu1_batch, -2)  # outer product
    Sigma_batch = Sigma_base + mu_outer / snr_factors
    
    # Convert back to eta2 (negative definite precision matrix)
    eta2_batch = -0.5 * jnp.linalg.inv(Sigma_batch)
    
    return process_batch(mu1_batch, eta2_batch, ef)


def process_batch(mu1_batch, eta2_batch, ef):
    """Process a batch of mu1 and eta2 to create final eta and mu arrays."""
    n_samples = mu1_batch.shape[0]
    
    # Compute eta1 from mu1 and eta2 (fix matrix multiplication order)
    eta1_batch = -2.0 * jnp.matmul(eta2_batch, jnp.expand_dims(mu1_batch, -1)).squeeze(-1)
    Sigma_batch = -0.5 * jnp.linalg.inv(eta2_batch)
    
    E_xx_batch = Sigma_batch + jnp.expand_dims(mu1_batch, -1) @ jnp.expand_dims(mu1_batch, -2)
        
    # Flatten all samples for regular format (tril conversion happens later)
    mu_dict = {'x': mu1_batch, 'xxT': 0.5*(E_xx_batch + jnp.transpose(E_xx_batch, (0, 2, 1)))}
    eta_dict = {'x': eta1_batch, 'xxT': 0.5*(eta2_batch + jnp.transpose(eta2_batch, (0, 2, 1)))}
        
    return ef.flatten_stats_or_eta(eta_dict), ef.flatten_stats_or_eta(mu_dict)

def main(dim=3, difficulty='Easy', n_train=500, n_val=100, n_test=100, seed=42, compute_cov_tt=False, ensure_extreme_in_train=True, tril_format=False):
    """Generate comparison data."""
    print("Generating Dataset for Model Comparison")
    print("="*55)
    
    # Generate data with specified difficulty
    data = generate_gaussian_data(
        dim=dim,
        n_train=n_train,
        n_val=n_val, 
        n_test=n_test,
        seed=seed,
        difficulty=difficulty,
        compute_cov_tt=compute_cov_tt,
        ensure_extreme_in_train=ensure_extreme_in_train,
        tril_format=tril_format
    )
    
    # Save to data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    filename_suffix = "_tril" if tril_format else ""
    data_file = data_dir / f"{difficulty.lower()}_{dim}d_gaussian{filename_suffix}.pkl"
    
    with open(data_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n{difficulty} dataset saved to: {data_file}")
    print(f"Dataset statistics:")
    print(f"  η dimension: {data['metadata']['eta_dim']}")
    print(f"  μ dimension: {data['metadata']['mu_T_dim']}")
    print(f"  Training samples: {data['metadata']['n_train']}")
    print(f"  Validation samples: {data['metadata']['n_val']}")
    print(f"  Test samples: {data['metadata']['n_test']}")
    
    # Print some statistics
    eta_train = data['train']['eta']
    mu_train = data['train']['mu_T']
    
    print(f"\nData ranges:")
    if isinstance(eta_train, dict):
        # For tril format, concatenate x and xxT_tril for statistics
        eta_flat = jnp.concatenate([eta_train['x'], eta_train['xxT_tril']], axis=1)
        mu_flat = jnp.concatenate([mu_train['x'], mu_train['xxT_tril']], axis=1)
        print(f"  η range: [{jnp.min(eta_flat):.3f}, {jnp.max(eta_flat):.3f}]")
        print(f"  μ range: [{jnp.min(mu_flat):.3f}, {jnp.max(mu_flat):.3f}]")
        print(f"  η std: {jnp.std(eta_flat):.3f}")
        print(f"  μ std: {jnp.std(mu_flat):.3f}")
    else:
        print(f"  η range: [{jnp.min(eta_train):.3f}, {jnp.max(eta_train):.3f}]")
        print(f"  μ range: [{jnp.min(mu_train):.3f}, {jnp.max(mu_train):.3f}]")
        print(f"  η std: {jnp.std(eta_train):.3f}")
        print(f"  μ std: {jnp.std(mu_train):.3f}")
    
    # Check condition numbers to show difficulty
    print(f"\nDifficulty indicators:")
    if isinstance(eta_train, dict):
        sample_etas = {k: v[:10] for k, v in eta_train.items()}
    else:
        sample_etas = eta_train[:10]
    
    # Use the correct exponential family for condition number calculation
    if tril_format:
        ef_cond = ef_factory("multivariate_normal_tril", x_shape=(dim,))
    else:
        ef_cond = ef_factory("multivariate_normal", x_shape=(dim,))
    
    condition_numbers = []
    for i in range(10):
        if isinstance(eta_train, dict):
            # For dictionary format, extract the i-th sample
            eta_dict = {k: v[i] for k, v in sample_etas.items()}
        else:
            eta_dict = ef_cond.unflatten_stats_or_eta(sample_etas[i])
        
        if tril_format:
            # For tril format, we need to reconstruct the full matrix
            eta2_tril = eta_dict['xxT_tril']
            eta2 = ef_cond.unflatten_LT(eta2_tril)
        else:
            eta2 = eta_dict['xxT']
        try:
            cond_num = jnp.linalg.cond(eta2)
            condition_numbers.append(float(cond_num))
        except:
            condition_numbers.append(float('inf'))
    
    print(f"  Sample condition numbers: {condition_numbers[:5]}")
    print(f"  Mean condition number: {np.mean([c for c in condition_numbers if np.isfinite(c)]):.1f}")
    
    print(f"\n✓ {difficulty} dataset ready for model comparison!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Gaussian dataset with different difficulty levels')
    parser.add_argument('--difficulty', choices=['Easy', 'Medium', 'Hard'], default='Easy',
                        help='Difficulty level of the dataset (default: Easy)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--dim', type=int, default=3,
                        help='Dimension of the dataset (default: 3)')
    parser.add_argument('--n_train', type=int, default=500,
                        help='Number of training samples (default: 500)')
    parser.add_argument('--n_val', type=int, default=100,
                        help='Number of validation samples (default: 100)')
    parser.add_argument('--n_test', type=int, default=100,
                        help='Number of test samples (default: 100)')
    parser.add_argument('--compute_cov_tt', action='store_true', default=False,
                        help='Compute the TT covariance matrix (default: False)')
    parser.add_argument('--ensure_extreme_in_train', action='store_true', default=True,
                        help='Ensure extreme values are included in training set (default: True)')
    parser.add_argument('--tril_format', action='store_true', default=False,
                        help='Use triangular parameterization format (default: False)')
    args = parser.parse_args()
    main(args.dim, args.difficulty, args.n_train, args.n_val, args.n_test, args.seed, args.compute_cov_tt, args.ensure_extreme_in_train, args.tril_format)
