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

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ef import ef_factory

def generate_gaussian_data(dim: int = 3, n_train: int = 1000, n_val: int = 250, 
                                         n_test: int = 250, seed: int = 42, difficulty = 'Easy', compute_cov_tt=False):
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

    # Create exponential family
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
    
    # Split into train/val/test
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
    
    # Create data dictionary
    data = {
        'train': {'eta': eta_train, 'mu_T': mu_train, 'cov_TT': cov_train},
        'val': {'eta': eta_val, 'mu_T': mu_val, 'cov_TT': cov_val},
        'test': {'eta': eta_test, 'mu_T': mu_test, 'cov_TT': cov_test},
        'metadata': {
            'n_train': n_train,
            'n_val': n_val,
            'n_test': n_test,
            'seed': seed,
            'eta_dim': eta_train.shape[1],
            'mu_T_dim': mu_train.shape[1],
            'exponential_family': f'multivariate_normal_{dim}d_{difficulty.lower()}',
            'ef_distribution_name': 'multivariate_normal',  # Name for ef_factory
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
        
    # Flatten all samples
    mu_dict = {'x': mu1_batch, 'xxT': 0.5*(E_xx_batch + jnp.transpose(E_xx_batch, (0, 2, 1)))}
    eta_dict = {'x': eta1_batch, 'xxT': 0.5*(eta2_batch + jnp.transpose(eta2_batch, (0, 2, 1)))}
        
    return ef.flatten_stats_or_eta(eta_dict), ef.flatten_stats_or_eta(mu_dict)

def main(dim=3, difficulty='Easy', n_train=500, n_val=100, n_test=100, seed=42, compute_cov_tt=False):
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
        compute_cov_tt=compute_cov_tt
    )
    
    # Save to data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    data_file = data_dir / f"{difficulty.lower()}_{dim}d_gaussian.pkl"
    
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
    print(f"  η range: [{jnp.min(eta_train):.3f}, {jnp.max(eta_train):.3f}]")
    print(f"  μ range: [{jnp.min(mu_train):.3f}, {jnp.max(mu_train):.3f}]")
    print(f"  η std: {jnp.std(eta_train):.3f}")
    print(f"  μ std: {jnp.std(mu_train):.3f}")
    
    # Check condition numbers to show difficulty
    print(f"\nDifficulty indicators:")
    sample_etas = eta_train[:10]
    ef = ef_factory("multivariate_normal", x_shape=(dim,))
    
    condition_numbers = []
    for i in range(10):
        eta_dict = ef.unflatten_stats_or_eta(sample_etas[i])
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
    args = parser.parse_args()
    main(args.dim, args.difficulty, args.n_train, args.n_val, args.n_test, args.seed, args.compute_cov_tt)
