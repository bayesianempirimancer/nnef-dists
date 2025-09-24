#!/usr/bin/env python
"""Script to generate and save training data without training."""

import argparse
import json
import sys
import hashlib
import pickle
import time
from pathlib import Path
from typing import Tuple

# Use relative imports since we're in the src/ directory

import jax
import jax.numpy as jnp
import yaml
from jax import random

from ef import ef_factory, ExponentialFamily
from sampling import run_hmc

Array = jax.Array


def generate_eta_grid(num_points: int, eta_ranges: tuple, key: Array) -> Array:
    """Generate eta parameters for 1D distributions."""
    dims = len(eta_ranges)
    keys = random.split(key, dims)
    comps = []
    for i, (low, high) in enumerate(eta_ranges):
        comps.append(random.uniform(keys[i], (num_points,), minval=low, maxval=high))
    return jnp.stack(comps, axis=-1)


def generate_negative_definite_matrix(d: int, key: Array, min_eigenval: float = -4.0, max_eigenval: float = -0.1) -> Array:
    """Generate a symmetric negative definite d×d matrix."""
    # Generate random orthogonal matrix
    A = random.normal(key, (d, d))
    Q, _ = jnp.linalg.qr(A)
    
    # Generate negative eigenvalues
    eigenvals = random.uniform(key, (d,), minval=min_eigenval, maxval=max_eigenval)
    
    # Construct the matrix: Q @ diag(eigenvals) @ Q.T
    return Q @ jnp.diag(eigenvals) @ Q.T


def generate_multivariate_eta_grid(
    num_points: int, 
    eta_ranges: tuple, 
    ef: ExponentialFamily,
    key: Array
) -> Array:
    """Generate eta parameters for multivariate normal distributions."""
    from ef import MultivariateNormal
    
    if not isinstance(ef, MultivariateNormal):
        raise ValueError("This function only works with MultivariateNormal distributions")
    
    d = ef.x_shape[-1]  # Dimension of the multivariate normal
    
    # Split key for different parts
    k1, k2 = random.split(key, 2)
    
    # Generate linear terms (eta1) - these can be any real values
    eta1 = random.uniform(k1, (num_points, d), 
                         minval=eta_ranges[0][0], 
                         maxval=eta_ranges[0][1])
    
    # Generate matrix terms (eta2) - must be symmetric negative definite
    keys = random.split(k2, num_points)
    eta2_matrices = jnp.stack([generate_negative_definite_matrix(d, keys[i], min_eigenval=eta_ranges[1][0], max_eigenval=eta_ranges[1][1]) for i in range(num_points)])
    
    # Flatten the matrices and combine with linear terms
    eta2_flat = eta2_matrices.reshape(num_points, d * d)
    eta_combined = jnp.concatenate([eta1, eta2_flat], axis=-1)
    
    return eta_combined

# =============================================================================
# Empirical moments and covariance computation functions
# =============================================================================

def empirical_moments_from_samples(ef, samples: Array) -> Array:
    """
    Compute empirical moments from HMC samples.
    
    Args:
        ef: ExponentialFamily distribution object
        samples: Array of shape (n_samples, x_dim) containing HMC samples
        
    Returns:
        Empirical moments (flattened sufficient statistics)
    """
    # samples: (N, D_flat) -> reshape to (N, x_shape)
    x = jnp.reshape(samples, (samples.shape[0],) + ef.x_shape)
    stats = jax.vmap(ef.compute_stats)(x)
    
    # Flatten per-sample stats dict then average
    def flatten_one(d):
        return ef.flatten_stats_or_eta(d)
    flat = jax.vmap(flatten_one)(stats)
    return jnp.mean(flat, axis=0)

  
def empirical_moments_and_covariance_from_samples(ef, samples: Array) -> Tuple[Array, Array]:
    """
    Compute both empirical moments and covariance from HMC samples.
    
    Args:
        ef: ExponentialFamily distribution object
        samples: Array of shape (n_samples, x_dim) containing HMC samples
        
    Returns:
        Tuple of (moments, covariance_matrix)
    """
    # samples: (N, D_flat) -> reshape to (N, x_shape)
    x = jnp.reshape(samples, (samples.shape[0],) + ef.x_shape)
    # Get unflattened stats for each sample
    stats = jax.vmap(lambda x: ef.compute_stats(x, flatten=False))(x)
    
    # Flatten per-sample stats dict
    def flatten_one(d):
        return ef.flatten_stats_or_eta(d)
    flattened = jax.vmap(flatten_one)(stats)
    
    # Compute empirical moments (mean)
    moments = jnp.mean(flattened, axis=0)
    
    # Compute empirical covariance
    stats_centered = flattened - moments[None, :]
    n_samples = flattened.shape[0]
    cov_matrix = jnp.dot(stats_centered.T, stats_centered) / (n_samples - 1)
    
    return moments, cov_matrix


def compute_moments_with_covariance(eta: Array, ef, num_samples: int = 1000, 
                                  key: Array = None) -> Tuple[Array, Array]:
    """
    Compute moments and their covariance from HMC samples.
    
    Args:
        eta: Natural parameters
        ef: ExponentialFamily distribution object
        num_samples: Number of HMC samples to generate
        key: Random key for HMC sampling
        
    Returns:
        Tuple of (moments, covariance_matrix)
    """    
    if key is None:
        key = random.PRNGKey(42)
    
    # Generate HMC samples
    samples = run_hmc(eta, ef, num_samples=num_samples, key=key)
    return empirical_moments_and_covariance_from_samples(ef, samples)


def identify_extreme_points(eta: Array, mu_T: Array, num_extreme_per_dim: int = 2) -> Array:
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


def build_dataset_with_covariance(
    ef: ExponentialFamily,
    train_points: int,
    val_points: int,
    eta_ranges: tuple,
    sampler_cfg: dict,
    seed: int,
    ensure_extreme_in_train: bool = True,
) -> tuple:
    """
    Build dataset with both moments and covariance matrices.
    
    Args:
        ensure_extreme_in_train: If True, identify extreme points and ensure they're in training set
    
    Returns:
        Tuple of (train_data, val_data) where each contains:
        - 'eta': natural parameters
        - 'y': moments (sufficient statistics)
        - 'cov': covariance matrices of sufficient statistics
    """
    key = random.PRNGKey(seed)
    
    if ensure_extreme_in_train:
        # Generate a larger pool of data points to identify extremes
        total_points = train_points + val_points + 200  # Extra points for extreme identification
        k_pool, k_pos = random.split(key, 2)
        
        # Use appropriate eta generation method based on distribution type
        from ef import MultivariateNormal
        if isinstance(ef, MultivariateNormal):
            etas_pool = generate_multivariate_eta_grid(total_points, eta_ranges, ef, k_pool)
        else:
            etas_pool = generate_eta_grid(total_points, eta_ranges, k_pool)
        
        # Generate mu_T for all points to identify extremes
        def simulate_all(etas: Array, key: Array) -> tuple:
            def one(eta, k):
                start_time = time.time()
                logp = ef.make_logdensity_fn(eta)
                init_pos = sampler_cfg.get("initial_position", None)
                if init_pos is None:
                    init_pos = jnp.zeros((ef.x_dim,))
                else:
                    init_pos = jnp.asarray(init_pos).reshape((ef.x_dim,))
                
                # Generate HMC samples
                samples = run_hmc(
                    logp,
                    num_samples=sampler_cfg["num_samples"],
                    num_warmup=sampler_cfg["num_warmup"],
                    step_size=sampler_cfg["step_size"],
                    num_integration_steps=sampler_cfg["num_integration_steps"],
                    initial_position=init_pos,
                    seed=k,
                )
                
                # Compute both moments and covariance
                moments, cov_matrix = empirical_moments_and_covariance_from_samples(ef, samples)
                sampling_time = time.time() - start_time
                return moments, cov_matrix, sampling_time

            keys = random.split(key, etas.shape[0])
            results = jax.vmap(one)(etas, keys)
            moments, covariances, sampling_times = results
            return moments, covariances, sampling_times
        
        print(f"Generating {total_points} data points to identify extreme values...")
        mu_T_pool, cov_pool, sampling_times_pool = simulate_all(etas_pool, k_pos)
        
        # Identify extreme points
        extreme_mask = identify_extreme_points(etas_pool, mu_T_pool)
        extreme_indices = jnp.where(extreme_mask)[0]
        non_extreme_indices = jnp.where(~extreme_mask)[0]
        
        print(f"Identified {len(extreme_indices)} extreme points out of {total_points} total points")
        
        # Shuffle non-extreme points
        k_shuffle = random.fold_in(k_pos, 1)
        shuffled_indices = random.permutation(k_shuffle, non_extreme_indices)
        
        # Allocate points: extremes go to training, rest are split between train/val
        num_extreme = len(extreme_indices)
        remaining_train = train_points - num_extreme
        remaining_val = val_points
        
        # Take extreme points for training
        train_eta_extreme = etas_pool[extreme_indices]
        train_mu_T_extreme = mu_T_pool[extreme_indices]
        train_cov_extreme = cov_pool[extreme_indices]
        train_times_extreme = sampling_times_pool[extreme_indices]
        
        # Take remaining points for training and validation
        train_eta_regular = etas_pool[shuffled_indices[:remaining_train]]
        train_mu_T_regular = mu_T_pool[shuffled_indices[:remaining_train]]
        train_cov_regular = cov_pool[shuffled_indices[:remaining_train]]
        train_times_regular = sampling_times_pool[shuffled_indices[:remaining_train]]
        
        val_eta = etas_pool[shuffled_indices[remaining_train:remaining_train + remaining_val]]
        val_mu_T = mu_T_pool[shuffled_indices[remaining_train:remaining_train + remaining_val]]
        val_cov = cov_pool[shuffled_indices[remaining_train:remaining_train + remaining_val]]
        val_times = sampling_times_pool[shuffled_indices[remaining_train:remaining_train + remaining_val]]
        
        # Combine extreme and regular training data
        etas_train = jnp.concatenate([train_eta_extreme, train_eta_regular], axis=0)
        mu_T_train = jnp.concatenate([train_mu_T_extreme, train_mu_T_regular], axis=0)
        cov_train = jnp.concatenate([train_cov_extreme, train_cov_regular], axis=0)
        times_train = jnp.concatenate([train_times_extreme, train_times_regular], axis=0)
        
        print(f"Training set: {len(etas_train)} points ({num_extreme} extreme + {remaining_train} regular)")
        print(f"Validation set: {len(val_eta)} points")
        
        # For extreme value case, we already have the data
        train_data = {"eta": etas_train, "y": mu_T_train, "cov": cov_train, "sampling_times": times_train}
        val_data = {"eta": val_eta, "y": val_mu_T, "cov": val_cov, "sampling_times": val_times}
        
    else:
        # Original logic without extreme value preservation
        k_tr, k_val, k_pos = random.split(key, 3)
        
        # Use appropriate eta generation method based on distribution type
        from ef import MultivariateNormal
        if isinstance(ef, MultivariateNormal):
            etas_train = generate_multivariate_eta_grid(train_points, eta_ranges, ef, k_tr)
            etas_val = generate_multivariate_eta_grid(val_points, eta_ranges, ef, k_val)
        else:
            etas_train = generate_eta_grid(train_points, eta_ranges, k_tr)
            etas_val = generate_eta_grid(val_points, eta_ranges, k_val)

        # Original simulation logic for non-extreme case
        def simulate(etas: Array, key: Array) -> tuple:
            def one(eta, k):
                start_time = time.time()
                logp = ef.make_logdensity_fn(eta)
                init_pos = sampler_cfg.get("initial_position", None)
                if init_pos is None:
                    init_pos = jnp.zeros((ef.x_dim,))
                else:
                    init_pos = jnp.asarray(init_pos).reshape((ef.x_dim,))
                
                # Generate HMC samples
                samples = run_hmc(
                    logp,
                    num_samples=sampler_cfg["num_samples"],
                    num_warmup=sampler_cfg["num_warmup"],
                    step_size=sampler_cfg["step_size"],
                    num_integration_steps=sampler_cfg["num_integration_steps"],
                    initial_position=init_pos,
                    seed=k,
                )
                
                # Compute both moments and covariance
                moments, cov_matrix = empirical_moments_and_covariance_from_samples(ef, samples)
                sampling_time = time.time() - start_time
                return moments, cov_matrix, sampling_time

            keys = random.split(key, etas.shape[0])
            results = jax.vmap(one)(etas, keys)
            moments, covariances, sampling_times = results
            return moments, covariances, sampling_times

        y_train, cov_train, times_train = simulate(etas_train, k_pos)
        y_val, cov_val, times_val = simulate(etas_val, random.fold_in(k_pos, 1))
        
        train_data = {"eta": etas_train, "y": y_train, "cov": cov_train, "sampling_times": times_train}
        val_data = {"eta": etas_val, "y": y_val, "cov": cov_val, "sampling_times": times_val}
    
    return train_data, val_data

def get_data_hash(cfg):
    """Generate a hash for the dataset configuration to use as cache key."""
    data_config = {
        "ef": cfg["ef"],
        "grid": cfg["grid"],
        "sampling": cfg["sampling"],
        "optim": {"seed": cfg["optim"]["seed"]}
    }
    config_str = json.dumps(data_config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


def save_training_data(train_data, val_data, data_file, config_hash, ef=None):
    """Save training data to file with metadata using JAX native format."""
    # Create test data by splitting validation data (50/50 split)
    import jax.numpy as jnp
    val_size = val_data["eta"].shape[0]
    test_size = val_size // 2
    val_size = val_size - test_size
    
    # Split validation data into val and test
    val_eta = val_data["eta"][:val_size]
    val_y = val_data["y"][:val_size]
    test_eta = val_data["eta"][val_size:]
    test_y = val_data["y"][val_size:]
    
    # Add covariance data if available
    val_cov = val_data.get("cov", None)
    test_cov = None
    if val_cov is not None:
        val_cov = val_cov[:val_size]
        test_cov = val_cov[val_size:]
    
    data_to_save = {
        "config_hash": config_hash,
        "train": {
            "eta": train_data["eta"],
            "mu_T": train_data["y"]
        },
        "val": {
            "eta": val_eta,
            "mu_T": val_y
        },
        "test": {
            "eta": test_eta,
            "mu_T": test_y
        },
        "shapes": {
            "train_eta": list(train_data["eta"].shape),
            "train_y": list(train_data["y"].shape),
            "val_eta": list(val_eta.shape),
            "val_y": list(val_y.shape),
            "test_eta": list(test_eta.shape),
            "test_y": list(test_y.shape)
        }
    }
    
    # Add exponential family metadata if ef is provided
    if ef is not None:
        import jax.numpy as jnp
        data_to_save["metadata"] = {
            "eta_dim": ef.eta_dim,
            "mu_T_dim": ef.eta_dim,  # Same as eta_dim for expectation parameters
            "ef_distribution_name": ef.__class__.__name__.lower().replace('_', '_'),  # e.g., multivariate_normal
            "x_shape": list(ef.x_shape),
            "x_dim": int(jnp.prod(jnp.array(ef.x_shape))),
            "exponential_family": f"{ef.__class__.__name__.lower()}_{ef.x_shape[0]}d" if len(ef.x_shape) == 1 else f"{ef.__class__.__name__.lower()}_{ef.x_shape}"
        }
    
    # Include covariance data if available
    if "cov" in train_data:
        data_to_save["train_cov"] = train_data["cov"]
        data_to_save["shapes"]["train_cov"] = list(train_data["cov"].shape)
    if val_cov is not None:
        data_to_save["val_cov"] = val_cov
        data_to_save["shapes"]["val_cov"] = list(val_cov.shape)
    if test_cov is not None:
        data_to_save["test_cov"] = test_cov
        data_to_save["shapes"]["test_cov"] = list(test_cov.shape)
    
    # Add timing data if available
    if "sampling_times" in train_data:
        data_to_save["train_sampling_times"] = train_data["sampling_times"]
        data_to_save["shapes"]["train_sampling_times"] = list(train_data["sampling_times"].shape)
        
        # Calculate total sampling time
        total_sampling_time = float(jnp.sum(train_data["sampling_times"]))
        data_to_save["total_sampling_time"] = total_sampling_time
        
        # Calculate average sampling time per eta
        avg_sampling_time_per_eta = float(jnp.mean(train_data["sampling_times"]))
        data_to_save["avg_sampling_time_per_eta"] = avg_sampling_time_per_eta
        
        print(f"Total sampling time: {total_sampling_time:.2f} seconds")
        print(f"Average sampling time per eta: {avg_sampling_time_per_eta:.4f} seconds")
    
    if "sampling_times" in val_data:
        val_times = val_data["sampling_times"][:val_size]
        test_times = val_data["sampling_times"][val_size:] if val_size < len(val_data["sampling_times"]) else None
        
        data_to_save["val_sampling_times"] = val_times
        data_to_save["shapes"]["val_sampling_times"] = list(val_times.shape)
        
        if test_times is not None:
            data_to_save["test_sampling_times"] = test_times
            data_to_save["shapes"]["test_sampling_times"] = list(test_times.shape)
    
    with open(data_file, "wb") as f:
        pickle.dump(data_to_save, f)


def main():
    parser = argparse.ArgumentParser(description="Generate training data")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--output", type=str, help="Output data file path (optional)")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if file exists")
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Create exponential family
    ef_params = cfg["ef"].copy()
    ef_name = ef_params.pop("name")
    ef = ef_factory(ef_name, **ef_params)
    print(f"Created {ef.__class__.__name__}: x_shape={ef.x_shape}, eta_dim={ef.eta_dim}")

    # Generate data hash and determine output file
    config_hash = get_data_hash(cfg)
    if args.output:
        data_file = Path(args.output)
    else:
        # Use relative path from project root
        project_root = Path(__file__).parent.parent
        data_file = project_root / "data" / f"training_data_{config_hash}.pkl"
    
    print(f"Config hash: {config_hash}")
    print(f"Output file: {data_file}")

    # Check if data already exists
    if data_file.exists() and not args.force:
        print(f"Data file already exists: {data_file}")
        print("Use --force to regenerate")
        return

    # Generate the dataset
    print("\nGenerating training data...")
    print(f"Training points: {cfg['grid']['num_train_points']}")
    print(f"Validation points: {cfg['grid']['num_val_points']}")
    print(f"Eta ranges: {cfg['grid']['eta_ranges']}")
    print(f"Sampling config: {cfg['sampling']}")
    
    train_data, val_data = build_dataset_with_covariance(
        ef=ef,
        train_points=cfg["grid"]["num_train_points"],
        val_points=cfg["grid"]["num_val_points"],
        eta_ranges=tuple(tuple(r) for r in cfg["grid"]["eta_ranges"]),
        sampler_cfg=cfg["sampling"],
        seed=cfg["optim"]["seed"],
        ensure_extreme_in_train=True,  # Enable extreme value preservation
    )

    print(f"\nDataset generated successfully!")
    print(f"Train data: eta {train_data['eta'].shape}, y {train_data['y'].shape}, cov {train_data['cov'].shape}")
    print(f"Val data: eta {val_data['eta'].shape}, y {val_data['y'].shape}, cov {val_data['cov'].shape}")
    
    # Print timing information if available
    if "sampling_times" in train_data:
        total_sampling_time = float(jnp.sum(train_data["sampling_times"]))
        avg_sampling_time_per_eta = float(jnp.mean(train_data["sampling_times"]))
        print(f"\nSampling timing information:")
        print(f"  Total sampling time: {total_sampling_time:.2f} seconds")
        print(f"  Average sampling time per eta: {avg_sampling_time_per_eta:.4f} seconds")
        print(f"  Number of eta values: {len(train_data['sampling_times'])}")
        print(f"  Total samples generated: {len(train_data['sampling_times']) * cfg['sampling']['num_samples']}")

    # Save the data
    print(f"\nSaving data to {data_file}")
    data_file.parent.mkdir(parents=True, exist_ok=True)
    save_training_data(train_data, val_data, data_file, config_hash, ef=ef)
    
    print("✅ Data generation complete!")


if __name__ == "__main__":
    main()
