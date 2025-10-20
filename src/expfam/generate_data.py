#!/usr/bin/env python
"""Script to generate and save training data without training."""

import argparse
import json
import hashlib
import pickle
import time
from pathlib import Path
from typing import Tuple, Dict, Any

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


def generate_eta_grid_logscale(num_points: int, eta_ranges: tuple, key: Array) -> Array:
    """
    Generate eta parameters using log scale to emphasize small values.
    
    This function generates eta values such that most points have small norms,
    emphasizing difficult cases and deemphasizing easy cases.
    
    Args:
        num_points: Number of eta points to generate
        eta_ranges: Tuple of (min, max) ranges for each eta dimension
        key: JAX random key
        
    Returns:
        Array of shape (num_points, num_dims) with eta values
    """
    dims = len(eta_ranges)
    keys = random.split(key, dims)
    comps = []
    
    for i, (low, high) in enumerate(eta_ranges):
        # Generate uniform samples in log space
        log_low = jnp.log(low)
        log_high = jnp.log(high)
        
        # Sample uniformly in log space
        log_samples = random.uniform(keys[i], (num_points,), minval=log_low, maxval=log_high)
        
        # Convert back to linear space
        eta_samples = jnp.exp(log_samples)
        comps.append(eta_samples)
    
    return jnp.stack(comps, axis=-1)


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


def generate_negative_definite_matrix(d: int, key: Array, min_eigenval: float = -4.0, max_eigenval: float = -0.1) -> Array:
    """Generate a symmetric negative definite d×d matrix."""
    # Generate random orthogonal matrix
    A = random.normal(key, (d, d))
    Q, _ = jnp.linalg.qr(A)
    
    # Generate negative eigenvalues
    eigenvals = random.uniform(key, (d,), minval=min_eigenval, maxval=max_eigenval)
    
    # Construct the matrix: Q @ diag(eigenvals) @ Q.T
    return Q @ jnp.diag(eigenvals) @ Q.T


def compute_moments_and_covariance(ef: ExponentialFamily, samples: Array) -> Tuple[Array, Array]:
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


def generate_data(
    ef: ExponentialFamily,
    num_points: int,
    eta_ranges: tuple,
    sampler_cfg: dict,
    seed: int,
    use_logscale: bool = True,
) -> Dict[str, Any]:
    """
    Generate a single dataset with eta, mu_T, and cov_TT.
    
    Returns:
        Dictionary with keys: 'eta', 'mu_T', 'cov_TT'
    """
    key = random.PRNGKey(seed)
    k_eta, k_sim = random.split(key, 2)
    
    # Generate eta parameters
    from ef import MultivariateNormal
    if isinstance(ef, MultivariateNormal):
        etas = generate_multivariate_eta_grid(num_points, eta_ranges, ef, k_eta)
    else:
        if use_logscale:
            etas = generate_eta_grid_logscale(num_points, eta_ranges, k_eta)
        else:
            etas = generate_eta_grid(num_points, eta_ranges, k_eta)

    # Generate moments and covariance for all points
    def simulate_one(eta: Array, key: Array) -> Tuple[Array, Array, float]:
        eta_start_time = time.time()
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
            seed=key,
        )
        
        # Compute both moments and covariance
        moments, cov_matrix = compute_moments_and_covariance(ef, samples)
        eta_time = time.time() - eta_start_time
        return moments, cov_matrix, eta_time

    print(f"Generating {num_points} data points...")
    start_time = time.time()
    keys = random.split(k_sim, num_points)
    results = jax.vmap(simulate_one)(etas, keys)
    mu_T, cov_TT, eta_times = results
    
    # Shuffle the data
    indices = random.permutation(k_sim, jnp.arange(num_points))
    etas = etas[indices]
    mu_T = mu_T[indices]
    cov_TT = cov_TT[indices]
    eta_times = eta_times[indices]

    total_time = time.time() - start_time
    avg_time_per_eta = float(jnp.mean(eta_times))

    return {
        "eta": etas,
        "mu_T": mu_T,
        "cov_TT": cov_TT,
        "timing_info": {
            "total_generation_time": total_time,
            "avg_time_per_eta": avg_time_per_eta,
            "eta_times": eta_times
        }
    }


def ttv_split(data: Dict[str, Any], n_train: int, n_test: int, n_val: int, num_samples: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Split data into train, test, and validation sets with exact counts.
    
    Args:
        data: Dictionary with keys 'eta', 'mu_T', 'cov_TT'
        n_train: Number of training points
        n_test: Number of test points
        n_val: Number of validation points
        num_samples: Number of samples used for HMC
        
    Returns:
        Tuple of (train_data, test_data, val_data) with same keys as input plus 'expected_MSE'
    """
    n_total = data["eta"].shape[0]
    expected_total = n_train + n_test + n_val
    
    if n_total != expected_total:
        raise ValueError(f"Total data points ({n_total}) doesn't match expected total ({expected_total})")
    
    train_data = {
        "eta": data["eta"][:n_train],
        "mu_T": data["mu_T"][:n_train],
        "cov_TT": data["cov_TT"][:n_train]
    }
    
    test_data = {
        "eta": data["eta"][n_train:n_train + n_test],
        "mu_T": data["mu_T"][n_train:n_train + n_test],
        "cov_TT": data["cov_TT"][n_train:n_train + n_test]
    }
    
    val_data = {
        "eta": data["eta"][n_train + n_test:],
        "mu_T": data["mu_T"][n_train + n_test:],
        "cov_TT": data["cov_TT"][n_train + n_test:]
    }
    
    # Add timing info to each split if available
    if "timing_info" in data:
        train_data["timing_info"] = {
            "eta_times": data["timing_info"]["eta_times"][:n_train]
        }
        test_data["timing_info"] = {
            "eta_times": data["timing_info"]["eta_times"][n_train:n_train + n_test]
        }
        val_data["timing_info"] = {
            "eta_times": data["timing_info"]["eta_times"][n_train + n_test:]
        }
    
    # Add expected MSE to each split
    train_cov_diag = jnp.diagonal(train_data["cov_TT"], axis1=-2, axis2=-1)
    test_cov_diag = jnp.diagonal(test_data["cov_TT"], axis1=-2, axis2=-1)
    val_cov_diag = jnp.diagonal(val_data["cov_TT"], axis1=-2, axis2=-1)
    
    train_data["expected_MSE"] = train_cov_diag / num_samples
    test_data["expected_MSE"] = test_cov_diag / num_samples
    val_data["expected_MSE"] = val_cov_diag / num_samples
    
    print(f"Split: {n_train} train, {n_test} test, {n_val} val")
    return train_data, test_data, val_data


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


def save_training_data(train_data, test_data, val_data, data_file, config_hash, ef=None, timing_info=None, sampler_cfg=None):
    """Save training data to file with metadata."""
    data_to_save = {
        "config_hash": config_hash,
        "train": train_data,
        "test": test_data,
        "val": val_data,
        "shapes": {
            "train_eta": list(train_data["eta"].shape),
            "train_mu_T": list(train_data["mu_T"].shape),
            "train_cov_TT": list(train_data["cov_TT"].shape),
            "test_eta": list(test_data["eta"].shape),
            "test_mu_T": list(test_data["mu_T"].shape),
            "test_cov_TT": list(test_data["cov_TT"].shape),
            "val_eta": list(val_data["eta"].shape),
            "val_mu_T": list(val_data["mu_T"].shape),
            "val_cov_TT": list(val_data["cov_TT"].shape),
        }
    }
    
    # Add exponential family metadata if ef is provided
    data_to_save["metadata"] = {
        "eta_dim": ef.eta_dim,
        "mu_T_dim": ef.eta_dim,
        "ef_distribution_name": ef.__class__.__name__,
        "x_shape": list(ef.x_shape),
        "x_dim": int(jnp.prod(jnp.array(ef.x_shape))),
    }
    
    # Add timing metadata if available
    if timing_info is not None:
        data_to_save["metadata"]["total_generation_time"] = timing_info["total_generation_time"]
        data_to_save["metadata"]["avg_time_per_eta"] = timing_info["avg_time_per_eta"]
        data_to_save["metadata"]["num_eta_points"] = len(timing_info["eta_times"])
        data_to_save["metadata"]["total_hmc_samples"] = len(timing_info["eta_times"]) * 1000  # Assuming 1000 samples per eta
    
    # Add total expected MSE metadata
    data_to_save["metadata"]["total_expected_MSE_train"] = float(jnp.mean(train_data["expected_MSE"]))
    data_to_save["metadata"]["total_expected_MSE_test"] = float(jnp.mean(test_data["expected_MSE"]))
    data_to_save["metadata"]["total_expected_MSE_val"] = float(jnp.mean(val_data["expected_MSE"]))
    
    with open(data_file, "wb") as f:
        pickle.dump(data_to_save, f)


def main():
    parser = argparse.ArgumentParser(description="Generate training data")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--output", type=str, help="Output data file path (optional)")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if file exists")
    parser.add_argument("--logscale", action="store_true", default=False, help="Use log-scale eta generation to emphasize small values (default: False)")
    parser.add_argument("--no-logscale", action="store_false", dest="logscale", help="Disable log-scale eta generation")
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
    n_train = cfg["grid"]["num_train_points"]
    n_test = cfg["grid"]["num_test_points"]
    n_val = cfg["grid"]["num_val_points"]
    total_points = n_train + n_test + n_val
    
    print(f"Train points: {n_train}")
    print(f"Test points: {n_test}")
    print(f"Val points: {n_val}")
    print(f"Total points: {total_points}")
    print(f"Eta ranges: {cfg['grid']['eta_ranges']}")
    print(f"Sampling config: {cfg['sampling']}")
    
    # Generate single dataset
    data = generate_data(
        ef=ef,
        num_points=total_points,
        eta_ranges=tuple(tuple(r) for r in cfg["grid"]["eta_ranges"]),
        sampler_cfg=cfg["sampling"],
        seed=cfg["optim"]["seed"],
        use_logscale=args.logscale,
    )

    # Split into train/test/val with exact counts
    train_data, test_data, val_data = ttv_split(data, n_train, n_test, n_val, cfg["sampling"]["num_samples"])

    print(f"\nDataset generated successfully!")
    print(f"Train data: eta {train_data['eta'].shape}, mu_T {train_data['mu_T'].shape}, cov_TT {train_data['cov_TT'].shape}")
    print(f"Test data: eta {test_data['eta'].shape}, mu_T {test_data['mu_T'].shape}, cov_TT {test_data['cov_TT'].shape}")
    print(f"Val data: eta {val_data['eta'].shape}, mu_T {val_data['mu_T'].shape}, cov_TT {val_data['cov_TT'].shape}")

    # Print timing information
    if "timing_info" in data:
        timing_info = data["timing_info"]
        print(f"\nTiming information:")
        print(f"  Total generation time: {timing_info['total_generation_time']:.2f} seconds")
        print(f"  Average time per eta: {timing_info['avg_time_per_eta']:.4f} seconds")
        print(f"  Number of eta points: {len(timing_info['eta_times'])}")
    
    # Print expected MSE information
    print(f"\nExpected MSE information:")
    print(f"  Train expected MSE shape: {train_data['expected_MSE'].shape}")
    print(f"  Train expected MSE range: [{jnp.min(train_data['expected_MSE']):.6f}, {jnp.max(train_data['expected_MSE']):.6f}]")
    print(f"  Train total expected MSE: {jnp.mean(train_data['expected_MSE']):.6f}")
    print(f"  Test total expected MSE: {jnp.mean(test_data['expected_MSE']):.6f}")
    print(f"  Val total expected MSE: {jnp.mean(val_data['expected_MSE']):.6f}")

    # Save the data
    print(f"\nSaving data to {data_file}")
    data_file.parent.mkdir(parents=True, exist_ok=True)
    timing_info = data.get("timing_info", None)
    save_training_data(train_data, test_data, val_data, data_file, config_hash, ef=ef, timing_info=timing_info, sampler_cfg=cfg["sampling"])
    
    print("✅ Data generation complete!")


if __name__ == "__main__":
    main()