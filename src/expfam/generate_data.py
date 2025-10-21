#!/usr/bin/env python
"""Thin wrapper around DataGenerator to produce datasets."""

import argparse
import json
import hashlib
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import jax
import jax.numpy as jnp
import yaml
from jax import random

from .ef import ef_factory
from .data_generator import DataGenerator, SamplingConfig

Array = jax.Array


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


def save_data(data: Dict[str, Any], data_file: Path, config_hash: str, ef):
    """Save data to file with metadata."""
    data_to_save = {
        "config_hash": config_hash,
        "eta": data["eta"],
        "mu_T": data["mu_T"],
        "cov_TT": data["cov_TT"],
        "ess": data["ess"],
        "shapes": {
            "eta": list(data["eta"].shape),
            "mu_T": list(data["mu_T"].shape),
            "cov_TT": list(data["cov_TT"].shape),
            "ess": list(data["ess"].shape),
        }
    }
    
    # Add exponential family metadata
    data_to_save["metadata"] = {
        "eta_dim": ef.eta_dim,
        "mu_T_dim": ef.eta_dim,
        "ef_distribution_name": ef.__class__.__name__,
        "x_shape": list(ef.x_shape),
        "x_dim": int(jnp.prod(jnp.array(ef.x_shape))),
    }
    
    # Add timing metadata if available
    if "timing_info" in data:
        timing_info = data["timing_info"]
        data_to_save["metadata"]["total_generation_time"] = timing_info["total_generation_time"]
        data_to_save["metadata"]["avg_time_per_eta"] = timing_info["avg_time_per_eta"]
        data_to_save["metadata"]["num_eta_points"] = len(timing_info["eta_times"])
        data_to_save["metadata"]["total_hmc_samples"] = len(timing_info["eta_times"]) * 1000  # Assuming 1000 samples per eta
    
    # Add ESS metadata
    data_to_save["metadata"]["avg_ess"] = float(jnp.mean(data["ess"]))
    data_to_save["metadata"]["min_ess"] = float(jnp.min(data["ess"]))
    data_to_save["metadata"]["max_ess"] = float(jnp.max(data["ess"]))
    
    with open(data_file, "wb") as f:
        pickle.dump(data_to_save, f)


def main():
    parser = argparse.ArgumentParser(description="Generate training data")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--output", type=str, help="Output data file path (optional)")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if file exists")
    parser.add_argument("--logscale", action="store_true", default=False, help="Use log-scale eta generation to emphasize small values (default: False)")
    parser.add_argument("--no-logscale", action="store_false", dest="logscale", help="Disable log-scale eta generation")
    # Debug controls
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with fewer eta points and samples")
    parser.add_argument("--debug-num-points", type=int, default=4, help="Num eta points in debug mode")
    parser.add_argument("--debug-num-samples", type=int, default=128, help="Num HMC samples in debug mode")
    parser.add_argument("--debug-num-warmup", type=int, default=64, help="Num warmup steps in debug mode")
    
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Create exponential family
    ef_params = cfg["ef"].copy()
    ef_name = ef_params.pop("name")
    ef = ef_factory(ef_name, **ef_params)
    print(f"Created {ef.__class__.__name__}: x_shape={ef.x_shape}, eta_dim={ef.eta_dim}")

    # Determine output file using number of datapoints
    num_points = cfg["grid"]["num_points"]
    if args.debug:
        num_points = min(num_points, args.debug_num_points)
    
    if args.output:
        data_file = Path(args.output)
    else:
        project_root = Path(__file__).resolve().parents[2]
        # Use ef.name and number of datapoints for sensible filenames
        data_file = project_root / "data" / f"{ef.name}_data_{num_points}.pkl"
    
    print(f"Output file: {data_file}")

    # Check if data already exists
    if data_file.exists() and not args.force:
        print(f"Data file already exists: {data_file}")
        print("Use --force to regenerate")
        return

    # Build DataGenerator
    print("\nGenerating training data...")
    overall_start_time = time.time()

    scfg = cfg["sampling"]
    if args.debug:
        scfg = dict(scfg)
        scfg["num_samples"] = min(int(scfg["num_samples"]), args.debug_num_samples)
        scfg["num_warmup"] = min(int(scfg["num_warmup"]), args.debug_num_warmup)

    dg = DataGenerator(ef, SamplingConfig(**scfg))

    # Debug: confirm sampling counts
    print(f"Sampling setup: num_chains={dg.sampling_config.num_chains}, "
          f"samples_per_chain={dg.sampling_config.num_samples}, "
          f"total_draws_per_eta={dg.sampling_config.num_chains * dg.sampling_config.num_samples}")

    # Sample eta: log-uniform magnitude in a configurable range
    key = random.PRNGKey(cfg["optim"]["seed"])
    k1, _ = random.split(key)
    eta_range_cfg = cfg.get("grid", {}).get("eta_range", [1e-3, 1e2])
    
    # Handle both old format (list) and new format (dict)
    if isinstance(eta_range_cfg, dict):
        # New format: per-dimension bounds
        input_bounds = {
            stat_name: (float(eta_range_cfg[stat_name][0]), float(eta_range_cfg[stat_name][1]))
            for stat_name in ef.stat_shapes.keys()
        }
    else:
        # Old format: single range for all dimensions
        abs_range = (float(eta_range_cfg[0]), float(eta_range_cfg[1]))
        input_bounds = {
            stat_name: abs_range for stat_name in ef.stat_shapes.keys()
        }
    
    # Batch eta generation for memory efficiency
    batch_size = int(cfg.get("grid", {}).get("batch_size", 256))
    num_batches = (num_points + batch_size - 1) // batch_size
    print(f"Processing {num_points} eta in {num_batches} batches of size {batch_size}")

    collected = {"eta": [], "mu_T": [], "cov_TT": [], "ess": []}
    for b in range(num_batches):
        start = b * batch_size
        end = min(start + batch_size, num_points)
        this_n = end - start
        k_b = random.fold_in(k1, b)
        # Use the distribution's valid eta sampler
        eta_batch = ef.valid_eta_sampler(k_b, input_bounds, this_n, sample_spread='log')
        batch_data = dg.generate_data(eta_batch, output_format="dict")
        for k in collected:
            collected[k].append(batch_data[k])
    data = {k: jnp.concatenate(v, axis=0) for k, v in collected.items()}

    # Reject chains with ESS < 0.1 * chain_length, retry once, then top-up with new etas
    chain_length = dg.sampling_config.num_chains * dg.sampling_config.num_samples
    ess_threshold = 0.1 * chain_length

    def filter_data(d):
        e = jnp.asarray(d["ess"])  # (B,)
        m = e >= ess_threshold
        return {
            "eta": jnp.asarray(d["eta"])[m],
            "mu_T": jnp.asarray(d["mu_T"])[m],
            "cov_TT": jnp.asarray(d["cov_TT"])[m],
            "ess": e[m],
        }, m

    filtered, mask = filter_data(data)
    kept = int(filtered["ess"].shape[0])
    if kept < num_points:
        print(f"Low-ESS filter kept {kept}/{num_points} (threshold {ess_threshold:.1f}). Retrying rejected...")

        # Retry once on rejected indices with a different seed
        rejected_mask = jnp.logical_not(mask)
        num_rejected = int(jnp.sum(rejected_mask))
        if num_rejected > 0:
            # Change seed for retry to get new chains
            dg.sampling_config = SamplingConfig(**{**scfg, "seed": int(cfg["optim"]["seed"]) + 1})
            # Generate new eta values for retry using valid_eta_sampler
            k_retry = random.fold_in(k1, num_batches)  # Use a different key for retry
            eta_rejected = ef.valid_eta_sampler(k_retry, input_bounds, num_rejected, sample_spread='log')
            retry_data = dg.generate_data(eta_rejected, output_format="dict")
            retry_filtered, retry_valid_mask = filter_data(retry_data)

            # Concatenate kept + newly valid
            for k in ["eta", "mu_T", "cov_TT", "ess"]:
                filtered[k] = jnp.concatenate([filtered[k], retry_filtered[k]], axis=0)
            kept = int(filtered["ess"].shape[0])

        # If still below target, top-up by sampling new etas until we hit num_points
        seed_bump = 2
        while kept < num_points:
            remaining = num_points - kept
            # New seed for eta generation and sampling
            k_eta = random.PRNGKey(int(cfg["optim"]["seed"]) + 1000 + seed_bump)
            new_eta = ef.valid_eta_sampler(k_eta, input_bounds, remaining, sample_spread='log')
            dg.sampling_config = SamplingConfig(**{**scfg, "seed": int(cfg["optim"]["seed"]) + 1000 + seed_bump})
            new_data = dg.generate_data(new_eta, output_format="dict")
            new_filtered, _ = filter_data(new_data)

            # Append whatever passed; if none pass, increase seed and continue
            if new_filtered["ess"].shape[0] > 0:
                take = int(min(new_filtered["ess"].shape[0], remaining))
                for k in ["eta", "mu_T", "cov_TT", "ess"]:
                    filtered[k] = jnp.concatenate([filtered[k], new_filtered[k][:take]], axis=0)
                kept = int(filtered["ess"].shape[0])
            seed_bump += 1

    # Trim to exactly num_points if we overshot
    if filtered["ess"].shape[0] > num_points:
        idx = jnp.arange(filtered["ess"].shape[0])[:num_points]
        for k in ["eta", "mu_T", "cov_TT", "ess"]:
            filtered[k] = filtered[k][idx]

    data = filtered
    total_time = time.time() - overall_start_time

    print(f"\nDataset generated successfully!")
    print(f"Data: eta {data['eta'].shape}, mu_T {data['mu_T'].shape}, cov_TT {data['cov_TT'].shape}, ess {data['ess'].shape}")

    # Print timing information
    if "timing_info" in data:
        timing_info = data["timing_info"]
        print(f"\nTiming information:")
        print(f"  Total generation time: {timing_info['total_generation_time']:.2f} seconds")
        print(f"  Average time per eta: {timing_info['avg_time_per_eta']:.4f} seconds")
        print(f"  Number of eta points: {len(timing_info['eta_times'])}")
    
    # Print ESS information
        print(f"\nEffective Sample Size (ESS) information:")
    print(f"  ESS range: [{jnp.min(data['ess']):.1f}, {jnp.max(data['ess']):.1f}]")
    print(f"  Average ESS: {jnp.mean(data['ess']):.1f}")
    print(f"  Total generation time: {total_time:.2f} seconds")

    # Save the data
    print(f"\nSaving data to {data_file}")
    data_file.parent.mkdir(parents=True, exist_ok=True)
    # Generate config hash just for saving in the data file
    config_hash = get_data_hash(cfg)
    save_data(data, data_file, config_hash, ef)
    
    print("✅ Data generation complete!")


if __name__ == "__main__":
    main()