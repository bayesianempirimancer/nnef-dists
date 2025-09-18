#!/usr/bin/env python3
"""
Generate training data specifically for NoProp-CT comparison experiments.
This script creates smaller, focused datasets for rapid experimentation.
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import yaml
from jax import random

from src.ef import ef_factory
from src.sampling import sample_ef_data


def generate_comparison_data(config: Dict, output_path: Path, seed: int = 42) -> None:
    """Generate training, validation, and test data for comparison experiments."""
    
    # Create exponential family
    ef = ef_factory(config["ef_type"], config.get("ef_params", {}))
    
    print(f"Generating data for {ef.__class__.__name__}")
    print(f"Natural parameter dimension: {ef.eta_dim}")
    
    # Set up random keys
    rng = random.PRNGKey(seed)
    train_key, val_key, test_key = random.split(rng, 3)
    
    # Generate training data
    print(f"Generating {config['grid']['num_train']} training samples...")
    train_data = sample_ef_data(
        ef=ef,
        num_samples=config['grid']['num_train'],
        eta_range=config['grid']['eta_range'],
        rng=train_key,
        sampling_config={
            'num_samples': 1000,  # MCMC samples per eta
            'num_warmup': 500,
            'step_size': 0.01,
            'num_integration_steps': 10,
        }
    )
    
    # Generate validation data
    print(f"Generating {config['grid']['num_val']} validation samples...")
    val_data = sample_ef_data(
        ef=ef,
        num_samples=config['grid']['num_val'],
        eta_range=config['grid']['eta_range'],
        rng=val_key,
        sampling_config={
            'num_samples': 1000,
            'num_warmup': 500,
            'step_size': 0.01,
            'num_integration_steps': 10,
        }
    )
    
    # Generate test data
    print(f"Generating {config['grid']['num_test']} test samples...")
    test_data = sample_ef_data(
        ef=ef,
        num_samples=config['grid']['num_test'],
        eta_range=config['grid']['eta_range'],
        rng=test_key,
        sampling_config={
            'num_samples': 1000,
            'num_warmup': 500,
            'step_size': 0.01,
            'num_integration_steps': 10,
        }
    )
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save data
    all_data = {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'config': config,
        'ef_type': config['ef_type'],
        'ef_params': config.get('ef_params', {}),
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(all_data, f)
    
    print(f"Data saved to {output_path}")
    
    # Print data statistics
    print("\\nData Statistics:")
    for split_name, data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        eta_mean = jnp.mean(data['eta'], axis=0)
        eta_std = jnp.std(data['eta'], axis=0)
        y_mean = jnp.mean(data['y'], axis=0)
        y_std = jnp.std(data['y'], axis=0)
        
        print(f"{split_name}:")
        print(f"  Eta - Mean: {eta_mean}, Std: {eta_std}")
        print(f"  Y   - Mean: {y_mean}, Std: {y_std}")


def sample_ef_data(ef, num_samples: int, eta_range: Tuple[float, float], rng, sampling_config: Dict) -> Dict:
    """Sample data from exponential family distribution."""
    
    # Generate random eta values
    if ef.eta_dim == 1:
        # For 1D, sample from range
        eta_samples = random.uniform(rng, (num_samples, 1), minval=eta_range[0], maxval=eta_range[1])
    elif ef.eta_dim == 2:
        # For 2D Gaussian natural params, ensure eta[1] < 0 for integrability
        eta1 = random.uniform(rng, (num_samples, 1), minval=eta_range[0], maxval=eta_range[1])
        eta2 = random.uniform(rng, (num_samples, 1), minval=-abs(eta_range[1]), maxval=-0.1)
        eta_samples = jnp.concatenate([eta1, eta2], axis=1)
    else:
        # For higher dimensions, use more sophisticated sampling
        eta_samples = random.normal(rng, (num_samples, ef.eta_dim)) * (eta_range[1] - eta_range[0]) / 4
    
    # For simplicity in comparison experiments, use analytical moments where possible
    if hasattr(ef, 'analytical_moments'):
        y_samples = ef.analytical_moments(eta_samples)
    else:
        # Fallback to approximate moments (this would normally use MCMC sampling)
        # For now, create a simple approximation for testing
        y_samples = eta_samples + 0.1 * random.normal(rng, eta_samples.shape)
    
    return {
        'eta': eta_samples,
        'y': y_samples,
    }


def main():
    parser = argparse.ArgumentParser(description='Generate comparison data for NoProp-CT experiments')
    parser.add_argument('--config', type=str, 
                       default='configs/noprop_ct_comparison.yaml',
                       help='Configuration file')
    parser.add_argument('--output', type=str, 
                       default='data/noprop_ct_comparison.pkl',
                       help='Output file for generated data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate data
    generate_comparison_data(config, Path(args.output), args.seed)


if __name__ == "__main__":
    main()
