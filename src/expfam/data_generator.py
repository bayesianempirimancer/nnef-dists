#!/usr/bin/env python
"""
Clean data generator class for exponential family distributions.

This module provides a DataGenerator class that can sample from and compute
expectations for exponential family distributions using MCMC sampling.
"""

from __future__ import annotations

import time
from typing import Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
from jax import random

try:
    import blackjax
    BLACKJAX_AVAILABLE = True
except ImportError:
    BLACKJAX_AVAILABLE = False
    blackjax = None

from .ef import ExponentialFamily

Array = jax.Array


@dataclass
class SamplingConfig:
    """Configuration for MCMC sampling."""
    num_samples: int = 1000
    num_warmup: int = 500
    step_size: float = 0.1
    num_integration_steps: int = 10
    num_chains: int = 5
    parallel_strategy: str = "vmap"  # "vmap", "pmap", "sequential"
    seed: int = 42


@dataclass
class ConstraintConfig:
    """Configuration for constraints on eta and x."""
    eta_bounds: Optional[Tuple[Array, Array]] = None  # (lower, upper) bounds for eta
    x_bounds: Optional[Tuple[Array, Array]] = None    # (lower, upper) bounds for x
    eta_constraint_fn: Optional[Callable[[Array], bool]] = None
    x_constraint_fn: Optional[Callable[[Array], bool]] = None


class DataGenerator:
    """
    Clean data generator for exponential family distributions.
    
    This class provides methods to sample from and compute expectations
    for exponential family distributions using MCMC sampling with
    configurable constraints and parallelization strategies.
    """
    
    def __init__(
        self,
        ef_distribution: ExponentialFamily,
        sampling_config: Optional[SamplingConfig] = None
    ):
        """
        Initialize the data generator.
        
        Args:
            ef_distribution: Exponential family distribution instance
            sampling_config: Configuration for MCMC sampling
            
        Raises:
            ImportError: If blackjax is not available
        """
        if not BLACKJAX_AVAILABLE:
            raise ImportError(
                "blackjax is required for DataGenerator. "
                "Please install it with: pip install blackjax"
            )
        
        self.sampling_config = sampling_config or SamplingConfig()
        self.ef = ef_distribution
        # Default constraints: use ef-provided bounds if present
        try:
            ef_eta_bounds = getattr(self.ef, 'eta_bounds', None)
            if callable(ef_eta_bounds):
                ef_eta_bounds = ef_eta_bounds()
        except Exception:
            ef_eta_bounds = None
        self.constraint_config = ConstraintConfig(
            eta_bounds=ef_eta_bounds,
            x_bounds=getattr(self.ef, 'x_bounds', None),
            x_constraint_fn=getattr(self.ef, 'x_constraint_fn', None)
        )
    
    def _create_log_density_fn(self, eta: Array) -> Callable[[Array], Array]:
        """Create log density function for the exponential family with given eta."""
        # Use the existing flattened_log_density_fn method from the exponential family
        # The EF will use its own x_bounds and x_constraint_fn properties
        return self.ef.flattened_log_density_fn(eta)
    
    def _apply_eta_constraints(self, eta: Array) -> Array:
        """Apply constraints to eta parameters."""
        # Skip eta constraints since we use valid_eta_sampler to generate valid eta values
        return eta
        
        # Apply eta bounds if any
        if self.constraint_config.eta_bounds is not None:
            bounds = self.constraint_config.eta_bounds
            if callable(bounds):
                bounds = bounds()
            
            # Handle dictionary-based bounds
            if isinstance(bounds, dict):
                # Convert dictionary bounds to per-dimension bounds
                stat_names = list(bounds.keys())
                lower_bounds = []
                upper_bounds = []
                
                # eta has shape (eta_dim,) - iterate over feature dimensions
                for i in range(eta.shape[0]):
                    if i < len(stat_names):
                        stat_name = stat_names[i]
                        if stat_name in bounds:
                            lower, upper = bounds[stat_name]
                        else:
                            lower, upper = -jnp.inf, jnp.inf
                    else:
                        lower, upper = -jnp.inf, jnp.inf
                    lower_bounds.append(lower)
                    upper_bounds.append(upper)
                
                lower = jnp.array(lower_bounds)
                upper = jnp.array(upper_bounds)
            else:
                # Handle tuple-based bounds (legacy)
                lower, upper = bounds
                # Flatten dict-shaped bounds if needed
                if isinstance(lower, dict):
                    lower = self.ef.flatten_stats_or_eta(lower)
                if isinstance(upper, dict):
                    upper = self.ef.flatten_stats_or_eta(upper)
            
            eta = jnp.clip(eta, lower, upper)
        
        # Apply eta constraint function if any
        if self.constraint_config.eta_constraint_fn is not None:
            # Note: This is a simple implementation - in practice you might want
            # to project eta to satisfy constraints rather than just clipping
            pass
        
        return eta
    
    def _sample_multiple_chains_impl(
        self, 
        eta: Array, 
        key: Array
    ) -> Array:
        """Sample from multiple chains for given eta."""
        # Create log density function with current eta
        log_density_fn = self._create_log_density_fn(eta)
        
        # Generate initial positions for each chain
        chain_keys = random.split(key, self.sampling_config.num_chains)
        initial_positions = jax.vmap(
            lambda k: random.normal(k, (self.ef.x_dim,))
        )(chain_keys)
        
        # Run HMC sampling for each chain
        chain_samples = jax.vmap(
            lambda init_pos, k: self._run_hmc(
                log_density_fn=log_density_fn,
                initial_position=init_pos,
                key=k
            )
        )(initial_positions, chain_keys)

        # Debug: confirm shapes
        print(f"[DEBUG] chain_samples shape: {chain_samples.shape} -> (num_chains, num_samples, x_dim)")
        
        # Combine samples from all chains: (num_chains, num_samples, x_dim) -> (num_chains * num_samples, x_dim)
        return chain_samples.reshape(-1, self.ef.x_dim)
    
    def _run_hmc(
        self, 
        log_density_fn: Callable[[Array], Array], 
        initial_position: Array, 
        key: Array
    ) -> Array:
        """Run HMC sampling using BlackJAX."""
        dim = jnp.size(initial_position)
        
        # Initialize HMC
        hmc = blackjax.hmc(
            logdensity_fn=log_density_fn,
            step_size=self.sampling_config.step_size,
            inverse_mass_matrix=jnp.ones((dim,)),
            num_integration_steps=self.sampling_config.num_integration_steps
        )
        
        initial_state = hmc.init(jnp.atleast_1d(initial_position))
        
        @jax.jit
        def one_step(state, k):
            k1, _ = random.split(k)
            state, _ = hmc.step(k1, state)
            return state, state.position
        
        # Warmup (scan for speed)
        keys = random.split(key, self.sampling_config.num_warmup)
        state, _ = jax.lax.scan(lambda st, kk: (one_step(st, kk)[0], None), initial_state, keys)
        
        # Sampling
        keys = random.split(key, self.sampling_config.num_samples)
        state, positions = jax.lax.scan(one_step, state, keys)
        
        return jnp.asarray(positions)
    
    def _compute_expectations_single_impl(
        self, 
        eta: Array, 
        samples: Array
    ) -> Tuple[Array, Array, Array]:
        """Compute expectations for a single eta using samples."""
        # Compute sufficient statistics for all samples; then flatten
        stats = jax.vmap(lambda x: self.ef.compute_stats(x.reshape(self.ef.x_shape), flatten=False))(samples)
        flat_stats = jax.vmap(self.ef.flatten_stats_or_eta)(stats)  # (N, S)
        
        # Expected sufficient statistics (mean, flattened)
        mu_T = jnp.mean(flat_stats, axis=0)  # (S,)
        
        # Covariance matrix over flattened stats
        centered = flat_stats - mu_T[None, :]
        n = centered.shape[0]
        cov_TT = (centered.T @ centered) / jnp.maximum(n - 1, 1)
        
        # Compute effective sample size (simplified)
        ess = self._compute_ess(samples)
        
        return mu_T, cov_TT, ess
    
    def _compute_ess(self, samples: Array) -> Array:
        """Compute effective sample size (simplified implementation)."""
        # This is a simplified ESS computation
        # In practice, you might want to use more sophisticated methods
        n_samples = samples.shape[0]
        
        # Simple autocorrelation-based ESS
        if n_samples < 10:
            return jnp.asarray(n_samples, dtype=jnp.float32)
        
        # Compute lag-1 autocorrelation
        centered = samples - jnp.mean(samples, axis=0)
        autocorr = jnp.sum(centered[:-1] * centered[1:]) / jnp.sum(centered * centered)
        
        # ESS formula
        ess = n_samples / (1 + 2 * autocorr)
        return jnp.clip(ess, 1.0, n_samples).astype(jnp.float32)
    
    def sample(self, eta_batch: Array) -> Array:
        """
        Sample from the exponential family distribution for a batch of eta values.
        
        Args:
            eta_batch: Array of shape (batch_size, eta_dim) containing natural parameters
            
        Returns:
            Array of shape (batch_size, num_chains * num_samples, x_dim) containing samples
        """
        batch_size = eta_batch.shape[0]
        
        # Apply eta constraints
        eta_batch = jax.vmap(self._apply_eta_constraints)(eta_batch)
        
        # Generate random keys for each eta
        key = random.PRNGKey(self.sampling_config.seed)
        keys = random.split(key, batch_size)
        
        if self.sampling_config.parallel_strategy == "vmap":
            # Vectorized sampling across batch
            samples = jax.vmap(
                lambda eta, k: self._sample_multiple_chains_impl(eta, k)
            )(eta_batch, keys)
            
        elif self.sampling_config.parallel_strategy == "pmap":
            # Multi-device parallel sampling
            samples = jax.pmap(
                lambda eta, k: self._sample_multiple_chains_impl(eta, k)
            )(eta_batch, keys)
            
        else:  # sequential
            # Sequential sampling
            samples = []
            for i in range(batch_size):
                sample = self._sample_multiple_chains_impl(eta_batch[i], keys[i])
                samples.append(sample)
            samples = jnp.stack(samples)
        
        return samples
    
    def compute_expectations(self, eta_batch: Array) -> Tuple[Array, Array, Array, Array]:
        """
        Compute expected sufficient statistics for a batch of eta values.
        
        Args:
            eta_batch: Array of shape (batch_size, eta_dim) containing natural parameters
            
        Returns:
            Tuple of (eta, mu_T, cov_TT, ess) where:
            - eta: Input eta values (shape: batch_size, eta_dim)
            - mu_T: Expected sufficient statistics (shape: batch_size, mu_dim)
            - cov_TT: Covariance matrices (shape: batch_size, mu_dim, mu_dim)
            - ess: Effective sample sizes (shape: batch_size,)
            
        Note:
            Uses num_chains * num_samples total samples per eta for better estimates
        """
        # First sample from the distribution
        samples = self.sample(eta_batch)
        
        # Compute expectations for each eta
        if self.sampling_config.parallel_strategy in ["vmap", "pmap"]:
            # Vectorized computation
            mu_T, cov_TT, ess = jax.vmap(self._compute_expectations_single_impl)(
                eta_batch, samples
            )
        else:
            # Sequential computation
            results = []
            for i in range(eta_batch.shape[0]):
                mu_T_i, cov_TT_i, ess_i = self._compute_expectations_single_impl(
                    eta_batch[i], samples[i]
                )
                results.append((mu_T_i, cov_TT_i, ess_i))
            
            mu_T = jnp.stack([r[0] for r in results])
            cov_TT = jnp.stack([r[1] for r in results])
            ess = jnp.stack([r[2] for r in results])
        
        # Apply reparameterization to eta before returning
        eta_reparam = self.ef.reparam_eta(eta_batch, flatten=True)
        
        return eta_reparam, mu_T, cov_TT, ess
    
    def generate_data(
        self, 
        eta_batch: Array,
        output_format: str = "dict"
    ) -> Union[Dict[str, Array], Tuple[Array, ...]]:
        """
        Generate complete data for a batch of eta values.
        
        Args:
            eta_batch: Array of shape (batch_size, eta_dim) containing natural parameters
            output_format: Output format - "dict" or "tuple"
            
        Returns:
            If output_format="dict": Dict with keys 'eta', 'mu_T', 'cov_TT', 'ess'
            If output_format="tuple": Tuple of (eta, mu_T, cov_TT, ess)
        """
        eta, mu_T, cov_TT, ess = self.compute_expectations(eta_batch)
        
        if output_format == "dict":
            return {
                'eta': eta,
                'mu_T': mu_T,
                'cov_TT': cov_TT,
                'ess': ess
            }
        else:
            return eta, mu_T, cov_TT, ess
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the data generator."""
        return {
            'ef_distribution': self.ef.name,
            'x_shape': self.ef.x_shape,
            'x_dim': self.ef.x_dim,
            'eta_dim': self.ef.eta_dim,
            'stat_names': self.ef.stat_names,
            'sampling_config': {
                'num_samples': self.sampling_config.num_samples,
                'num_warmup': self.sampling_config.num_warmup,
                'step_size': self.sampling_config.step_size,
                'num_integration_steps': self.sampling_config.num_integration_steps,
                'num_chains': self.sampling_config.num_chains,
                'parallel_strategy': self.sampling_config.parallel_strategy,
                'seed': self.sampling_config.seed
            },
            'constraint_config': {
                'has_eta_bounds': self.constraint_config.eta_bounds is not None,
                'has_x_bounds': self.constraint_config.x_bounds is not None,
                'has_eta_constraint_fn': self.constraint_config.eta_constraint_fn is not None,
                'has_x_constraint_fn': self.constraint_config.x_constraint_fn is not None
            }
        }
