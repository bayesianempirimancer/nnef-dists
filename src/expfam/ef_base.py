from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union, Optional
from functools import cached_property

from flax.typing import T
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

Array = jax.Array


class ExponentialFamily:
    """Base interface for exponential-family distributions using dict-based stats.

    Implementations define immutable `x_shape` and a specification of sufficient
    statistics via `stat_specs()` mapping stat name -> stat tensor shape. Natural
    parameters `eta` can be provided as a matching dict
    """

    x_shape: Tuple[int, ...]
    name: str

    # ----- Required subclass API -----

    @cached_property
    def stat_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Return mapping from stat name -> shape (no batch dims)."""
        raise NotImplementedError
        
    def _compute_stats(self, x: Array) -> Dict[str, Array]:
        raise NotImplementedError

    # ----- Optional subclass API -----
    def reparam_eta(self, eta: Union[Dict[str, Array], Array], flatten: bool = False) -> Union[Dict[str, Array], Array]:
        """
        Apply reparameterization to eta values.
        
        This method allows distributions to transform eta values to ensure they satisfy
        constraints (e.g., negative definiteness for multivariate Gaussians).
        
        Args:
            eta: Natural parameters as either a dictionary or flattened array
            flatten: If True, return flattened array; if False, return dictionary (default)
            
        Returns:
            Reparameterized eta in the same format as input, unless flatten is specified
        """
        # Handle input format
        input_is_array = isinstance(eta, (jnp.ndarray, Array))
        
        if input_is_array:
            # Convert flattened array to dict
            eta_dict = self.unflatten_stats_or_eta(eta)
        else:
            eta_dict = eta
        
        # Apply reparameterization (default: no-op)
        eta_reparam_dict = self._reparam_eta(eta_dict)
        
        # Handle output format
        if flatten:
            return self.flatten_stats_or_eta(eta_reparam_dict)
        else:
            return eta_reparam_dict
    
    def _reparam_eta(self, eta: Dict[str,Array]) -> Dict[str,Array]:
        """Internal reparameterization method - override in subclasses as needed."""
        return eta

    # ----- Properties -----
    @property
    def x_bounds(self) -> Optional[Tuple[Array, Array]]:
        """Default: no bounds for x values"""
        return None
    
    @property
    def x_constraint_fn(self) -> Optional[Callable[[Array], bool]]:
        """Default: no constraints on x values"""
        return None
    
    def eta_bounds(self) -> Optional[Dict[str, Tuple[float, float]]]:
        """Return bounds for each eta dimension as a dictionary.
        
        Returns:
            Dict mapping dimension names to (lower_bound, upper_bound) tuples.
            Use -jnp.inf for unbounded lower, jnp.inf for unbounded upper.
            Return None if no constraints.
        """
        return None

    @property
    def eta_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return self.stat_shapes

    @property
    def stat_names(self) -> List[str]:
        return list(self.stat_shapes.keys())

    @cached_property
    def x_dim(self) -> int:
        size = 1
        for n in self.x_shape:
            size *= n
        return size

    @cached_property
    def eta_dim(self) -> int:
        total = 0
        for shape in self.stat_shapes.values():
            size = 1
            for n in shape:
                size *= n
            total += size
        return total

    @property
    def stats_dim(self) -> int:
        return self.eta_dim

    @cached_property
    def stat_sizes(self) -> Dict[str, int]:
        """Pre-computed sizes of each statistic for efficient flattening/unflattening."""
        sizes = {}
        for name, shape in self.stat_shapes.items():
            if len(shape) > 0:
                size = 1
                for n in shape:
                    size *= n
                sizes[name] = size
            else:
                sizes[name] = 1
        return sizes

    def compute_stats(self, x: Array, flatten: bool = True) -> Dict[str, Array]:
        stats = self._compute_stats(x)
        if flatten:
            return self.flatten_stats_or_eta(stats)
        else:
            return stats

    def compute_expected_stats(self, x: Array, sample_axis: Tuple[int, ...] = None, flatten: bool = False) -> Dict[str, Array]:
        stats = self._compute_stats(x)
        if sample_axis is None: 
            sample_axis = tuple(range(x.ndim - len(self.x_shape)))
        
        # Compute expected stats for each stat component separately
        expected_stats = {}
        for name, stat_val in stats.items():
            expected_stats[name] = jnp.mean(stat_val, axis=sample_axis)
        
        if flatten: 
            return self.flatten_stats_or_eta(expected_stats)
        else:
            return expected_stats        

    def flatten_stats_or_eta(self, stats: Dict[str, Array]) -> Array:
        if isinstance(stats, Array):
            return stats
        
        # Use cached stat sizes for efficiency
        pieces = []
        for name in self.stat_names:
            stat_size = self.stat_sizes[name]
            v = jnp.asarray(stats[name])
            stat_shape = self.stat_shapes[name]
            
            if len(stat_shape) > 0:
                # Preserve batch dimensions, flatten only the stat dimensions
                batch_shape = v.shape[:-len(stat_shape)]
                pieces.append(jnp.reshape(v, batch_shape + (stat_size,)))
            else:
                # Scalar stat
                pieces.append(jnp.reshape(v, v.shape[:-1] + (1,)))
        
        return jnp.concatenate(pieces, axis=-1) if pieces else jnp.zeros((0,))

    def unflatten_stats_or_eta(self, eta: Union[Array, Dict[str, Array]]) -> Dict[str, Array]:
        if isinstance(eta, dict):
            return eta
        vec = jnp.asarray(eta)
        batch_shape = vec.shape[:-1]
        result: Dict[str, Array] = {}
        idx = 0
        for name in self.stat_names:
            shape = self.stat_shapes[name]
            size = self.stat_sizes[name]
            sl = vec[...,idx : idx + size]
            result[name] = jnp.reshape(sl, batch_shape + shape)
            idx += size
        return result

    def log_unnormalized(self, x: Array, eta: Union[Array, Dict[str, Array]], reparameterize: bool = True) -> Array:
        # Compute the base log probability
        eta = self.unflatten_stats_or_eta(eta)
        if reparameterize:
            eta = self._reparam_eta(eta)
        stats = self.compute_stats(x, flatten=False)
        logp = 0.0
        for name, shape in self.stat_shapes.items():
            num_axes = len(shape)
            axes = () if num_axes == 0 else tuple(range(-num_axes, 0))
            logp += jnp.sum(stats[name] *  eta[name], axis=axes)
        base_logp = logp
        
        # Apply x constraint function if defined as class property
        if self.x_constraint_fn is not None:
            # Use JAX-compatible constraint checking
            # x_constraint_fn should return boolean array with same batch shape as x
            constraint_satisfied = self.x_constraint_fn(x)
            
            # If constraint_satisfied is an array, we need to reduce over the x_shape dimensions
            # but preserve batch dimensions
            if constraint_satisfied.ndim > 0:
                # Reduce over the last len(x_shape) dimensions (the x_shape dimensions)
                # This preserves batch dimensions but reduces over the actual x elements
                reduce_axes = tuple(range(-len(self.x_shape), 0)) if len(self.x_shape) > 0 else ()
                constraint_satisfied = jnp.all(constraint_satisfied, axis=reduce_axes)
            
            constraint_violated = jnp.logical_not(constraint_satisfied)
            # Use jnp.where for JAX-compatible conditional
            base_logp = jnp.where(constraint_violated, -jnp.inf, base_logp)
        
        # Apply x bounds if defined as class property
        if self.x_bounds is not None:
            lower, upper = self.x_bounds
            # Use JAX-compatible bounds checking
            # For bounds, we need to check element-wise and then reduce over x_shape dimensions
            bounds_violated = jnp.any(x < lower, axis=-1) | jnp.any(x > upper, axis=-1)
            # Use jnp.where for JAX-compatible conditional
            base_logp = jnp.where(bounds_violated, -jnp.inf, base_logp)
        
        return base_logp

    def flattened_log_density_fn(self, eta: Union[Array, Dict[str, Array]], reparameterize: bool = True) -> Callable[[Array], Array]:
        """Returns a function f(x_flat) expecting flattened `x` of length `x_dim`.

        The function reshapes the flat vector back to `x_shape` and evaluates log p~.
        Works with both single flattened x and batches of flattened x's.
        Constraints are applied in the log_unnormalized function.
        
        Args:
            eta: Natural parameters
            reparameterize: Whether to apply reparameterization to eta (default: True)
        """

        def _fn(x_flat: Array) -> Array:
            # Handle both single flattened x and batches of flattened x's
            batch_shape = x_flat.shape[:-1] if x_flat.ndim > 0 else ()
            x = jnp.reshape(x_flat, batch_shape + self.x_shape)
            
            # Constraints are now handled in log_unnormalized
            return self.log_unnormalized(x, eta, reparameterize=reparameterize)

        return _fn

    def valid_eta_sampler(self, 
        key: jr.PRNGKey,
        input_bounds: Dict[str, Tuple[float, float]],
        num_points: int, 
        sample_spread = 'log') -> Array:

        """Sample valid eta values for this exponential family.
        
        Args:
            num_points: Number of eta samples to generate
            key: JAX random key for sampling
            input_bounds: Optional additional constraints on eta values.
                         Dict mapping dimension names to (lower, upper) bounds.
                         These will be intersected with eta_bounds().
        
        Returns:
            Array of shape (num_points, eta_dim) containing valid eta samples
        """
        if input_bounds.keys() != self.stat_shapes.keys():
            raise ValueError(f"Input bounds must match the distribution's stat_shapes. Keys: {input_bounds.keys()}, Stat shapes: {self.stat_shapes.keys()}")

        # Use input bounds directly (they should already be properly constrained by the config)
        # If intersection is needed, it should be done at the config level, not here
        final_bounds = input_bounds.copy()

        eta = {}
        for name, (lower, upper) in final_bounds.items():
            if sample_spread == 'uniform':
                key, subkey = jr.split(key)
                eta[name] = jr.uniform(subkey, (num_points,) + self.stat_shapes[name], minval=lower, maxval=upper)
            elif sample_spread == 'log':
                key, subkey = jr.split(key)
                if upper > 0 and lower > 0:
                    eta[name] = jnp.exp(jr.uniform(subkey, (num_points,) + self.stat_shapes[name], minval=jnp.log(lower), maxval=jnp.log(upper)))
                elif upper < 0 and lower < 0: 
                    eta[name] = -jnp.exp(jr.uniform(subkey, (num_points,) + self.stat_shapes[name], minval=jnp.log(-upper), maxval=jnp.log(-lower)))
                else: # log makes no sense using uniform
                    eta[name] = jr.uniform(subkey, (num_points,) + self.stat_shapes[name], minval=lower, maxval=upper)

        return self.flatten_stats_or_eta(eta)


