from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union
from functools import cached_property

import jax
import jax.numpy as jnp

Array = jax.Array


class ExponentialFamily:
    """Base interface for exponential-family distributions using dict-based stats.

    Implementations define immutable `x_shape` and a specification of sufficient
    statistics via `stat_specs()` mapping stat name -> stat tensor shape. Natural
    parameters `eta` can be provided as a matching dict
    """

    x_shape: Tuple[int, ...]

    # ----- Required subclass API -----
    @cached_property
    def stat_specs(self) -> Dict[str, Tuple[int, ...]]:
        """Return mapping from stat name -> shape (no batch dims)."""
        raise NotImplementedError

    def stat_names(self) -> List[str]:
        return list(self.stat_specs.keys())

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

    def _compute_stats(self, x: Array) -> Dict[str, Array]:
        raise NotImplementedError

    def flatten_stats_or_eta(self, stats: Dict[str, Array]) -> Array:
        pieces = []
        for name in self.stat_names():
            v = jnp.asarray(stats[name])
            # Preserve batch dimensions, flatten only the stat dimensions
            stat_shape = self.stat_specs[name]
            stat_size = int(jnp.prod(jnp.array(stat_shape))) if len(stat_shape) > 0 else 1
            batch_shape = v.shape[:-len(stat_shape)] if len(stat_shape) > 0 else v.shape[:-1]
            pieces.append(jnp.reshape(v, batch_shape + (stat_size,)))
        if pieces:
            return jnp.concatenate(pieces, axis=-1)
        else:
            return jnp.zeros((0,))

    def unflatten_stats_or_eta(self, eta: Union[Array, Dict[str, Array]]) -> Dict[str, Array]:
        if isinstance(eta, dict):
            return eta
        vec = jnp.asarray(eta)
        batch_shape = vec.shape[:-1]
        result: Dict[str, Array] = {}
        idx = 0
        for name, shape in self.stat_specs.items():
            size = int(jnp.prod(jnp.array(shape))) if len(shape) > 0 else 1
            sl = vec[...,idx : idx + size]
            result[name] = jnp.reshape(sl, batch_shape + shape)
            idx += size
        return result

    def log_unnormalized(self, x: Array, eta: Union[Array, Dict[str, Array]]) -> Array:

        if isinstance(eta, Array):
            stats = self.compute_stats(x, flatten=True)
            assert stats.shape[-1] == self.eta_dim, f"Flattened shape mismatch: {stats.shape} != {self.eta_dim}"
            return jnp.sum(stats * eta, axis=-1)
        else:
            stats = self.compute_stats(x, flatten=False)
            logp = 0.0
            for name, shape in self.stat_specs.items():
                t_val = stats[name]
                e_val = eta[name]
                shape_axis = self.stat_specs[name]
                num_axes = len(shape)
                assert t_val.shape[-num_axes:] == shape, f"Shape mismatch for {name}: {t_val.shape} != {shape}"
                assert e_val.shape[-num_axes:] == shape, f"Shape mismatch for {name}: {e_val.shape} != {shape}"
                axes = () if num_axes == 0 else tuple(range(-num_axes, 0))
                logp += jnp.sum(t_val * e_val, axis=axes)
        return logp

    def make_logdensity_fn(self, eta: Union[Array, Dict[str, Array]]) -> Callable[[Array], Array]:
        """Returns a function f(x_flat) expecting flattened `x` of length `x_dim`.

        The function reshapes the flat vector back to `x_shape` and evaluates log p~.
        """
        eta_dict = self.unflatten_stats_or_eta(eta)

        def _fn(x_flat: Array) -> Array:
            x = jnp.reshape(x_flat, self.x_shape)
            return self.log_unnormalized(x, eta_dict)

        return _fn
    
    def default_initial_position(self) -> Array:
        """Default initial position for sampling: zeros of shape x_shape."""
        return jnp.zeros(self.x_shape)

    @property
    def x_dim(self) -> int:
        size = 1
        for n in self.x_shape:
            size *= n
        return size

    @property
    def eta_dim(self) -> int:
        total = 0
        for shape in self.stat_specs.values():
            size = 1
            for n in shape:
                size *= n
            total += size
        return total

    @property
    def eta_shape(self) -> Dict[str, Tuple[int, ...]]:
        return self.stat_specs

def ef_factory(name: str, **kwargs) -> ExponentialFamily:
    n = name.lower()
    if n in {"gaussian_1d", "gauss1d", "gaussian"}:
        return GaussianNatural1D()
    elif n in {"mv_normal", "multivariate_normal"}:
        x_shape = kwargs.get("x_shape", (2,))  # default 2D
        # Convert list to tuple if needed
        if isinstance(x_shape, list):
            x_shape = tuple(x_shape)
        return MultivariateNormal(x_shape=x_shape)
    elif n in {"mv_normal_tril", "multivariate_normal_tril"}:
        x_shape = kwargs.get("x_shape", (2,))  # default 2D
        # Convert list to tuple if needed
        if isinstance(x_shape, list):
            x_shape = tuple(x_shape)
        return MultivariateNormal_tril(x_shape=x_shape)
    raise ValueError(f"Unknown EF name: {name}")

@dataclass(frozen=True)
class GaussianNatural1D(ExponentialFamily):
    """
    Univariate Gaussian in natural parameterization with T(x) = [x, x^2].
    log p(x | eta) ∝ eta1 * x + eta2 * x^2  (base measure constant 0)
    Integrability requires eta2 < 0.
    """
    x_shape: Tuple[int, ...] = (1,)

    @cached_property
    def stat_specs(self) -> Dict[str, Tuple[int, ...]]:
        return {"x": (1,), "x2": (1,)}

    def _compute_stats(self, x: Array) -> Dict[str, Array]:
        return {"x": x, "x2": x ** 2}


@dataclass(frozen=True)
class MultivariateNormal(ExponentialFamily):
    """
    Multivariate normal in natural parameterization with T(x) = [x, x^T x].
    log p(x | eta) ∝ eta1 . x + eta2 . (x *x^T)
    Dictionary keys are sensibly named 'x' and 'xxT' and x is assumed to be in vector format, 

    Integrability requires eta2 to be negative definite.  
    """
    x_shape: Tuple[int,]

    @cached_property
    def stat_specs(self) -> Dict[str, Tuple[int, ...]]:
        return {"x": (self.x_shape[-1],), "xxT": (self.x_shape[-1], self.x_shape[-1])}

    def _compute_stats(self, x: Array) -> Dict[str, Array]:
        return {"x": x, "xxT": x[...,None] * x[...,None,:]}

@dataclass(frozen=True)
class MultivariateNormal_tril(MultivariateNormal):
    """
    Multivariate normal in natural parameterization with T(x) = [x, lower_triangular(xx^T)].
    This avoids overparameterization by storing only the lower triangular part of xx^T.
    Uses standard lower triangular convention compatible with JAX/NumPy.
    """
    x_shape: Tuple[int,]

    @cached_property
    def stat_specs(self) -> Dict[str, Tuple[int, ...]]:
        return {"x": (self.x_shape[-1],), "xxT_tril": (self.x_shape[-1]*(self.x_shape[-1]+1)//2,)}

    @cached_property
    def tril_mask(self) -> Array:
        n = self.x_shape[-1]
        return jnp.tril(jnp.ones((n,n), dtype=bool))
    
    @cached_property
    def tril_indices(self) -> Array:
        return jnp.where(self.tril_mask.flatten())[0]

    def flatten_LT(self, x: Array) -> Array:
        n = self.x_shape[-1]
        batch_shape = x.shape[:-2]
        assert x.shape[-2:] == (n, n), f"Shape mismatch: {x.shape[-2:]} != {n, n}"
        flat_x = x.reshape(-1, n * n)
        flat_x = flat_x[:,self.tril_indices]
        return flat_x.reshape(batch_shape + (n*(n+1)//2,))

    def unflatten_LT(self, x: Array) -> Array:
        n = self.x_shape[-1]
        expected_shape = n*(n+1)//2
        batch_shape = x.shape[:-1]
        assert expected_shape == x.shape[-1], f"Shape mismatch: Terminal shape of {x.shape} != {n*(n+1)//2}"

        flat_x = x.reshape(-1, expected_shape)
        batch_size = int(jnp.prod(jnp.array(batch_shape)))
        lt_mat_flat = jnp.zeros((batch_size, n*n))

        lt_mat_flat = lt_mat_flat.at[:,self.tril_indices].set(flat_x)
        return lt_mat_flat.reshape(batch_shape + (n,n))

    def _compute_stats(self, x: Array) -> Dict[str, Array]:
        xxT = x[...,None] * x[...,None,:]
        return {"x": x, "xxT_tril": self.flatten_LT(xxT)}

    def LTnat_to_standard_nat(self, eta: Dict[str, Array]) -> Dict[str, Array]:
        eta = self.unflatten_stats_or_eta(eta)
        xxT = self.unflatten_LT(eta["xxT_tril"])
        xxT = xxT + xxT.T - jnp.diag(jnp.diag(xxT))
        return {"x": eta["x"], "xxT": xxT}

    def standard_nat_to_LTnat(self, eta: Dict[str, Array]) -> Dict[str, Array]:
        xxT = 2*eta["xxT"] - jnp.diag(jnp.diag(2*eta["xxT"]))
        xxT = xxT*self.tril_mask
        return {"x": eta["x"], "xxT_tril": self.flatten_LT(eta["xxT"])}

    
