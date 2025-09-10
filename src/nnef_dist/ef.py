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

    def compute_stats(self, x: Array) -> Dict[str, Array]:
        """Return dict of sufficient statistics evaluated at x.

        Each value has trailing shape equal to stat_specs()[name]."""
        raise NotImplementedError

    # ----- Generic implementations -----
    def stat_names(self) -> List[str]:
        return list(self.stat_specs.keys())

    def flatten_stats_or_eta(self, stats: Dict[str, Array]) -> Array:
        pieces = []
        for name in self.stat_names():
            v = jnp.asarray(stats[name])
            pieces.append(jnp.reshape(v, (-1,)))
        return jnp.concatenate(pieces) if pieces else jnp.zeros((0,))

    def unflatten_stats_or_eta(self, eta: Union[Array, Dict[str, Array]]) -> Dict[str, Array]:
        if isinstance(eta, dict):
            return eta
        vec = jnp.asarray(eta)
        result: Dict[str, Array] = {}
        idx = 0
        for name, shape in self.stat_specs.items():
            size = int(jnp.prod(jnp.array(shape))) if len(shape) > 0 else 1
            sl = vec[idx : idx + size]
            result[name] = jnp.reshape(sl, shape)
            idx += size
        return result

    def log_unnormalized(self, x: Array, eta: Union[Array, Dict[str, Array]]) -> Array:
        stats = self.compute_stats(x)
        eta_dict = self.unflatten_stats_or_eta(eta)  # does nothing if eta is a dict
        logp = 0.0
        for name, shape in self.stat_specs.items():
            t_val = stats[name]
            e_val = eta_dict[name]
            
            assert t_val.shape[-len(shape):] == shape, f"Shape mismatch for {name}: {t_val.shape} != {shape}"
            assert e_val.shape[-len(shape):] == shape, f"Shape mismatch for {name}: {e_val.shape} != {shape}"
            num_axes = len(shape)
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
        return MultivariateNormal(x_shape=x_shape)
    raise ValueError(f"Unknown EF name: {name}")

@dataclass(frozen=True)
class GaussianNatural1D(ExponentialFamily):
    """
    Univariate Gaussian in natural parameterization with T(x) = [x, x^2].
    log p(x | eta) ∝ eta1 * x + eta2 * x^2  (base measure constant 0)
    Integrability requires eta2 < 0.
    """
    x_shape: Tuple[int, ...] = ()

    @cached_property
    def stat_specs(self) -> Dict[str, Tuple[int, ...]]:
        return {"x": (), "x2": ()}

    def compute_stats(self, x: Array) -> Dict[str, Array]:
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

    def compute_stats(self, x: Array) -> Dict[str, Array]:
        return {"x": x, "xxT": x[...,None] * x[...,None,:]}

    def compute_expected_stats(self, x: Array, sample_axis: Tuple[int, ...] = None) -> Dict[str, Array]:
        if sample_axis is None: 
            sample_axis = tuple(range(0,x.ndim - 1))        
        return {"x": jnp.mean(x, axis=sample_axis), "xxT": jnp.mean(x[...,None] * x[...,None,:], axis=sample_axis)}


    
