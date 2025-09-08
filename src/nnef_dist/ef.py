from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import jax
import jax.numpy as jnp


Array = jax.Array


class ExponentialFamily:
    """Base interface for exponential-family distributions.

    Implementations must define immutable `x_shape` and `t_shape` (shape of T(x)).
    The natural parameter `eta` must have the same shape as `t_shape`.
    """

    # Immutable shapes (tuples). Examples: () for scalar, (d,), (h, w, c), etc.
    x_shape: Tuple[int, ...]
    t_shape: Tuple[int, ...]

    def log_unnormalized(self, x: Array, eta: Array) -> Array:
        """Returns log p~(x | eta). Shape: broadcastable to (...,).

        x has shape `x_shape` (or batch dims + x_shape). eta has shape `t_shape` (or batch dims + t_shape).
        """
        raise NotImplementedError

    def sufficient_statistic(self, x: Array) -> Array:
        """Returns T(x) with shape `t_shape` (or batch dims + t_shape)."""
        raise NotImplementedError

    def make_logdensity_fn(self, eta: Array) -> Callable[[Array], Array]:
        """Returns a function f(x_flat) expecting flattened `x` of length `x_dim`.

        The function reshapes the flat vector back to `x_shape` and evaluates log p~.
        """
        eta = jnp.asarray(eta)

        def _fn(x_flat: Array) -> Array:
            x = jnp.reshape(x_flat, self.x_shape)
            return self.log_unnormalized(x, eta)

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
    def t_dim(self) -> int:
        size = 1
        for n in self.t_shape:
            size *= n
        return size

    @property
    def eta_shape(self) -> Tuple[int, ...]:
        return self.t_shape


@dataclass(frozen=True)
class GaussianNatural1D(ExponentialFamily):
    """
    Univariate Gaussian in natural parameterization with T(x) = [x, x^2].
    log p(x | eta) ∝ eta1 * x + eta2 * x^2  (base measure constant 0)
    Integrability requires eta2 < 0.
    """

    x_shape: Tuple[int, ...] = ()
    t_shape: Tuple[int, ...] = (2,)

    def sufficient_statistic(self, x: Array) -> Array:
        x = jnp.asarray(x)
        t1 = x
        t2 = x ** 2
        return jnp.stack([t1, t2], axis=-1)

    def log_unnormalized(self, x: Array, eta: Array) -> Array:
        t = self.sufficient_statistic(x)
        return jnp.sum(t * eta, axis=-1)

    def recommended_eta_ranges(self) -> List[Tuple[float, float]]:
        return [(-2.0, 2.0), (-1.5, -0.1)]


def ef_factory(name: str) -> ExponentialFamily:
    n = name.lower()
    if n in {"gaussian_1d", "gauss1d", "gaussian"}:
        return GaussianNatural1D()
    raise ValueError(f"Unknown EF name: {name}")


