from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp


Array = jax.Array


def sufficient_statistic_poly1d(x: Array) -> Array:
    """
    T(x) for a 1D polynomial EF up to degree 2: [x, x^2].
    Returns shape (..., 2).
    """
    x = jnp.asarray(x)
    t1 = x
    t2 = x ** 2
    return jnp.stack([t1, t2], axis=-1)


@dataclass
class PolynomialEF1D:
    """
    A simple univariate exponential family with T(x) = [x, x^2].

    log p(x | eta) ∝ eta1 * x + eta2 * x^2  (base measure constant 0)

    Integrability requires eta2 < 0.
    """

    # No parameters beyond the choice of T for now
    def log_unnormalized(self, x: Array, eta: Array) -> Array:
        t = sufficient_statistic_poly1d(x)
        return jnp.sum(t * eta, axis=-1)


def log_unnormalized_density(x: Array, eta: Array) -> Array:
    """Convenience: log p~(x|eta) for the PolynomialEF1D."""
    return PolynomialEF1D().log_unnormalized(x, eta)


def make_logdensity_fn(eta: Array) -> Callable[[Array], Array]:
    """
    Returns a function f(x) = log p~(x|eta) for HMC.

    eta: shape (2,) for [eta1, eta2]. Caller must ensure eta[1] < 0.
    """
    eta = jnp.asarray(eta)

    def _fn(x: Array) -> Array:
        return log_unnormalized_density(x, eta)

    return _fn


def sampleable_eta_bounds() -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Recommended default bounds for eta1 and eta2.
    Chosen to avoid extremely peaked/flat distributions: eta2 in [-2.0, -0.05].
    """
    return (-2.0, 2.0), (-2.0, -0.05)


