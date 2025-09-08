from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import blackjax
from jax import random


Array = jax.Array


def run_hmc(
    logdensity_fn: Callable[[Array], Array],
    num_samples: int,
    num_warmup: int,
    step_size: float,
    num_integration_steps: int,
    initial_position: Array,
    seed: int = 0,
) -> Array:
    """Run HMC with BlackJAX for a vectorized position.

    initial_position: shape (D,) for flattened x.
    Returns an array of shape (num_samples, D) of draws after warmup.
    """
    key = random.PRNGKey(seed)
    dim = jnp.size(initial_position)
    hmc = blackjax.hmc(logdensity_fn, step_size=step_size, inverse_mass_matrix=jnp.ones((dim,)))
    initial_state = hmc.init(jnp.atleast_1d(initial_position))

    def one_step(state, k):
        k1, _ = random.split(k)
        state, _ = hmc.step(k1, state, num_integration_steps)
        return state, state.position

    # Warmup (simple placeholder; could use blackjax.window_adaptation)
    state = initial_state
    keys = random.split(key, num_warmup)
    for k in keys:
        state, _ = one_step(state, k)

    # Sampling
    keys = random.split(key, num_samples)
    state, positions = jax.lax.scan(one_step, state, keys)
    return jnp.asarray(positions)


