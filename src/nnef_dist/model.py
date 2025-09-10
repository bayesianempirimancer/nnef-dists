from __future__ import annotations

from typing import Sequence, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp


Array = jax.Array


def get_activation(name: str) -> Callable[[Array], Array]:
    if name == "relu":
        return nn.relu
    if name == "gelu":
        return nn.gelu
    if name == "tanh":
        return nn.tanh
    if name == "swish":
        return nn.swish
    raise ValueError(f"Unknown activation: {name}")


class MomentMLP(nn.Module):
    hidden_sizes: Sequence[int]
    activation: str = "swish"
    output_dim: int = 2  # flattened dim of E[T(x)]

    @nn.compact
    def __call__(self, eta: Array) -> Array:
        x = eta
        act = get_activation(self.activation)
        for size in self.hidden_sizes:
            x = nn.Dense(size)(x)
            x = act(x)
        x = nn.Dense(self.output_dim)(x)
        return x


