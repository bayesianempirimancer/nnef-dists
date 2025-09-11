from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Tuple, Callable

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.core import FrozenDict
from jax import random

from ef import ExponentialFamily, GaussianNatural1D
from model import nat2statMLP

Array = jax.Array

class TrainState(train_state.TrainState):
    batch_stats: FrozenDict | None = None


def create_train_state(rng: Array, model: nat2statMLP, ef: ExponentialFamily, learning_rate: float) -> TrainState:
    params = model.init(rng, jnp.zeros((1, ef.eta_dim)))
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params["params"], tx=tx)


def loss_fn(params, model: nat2statMLP, batch_eta: Array, batch_target: Array) -> Tuple[Array, Dict[str, Array]]:
    preds = model.apply({"params": params}, batch_eta)
    loss = jnp.mean(jnp.square(preds - batch_target))
    return loss, {"mse": loss}


def train_moment_net(
    ef: ExponentialFamily,
    train_data: Dict[str, Array],
    val_data: Dict[str, Array],
    hidden_sizes: Tuple[int, ...],
    activation: str,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
    seed: int,
) -> Tuple[TrainState, Dict[str, Array]]:
    rng = random.PRNGKey(seed)
    model = nat2statMLP(ef, hidden_sizes=hidden_sizes, activation=activation, output_dim=ef.eta_dim)
    state = create_train_state(rng, model, ef, learning_rate)

    num_train = train_data["eta"].shape[0]
    steps_per_epoch = max(1, (num_train + batch_size - 1) // batch_size)

    @jax.jit
    def train_step(state: TrainState, batch_eta: Array, batch_y: Array) -> Tuple[TrainState, Dict[str, Array]]:
        grad_fn = jax.value_and_grad(lambda p: loss_fn(p, model, batch_eta, batch_y), has_aux=True)
        (loss, metrics), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    history = {"train_mse": [], "val_mse": []}
    indices = jnp.arange(num_train)

    for epoch in range(num_epochs):
        # Shuffle
        perm_key = random.fold_in(rng, epoch)
        perm = random.permutation(perm_key, indices)
        eta_shuf = train_data["eta"][perm]
        y_shuf = train_data["y"][perm]

        # Mini-batch training
        for step in range(steps_per_epoch):
            start = step * batch_size
            end = jnp.minimum(start + batch_size, num_train)
            batch_eta = eta_shuf[start:end]
            batch_y = y_shuf[start:end]
            state, metrics = train_step(state, batch_eta, batch_y)

        # Epoch metrics
        train_pred = model.apply({"params": state.params}, train_data["eta"])  # type: ignore[arg-type]
        train_mse = jnp.mean(jnp.square(train_pred - train_data["y"]))
        val_pred = model.apply({"params": state.params}, val_data["eta"])  # type: ignore[arg-type]
        val_mse = jnp.mean(jnp.square(val_pred - val_data["y"]))
        history["train_mse"].append(float(train_mse))
        history["val_mse"].append(float(val_mse))

    return state, history


