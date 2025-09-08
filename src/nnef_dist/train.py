from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.core import FrozenDict
from jax import random

from .ef import make_logdensity_fn, sufficient_statistic_poly1d
from .sampling import run_hmc
from .model import MomentMLP


Array = jax.Array


class TrainState(train_state.TrainState):
    batch_stats: FrozenDict | None = None


def generate_eta_grid(num_points: int, eta1_range: Tuple[float, float], eta2_range: Tuple[float, float], key: Array) -> Array:
    eta1 = random.uniform(key, (num_points,), minval=eta1_range[0], maxval=eta1_range[1])
    # Ensure negative eta2 for integrability
    eta2 = random.uniform(key, (num_points,), minval=eta2_range[0], maxval=eta2_range[1])
    return jnp.stack([eta1, eta2], axis=-1)


def empirical_moments_from_samples(samples: Array) -> Array:
    t = sufficient_statistic_poly1d(samples)
    return jnp.mean(t, axis=0)


def build_dataset(
    train_points: int,
    val_points: int,
    eta1_range: Tuple[float, float],
    eta2_range: Tuple[float, float],
    sampler_cfg: Dict,
    seed: int,
) -> Tuple[Dict[str, Array], Dict[str, Array]]:
    key = random.PRNGKey(seed)
    k_tr, k_val, k_pos = random.split(key, 3)
    etas_train = generate_eta_grid(train_points, eta1_range, eta2_range, k_tr)
    etas_val = generate_eta_grid(val_points, eta1_range, eta2_range, k_val)

    def simulate(etas: Array, key: Array) -> Array:
        def one(eta, k):
            logp = make_logdensity_fn(eta)
            samples = run_hmc(
                logp,
                num_samples=sampler_cfg["num_samples"],
                num_warmup=sampler_cfg["num_warmup"],
                step_size=sampler_cfg["step_size"],
                num_integration_steps=sampler_cfg["num_integration_steps"],
                initial_position=sampler_cfg.get("initial_position", 0.0),
                seed=int(k[0]),
            )
            return empirical_moments_from_samples(samples)

        keys = random.split(key, etas.shape[0])
        ys = jax.vmap(one)(etas, keys)
        return ys

    y_train = simulate(etas_train, k_pos)
    y_val = simulate(etas_val, random.fold_in(k_pos, 1))
    return {"eta": etas_train, "y": y_train}, {"eta": etas_val, "y": y_val}


def create_train_state(rng: Array, model: MomentMLP, learning_rate: float) -> TrainState:
    params = model.init(rng, jnp.zeros((1, 2)))
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params["params"], tx=tx)


def loss_fn(params, model: MomentMLP, batch_eta: Array, batch_target: Array) -> Tuple[Array, Dict[str, Array]]:
    preds = model.apply({"params": params}, batch_eta)
    loss = jnp.mean(jnp.square(preds - batch_target))
    return loss, {"mse": loss}


def train_moment_net(
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
    model = MomentMLP(hidden_sizes=hidden_sizes, activation=activation)
    state = create_train_state(rng, model, learning_rate)

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


