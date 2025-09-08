#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import yaml

from nnef_dist.train import build_dataset, train_moment_net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--out_dir", type=str, default="artifacts/gaussian_moment_net", help="Artifacts dir")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data, val_data = build_dataset(
        train_points=cfg["grid"]["num_train_points"],
        val_points=cfg["grid"]["num_val_points"],
        eta1_range=tuple(cfg["grid"]["eta1_range"]),
        eta2_range=tuple(cfg["grid"]["eta2_range"]),
        sampler_cfg=cfg["sampling"],
        seed=cfg["optim"]["seed"],
    )

    state, history = train_moment_net(
        train_data=train_data,
        val_data=val_data,
        hidden_sizes=tuple(cfg["model"]["hidden_sizes"]),
        activation=cfg["model"].get("activation", "tanh"),
        learning_rate=cfg["optim"]["learning_rate"],
        num_epochs=cfg["optim"]["num_epochs"],
        batch_size=cfg["optim"]["batch_size"],
        seed=cfg["optim"]["seed"],
    )

    # Save params and history
    params_file = out_dir / "params.json"
    hist_file = out_dir / "history.json"
    with params_file.open("w") as f:
        json.dump({"params": jnp.array(state.params).tolist()}, f)
    with hist_file.open("w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved params to {params_file}")
    print(f"Saved history to {hist_file}")


if __name__ == "__main__":
    main()


