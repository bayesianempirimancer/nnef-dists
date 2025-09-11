from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SamplingConfig:
    num_samples: int = 2_000
    num_warmup: int = 1_000
    step_size: float = 0.1
    num_integration_steps: int = 10
    initial_position: float = 0.0


@dataclass
class ModelConfig:
    hidden_sizes: List[int]
    activation: str = "tanh"


@dataclass
class OptimConfig:
    learning_rate: float = 1e-3
    num_epochs: int = 200
    batch_size: int = 64
    seed: int = 0


@dataclass
class GridConfig:
    num_train_points: int
    num_val_points: int
    eta1_range: Tuple[float, float]
    eta2_range: Tuple[float, float]


@dataclass
class DataConfig:
    family: str = "poly1d"
    # Currently fixed to T(x) = [x, x^2] for stability.
    max_degree: int = 2



