# nnef-dists

Fast variational inference with non‑named exponential-family distributions using neural networks to learn the mapping between natural parameters and expected sufficient statistics. Features both direct ET networks and geometric-flow approaches, plus a unified data-generation and training pipeline.

## Overview

For an exponential family with natural parameters $\eta$ and sufficient statistics $T(x)$, inference by message passing requires the mapping $\mu_T(\eta) = \langle T(x) \mid \eta \rangle$. Named exponential-family distributions are easy because they have closed-form, invertible expressions for $\mu_T(\eta)$. In general,

$$ \log p(x\mid\eta) = \eta\cdot T(x) - A(\eta) $$

and $\mu_T(\eta)$ may be unknown. This project learns $\mu_T(\eta)$ for arbitrary exponential families by training compact neural networks on samples conditioned on $\eta$.

## Project Structure (Updated)

Active code lives under `src/` with EF definitions, data generation, and production model trainers. Legacy trainers remain under `src/training/` and should be avoided.

```
src/
├── expfam/                     # Exponential family distributions and data generation
│   ├── ef_base.py             # ExponentialFamily base class
│   ├── ef.py                  # Distribution implementations
│   ├── data_generator.py      # BlackJAX-based sampling and expectations
│   └── generate_data.py       # CLI/script to generate datasets
└── models/
    ├── __init__.py             # Model registry/imports
    ├── base_config.py          # BaseConfig
    ├── base_model.py           # BaseModel[T]
    ├── base_trainer.py         # BaseETTrainer (80-10-10 randomized splitter)
    ├── base_training_config.py # BaseTrainingConfig
    ├── mlp_et/                 # MLP ET model + train.py
    ├── glu_et/                 # GLU ET model + train.py
    ├── quadratic_et/           # Quadratic ET model + train.py
    ├── glow_et/                # Glow ET model + train.py
    ├── geometric_flow_et/      # Geometric Flow ET model + train.py
    └── noprop_*                # NoProp variants

```

- Data configs for sampling are in `data/configs/*.yaml`.
- Archive/legacy directories under `src/models/archive/` and `src/training/` should be ignored for new development.

## End-to-End Pipeline

1) Define an exponential-family (EF) distribution
- Implement an `ExponentialFamily` subclass in `src/expfam/ef.py` (see `MultivariateNormal`, `MultivariateNormal_tril`, `RectifiedGaussian`, etc.).
- Specify `x_shape`, `stat_shapes`, `_compute_stats(x)`; optionally `eta_bounds()` and/or `reparam_eta()`.
- For parameter validity, optionally implement `reparam_eta` (e.g., enforce negative-definite precision via tril/full reparameterization).

2) Create a sampling config
- Add a YAML file under `data/configs/` (e.g., `data/configs/multivariate_normal.yaml`).
- Specify EF name, `x_shape`, and sampling parameters. `SamplingConfig` supports: `num_samples`, `num_warmup`, `step_size`, `num_integration_steps`, `num_chains` (default 5), `parallel_strategy` (vmap/pmap/sequential), `seed`.

3) Generate samples
- Use `src/expfam/generate_data.py` to produce datasets using BlackJAX HMC via `DataGenerator`.
- Output is a single dict with exactly these keys: `eta`, `mu_T`, `cov_TT`, `ess` saved to an appropriately named file in the data directory.

Example:
```bash
conda activate numpyro
python -m src.expfam.generate_data --config data/configs/multivariate_normal.yaml --force
```

4) Train a model to learn μ_T(η)
- Use `train.py` under `src/models/<model>/` (e.g., `src/models/mlp_et/train.py`).
- `BaseETTrainer` loads the single-format data, validates keys, shuffles, and splits 80‑10‑10 (default seed 101).

Example:
```bash
conda activate numpyro
python src/models/mlp_et/train.py --data data/mvn.pkl --epochs 100
```

## Data Format (Single Source)

All generated datasets must be a dict with exactly:

```python
{
  'eta':  <array>,
  'mu_T': <array>,
  'cov_TT': <array>,
  'ess': <array>
}
```

Old pre-split formats are not supported.

## Notes on EF Implementations

- `flattened_log_density_fn(η)` returns a callable for batches of flattened `x`.
- Constraints (`x_bounds`, `x_constraint_fn`) are enforced inside `log_unnormalized` in a JAX‑compatible way.
- Multivariate normals share a unified `_reparam_eta` logic to ensure negative‑definite precision; `MultivariateNormal_tril` accepts `xxT_tril` at the interface, converts internally to full, reuses the same core logic, and returns tril format where appropriate.

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd nnef-dists

# Install
your_env> pip install -e .
# Recommended conda env
conda activate numpyro
```

### 2. Define an Exponential Family Distribution

Add your distribution to `src/expfam/ef.py`:

```python
@dataclass(frozen=True)
class YourDistribution(ExponentialFamily):
    x_shape: Tuple[int,]
    @cached_property
    def stat_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {"stat1": (self.x_shape[-1],), "stat2": (self.x_shape[-1], self.x_shape[-1])}
    def _compute_stats(self, x: Array) -> Dict[str, Array]:
        return {"stat1": x, "stat2": x[..., None] * x[..., None, :]}
```

### 3. Generate Training Data (via config)

```bash
python -m src.expfam.generate_data --config data/configs/your_dist.yaml --output data/your_dist.pkl
```

### 4. Train a Model

```bash
python src/models/mlp_et/train.py --data data/your_dist.pkl --epochs 100
```