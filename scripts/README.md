# Scripts Directory

This directory contains organized scripts for training and experimenting with neural networks for natural parameter to statistics mapping.

## Directory Structure

```
scripts/
├── models/           # Individual model training scripts
├── experiments/      # Comprehensive experiments and comparisons
├── archive/          # Old scripts (kept for reference)
└── README.md         # This file
```

## Model Training Scripts (`scripts/training/`)

Each model has its own training script with standardized configuration:

### Traditional Approaches
- `train_mlp_ET.py` - Standard Multi-Layer Perceptron ET networks
- `train_glu_ET.py` - Gated Linear Unit ET networks  
- `train_quadratic_resnet_ET.py` - Quadratic ResNet ET networks
- `train_mlp_logZ.py` - MLP LogZ networks (gradient-based)
- `train_quadratic_resnet_logZ.py` - Quadratic ResNet LogZ networks

### Novel Flow-Based Approaches
- `train_geometric_flow_ET.py` - **NEW** Geometric Flow ET networks with continuous dynamics
- `train_glow_net_ET.py` - Glow networks using normalizing flows with affine coupling
- `train_invertible_nn_ET.py` - Invertible neural networks

### Specialized Models
- `train_noprop_ct_ET.py` - NoProp-CT continuous-time models
- `train_convex_nn_logZ.py` - Convex neural networks

### Usage

Each script can be run independently:

```bash
# Traditional ET Networks
python scripts/training/train_mlp_ET.py
python scripts/training/train_glu_ET.py
python scripts/training/train_quadratic_resnet_ET.py

# LogZ Networks  
python scripts/training/train_mlp_logZ.py

# Geometric Flow Networks (Novel)
python scripts/training/train_geometric_flow_ET.py --save-dir artifacts/geometric_flow

# Flow-based approaches
python scripts/training/train_glow_net_ET.py
python scripts/training/train_invertible_nn_ET.py
```

### Configuration

Edit the configuration section at the top of each script to modify:
- Network architecture (layers, units, activation)
- Training parameters (learning rate, batch size, epochs)
- Loss function type
- Output directory

### Geometric Flow Networks - Special Configuration

The geometric flow approach has additional parameters:
- `matrix_rank`: Rank of flow matrix A (default: μ_dim)
- `n_time_steps`: Integration steps (default: 3, minimal due to smoothness)
- `smoothness_weight`: Penalty for large derivatives (default: 1e-3)
- `time_embed_dim`: Sinusoidal time embedding dimension (default: 16)
- `max_freq`: Maximum frequency for time embeddings (default: 10.0)

## Experiment Scripts (`scripts/experiments/`)

Comprehensive experiments that compare multiple models:

- `comprehensive_deep_narrow.py` - Systematic comparison of deep narrow vs wide shallow
- `deep_narrow_experiment.py` - Basic deep narrow networks experiment

### Usage

```bash
python scripts/experiments/comprehensive_deep_narrow.py
```

## Configuration System

All scripts use the standardized configuration system from `src/config.py`:

- `NetworkConfig` - Architecture parameters
- `TrainingConfig` - Training parameters  
- `ModelSpecificConfig` - Model-specific parameters
- `ExperimentConfig` - Experiment settings
- `FullConfig` - Combined configuration

## Plotting

All plotting utilities are in the `plotting/` directory:

- `model_comparison.py` - Standardized comparison plots

## Archive

Old scripts are preserved in `scripts/archive/` for reference but should not be used for new experiments.
