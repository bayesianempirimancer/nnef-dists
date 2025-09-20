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

Each model has its own training script with standardized configuration at the top:

- `train_standard_mlp.py` - Standard Multi-Layer Perceptron
- `train_deep_flow.py` - Deep Flow Network with diffusion training
- `train_quadratic_resnet.py` - Quadratic ResNet with adaptive mixing
- `train_noprop_ct.py` - NoProp-CT continuous-time model (TODO)
- `train_diffusion.py` - Diffusion-based moment network (TODO)

### Usage

Each script can be run independently:

```bash
python scripts/training/train_standard_mlp.py
python scripts/training/train_deep_flow.py
python scripts/training/train_quadratic_resnet.py
```

### Configuration

Edit the configuration section at the top of each script to modify:
- Network architecture (layers, units, activation)
- Training parameters (learning rate, batch size, epochs)
- Loss function type
- Output directory

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
