# Scripts Directory

This directory contains organized scripts for training and experimenting with neural networks for natural parameter to statistics mapping.

## Directory Structure

```
scripts/
├── training/           # Individual model training scripts (11 models)
├── debug/             # Outdated and experimental scripts
├── plotting/
│   ├── plot_training_results.py     # Standardized plotting functions
│   └── create_comparison_analysis.py # Analysis and visualization
├── test_all_training_scripts.py     # Quick validation with small architectures
├── run_comprehensive_model_comparison.py  # Full-scale model comparison
├── list_available_models.py         # Model listing and overview
└── README.md         # This file
```

## Model Training Scripts (`scripts/training/`)

Each model has its own training script with standardized configuration:

### Traditional ET Approaches
- `train_mlp_ET.py` - Standard Multi-Layer Perceptron ET networks
- `train_glu_ET.py` - Gated Linear Unit ET networks  
- `train_quadratic_resnet_ET.py` - Quadratic ResNet ET networks
- `train_invertible_nn_ET.py` - Invertible neural networks
- `train_noprop_ct_ET.py` - NoProp-CT continuous-time models

### Novel Flow-Based Approaches
- `train_geometric_flow_ET.py` - **NEW** Geometric Flow ET networks with continuous dynamics
- `train_glow_ET.py` - Glow networks using normalizing flows with affine coupling

### Log Normalizer Approaches
- `train_mlp_logZ.py` - MLP LogZ networks (gradient-based)
- `train_glu_logZ.py` - GLU LogZ networks
- `train_quadratic_resnet_logZ.py` - Quadratic ResNet LogZ networks
- `train_convex_nn_logZ.py` - Convex neural networks

### Training Templates
- `ET_training_template.py` - Template for creating new ET training scripts
- `logZ_training_template.py` - Template for creating new LogZ training scripts

### Usage

Each script can be run independently:

```bash
# Traditional ET Networks
python scripts/training/train_mlp_ET.py
python scripts/training/train_glu_ET.py
python scripts/training/train_quadratic_resnet_ET.py

# LogZ Networks  
python scripts/training/train_mlp_logZ.py
python scripts/training/train_glu_logZ.py

# Geometric Flow Networks (Novel)
python scripts/training/train_geometric_flow_ET.py

# Flow-based approaches
python scripts/training/train_glow_ET.py
python scripts/training/train_invertible_nn_ET.py

# Specialized Networks
python scripts/training/train_noprop_ct_ET.py
python scripts/training/train_convex_nn_logZ.py
```

### Standardized Features

All training scripts now include:
- **Standardized data loading** from `data/easy_3d_gaussian.pkl`
- **Memory optimization** with automatic `cov_tt` purging
- **Standardized plotting** using `scripts/plot_training_results.py`
- **Consistent output directories** in `artifacts/ET_models/` or `artifacts/logZ_models/`
- **MSE loss with L1 regularization** (no covariance-based losses)
- **Model definitions** imported from `src/models/` directory

## Quick Testing (`test_all_training_scripts.py`)

Tests all 11 models with small architectures for quick validation:

```bash
# Test all models with small networks (20 epochs, ~1K parameters)
python scripts/test_all_training_scripts.py

# Analyze test results
python scripts/plotting/create_comparison_analysis.py --mode test
```

**Features:**
- **Small architectures**: 2 layers × 32 units (~1K parameters each)
- **Quick training**: 20 epochs with small dataset (200 train, 50 val/test)
- **Compatible output**: Creates files compatible with analysis script
- **Results in**: `artifacts/tests/` directory

## Comprehensive Comparison (`run_comprehensive_model_comparison.py`)

Full-scale comparison of all 11 models with standardized architectures:

```bash
# Train all models with comparable architectures
python scripts/run_comprehensive_model_comparison.py --data data/easy_3d_gaussian.pkl --ef gaussian_3d

# Force retrain (skip completed models)
python scripts/run_comprehensive_model_comparison.py --data data/easy_3d_gaussian.pkl --ef gaussian_3d --force-retrain
```

**Features:**
- **Standardized architectures**: 12 layers × 128 units (Glow: 24 layers)
- **Comparable parameter counts**: ~50K-200K parameters per model
- **200 epochs**: Full training with early stopping
- **Skip logic**: Automatically skips already completed models
- **Timing tracking**: Training time and inference time per sample
- **Results in**: `artifacts/ET_models/` and `artifacts/logZ_models/`

## Analysis and Visualization (`plotting/create_comparison_analysis.py`)

Creates comprehensive analysis plots and tables:

```bash
# Analyze comprehensive results
python scripts/plotting/create_comparison_analysis.py --mode full

# Analyze test results
python scripts/plotting/create_comparison_analysis.py --mode test

# Custom output directory
python scripts/plotting/create_comparison_analysis.py --mode full --output artifacts/my_analysis
```

**Features:**
- **Two modes**: `full` (comprehensive results) or `test` (small architecture results)
- **Comprehensive plots**: 8-panel comparison with performance metrics
- **Performance tables**: CSV and formatted text output
- **Model ranking**: Sorted by MSE with timing information
- **Results in**: `artifacts/comprehensive_comparison/` (or custom directory)

## Data Generation (`src/utils/generate_normal_data.py`)

Utilities for generating training datasets:

```bash
# Generate challenging datasets
python src/utils/generate_normal_data.py
```

## Model Listing (`list_available_models.py`)

Overview of all available models and configurations:

```bash
# List all models and their configurations
python scripts/list_available_models.py
```

**Features:**
- **Model overview**: All 11 models with architecture details
- **Configuration summary**: Parameter counts and training settings
- **Artifacts structure**: Directory organization overview

## Standardized Plotting (`plot_training_results.py`)

Centralized plotting functions used by all training scripts:

```python
from scripts.plot_training_results import plot_training_results, plot_model_comparison, save_results_summary

# Create individual model plots
plot_training_results(trainer, eta_data, ground_truth, predictions, losses, config, model_name)

# Create comparison plots
plot_model_comparison(results, output_dir)

# Save results summary
save_results_summary(results, output_dir)
```

## Configuration System

All scripts use the standardized configuration system from `src/config.py`:

- `NetworkConfig` - Architecture parameters (hidden sizes, activation, etc.)
- `TrainingConfig` - Training parameters (learning rate, epochs, batch size, etc.)
- `FullConfig` - Combined configuration for complete model setup

## Workflow Examples

### 1. Quick Development Cycle
```bash
# Test all models quickly
python scripts/test_all_training_scripts.py

# Analyze test results
python scripts/plotting/create_comparison_analysis.py --mode test
```

### 2. Full Research Pipeline
```bash
# Run comprehensive comparison
python scripts/run_comprehensive_model_comparison.py --data data/easy_3d_gaussian.pkl --ef gaussian_3d

# Analyze comprehensive results
python scripts/plotting/create_comparison_analysis.py --mode full
```

### 3. Individual Model Development
```bash
# Train specific model
python scripts/training/train_geometric_flow_ET.py

# List available models
python scripts/list_available_models.py
```

## Archive (`scripts/debug/`)

Outdated and experimental scripts are preserved in `scripts/debug/` for reference but should not be used for new experiments:

- `quick_model_test.py`
- `run_all_model_comparison.sh`
- `run_all_models_comprehensive.py`
- `run_comprehensive_comparison.py`
- `run_model_comparison.py`
- `test_all_models.py`
- `train_comparison_models.py`

## Notes

- All training scripts use **MSE loss with L1 regularization**
- **No covariance-based losses** are used (purged from memory)
- All models use **standardized data loading** from `easy_3d_gaussian.pkl`
- **Memory optimization** with automatic `cov_tt` purging
- **Consistent output format** for seamless analysis integration