# Scripts Directory

This directory contains a clean, organized system for training and analyzing neural networks for exponential family distributions. The system uses proper Python package structure with clean imports and model-agnostic functionality.

## Directory Structure

```
scripts/
├── __init__.py                    # Package initialization
├── load_model_and_data.py        # Model-agnostic loader (works with all model types)
├── train_mlp_et_simple.py        # MLP ET training script
├── train_mlp_et_laplace.py       # Alternative MLP training script
├── train_mlp_et_equal_sizes.py   # MLP training with equal dataset sizes
├── plotting/                     # Clean plotting system
│   ├── __init__.py
│   └── plot_learning_errors.py   # 4-panel learning error analysis
├── archive/                      # Archived scripts
├── debug/                        # Debug utilities
└── README.md                     # This file
```

## Key Features

- **✅ Model-Agnostic**: Works with all 6 supported model types
- **✅ Clean Imports**: No `sys.path` manipulation, uses proper Python package structure
- **✅ Automatic Data Inference**: Plotting system automatically finds data files from training results
- **✅ Integrated Workflow**: Training automatically creates analysis plots
- **✅ Production Ready**: Professional package structure with proper imports

## Supported Model Types

The system works with all model types in the nnef-dists codebase:

- `geometric_flow` - Geometric Flow ET networks
- `noprop_geometric_flow` - NoProp Geometric Flow ET networks  
- `mlp_et` - Multi-Layer Perceptron ET networks
- `glow_et` - Glow ET networks
- `glu_et` - Gated Linear Unit ET networks
- `quadratic_et` - Quadratic ET networks

## Usage

### Prerequisites

Set the `PYTHONPATH` environment variable to the project root:

```bash
export PYTHONPATH=/path/to/nnef-dists:$PYTHONPATH
```

### 1. Training Models

Train an MLP ET model:

```bash
# Basic training
PYTHONPATH=/path/to/nnef-dists python -m scripts.train_mlp_et_simple \
  --data data/training_data.pkl \
  --epochs 50 \
  --model-name my_model

# With custom learning rate
PYTHONPATH=/path/to/nnef-dists python -m scripts.train_mlp_et_simple \
  --data data/training_data.pkl \
  --epochs 100 \
  --learning-rate 0.001 \
  --model-name my_model_lr001
```

**Training automatically:**
- Creates model artifacts in `artifacts/[model_name]/`
- Saves training results, config, and model parameters
- Generates 4-panel learning error analysis plot
- Records data file path for automatic inference

### 2. Loading Models and Data

Load any trained model using the model-agnostic loader:

```bash
# Load model and results only
PYTHONPATH=/path/to/nnef-dists python -m scripts.load_model_and_data \
  --model-dir artifacts/my_model

# Load with specific data file
PYTHONPATH=/path/to/nnef-dists python -m scripts.load_model_and_data \
  --model-dir artifacts/my_model \
  --data data/training_data.pkl
```

**Features:**
- Automatically infers data file from training results (when available)
- Works with any model type
- Makes predictions on test data
- Shows model information and training metrics

### 3. Creating Analysis Plots

Generate 4-panel learning error analysis plots:

```bash
# Auto-infer data file from results
PYTHONPATH=/path/to/nnef-dists python -m scripts.plotters.plot_learning_errors \
  --model-dir artifacts/my_model

# Specify data file explicitly
PYTHONPATH=/path/to/nnef-dists python -m scripts.plotters.plot_learning_errors \
  --model-dir artifacts/my_model \
  --data data/training_data.pkl

# Custom output location
PYTHONPATH=/path/to/nnef-dists python -m scripts.plotters.plot_learning_errors \
  --model-dir artifacts/my_model \
  --save my_custom_plot.png
```

**4-Panel Analysis Includes:**
1. **Training History**: Loss curves with theoretical minimum MSE lines
2. **Predictions vs True**: Scatter plot of predicted vs actual mu_T values
3. **Error vs True mu_T**: Residual analysis showing prediction errors
4. **Error vs ||eta||**: Error magnitude as a function of input norm

## Working Example

Here's a complete workflow example:

```bash
# 1. Set up environment
export PYTHONPATH=/home/jebeck/GitHub/nnef-dists:$PYTHONPATH
cd /home/jebeck/GitHub/nnef-dists

# 2. Train a model (automatically creates analysis plot)
python -m scripts.train_mlp_et_simple \
  --data data/training_data_2ae1e5d27f80bd60b739390026cc5465.pkl \
  --epochs 20 \
  --model-name example_model

# 3. Check what was created
ls -la artifacts/example_model/
# Output: config.json, learning_errors.png, model_params.pkl, training_results.pkl

# 4. Load the model and make predictions
python -m scripts.load_model_and_data \
  --model-dir artifacts/example_model

# 5. Create additional analysis plots
python -m scripts.plotters.plot_learning_errors \
  --model-dir artifacts/example_model \
  --save artifacts/example_model/custom_analysis.png
```

## Python API Usage

You can also use the system programmatically:

```python
from scripts.load_model_and_data import load_model_and_data, make_predictions

# Load everything
config, results, data, model, params, metadata = load_model_and_data(
    "artifacts/my_model"
)

# Make predictions
predictions = make_predictions(model, params, test_data)

# Access training results
print(f"Final train loss: {results['final_train_loss']}")
print(f"Final val loss: {results['final_val_loss']}")
print(f"Training time: {results['training_time']} seconds")
```

## File Structure

### Training Artifacts

Each training run creates a directory in `artifacts/[model_name]/` with:

- `config.json` - Model configuration
- `training_results.pkl` - Training metrics and results
- `model_params.pkl` - Trained model parameters
- `learning_errors.png` - 4-panel analysis plot (auto-generated)

### Training Results

The `training_results.pkl` file contains:

```python
{
    'train_losses': [...],           # Training loss per epoch
    'val_losses': [...],             # Validation loss per epoch
    'final_train_loss': float,       # Final training loss
    'final_val_loss': float,         # Final validation loss
    'best_val_loss': float,          # Best validation loss
    'training_time': float,          # Total training time (seconds)
    'inference_time': float,         # Inference time for batch
    'inference_time_per_sample': float,  # Time per individual sample
    'inference_batch_size': int,     # Batch size used for timing
    'config': dict,                  # Model configuration
    'data_file': str                 # Path to training data file
}
```

## Advanced Usage

### Custom Model Types

The system is model-agnostic and works with any model type supported by the training infrastructure:

```python
from scripts.load_model_and_data import get_supported_model_types

print("Supported model types:", get_supported_model_types())
# Output: ['geometric_flow', 'noprop_geometric_flow', 'mlp_et', 'glow_et', 'glu_et', 'quadratic_et']
```

### Batch Processing

Process multiple models:

```bash
# Train multiple models
for model_name in model1 model2 model3; do
    python -m scripts.train_mlp_et_simple \
      --data data/training_data.pkl \
      --epochs 50 \
      --model-name $model_name
done

# Analyze all models
for model_dir in artifacts/model*; do
    python -m scripts.plotters.plot_learning_errors --model-dir $model_dir
done
```

## Troubleshooting

### Import Errors

If you get import errors, make sure `PYTHONPATH` is set correctly:

```bash
# Check current PYTHONPATH
echo $PYTHONPATH

# Set it correctly
export PYTHONPATH=/path/to/nnef-dists:$PYTHONPATH
```

### Missing Data Files

If the plotting system can't find the data file:

1. Check that the training results contain the data file path:
   ```python
   import pickle
   with open('artifacts/my_model/training_results.pkl', 'rb') as f:
       results = pickle.load(f)
   print(results.get('data_file'))
   ```

2. Specify the data file explicitly:
   ```bash
   python -m scripts.plotters.plot_learning_errors \
     --model-dir artifacts/my_model \
     --data data/training_data.pkl
   ```

## Archive and Debug

- `archive/` - Contains old training scripts for reference
- `debug/` - Debug utilities and experimental scripts

These directories are preserved for reference but should not be used for new experiments.

## TODO

### Model Status
- **✅ MLP ET Network**: Fully debugged and tested with all improvements
  - ✅ Eta embedding support
  - ✅ Corrected ResNet architecture
  - ✅ Parameter sharing control
  - ✅ Layer normalization disabled by default
  - ✅ Dimension-expanding architecture
  - ✅ All config fields properly implemented

- **⚠️ Other ET Networks**: Improvements propagated but need testing
  - **GLU ET Network**: Architecture updated, needs validation
  - **Quadratic ET Network**: Architecture updated, needs validation  
  - **NoProp CT ET Network**: Partial updates applied, needs completion
  - **NoProp Geometric Flow ET Network**: Uses different architecture pattern
  - **Glow ET Network**: Uses flow-based architecture, needs validation

### Pending Tasks
- [ ] Test all training scripts to ensure they work correctly with the current codebase
- [ ] Homogenize data loading and creation scripts for consistency
- [ ] Implement Laplace approximation for estimation of mu_T
- [ ] Update .gitignore to ignore contents of artifacts directory but keep the directory itself
- [ ] Add papers directory to .gitignore
- [ ] Rename load_model_and_data.py to load_model_and_results.py and move to scripts/loaders directory
- [ ] Validate GLU ET, Quadratic ET, and other networks with training runs
- [ ] Complete NoProp CT ET network updates

## Notes

- All scripts use **clean imports** with no `sys.path` manipulation
- **Model-agnostic design** works with any supported model type
- **Automatic data inference** eliminates the need to specify data files manually
- **Integrated workflow** provides seamless training and analysis
- **Production-ready** with proper Python package structure
- **Only MLP ET network is fully debugged** - other networks need validation