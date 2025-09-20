#!/usr/bin/env python3
"""
Training script for GLU Network model.

This script trains, evaluates, and plots results for a GLU Network
on the natural parameter to statistics mapping task.
"""

import sys
from pathlib import Path
import time
import json
import jax.numpy as jnp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import FullConfig
from src.models.glu_network import create_model_and_trainer
from src.data_utils import load_3d_gaussian_data, compute_ground_truth_3d_tril
from plotting.model_comparison import create_comprehensive_report


# =============================================================================
# CONFIGURATION - Edit these parameters to modify the experiment
# =============================================================================

# Deep narrow GLU configuration
CONFIG = FullConfig()

# Network architecture - Deep narrow for GLU
CONFIG.network.hidden_sizes = [80] * 10  # 10 GLU layers x 80 units each
CONFIG.network.activation = "tanh"
CONFIG.network.use_feature_engineering = True
CONFIG.network.dropout_rate = 0.1  # GLU can handle some dropout
CONFIG.network.output_dim = 9  # For tril format

# Training parameters optimized for GLU
CONFIG.training.learning_rate = 8e-4
CONFIG.training.num_epochs = 100
CONFIG.training.batch_size = 32
CONFIG.training.patience = 20
CONFIG.training.weight_decay = 1e-6
CONFIG.training.gradient_clip_norm = 1.0

# Experiment settings
CONFIG.experiment.experiment_name = "glu_network_deep_narrow"
CONFIG.experiment.output_dir = "artifacts/glu_network_results"

# Loss Function
LOSS_FUNCTION = "mse"  # Standard MSE loss

# =============================================================================


if __name__ == "__main__":
    # Import the main function from the model file
    from src.models.glu_network import main
    main()
