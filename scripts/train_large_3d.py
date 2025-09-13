#!/usr/bin/env python
"""Training script for the large 3D multivariate Gaussian model."""

import sys
import argparse
from pathlib import Path

# Add parent directory to path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import yaml
import matplotlib.pyplot as plt

from src.ef import ef_factory
from src.model import nat2statMLP
from src.train import train_moment_net
from src.data_utils import load_training_data
from src.generate_data import get_data_hash
from plot_3d_results import plot_3d_multivariate_results

def train_large_3d_multivariate():
    """Train the large 3D multivariate Gaussian model with the configuration from multivariate_3d_large.yaml."""
    
    # Load the large configuration
    config_file = "configs/multivariate_3d_large.yaml"
    print(f"Loading configuration from {config_file}")
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Get the config hash to identify the corresponding data file
    config_hash = get_data_hash(cfg)
    data_file = f"data/training_data_{config_hash}.pkl"
    
    print(f"Config hash: {config_hash}")
    print(f"Looking for data file: {data_file}")
    
    # Check if data file exists
    if not Path(data_file).exists():
        print(f"❌ Data file {data_file} not found!")
        print("Please generate the data first by running:")
        print(f"python src/generate_data.py --config {config_file}")
        return None, None
    
    # Load the dataset
    print(f"Loading training data from {data_file}")
    train_data, val_data, data_config_hash = load_training_data(data_file)
    
    print(f"Data config hash: {data_config_hash}")
    print(f"Train data: eta {train_data['eta'].shape}, y {train_data['y'].shape}")
    print(f"Val data: eta {val_data['eta'].shape}, y {val_data['y'].shape}")
    
    # Create exponential family (3D multivariate normal)
    ef = ef_factory("multivariate_normal", x_shape=cfg["ef"]["x_shape"])
    print(f"Created {ef.__class__.__name__}: x_shape={ef.x_shape}, eta_dim={ef.eta_dim}")
    
    # Get model parameters from config
    hidden_sizes = tuple(cfg["optim"]["hidden_sizes"])
    activation = cfg["optim"]["activation"]
    learning_rate = cfg["optim"]["learning_rate"]
    num_epochs = cfg["optim"]["num_epochs"]
    batch_size = cfg["optim"]["batch_size"]
    seed = cfg["optim"]["seed"]
    
    # Create model
    model = nat2statMLP(
        dist=ef,
        hidden_sizes=hidden_sizes,
        activation=activation,
        output_dim=ef.eta_dim
    )
    print(f"Created model with hidden sizes: {hidden_sizes}")
    
    print(f"\nStarting training...")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Total train samples: {train_data['eta'].shape[0]}")
    print(f"Total val samples: {val_data['eta'].shape[0]}")
    print(f"Input dimension: {ef.eta_dim}")
    print(f"Output dimension: {ef.eta_dim}")
    
    # Train the model
    state, history = train_moment_net(
        ef=ef,
        train_data=train_data,
        val_data=val_data,
        hidden_sizes=hidden_sizes,
        activation=activation,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        seed=seed,
    )
    
    print(f"\nTraining completed!")
    print(f"Final train MSE: {history['train_mse'][-1]:.6f}")
    print(f"Final val MSE: {history['val_mse'][-1]:.6f}")
    
    # Save the training history
    history_file = f"artifacts/large_3d_training_history_{config_hash}.pkl"
    Path("artifacts").mkdir(exist_ok=True)
    with open(history_file, "wb") as f:
        pickle.dump({
            "history": history, 
            "config_hash": config_hash,
            "config": cfg
        }, f)
    print(f"Training history saved to {history_file}")
    
    # Generate plots
    print(f"\nGenerating plots...")
    try:
        plot_3d_multivariate_results(data_file, history_file)
        print(f"✅ Plots generated successfully!")
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
    
    return state, history

def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description="Train large 3D multivariate Gaussian model")
    parser.add_argument("--config", default="configs/multivariate_3d_large.yaml",
                       help="Path to configuration file")
    parser.add_argument("--plot-only", action="store_true", 
                       help="Only generate plots, skip training")
    parser.add_argument("--data-file", 
                       help="Path to training data file (optional, will auto-detect from config)")
    parser.add_argument("--history-file", 
                       help="Path to training history file (optional)")
    
    args = parser.parse_args()
    
    if args.plot_only:
        print("Generating plots only...")
        
        # Auto-detect files if not provided
        if not args.data_file or not args.history_file:
            with open(args.config, "r") as f:
                cfg = yaml.safe_load(f)
            config_hash = get_data_hash(cfg)
            
            if not args.data_file:
                args.data_file = f"data/training_data_{config_hash}.pkl"
            if not args.history_file:
                args.history_file = f"artifacts/large_3d_training_history_{config_hash}.pkl"
        
        try:
            plot_3d_multivariate_results(args.data_file, args.history_file)
            print("✅ Plots generated successfully!")
        except Exception as e:
            print(f"❌ Error generating plots: {e}")
    else:
        train_large_3d_multivariate()

if __name__ == "__main__":
    main()
