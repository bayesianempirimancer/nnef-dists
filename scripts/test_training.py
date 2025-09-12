#!/usr/bin/env python
"""Test script for the training routine."""

import sys
from pathlib import Path

# Add parent directory to path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle

from src.ef import ef_factory
from src.model import nat2statMLP
from src.train import train_moment_net
from src.data_utils import load_training_data

def test_training():
    """Test the training routine with the small Gaussian 1D dataset."""
    
    # Load the small dataset
    data_file = "data/training_data_aee3fc097f906c743d8dae9a130ea1eb.pkl"
    print(f"Loading training data from {data_file}")
    train_data, val_data, config_hash = load_training_data(data_file)
    
    print(f"Config hash: {config_hash}")
    print(f"Train data: eta {train_data['eta'].shape}, y {train_data['y'].shape}")
    print(f"Val data: eta {val_data['eta'].shape}, y {val_data['y'].shape}")
    
    # Create exponential family (Gaussian 1D)
    ef = ef_factory("gaussian_1d")
    print(f"Created {ef.__class__.__name__}: x_shape={ef.x_shape}, eta_dim={ef.eta_dim}")
    
    # Create model
    hidden_sizes = (32, 16)
    activation = "tanh"
    model = nat2statMLP(
        dist=ef,
        hidden_sizes=hidden_sizes,
        activation=activation,
        output_dim=ef.eta_dim
    )
    print(f"Created model with hidden sizes: {hidden_sizes}")
    
    # Training parameters
    learning_rate = 0.001
    num_epochs = 20  # Small number for testing
    batch_size = 16
    seed = 42
    
    print(f"\nStarting training...")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    
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
    
    # Save just the training history (state can't be pickled easily)
    history_file = "artifacts/test_training_history.pkl"
    Path("artifacts").mkdir(exist_ok=True)
    with open(history_file, "wb") as f:
        pickle.dump({"history": history, "config_hash": config_hash}, f)
    print(f"Training history saved to {history_file}")
    
    # Generate plots
    print(f"\nGenerating plots...")
    data_file = "data/training_data_aee3fc097f906c743d8dae9a130ea1eb.pkl"
    try:
        plot_gaussian_1d_results(data_file, history_file)
        print(f"✅ Plots generated successfully!")
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
    
    return state, history

def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description="Test training and plotting")
    parser.add_argument("--plot-only", action="store_true", 
                       help="Only generate plots, skip training")
    parser.add_argument("--data-file", default="data/training_data_aee3fc097f906c743d8dae9a130ea1eb.pkl",
                       help="Path to training data file")
    parser.add_argument("--history-file", default="artifacts/test_training_history.pkl",
                       help="Path to training history file")
    
    args = parser.parse_args()
    
    if args.plot_only:
        print("Generating plots only...")
        try:
            plot_gaussian_1d_results(args.data_file, args.history_file)
            print("✅ Plots generated successfully!")
        except Exception as e:
            print(f"❌ Error generating plots: {e}")
    else:
        test_training()


if __name__ == "__main__":
    main()
