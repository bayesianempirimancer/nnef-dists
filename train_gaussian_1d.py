#!/usr/bin/env python
"""Train the model on the generated Gaussian 1D medium dataset."""

import sys
from pathlib import Path

# Add src to path so we can import without installing
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pickle
import yaml
from ef import ef_factory
from model import nat2statMLP
from notebooks.simple_training import train_model_simple

def load_training_data(data_file):
    """Load training data from pickle file."""
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    
    train_data = {
        "eta": data["train_eta"],
        "y": data["train_y"]
    }
    val_data = {
        "eta": data["val_eta"],
        "y": data["val_y"]
    }
    
    return train_data, val_data

def main():
    # Load config
    config_file = "configs/gaussian_1d_medium.yaml"
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Create exponential family
    ef_params = cfg["ef"].copy()
    ef_name = ef_params.pop("name")
    ef = ef_factory(ef_name, **ef_params)
    print(f"Created {ef.__class__.__name__}: x_shape={ef.x_shape}, eta_dim={ef.eta_dim}")
    
    # Load training data
    data_file = "data/training_data_da1a4390c850ef5960807690317cc47b.pkl"
    print(f"Loading training data from {data_file}")
    train_data, val_data = load_training_data(data_file)
    
    print(f"Train data: eta {train_data['eta'].shape}, y {train_data['y'].shape}")
    print(f"Val data: eta {val_data['eta'].shape}, y {val_data['y'].shape}")
    
    # Create model
    hidden_sizes = tuple(cfg["optim"]["hidden_sizes"])
    model = nat2statMLP(
        dist=ef,
        hidden_sizes=hidden_sizes,
        activation=cfg["optim"]["activation"],
        output_dim=ef.stat_dim
    )
    print(f"Created model with hidden sizes: {hidden_sizes}")
    
    # Train model
    print("\nStarting training...")
    state, history = train_model_simple(
        model=model,
        train_data=train_data,
        val_data=val_data,
        num_epochs=cfg["optim"]["num_epochs"],
        learning_rate=cfg["optim"]["learning_rate"],
        batch_size=cfg["optim"]["batch_size"],
        seed=cfg["optim"]["seed"]
    )
    
    print("\nTraining completed!")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    
    # Save model
    model_file = "artifacts/gaussian_1d_medium_model.pkl"
    Path("artifacts").mkdir(exist_ok=True)
    with open(model_file, "wb") as f:
        pickle.dump({"state": state, "history": history, "config": cfg}, f)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    main()
