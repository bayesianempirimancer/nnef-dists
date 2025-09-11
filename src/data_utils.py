"""Utilities for loading and working with training data."""

import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional

import jax
import jax.numpy as jnp
from jax import random, Array
from sampling import run_hmc

def load_training_data(data_file: str) -> Tuple[Dict[str, Array], Dict[str, Array], str]:
    """
    Load training data from a pickle file (JAX native format).
    
    Args:
        data_file: Path to the training data pickle file
        
    Returns:
        Tuple of (train_data, val_data, config_hash)
        where train_data and val_data are dicts with 'eta' and 'y' keys
    """
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
    
    # Include covariance data if available
    if "train_cov" in data:
        train_data["cov"] = data["train_cov"]
    if "val_cov" in data:
        val_data["cov"] = data["val_cov"]
    
    # Extract config hash from filename
    config_hash = Path(data_file).stem.split("_")[-1]
    
    return train_data, val_data, config_hash


def list_data_files(data_dir: str = "data") -> list[Path]:
    """List all available training data files."""
    data_path = Path(data_dir)
    
    # If relative path doesn't exist, try relative to current file's parent
    if not data_path.exists():
        # Try relative to the src directory (where this file is located)
        current_file = Path(__file__)
        project_root = current_file.parent.parent
        data_path = project_root / "data"
    
    if not data_path.exists():
        return []
    return list(data_path.glob("training_data_*.pkl"))


def load_latest_data(data_dir: str = "data") -> Tuple[Dict[str, Array], Dict[str, Array], str]:
    """Load the most recently created training data file."""
    data_files = list_data_files(data_dir)
    if not data_files:
        raise FileNotFoundError(f"No training data files found in {data_dir}")
    
    # Sort by modification time, newest first
    latest_file = sorted(data_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    return load_training_data(str(latest_file))


def inspect_data(train_data: Dict[str, Array], val_data: Dict[str, Array], config_hash: str = "unknown"):
    """Print information about the loaded data."""
    print(f"📊 Dataset Information:")
    print(f"Config hash: {config_hash}")
    print(f"Train data: eta {train_data['eta'].shape}, y {train_data['y'].shape}")
    print(f"Val data: eta {val_data['eta'].shape}, y {val_data['y'].shape}")
    
    print(f"\n📈 Data Statistics:")
    print(f"Train eta range: [{train_data['eta'].min():.3f}, {train_data['eta'].max():.3f}]")
    print(f"Train y range: [{train_data['y'].min():.3f}, {train_data['y'].max():.3f}]")
    print(f"Val eta range: [{val_data['eta'].min():.3f}, {val_data['eta'].max():.3f}]")
    print(f"Val y range: [{val_data['y'].min():.3f}, {val_data['y'].max():.3f}]")
    
    print(f"\n🔍 Sample Data:")
    print(f"First 3 train etas:\n{train_data['eta'][:3]}")
    print(f"First 3 train ys:\n{train_data['y'][:3]}")


# Convenience function for interactive use
def quick_load(data_file: Optional[str] = None) -> Tuple[Dict[str, Array], Dict[str, Array]]:
    """
    Quick load function for interactive use.
    
    Args:
        data_file: Optional path to specific data file. If None, loads the latest.
        
    Returns:
        Tuple of (train_data, val_data)
    """
    if data_file is None:
        train_data, val_data, config_hash = load_latest_data()
    else:
        train_data, val_data, config_hash = load_training_data(data_file)
    
    inspect_data(train_data, val_data, config_hash)
    return train_data, val_data


    
