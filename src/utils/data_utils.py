"""Utilities for loading and working with training data."""

import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
import jax.numpy as jnp

from jax import Array
from ..ef import MultivariateNormal_tril

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


def load_3d_gaussian_data(data_dir: Path, format: str = "tril") -> Tuple[Dict[str, Array], MultivariateNormal_tril]:
    """
    Load 3D Gaussian dataset and optionally convert to tril format.
    
    Args:
        data_dir: Directory containing data files
        format: "full" or "tril" format
        
    Returns:
        Tuple of (data_dict, exponential_family_object)
    """
    # Find largest 3D dataset
    suitable_files = []
    for data_file in data_dir.glob("*.pkl"):
        try:
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            
            if data["train_eta"].shape[1] == 12:  # 3D Gaussian
                suitable_files.append((data_file, data["train_eta"].shape[0]))
        except Exception:
            continue
    
    if not suitable_files:
        raise FileNotFoundError("No 3D Gaussian datasets found!")
    
    best_file, n_samples = max(suitable_files, key=lambda x: x[1])
    ef = MultivariateNormal_tril(x_shape=(3,))
    
    print(f"Loading 3D Gaussian data from {best_file.name}")
    print(f"Dataset size: {n_samples} training samples")
    
    with open(best_file, 'rb') as f:
        data = pickle.load(f)
    
    if format == "tril":
        # Convert from full matrix format to triangular format
        def convert_to_tril(y_full):
            """Convert from full 12D format to tril 9D format."""
            batch_size = y_full.shape[0]
            
            # Extract mean (first 3 components)
            mean = y_full[:, :3]
            
            # Extract full matrix (last 9 components, reshaped to 3x3)
            full_matrix = y_full[:, 3:].reshape(batch_size, 3, 3)
            
            # Convert to lower triangular using ef.flatten_LT
            tril_matrix = ef.flatten_LT(full_matrix)
            
            # Combine mean and triangular matrix
            return jnp.concatenate([mean, tril_matrix], axis=1)
        
        # Convert all datasets
        converted_data = {
            "train_eta": data["train_eta"],
            "train_y": convert_to_tril(data["train_y"]),
            "val_eta": data["val_eta"],
            "val_y": convert_to_tril(data["val_y"])
        }
        
        print(f"Converted from full format (12D) to tril format (9D)")
    else:
        converted_data = data
    
    print(f"Output dimension: {converted_data['train_y'].shape[1]}")
    
    return converted_data, ef


def compute_ground_truth_3d_tril(eta: jnp.ndarray, ef: MultivariateNormal_tril) -> jnp.ndarray:
    """
    Compute ground truth statistics for 3D Gaussian using analytical formulas.
    Returns in triangular format.
    
    For multivariate Gaussian: Σ^{-1} = -2η₂, μ = Ση₁
    
    Args:
        eta: Natural parameters [batch_size, 12] where first 3 are η₁, last 9 are η₂ (flattened)
        ef: MultivariateNormal_tril exponential family object
        
    Returns:
        Expected sufficient statistics [batch_size, 9] in tril format
    """
    batch_size = eta.shape[0]
    
    # Extract η₁ (first 3 components) and η₂ (last 9 components)
    eta1 = eta[:, :3]  # Shape: (batch_size, 3)
    eta2 = eta[:, 3:].reshape(batch_size, 3, 3)  # Shape: (batch_size, 3, 3)
    
    # Compute Σ^{-1} = -2η₂
    Sigma_inv = -2.0 * eta2  # Shape: (batch_size, 3, 3)
    
    # Compute Σ = (Σ^{-1})^{-1}
    Sigma = jnp.linalg.inv(Sigma_inv)  # Shape: (batch_size, 3, 3)
    
    # Compute μ = Ση₁
    mu = jnp.einsum('bij,bj->bi', Sigma, eta1)  # Shape: (batch_size, 3)
    
    # Expected sufficient statistics: [μ, Σ + μμ^T]
    mu_muT = jnp.einsum('bi,bj->bij', mu, mu)  # Shape: (batch_size, 3, 3)
    expected_xxT = Sigma + mu_muT  # Shape: (batch_size, 3, 3)
    
    # Convert to triangular format using ef.flatten_LT
    expected_xxT_tril = ef.flatten_LT(expected_xxT)
    
    # Combine mean and triangular part
    expected_stats = jnp.concatenate([mu, expected_xxT_tril], axis=1)  # Shape: (batch_size, 9)
    
    return expected_stats
