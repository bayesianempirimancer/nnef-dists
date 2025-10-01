#!/usr/bin/env python3
"""
Model-agnostic loading script that works with all model types.
Automatically infers data file location from training results.
"""

import pickle
import json
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Import the generic training infrastructure using absolute imports
from src.training.trainer_factory import create_model_and_trainer


def load_training_data(data_file: str) -> Dict[str, Any]:
    """Load training data from pickle file."""
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    return data


def load_training_results(results_file: str) -> Dict[str, Any]:
    """Load training results from pickle file."""
    with open(results_file, "rb") as f:
        results = pickle.load(f)
    return results


def load_model_config(config_file: str) -> Dict[str, Any]:
    """Load model configuration from JSON file."""
    with open(config_file, "r") as f:
        config = json.load(f)
    return config


def load_model_parameters(params_file: str) -> Any:
    """Load model parameters from pickle file."""
    with open(params_file, "rb") as f:
        params = pickle.load(f)
    return params


def load_model_and_data(model_dir: str, data_file: Optional[str] = None) -> Tuple[Dict, Dict, Dict, Any, Any, Dict]:
    """
    Load model, results, and optionally data for any model type.
    
    This function is model-agnostic and works with all supported model types:
    - geometric_flow
    - noprop_geometric_flow  
    - mlp_et
    - glow_et
    - glu_et
    - quadratic_et
    
    Args:
        model_dir: Path to model artifacts directory
        data_file: Optional path to training data pickle file. If None, will be inferred from results.
        
    Returns:
        Tuple of (config, results, data, model, params, metadata)
        - config: Model configuration dictionary
        - results: Training results dictionary
        - data: Training data dictionary (None if data_file not provided and not found in results)
        - model: Trained model instance
        - params: Model parameters
        - metadata: Data metadata (empty dict if no data loaded)
    """
    model_dir = Path(model_dir)
    
    # Load results first to check for data file path
    results = load_training_results(model_dir / "training_results.pkl")
    
    # Determine data file path
    if data_file is None:
        data_file = results.get('data_file')
        if data_file is None:
            print("Warning: No data file specified and not found in results. Data will not be loaded.")
            data = None
            metadata = {}
        else:
            print(f"Using data file from results: {data_file}")
            data = load_training_data(data_file)
            metadata = data.get('metadata', {})
    else:
        print(f"Using specified data file: {data_file}")
        data = load_training_data(data_file)
        metadata = data.get('metadata', {})
    
    # Load config and parameters
    config = load_model_config(model_dir / "config.json")
    params = load_model_parameters(model_dir / "model_params.pkl")
    
    # Extract model type from config
    model_type = config.get('model_type')
    if model_type is None:
        raise ValueError("Config file must contain 'model_type' field")
    
    # Create model using the generic training infrastructure
    print(f"Creating {model_type} model using training infrastructure...")
    model, trainer = create_model_and_trainer(model_type, config)
    
    return config, results, data, model, params, metadata


def make_predictions(model, params, eta_data: np.ndarray, batch_size: int = 100) -> jnp.ndarray:
    """
    Make predictions on data in batches.
    
    This function works with any model type that has an apply method.
    
    Args:
        model: Trained model instance
        params: Model parameters
        eta_data: Input data array
        batch_size: Batch size for prediction
        
    Returns:
        Predictions array
    """
    predictions = []
    
    for i in range(0, len(eta_data), batch_size):
        batch_eta = jnp.array(eta_data[i:i+batch_size])
        batch_pred = model.predict(params, batch_eta)
        predictions.append(batch_pred)
    
    return jnp.concatenate(predictions, axis=0)


def get_supported_model_types() -> list:
    """
    Get list of supported model types.
    
    Returns:
        List of supported model type strings
    """
    return [
        "geometric_flow",
        "noprop_geometric_flow", 
        "mlp_et",
        "glow_et",
        "glu_et",
        "quadratic_et"
    ]


def main():
    """Test the loading functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test model-agnostic loading")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to model directory")
    parser.add_argument("--data", type=str, help="Path to data file (optional, will be inferred from results if not provided)")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_dir}")
    
    try:
        config, results, data, model, params, metadata = load_model_and_data(args.model_dir, args.data)
        
        print("✅ Successfully loaded everything!")
        print(f"Model type: {config.get('model_type', 'unknown')}")
        print(f"Model name: {config.get('model_name', 'unknown')}")
        
        if data is not None:
            print(f"Data shapes: train={len(data['train']['eta'])}, val={len(data['val']['eta'])}, test={len(data['test']['eta'])}")
            
            # Test making predictions
            print("\nTesting predictions...")
            test_eta = jnp.array(data['test']['eta'][:10])
            test_pred = make_predictions(model, params, test_eta)
            print(f"Test predictions shape: {test_pred.shape}")
            print("✅ Predictions working!")
        else:
            print("No data loaded - predictions not tested")
        
        print(f"Training epochs: {len(results['train_losses'])}")
        print(f"Final train loss: {results['final_train_loss']:.6f}")
        print(f"Final val loss: {results['final_val_loss']:.6f}")
        
        # Show supported model types
        print(f"\nSupported model types: {get_supported_model_types()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()