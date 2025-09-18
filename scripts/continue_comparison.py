#!/usr/bin/env python3
"""
Continue the comprehensive comparison from where it left off.
This script runs only the remaining models (NoProp-CT and Improved INN).
"""

import sys
from pathlib import Path
import pickle
import jax
import jax.numpy as jnp
from jax import random
import optax
import matplotlib.pyplot as plt
import numpy as np
import time
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ef import GaussianNatural1D
from src.improved_inn import ImprovedInvertibleNet, ImprovedINNConfig, train_improved_inn
from src.noprop_ct import NoPropCTMomentNet, NoPropCTConfig, train_noprop_ct_moment_net
from scripts.run_noprop_ct_demo import load_existing_data


def analytical_solution(eta):
    """Analytical solution for Gaussian 1D: mu = -eta1/(2*eta2), sigma2 = -1/(2*eta2)"""
    eta1, eta2 = eta[:, 0], eta[:, 1]
    
    # Convert to mean and variance
    mu = -eta1 / (2 * eta2)
    sigma2 = -1 / (2 * eta2)
    
    # Expected sufficient statistics: E[x] = mu, E[x^2] = mu^2 + sigma^2
    E_x = mu
    E_x2 = mu**2 + sigma2
    
    return jnp.stack([E_x, E_x2], axis=-1)


def evaluate_model(model, params, test_data, name="Model"):
    """Comprehensive model evaluation."""
    
    pred = model.apply(params, test_data['eta'])
    
    # Basic metrics
    mse = float(jnp.mean(jnp.square(pred - test_data['y'])))
    mae = float(jnp.mean(jnp.abs(pred - test_data['y'])))
    rmse = float(jnp.sqrt(mse))
    
    # Component-wise metrics
    component_mse = jnp.mean(jnp.square(pred - test_data['y']), axis=0)
    component_mae = jnp.mean(jnp.abs(pred - test_data['y']), axis=0)
    
    # R-squared
    ss_res = jnp.sum(jnp.square(pred - test_data['y']))
    ss_tot = jnp.sum(jnp.square(test_data['y'] - jnp.mean(test_data['y'], axis=0)))
    r2 = float(1 - (ss_res / ss_tot))
    
    # Max absolute error
    max_error = float(jnp.max(jnp.abs(pred - test_data['y'])))
    
    return {
        'name': name,
        'predictions': pred,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'max_error': max_error,
        'component_mse': component_mse.tolist(),
        'component_mae': component_mae.tolist()
    }


def continue_comparison():
    """Continue the comparison with the remaining models."""
    
    print("🔄 CONTINUING COMPREHENSIVE COMPARISON")
    print("=" * 60)
    print("Running the remaining models: NoProp-CT and Improved INN")
    
    # Load data
    ef = GaussianNatural1D()
    train_data, val_data, test_data = load_existing_data("gaussian_1d")
    
    print(f"\nDataset: {train_data['eta'].shape[0]} training, {test_data['eta'].shape[0]} test samples")
    
    results = {}
    training_histories = {}
    
    # Get analytical baseline
    analytical_pred = analytical_solution(test_data['eta'])
    analytical_mse = float(jnp.mean(jnp.square(analytical_pred - test_data['y'])))
    print(f"Analytical MSE baseline: {analytical_mse:.8f}")
    
    # 1. NoProp-CT (Neural ODE) - Fixed configuration
    print(f"\n{'='*60}")
    print("1. NOPROP-CT (Neural ODE) - RETRY")
    print(f"{'='*60}")
    
    noprop_config = NoPropCTConfig(
        hidden_sizes=(128, 64),
        activation="tanh",
        learning_rate=1e-3,
        num_time_steps=20,
        ode_solver="euler",
        denoising_weight=1.0,
        consistency_weight=0.1
    )
    
    print("  Training NoProp-CT...")
    start_time = time.time()
    
    try:
        noprop_state, noprop_history = train_noprop_ct_moment_net(
            ef=ef,
            train_data=train_data,
            val_data=val_data,
            config=noprop_config,
            num_epochs=200,
            batch_size=64,
            seed=49
        )
        
        noprop_training_time = time.time() - start_time
        print(f"  Training completed in {noprop_training_time:.1f}s")
        
        # Evaluate NoProp-CT
        noprop_model = NoPropCTMomentNet(ef=ef, config=noprop_config)
        
        # Custom evaluation for NoProp-CT (handle different apply signature)
        noprop_pred = noprop_model.apply(noprop_state.params, test_data['eta'])
        
        results['noprop_ct'] = {
            'name': 'NoProp-CT',
            'predictions': noprop_pred,
            'mse': float(jnp.mean(jnp.square(noprop_pred - test_data['y']))),
            'mae': float(jnp.mean(jnp.abs(noprop_pred - test_data['y']))),
            'rmse': float(jnp.sqrt(jnp.mean(jnp.square(noprop_pred - test_data['y'])))),
            'r2': float(1 - jnp.sum(jnp.square(noprop_pred - test_data['y'])) / jnp.sum(jnp.square(test_data['y'] - jnp.mean(test_data['y'], axis=0)))),
            'max_error': float(jnp.max(jnp.abs(noprop_pred - test_data['y']))),
            'component_mse': jnp.mean(jnp.square(noprop_pred - test_data['y']), axis=0).tolist(),
            'component_mae': jnp.mean(jnp.abs(noprop_pred - test_data['y']), axis=0).tolist()
        }
        
        training_histories['noprop_ct'] = {
            'train_losses': noprop_history['train_losses'],
            'val_losses': noprop_history['val_losses'],
            'training_time': noprop_training_time,
            'epochs_trained': len(noprop_history['train_losses'])
        }
        
        print(f"  NoProp-CT MSE: {results['noprop_ct']['mse']:.8f}")
        print(f"  NoProp-CT MAE: {results['noprop_ct']['mae']:.8f}")
        
    except Exception as e:
        print(f"  ❌ NoProp-CT failed: {e}")
        results['noprop_ct'] = None
    
    # 2. Improved INN (Invertible Neural Network)
    print(f"\n{'='*60}")
    print("2. IMPROVED INN (Invertible Neural Network)")
    print(f"{'='*60}")
    
    inn_config = ImprovedINNConfig(
        num_layers=8,
        hidden_size=96,
        num_hidden_layers=3,
        activation="tanh",
        learning_rate=8e-4,
        clamp_alpha=2.0,
        log_det_weight=0.01,
        invertibility_weight=0.3,
        weight_decay=1e-5,
        use_geometric_preprocessing=False,  # Keep stable
        preprocessing_epsilon=1e-6
    )
    
    print("  Training Improved INN...")
    start_time = time.time()
    
    try:
        inn_state, inn_history = train_improved_inn(
            ef=ef,
            train_data=train_data,
            val_data=val_data,
            config=inn_config,
            num_epochs=150,
            batch_size=64,
            seed=50
        )
        
        inn_training_time = time.time() - start_time
        print(f"  Training completed in {inn_training_time:.1f}s")
        
        # Evaluate INN
        inn_model = ImprovedInvertibleNet(ef=ef, config=inn_config)
        inn_pred, _ = inn_model.apply(inn_state.params, test_data['eta'], reverse=False)
        
        results['improved_inn'] = {
            'name': 'Improved INN',
            'predictions': inn_pred,
            'mse': float(jnp.mean(jnp.square(inn_pred - test_data['y']))),
            'mae': float(jnp.mean(jnp.abs(inn_pred - test_data['y']))),
            'rmse': float(jnp.sqrt(jnp.mean(jnp.square(inn_pred - test_data['y'])))),
            'r2': float(1 - jnp.sum(jnp.square(inn_pred - test_data['y'])) / jnp.sum(jnp.square(test_data['y'] - jnp.mean(test_data['y'], axis=0)))),
            'max_error': float(jnp.max(jnp.abs(inn_pred - test_data['y']))),
            'component_mse': jnp.mean(jnp.square(inn_pred - test_data['y']), axis=0).tolist(),
            'component_mae': jnp.mean(jnp.abs(inn_pred - test_data['y']), axis=0).tolist()
        }
        
        training_histories['improved_inn'] = {
            'train_losses': inn_history['train_losses'],
            'val_losses': inn_history['val_losses'],
            'training_time': inn_training_time,
            'epochs_trained': len(inn_history['train_losses'])
        }
        
        print(f"  Improved INN MSE: {results['improved_inn']['mse']:.8f}")
        print(f"  Improved INN MAE: {results['improved_inn']['mae']:.8f}")
        
    except Exception as e:
        print(f"  ❌ Improved INN failed: {e}")
        results['improved_inn'] = None
    
    # Print summary of remaining models
    print(f"\n{'='*60}")
    print("REMAINING MODELS SUMMARY")
    print(f"{'='*60}")
    
    for model_name, result in results.items():
        if result is not None:
            print(f"{result['name']:<20} MSE: {result['mse']:.8f}  "
                  f"vs Analytical: {result['mse']/analytical_mse:.1f}x worse")
    
    # Save results
    output_dir = Path("artifacts/remaining_models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    results_for_json = {}
    for model_name, result in results.items():
        if result is not None:
            results_for_json[model_name] = {
                'name': result['name'],
                'mse': result['mse'],
                'mae': result['mae'],
                'rmse': result['rmse'],
                'r2': result['r2'],
                'max_error': result['max_error'],
                'component_mse': result['component_mse'],
                'component_mae': result['component_mae'],
                'training_time': training_histories[model_name]['training_time'],
                'epochs_trained': training_histories[model_name]['epochs_trained']
            }
    
    with open(output_dir / "remaining_models_results.json", 'w') as f:
        json.dump(results_for_json, f, indent=2)
    
    print(f"\n📊 Results saved to {output_dir}/")
    
    return results, training_histories


if __name__ == "__main__":
    print("🔄 Continuing comprehensive comparison...")
    
    try:
        results, histories = continue_comparison()
        print("\n✅ Remaining models completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
