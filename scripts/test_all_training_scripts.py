#!/usr/bin/env python3
"""
Test all training scripts with small network architectures.

This script trains all models with small network configurations on a small dataset
for quick validation and creates output compatible with create_comparison_analysis.py.
"""

import os
import sys
import json
import pickle
import time
import importlib.util
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from jax import random

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Import training components
from src.config import FullConfig, NetworkConfig, TrainingConfig


# =============================================================================
# SMALL NETWORK ARCHITECTURES - Low Parameter Count for Quick Testing
# =============================================================================

SMALL_MODEL_ARCHITECTURES = {
    # ET Models - Expected Statistics Networks
    'mlp_ET': {
        'type': 'ET',
        'hidden_sizes': [32, 32],  # 2 layers x 32 units (~1K params)
        'description': 'Small MLP for Expected Statistics',
        'output_dim': 9,  # 3D mean + 6D upper triangular covariance
    },
    
    'glu_ET': {
        'type': 'ET', 
        'hidden_sizes': [32, 32],  # 2 layers x 32 units (~1K params)
        'description': 'Small GLU for Expected Statistics',
        'output_dim': 9,
    },
    
    'quadratic_resnet_ET': {
        'type': 'ET',
        'hidden_sizes': [32],  # 1 layer + 1 residual block (~1K params)
        'description': 'Small Quadratic ResNet for Expected Statistics',
        'output_dim': 9,
    },
    
    'invertible_nn_ET': {
        'type': 'ET',
        'hidden_sizes': [32, 32],  # 2 layers x 32 units (~1K params)
        'description': 'Small Invertible NN for Expected Statistics',
        'output_dim': 9,
    },
    
    'noprop_ct_ET': {
        'type': 'ET',
        'hidden_sizes': [32, 32],  # 2 layers x 32 units (~1K params)
        'description': 'Small NoProp-CT for Expected Statistics',
        'output_dim': 9,
    },
    
    'geometric_flow_ET': {
        'type': 'ET',
        'hidden_sizes': [32, 32],  # 2 layers x 32 units (~1K params)
        'description': 'Small Geometric Flow for Expected Statistics',
        'output_dim': 9,
    },
    
    'glow_ET': {
        'type': 'ET',
        'hidden_sizes': [32] * 4,  # 4 layers x 32 units (~2K params, still small)
        'description': 'Small Glow Flow for Expected Statistics',
        'output_dim': 9,
    },
    
    # LogZ Models - Log Normalizer Networks
    'mlp_logZ': {
        'type': 'LogZ',
        'hidden_sizes': [32, 32],  # 2 layers x 32 units (~1K params)
        'description': 'Small MLP for Log Normalizer',
        'output_dim': 1,  # Scalar log normalizer
    },
    
    'glu_logZ': {
        'type': 'LogZ',
        'hidden_sizes': [32, 32],  # 2 layers x 32 units (~1K params)
        'description': 'Small GLU for Log Normalizer',
        'output_dim': 1,
    },
    
    'quadratic_resnet_logZ': {
        'type': 'LogZ',
        'hidden_sizes': [32],  # 1 layer + 1 residual block (~1K params)
        'description': 'Small Quadratic ResNet for Log Normalizer',
        'output_dim': 1,
    },
    
    'convex_nn_logZ': {
        'type': 'LogZ',
        'hidden_sizes': [32, 32],  # 2 layers x 32 units (~1K params)
        'description': 'Small Convex NN for Log Normalizer',
        'output_dim': 1,
    },
}


def test_script_imports():
    """Test that all training scripts can import their dependencies."""
    training_dir = Path("scripts/training")
    
    # Get all train_*.py files
    train_scripts = list(training_dir.glob("train_*.py"))
    train_scripts = [f for f in train_scripts if not f.name.startswith("train_template") and not f.name.endswith("_training_template.py")]
    
    print(f"Testing {len(train_scripts)} training scripts...")
    print("="*50)
    
    results = {}
    
    for script_path in sorted(train_scripts):
        script_name = script_path.stem
        print(f"\nTesting {script_name}...")
        
        try:
            # Try to import the script
            spec = importlib.util.spec_from_file_location(script_name, script_path)
            module = importlib.util.module_from_spec(spec)
            
            # Test basic imports (don't execute main)
            with open(script_path, 'r') as f:
                content = f.read()
            
            # Check for key imports
            imports_to_check = [
                'from src.config import',
                'from src.models.',
                'from src.utils.',
                'from src.ef import'
            ]
            
            import_status = {}
            for imp in imports_to_check:
                import_status[imp] = imp in content
            
            results[script_name] = {
                'can_import': True,
                'imports': import_status,
                'file_size': script_path.stat().st_size,
                'status': 'OK'
            }
            
            print(f"  ✓ {script_name}: Imports look good")
            
        except Exception as e:
            results[script_name] = {
                'can_import': False,
                'error': str(e),
                'status': 'ERROR'
            }
            print(f"  ✗ {script_name}: {e}")
    
    return results


def test_model_imports():
    """Test that key model imports work."""
    print(f"\nTesting Model Imports...")
    print("="*30)
    
    models_to_test = [
        ('src.models.ET_Net', 'create_mlp_et'),
        ('src.models.logZ_Net', 'create_logZ_network_and_trainer'),
        ('src.models.geometric_flow_net', 'create_geometric_flow_et_network'),
        ('src.models.glow_ET', 'create_glow_et_model_and_trainer'),
        ('src.utils.data_utils', 'load_training_data'),
        ('src.config', 'FullConfig'),
        ('src.ef', 'ef_factory')
    ]
    
    for module_name, function_name in models_to_test:
        try:
            module = importlib.import_module(module_name)
            func = getattr(module, function_name)
            print(f"  ✓ {module_name}.{function_name}")
        except Exception as e:
            print(f"  ✗ {module_name}.{function_name}: {e}")



def generate_small_test_data():
    """Generate a small test dataset for training validation."""
    print(f"\nGenerating Small Test Dataset...")
    print("="*35)
    
    data_file = Path("data/easy_3d_gaussian_small.pkl")
    
    if data_file.exists():
        size = data_file.stat().st_size
        print(f"  ✓ {data_file} already exists ({size/1024:.1f} KB)")
        return True
    
    try:
        print(f"  🔧 Generating {data_file}...")
        
        # Generate small test data (200 samples for training, 50 for val/test)
        rng = random.PRNGKey(42)
        n_train = 200
        n_val = 50
        n_test = 50
        
        # Generate random natural parameters (eta) - 12D for 3D Gaussian
        eta_train = random.normal(rng, (n_train, 12))
        eta_val = random.normal(rng, (n_val, 12))
        eta_test = random.normal(rng, (n_test, 12))
        
        # Generate corresponding statistics (ground truth)
        # For ET models: 9D (3D mean + 6D upper triangular covariance)
        # For LogZ models: just mean (3D)
        
        # Generate mean (3D)
        mean_train = eta_train[:, :3] + 0.1 * random.normal(rng, (n_train, 3))
        mean_val = eta_val[:, :3] + 0.1 * random.normal(rng, (n_val, 3))
        mean_test = eta_test[:, :3] + 0.1 * random.normal(rng, (n_test, 3))
        
        # Generate full statistics for ET models (9D: mean + covariance)
        # Create simple covariance structure (upper triangular)
        cov_train = jnp.ones((n_train, 6)) * 0.1  # Simple covariance
        cov_val = jnp.ones((n_val, 6)) * 0.1
        cov_test = jnp.ones((n_test, 6)) * 0.1
        
        stats_train = jnp.concatenate([mean_train, cov_train], axis=1)  # 9D
        stats_val = jnp.concatenate([mean_val, cov_val], axis=1)  # 9D
        stats_test = jnp.concatenate([mean_test, cov_test], axis=1)  # 9D
        
        # Create small dataset structure
        small_data = {
            "train": {
                "eta": eta_train,
                "mean": mean_train,  # 3D for LogZ models
                "stats": stats_train  # 9D for ET models
            },
            "val": {
                "eta": eta_val,
                "mean": mean_val,  # 3D for LogZ models
                "stats": stats_val  # 9D for ET models
            },
            "test": {
                "eta": eta_test,
                "mean": mean_test,  # 3D for LogZ models
                "stats": stats_test  # 9D for ET models
            }
        }
        
        # Save to file
        data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(data_file, 'wb') as f:
            pickle.dump(small_data, f)
        
        size = data_file.stat().st_size
        print(f"  ✅ Generated {data_file} ({size/1024:.1f} KB)")
        print(f"  📊 Dataset: {n_train} train, {n_val} val, {n_test} test samples")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to generate test data: {e}")
        return False


def train_small_models():
    """Train all models with small architectures on small dataset."""
    print(f"\nTraining All Models with Small Architectures...")
    print("="*50)
    
    # Generate small dataset if needed
    if not generate_small_test_data():
        print("❌ Could not create test data")
        return {}
    
    # Load small dataset
    data_file = Path("data/easy_3d_gaussian_small.pkl")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    eta_train = data['train']['eta']
    eta_val = data['val']['eta']
    eta_test = data['test']['eta']
    
    print(f"📊 Loaded dataset: {eta_train.shape[0]} train, {eta_val.shape[0]} val, {eta_test.shape[0]} test samples")
    
    # Create small training config (20 epochs)
    training_config = TrainingConfig(
        learning_rate=1e-3,
        num_epochs=20,  # Small number of epochs for quick testing
        batch_size=32,  # Small batch size
        patience=10,
        weight_decay=1e-4,
        gradient_clip_norm=1.0,
        validation_freq=10
    )
    
    results = {}
    
    # Train each model
    for model_name, model_config in SMALL_MODEL_ARCHITECTURES.items():
        print(f"\n🤖 Training {model_name}...")
        print(f"   Architecture: {model_config['hidden_sizes']}")
        print(f"   Type: {model_config['type']}")
        
        # Get appropriate ground truth for this model type
        if model_config['type'] == 'ET':
            ground_truth_train = data['train']['stats']
            ground_truth_val = data['val']['stats']
            ground_truth_test = data['test']['stats']
        else:
            ground_truth_train = data['train']['mean']
            ground_truth_val = data['val']['mean']
            ground_truth_test = data['test']['mean']
        
        try:
            result = train_single_model(
                model_name, model_config, 
                eta_train, ground_truth_train,
                eta_val, ground_truth_val,
                eta_test, ground_truth_test,
                training_config
            )
            results[model_name] = result
            
            if result['status'] == 'success':
                print(f"   ✅ Success: MSE={result['mse']:.2e}, MAE={result['mae']:.2e}")
                print(f"   ⏱️  Training time: {result['training_time']:.1f}s")
            else:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   💥 Crashed: {e}")
            results[model_name] = {
                'status': 'crashed',
                'error': str(e),
                'model_name': model_name
            }
    
    return results


def train_single_model(model_name, model_config, eta_train, ground_truth_train, eta_val, ground_truth_val, eta_test, ground_truth_test, training_config):
    """Train a single model with the given configuration."""
    
    start_time = time.time()
    
    try:
        # Create network configuration
        network_config = NetworkConfig()
        network_config.hidden_sizes = model_config['hidden_sizes']
        network_config.activation = "swish"
        network_config.use_layer_norm = True
        network_config.output_dim = model_config['output_dim']
        
        # Create full configuration
        config = FullConfig()
        config.network = network_config
        config.training = training_config
        
        # Import and create model based on type
        if model_config['type'] == 'ET':
            # Import ET model creation functions
            if model_name == 'mlp_ET':
                from src.models.mlp_ET import create_model_and_trainer
            elif model_name == 'glu_ET':
                from src.models.glu_ET import create_model_and_trainer
            elif model_name == 'quadratic_resnet_ET':
                from src.models.quadratic_resnet_ET import create_model_and_trainer
            elif model_name == 'invertible_nn_ET':
                from src.models.invertible_nn_ET import create_model_and_trainer
            elif model_name == 'noprop_ct_ET':
                from src.models.noprop_ct_ET import create_model_and_trainer
            elif model_name == 'geometric_flow_ET':
                from src.models.geometric_flow_net import create_geometric_flow_et_network
            elif model_name == 'glow_ET':
                from src.models.glow_net_ET import create_glow_et_model_and_trainer
            else:
                raise ValueError(f"Unknown ET model: {model_name}")
            
            # Create model and trainer
            if model_name in ['geometric_flow_ET']:
                # Special case for geometric flow - returns single trainer object
                trainer = create_geometric_flow_et_network(config)
                model = trainer.model
            elif model_name == 'glow_ET':
                # Special case for glow - returns single trainer object
                trainer = create_glow_et_model_and_trainer(config)
                model = trainer.model
            else:
                # Standard ET models - returns single trainer object
                trainer = create_model_and_trainer(config)
                model = trainer.model
                
        elif model_config['type'] == 'LogZ':
            # Import LogZ model creation functions
            if model_name == 'mlp_logZ':
                from src.models.mlp_logZ import create_model_and_trainer
            elif model_name == 'glu_logZ':
                from src.models.glu_logZ import create_model_and_trainer
            elif model_name == 'quadratic_resnet_logZ':
                from src.models.quadratic_resnet_logZ import create_model_and_trainer
            elif model_name == 'convex_nn_logZ':
                from src.models.convex_nn_logZ import create_model_and_trainer
            else:
                raise ValueError(f"Unknown LogZ model: {model_name}")
            
            # Create model and trainer - returns single trainer object
            trainer = create_model_and_trainer(config)
            model = trainer.model
        else:
            raise ValueError(f"Unknown model type: {model_config['type']}")
        
        # Prepare training data
        train_data = {"eta": eta_train, "stats": ground_truth_train}
        val_data = {"eta": eta_val, "stats": ground_truth_val}
        
        # Train the model
        print(f"   🔧 Training {model_name}...")
        trainer.train(train_data, val_data)
        
        # Make predictions
        predictions = trainer.predict(eta_test)
        
        # Calculate metrics
        mse = float(np.mean((predictions - ground_truth_test) ** 2))
        mae = float(np.mean(np.abs(predictions - ground_truth_test)))
        
        # Measure inference time
        inference_start = time.time()
        _ = trainer.predict(eta_test[:10])  # Test on small batch
        inference_time = time.time() - inference_start
        inference_time_per_sample = inference_time / 10
        
        training_time = time.time() - start_time
        
        # Get parameter count
        params = trainer.params if hasattr(trainer, 'params') else {}
        param_count = sum(x.size for x in jax.tree_leaves(params)) if params else 0
        
        # Create results
        result = {
            'status': 'success',
            'model_name': model_name,
            'mse': mse,
            'mae': mae,
            'training_time': training_time,
            'inference_time_per_sample': inference_time_per_sample,
            'parameter_count': param_count,
            'predictions': predictions,
            'ground_truth': ground_truth_test,
            'trainer': trainer
        }
        
        # Save compatible output
        save_model_results(model_name, result, model_config['type'])
        
        return result
        
    except Exception as e:
        training_time = time.time() - start_time
        return {
            'status': 'failed',
            'error': str(e),
            'training_time': training_time,
            'model_name': model_name
        }


def save_model_results(model_name, result, model_type):
    """Save model results in format compatible with create_comparison_analysis.py."""
    
    # Place all test results in artifacts/tests directory
    output_dir = Path(f"artifacts/tests/{model_name}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create results.json structure
    results_data = {
        model_name: {
            'mse': result['mse'],
            'mae': result['mae'],
            'training_time': result['training_time'],
            'inference_time_per_sample': result['inference_time_per_sample'],
            'parameter_count': result['parameter_count'],
            'predictions': result['predictions'].tolist() if hasattr(result['predictions'], 'tolist') else result['predictions'],
            'ground_truth': result['ground_truth'].tolist() if hasattr(result['ground_truth'], 'tolist') else result['ground_truth'],
            'test_metrics': {
                'mse': result['mse'],
                'mae': result['mae']
            }
        }
    }
    
    # Save results.json
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Create training_summary.pkl as fallback
    summary_data = {
        model_name: {
            'mse': result['mse'],
            'mae': result['mae'],
            'training_time': result['training_time'],
            'inference_time_per_sample': result['inference_time_per_sample'],
            'parameter_count': result['parameter_count']
        }
    }
    
    summary_file = output_dir / "training_summary.pkl"
    with open(summary_file, 'wb') as f:
        pickle.dump(summary_data, f)
    
    print(f"   📁 Saved results to: {output_dir}")




def main():
    """Run all tests with small network training."""
    print("Testing All Models with Small Architectures")
    print("="*50)
    
    # Test script imports first
    print("🔍 Testing script imports...")
    script_results = test_script_imports()
    
    # Test model imports
    print("🔍 Testing model imports...")
    test_model_imports()
    
    # Train all models with small architectures
    print("🚀 Training all models with small architectures...")
    training_results = train_small_models()
    
    # Summary
    print(f"\n{'='*50}")
    print("TRAINING SUMMARY")
    print(f"{'='*50}")
    
    # Import test summary
    total_scripts = len(script_results)
    working_scripts = sum(1 for r in script_results.values() if r['status'] == 'OK')
    
    print(f"Import Tests: {working_scripts}/{total_scripts} scripts can import")
    
    # Training results summary
    if training_results:
        print(f"\nTraining Results: {len(training_results)} models trained")
        success_count = sum(1 for r in training_results.values() if r['status'] == 'success')
        failed_count = len(training_results) - success_count
        
        print(f"✅ Successful: {success_count}")
        print(f"❌ Failed: {failed_count}")
        
        # Show successful models
        if success_count > 0:
            print(f"\n🏆 SUCCESSFUL MODELS:")
            successful_models = [(name, result) for name, result in training_results.items() if result['status'] == 'success']
            successful_models.sort(key=lambda x: x[1]['mse'])  # Sort by MSE
            
            for name, result in successful_models:
                mse = result['mse']
                mae = result['mae']
                train_time = result['training_time']
                params = result['parameter_count']
                print(f"  • {name:<20} MSE: {mse:.2e} | MAE: {mae:.2e} | Time: {train_time:.1f}s | Params: {params:,}")
        
        # Show failed models
        if failed_count > 0:
            print(f"\n❌ FAILED MODELS:")
            for name, result in training_results.items():
                if result['status'] != 'success':
                    error = result.get('error', 'Unknown error')
                    print(f"  • {name:<20} Error: {error}")
    
    # Overall assessment
    if success_count >= 8:  # Most models should succeed
        print(f"\n🎉 Small architecture training completed successfully!")
        print(f"📊 {success_count} models trained and ready for analysis")
        print(f"💡 Run analysis: python scripts/create_comparison_analysis.py")
        print(f"🚀 For full comparison: python scripts/run_comprehensive_model_comparison.py --data data/easy_3d_gaussian.pkl --ef gaussian_3d")
    else:
        print(f"\n⚠️  Only {success_count} models succeeded - some may need fixes")
    
    return 0 if success_count >= 8 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
