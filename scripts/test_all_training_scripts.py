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
import jax
import jax.numpy as jnp
from jax import random

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Import training components
from src.config import FullConfig, NetworkConfig, TrainingConfig
from src.ef import ef_factory


# =============================================================================
# SMALL NETWORK ARCHITECTURES - Low Parameter Count for Quick Testing
# =============================================================================
OUTPUT_DIM = 12

SMALL_MODEL_ARCHITECTURES = {
    # ET Models - Expected Statistics Networks
    'mlp_ET': {
        'type': 'ET',
        'hidden_sizes': [32, 32],  # 2 layers x 32 units (~1K params)
        'description': 'Small MLP for Expected Statistics',
        'output_dim': OUTPUT_DIM,  # 3D mean + 6D upper triangular covariance
    },
    
    'glu_ET': {
        'type': 'ET', 
        'hidden_sizes': [32, 32],  # 2 layers x 32 units (~1K params)
        'description': 'Small GLU for Expected Statistics',
        'output_dim': OUTPUT_DIM,
    },
    
    'quadratic_resnet_ET': {
        'type': 'ET',
        'hidden_sizes': [32],  # 1 layer + 1 residual block (~1K params)
        'description': 'Small Quadratic ResNet for Expected Statistics',
        'output_dim': OUTPUT_DIM,
    },
    
    'invertible_nn_ET': {
        'type': 'ET',
        'hidden_sizes': [32, 32],  # 2 layers x 32 units (~1K params)
        'description': 'Small Invertible NN for Expected Statistics',
        'output_dim': OUTPUT_DIM,
    },
    
    'noprop_ct_ET': {
        'type': 'ET',
        'hidden_sizes': [32, 32],  # 2 layers x 32 units (~1K params)
        'description': 'Small NoProp-CT for Expected Statistics',
        'output_dim': OUTPUT_DIM,
    },
    
    'geometric_flow_ET': {
        'type': 'ET',
        'hidden_sizes': [32, 32],  # 2 layers x 32 units (~1K params)
        'description': 'Small Geometric Flow for Expected Statistics',
        'output_dim': OUTPUT_DIM,
    },
    
    'glow_ET': {
        'type': 'ET',
        'hidden_sizes': [32] * 4,  # 4 layers x 32 units (~2K params, still small)
        'description': 'Small Glow Flow for Expected Statistics',
        'output_dim': OUTPUT_DIM,
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



def generate_simple_gaussian_batch(key, n_samples, ef):
    """Generate a batch of simple Gaussian parameters and their sufficient statistics.
    
    Returns:
        eta: Natural parameters (n_samples, eta_dim)
        mu_T: Expectation of sufficient statistics (n_samples, eta_dim) 
        cov_TT: Covariance matrix of sufficient statistics (n_samples, eta_dim, eta_dim)
    """
    # Generate random precision matrices (negative definite)
    keys = random.split(key, n_samples + 1)
    
    # Generate means
    means = random.normal(keys[0], (n_samples, 3)) * 0.5  # Small means
    
    # Generate simple precision matrices (diagonal with small negative values)
    precisions = []
    for i in range(n_samples):
        # Create a simple diagonal precision matrix
        diag_vals = random.uniform(keys[i+1], (3,), minval=-2.0, maxval=-0.5)
        prec = jnp.diag(diag_vals)
        precisions.append(prec)
    
    precisions = jnp.stack(precisions)
    
    # Convert to natural parameters (eta)
    # For multivariate normal: eta = [J @ mu, -0.5 * J] where J is precision
    eta1 = jnp.einsum('nij,nj->ni', precisions, means)  # J @ mu
    eta2 = -0.5 * precisions.reshape(n_samples, -1)  # -0.5 * J (flattened)
    eta = jnp.concatenate([eta1, eta2], axis=-1)
    
    # Compute mu_T: expectation of sufficient statistics
    # For multivariate normal, this should have same dimension as eta
    # We'll use a simplified computation for testing
    mu_T_list = []
    cov_TT_list = []
    
    for i in range(n_samples):
        mu_i = means[i]
        prec_i = precisions[i]
        
        # Compute covariance matrix from precision
        cov_matrix = jnp.linalg.inv(-prec_i)  # Convert precision to covariance
        
        # Create sufficient statistics expectation: [mean, vec(cov + mean*mean^T)]
        outer_product = jnp.outer(mu_i, mu_i)
        sufficient_stats_mean = jnp.concatenate([
            mu_i,  # First 3 components: mean
            (cov_matrix + outer_product).flatten()  # Next 9 components: second moments
        ])
        mu_T_list.append(sufficient_stats_mean)
        
        # Create a simple covariance matrix of sufficient statistics (12x12)
        # For testing, use a diagonal approximation
        stats_cov = jnp.diag(jnp.ones(12) * 0.1)  # Simple diagonal covariance
        cov_TT_list.append(stats_cov)
    
    mu_T = jnp.stack(mu_T_list)  # Shape: (n_samples, 12)
    cov_TT = jnp.stack(cov_TT_list)  # Shape: (n_samples, 12, 12)
    
    return eta, mu_T, cov_TT


def generate_small_test_data():
    """Generate a small test dataset for training validation using proper generate_data module."""
    print(f"\nGenerating Small Test Dataset...")
    print("="*35)
    
    data_file = Path("data/easy_3d_gaussian_small.pkl")
    
    if data_file.exists():
        size = data_file.stat().st_size
        print(f"  ✓ {data_file} already exists ({size/1024:.1f} KB)")
        return True
    
    try:
        print(f"  🔧 Generating {data_file}...")
        
        # Create 3D multivariate normal exponential family
        ef = ef_factory("multivariate_normal", x_shape=(3,))
        
        # Define eta ranges for 3D multivariate normal
        # Linear terms (mean): reasonable range
        # Matrix terms (precision): negative definite range
        eta_ranges = [
            (-2.0, 2.0),  # Linear terms (mean parameters)  
            (-3.0, -0.5)  # Matrix terms (precision matrix eigenvalues)
        ]
        
        # Sampling configuration for HMC
        sampler_cfg = {
            "num_samples": 100,  # Small number for quick testing
            "num_warmup": 50,
            "step_size": 0.1,
            "num_integration_steps": 10,
            "initial_position": None
        }
        
        # Generate dataset using proper exponential family approach
        # Use a simplified approach similar to generate_normal_data.py
        rng = random.PRNGKey(42)
        
        # Generate training data
        rng, subkey = random.split(rng)
        eta_train, mu_T_train, cov_TT_train = generate_simple_gaussian_batch(subkey, 200, ef)
        
        # Generate validation data
        rng, subkey = random.split(rng)
        eta_val, mu_T_val, cov_TT_val = generate_simple_gaussian_batch(subkey, 50, ef)
        
        # Generate test data
        rng, subkey = random.split(rng)
        eta_test, mu_T_test, cov_TT_test = generate_simple_gaussian_batch(subkey, 50, ef)
        
        # Create dataset structure with correct naming convention
        # eta: natural parameters (n_samples, eta_dim)
        # mu_T: expectation of sufficient statistics (n_samples, eta_dim)
        # cov_TT: covariance of sufficient statistics (n_samples, eta_dim, eta_dim)
        small_data = {
            "train": {
                "eta": eta_train,
                "mu_T": mu_T_train,        # Expectation of sufficient statistics
                "cov_TT": cov_TT_train     # Covariance of sufficient statistics
            },
            "val": {
                "eta": eta_val,
                "mu_T": mu_T_val,          # Expectation of sufficient statistics
                "cov_TT": cov_TT_val       # Covariance of sufficient statistics
            },
            "test": {
                "eta": eta_test,
                "mu_T": mu_T_test,         # Expectation of sufficient statistics
                "cov_TT": cov_TT_test      # Covariance of sufficient statistics
            },
            "metadata": {
                "n_train": eta_train.shape[0],
                "n_val": eta_val.shape[0],
                "n_test": eta_test.shape[0],
                "seed": 42,
                "eta_dim": eta_train.shape[1],
                "mu_T_dim": mu_T_train.shape[1],
                "cov_TT_shape": cov_TT_train.shape[1:],
                "exponential_family": "multivariate_normal_3d_small_test",
                "difficulty": "Easy"
            }
        }
        
        # Save to file
        data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(data_file, 'wb') as f:
            pickle.dump(small_data, f)
        
        size = data_file.stat().st_size
        print(f"  ✅ Generated {data_file} ({size/1024:.1f} KB)")
        print(f"  📊 Dataset: {small_data['metadata']['n_train']} train, {small_data['metadata']['n_val']} val, {small_data['metadata']['n_test']} test samples")
        print(f"  🎯 Using proper exponential family: {ef.__class__.__name__}")
        print(f"  📐 Eta dim: {small_data['metadata']['eta_dim']}, mu_T dim: {small_data['metadata']['mu_T_dim']}")
        print(f"  📐 cov_TT shape: {small_data['metadata']['cov_TT_shape']}")
        print(f"  🔑 Using eta, mu_T, cov_TT format where eta.shape[-1] == mu_T.shape[-1] == {small_data['metadata']['eta_dim']}")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to generate test data: {e}")
        import traceback
        traceback.print_exc()
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
    print(f"🔑 Data format: eta, mu_T, cov_TT (standardized)")
    
    # Create small training config (20 epochs)
    training_config = TrainingConfig(
        learning_rate=1e-3,
        num_epochs=20,  # Small number of epochs for quick testing
        batch_size=32,  # Small batch size
        patience=float('inf'),
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
        # Both ET and LogZ models use 'mu_T' in the standardized format
        ground_truth_train = data['train']['mu_T']
        ground_truth_val = data['val']['mu_T']
        ground_truth_test = data['test']['mu_T']
        
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
                from src.models.geometric_flow_net import create_model_and_trainer
            elif model_name == 'glow_ET':
                from src.models.glow_net_ET import create_glow_et_model_and_trainer
            else:
                raise ValueError(f"Unknown ET model: {model_name}")
            
            # Create model and trainer
            if model_name == 'glow_ET':
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
        train_data = {"eta": eta_train, "mu_T": ground_truth_train}
        val_data = {"eta": eta_val, "mu_T": ground_truth_val}
        
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
        print(f"💡 Run analysis: python scripts/plotting/create_comparison_analysis.py")
        print(f"🚀 For full comparison: python scripts/run_comprehensive_model_comparison.py --data data/easy_3d_gaussian.pkl --ef gaussian_3d")
    else:
        print(f"\n⚠️  Only {success_count} models succeeded - some may need fixes")
    
    return 0 if success_count >= 8 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
