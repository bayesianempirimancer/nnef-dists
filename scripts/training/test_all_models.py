#!/usr/bin/env python3
"""
Comprehensive test script for all neural network models.

This script tests all models in the src/models directory to ensure they give
sensible results. It trains each model briefly and reports performance metrics.

Usage:
    python scripts/training/test_all_models.py --epochs 50 --plot
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import with proper module structure
try:
    from config import FullConfig
    from utils.data_utils import generate_exponential_family_data
    from ef import MultivariateNormal
except ImportError:
    # Fallback for direct execution
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", Path(__file__).parent.parent.parent / "src" / "config.py")
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    FullConfig = config_module.FullConfig
    
    spec = importlib.util.spec_from_file_location("data_utils", Path(__file__).parent.parent.parent / "src" / "utils" / "data_utils.py")
    data_utils_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_utils_module)
    generate_exponential_family_data = data_utils_module.generate_exponential_family_data
    
    spec = importlib.util.spec_from_file_location("ef", Path(__file__).parent.parent.parent / "src" / "ef.py")
    ef_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ef_module)
    MultivariateNormal = ef_module.MultivariateNormal

def create_test_config():
    """Create a test configuration."""
    config = FullConfig()
    config.network.exp_family = "multivariate_normal_3d"
    config.network.hidden_sizes = [64, 64]
    config.network.activation = "swish"
    config.training.n_samples = 500
    config.training.learning_rate = 1e-3
    config.training.batch_size = 32
    return config

def test_logZ_model(model_name, model_class, trainer_class, config, eta_data, ground_truth, ef, epochs=50):
    """Test a logZ model (outputs scalar, uses gradients)."""
    print(f"\n{'='*60}")
    print(f"TESTING {model_name.upper()} LOGZ MODEL")
    print(f"{'='*60}")
    
    try:
        # Create trainer
        trainer = trainer_class(config)
        
        # Initialize parameters
        rng = random.PRNGKey(42)
        trainer.params = trainer.model.init(rng, eta_data[:1])
        
        param_count = sum(x.size for x in jax.tree.leaves(trainer.params))
        print(f"Model parameters: {param_count:,}")
        
        # Training loop
        losses = []
        start_time = time.time()
        
        for epoch in range(epochs):
            # Shuffle data
            rng, shuffle_rng = random.split(rng)
            indices = random.permutation(shuffle_rng, len(eta_data))
            eta_batch = eta_data[indices]
            target_batch = ground_truth[indices]
            
            # Train epoch
            trainer.params, trainer.opt_state, loss = trainer.train_epoch(
                trainer.params, trainer.opt_state, eta_batch, target_batch, ef
            )
            
            losses.append(float(loss))
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}: Loss = {loss:.6f}")
        
        training_time = time.time() - start_time
        
        # Compute final predictions
        predictions = trainer.compute_predictions(trainer.params, eta_data)
        
        # Performance metrics
        mse = np.mean((predictions - ground_truth) ** 2)
        mae = np.mean(np.abs(predictions - ground_truth))
        r2 = 1 - np.sum((ground_truth - predictions) ** 2) / np.sum((ground_truth - np.mean(ground_truth)) ** 2)
        
        print(f"\nFINAL PERFORMANCE:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  R²:  {r2:.4f}")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Parameters: {param_count:,}")
        
        # Check for numerical issues
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            print("  ⚠️  WARNING: Model produced NaN or Inf values!")
            return False
        
        if np.any(np.isnan(losses)) or np.any(np.isinf(losses)):
            print("  ⚠️  WARNING: Training produced NaN or Inf losses!")
            return False
        
        print("  ✅ Model training completed successfully!")
        return True
        
    except Exception as e:
        print(f"  ❌ ERROR: {str(e)}")
        return False

def test_ET_model(model_name, model_class, trainer_class, config, eta_data, ground_truth, ef, epochs=50):
    """Test an ET model (directly predicts expectations)."""
    print(f"\n{'='*60}")
    print(f"TESTING {model_name.upper()} ET MODEL")
    print(f"{'='*60}")
    
    try:
        # Create trainer
        trainer = trainer_class(config)
        
        # Initialize parameters
        rng = random.PRNGKey(42)
        trainer.params = trainer.model.init(rng, eta_data[:1])
        
        param_count = sum(x.size for x in jax.tree.leaves(trainer.params))
        print(f"Model parameters: {param_count:,}")
        
        # Training loop
        losses = []
        start_time = time.time()
        
        for epoch in range(epochs):
            # Shuffle data
            rng, shuffle_rng = random.split(rng)
            indices = random.permutation(shuffle_rng, len(eta_data))
            eta_batch = eta_data[indices]
            target_batch = ground_truth[indices]
            
            # Train epoch
            trainer.params, trainer.opt_state, loss = trainer.train_epoch(
                trainer.params, trainer.opt_state, eta_batch, target_batch, ef
            )
            
            losses.append(float(loss))
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}: Loss = {loss:.6f}")
        
        training_time = time.time() - start_time
        
        # Compute final predictions
        predictions = trainer.compute_predictions(trainer.params, eta_data)
        
        # Performance metrics
        mse = np.mean((predictions - ground_truth) ** 2)
        mae = np.mean(np.abs(predictions - ground_truth))
        r2 = 1 - np.sum((ground_truth - predictions) ** 2) / np.sum((ground_truth - np.mean(ground_truth)) ** 2)
        
        print(f"\nFINAL PERFORMANCE:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  R²:  {r2:.4f}")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Parameters: {param_count:,}")
        
        # Check for numerical issues
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            print("  ⚠️  WARNING: Model produced NaN or Inf values!")
            return False
        
        if np.any(np.isnan(losses)) or np.any(np.isinf(losses)):
            print("  ⚠️  WARNING: Training produced NaN or Inf losses!")
            return False
        
        print("  ✅ Model training completed successfully!")
        return True
        
    except Exception as e:
        print(f"  ❌ ERROR: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test all neural network models')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs for each model')
    parser.add_argument('--samples', type=int, default=500,
                       help='Number of training samples')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')
    
    args = parser.parse_args()
    
    # Set up output directory
    artifacts_dir = Path(__file__).parent.parent.parent / "artifacts"
    individual_tests_dir = artifacts_dir / "individual_tests"
    individual_tests_dir.mkdir(parents=True, exist_ok=True)
    
    print("COMPREHENSIVE MODEL TESTING")
    print("="*60)
    print(f"Testing all models with {args.epochs} epochs and {args.samples} samples")
    
    # Create test configuration
    config = create_test_config()
    config.training.n_samples = args.samples
    
    # Generate test data
    print("\nGenerating test data...")
    eta_data, ground_truth = generate_exponential_family_data(
        exp_family=config.network.exp_family,
        n_samples=args.samples,
        seed=42
    )
    
    print(f"Data shapes: eta={eta_data.shape}, ground_truth={ground_truth.shape}")
    
    # Create exponential family instance
    ef = MultivariateNormal(dim=3)
    
    # Test results
    results = {}
    
    # Test logZ models
    print("\n" + "="*60)
    print("TESTING LOGZ MODELS (Gradient-based)")
    print("="*60)
    
    # MLP LogZ
    try:
        from models.mlp_logZ import MLPLogNormalizerNetwork, MLPLogNormalizerTrainer
        results['mlp_logZ'] = test_logZ_model(
            'mlp', MLPLogNormalizerNetwork, MLPLogNormalizerTrainer, 
            config, eta_data, ground_truth, ef, args.epochs
        )
    except ImportError as e:
        print(f"❌ Could not import MLP LogZ: {e}")
        results['mlp_logZ'] = False
    
    # Quadratic ResNet LogZ
    try:
        from models.quadratic_resnet_logZ import QuadraticResNetLogNormalizer, QuadraticResNetLogNormalizerTrainer
        results['quadratic_resnet_logZ'] = test_logZ_model(
            'quadratic_resnet', QuadraticResNetLogNormalizer, QuadraticResNetLogNormalizerTrainer,
            config, eta_data, ground_truth, ef, args.epochs
        )
    except ImportError as e:
        print(f"❌ Could not import Quadratic ResNet LogZ: {e}")
        results['quadratic_resnet_logZ'] = False
    
    # GLU LogZ
    try:
        from models.glu_logZ import GLULogNormalizerNetwork, GLULogNormalizerTrainer
        results['glu_logZ'] = test_logZ_model(
            'glu', GLULogNormalizerNetwork, GLULogNormalizerTrainer,
            config, eta_data, ground_truth, ef, args.epochs
        )
    except ImportError as e:
        print(f"❌ Could not import GLU LogZ: {e}")
        results['glu_logZ'] = False
    
    # Test ET models
    print("\n" + "="*60)
    print("TESTING ET MODELS (Direct prediction)")
    print("="*60)
    
    # Standard MLP ET
    try:
        from models.standard_mlp_ET import StandardMLPNetwork, StandardMLPTrainer
        results['standard_mlp_ET'] = test_ET_model(
            'standard_mlp', StandardMLPNetwork, StandardMLPTrainer,
            config, eta_data, ground_truth, ef, args.epochs
        )
    except ImportError as e:
        print(f"❌ Could not import Standard MLP ET: {e}")
        results['standard_mlp_ET'] = False
    
    # Deep Flow ET
    try:
        from models.glow_net_ET import GlowNetworkET, GlowTrainerET
        results['glow_net_ET'] = test_ET_model(
            'glow_net', GlowNetworkET, GlowTrainerET,
            config, eta_data, ground_truth, ef, args.epochs
        )
    except ImportError as e:
        print(f"❌ Could not import Glow Network ET: {e}")
        results['glow_net_ET'] = False
    
    # GLU ET
    try:
        from models.glu_ET import GLUNetwork, GLUTrainer
        results['glu_ET'] = test_ET_model(
            'glu', GLUNetwork, GLUTrainer,
            config, eta_data, ground_truth, ef, args.epochs
        )
    except ImportError as e:
        print(f"❌ Could not import GLU ET: {e}")
        results['glu_ET'] = False
    
    # Invertible NN ET
    try:
        from models.invertible_nn_ET import InvertibleNNNetwork, InvertibleNNTrainer
        results['invertible_nn_ET'] = test_ET_model(
            'invertible_nn', InvertibleNNNetwork, InvertibleNNTrainer,
            config, eta_data, ground_truth, ef, args.epochs
        )
    except ImportError as e:
        print(f"❌ Could not import Invertible NN ET: {e}")
        results['invertible_nn_ET'] = False
    
    # NoProp CT ET
    try:
        from models.noprop_ct_ET import NoPropCTNetwork, NoPropCTTrainer
        results['noprop_ct_ET'] = test_ET_model(
            'noprop_ct', NoPropCTNetwork, NoPropCTTrainer,
            config, eta_data, ground_truth, ef, args.epochs
        )
    except ImportError as e:
        print(f"❌ Could not import NoProp CT ET: {e}")
        results['noprop_ct_ET'] = False
    
    # Quadratic ResNet ET
    try:
        from models.quadratic_resnet_ET import QuadraticResNetNetwork, QuadraticResNetTrainer
        results['quadratic_resnet_ET'] = test_ET_model(
            'quadratic_resnet', QuadraticResNetNetwork, QuadraticResNetTrainer,
            config, eta_data, ground_truth, ef, args.epochs
        )
    except ImportError as e:
        print(f"❌ Could not import Quadratic ResNet ET: {e}")
        results['quadratic_resnet_ET'] = False
    
    # Summary
    print("\n" + "="*60)
    print("TESTING SUMMARY")
    print("="*60)
    
    successful_models = []
    failed_models = []
    
    for model_name, success in results.items():
        if success:
            successful_models.append(model_name)
        else:
            failed_models.append(model_name)
    
    print(f"\n✅ SUCCESSFUL MODELS ({len(successful_models)}):")
    for model in successful_models:
        print(f"  - {model}")
    
    if failed_models:
        print(f"\n❌ FAILED MODELS ({len(failed_models)}):")
        for model in failed_models:
            print(f"  - {model}")
    
    print(f"\nOverall: {len(successful_models)}/{len(results)} models working correctly")
    
    if len(successful_models) == len(results):
        print("🎉 All models are working correctly!")
    else:
        print("⚠️  Some models need attention.")

if __name__ == "__main__":
    main()
