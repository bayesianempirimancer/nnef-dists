#!/usr/bin/env python3
"""
Debug script to test individual model implementations.

This script quickly tests each standardized model to ensure they work
correctly with the new configuration system and base classes.
"""

import sys
from pathlib import Path
import time
import jax.numpy as jnp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import FullConfig
from src.data_utils import load_3d_gaussian_data, compute_ground_truth_3d_tril


def create_debug_config():
    """Create a minimal configuration for quick testing."""
    config = FullConfig()
    
    # Small architecture for fast debugging
    config.network.hidden_sizes = [32, 32]  # 2 layers x 32 units
    config.network.activation = "tanh"
    config.network.use_feature_engineering = True
    config.network.output_dim = 9
    
    # Quick training
    config.training.learning_rate = 1e-3
    config.training.num_epochs = 10  # Very short for debugging
    config.training.batch_size = 16
    config.training.patience = 5
    
    return config


def test_model(model_name, create_trainer_fn, train_data, val_data, test_data, ground_truth):
    """Test a single model implementation."""
    
    print(f"\n🔧 Testing {model_name}")
    
    try:
        # Create configuration
        config = create_debug_config()
        
        # Model-specific adjustments
        if "Flow" in model_name:
            config.model_specific.num_flow_layers = 5  # Small for debugging
            config.model_specific.flow_hidden_size = 32
        elif "INN" in model_name or "Invertible" in model_name:
            config.model_specific.coupling_type = "additive"
            config.model_specific.use_actnorm = True
        elif "NoProp" in model_name:
            config.model_specific.num_time_steps = 5
            config.model_specific.time_horizon = 1.0
        
        start_time = time.time()
        
        # Create trainer
        trainer = create_trainer_fn(config)
        
        # Test initialization
        sample_input = train_data['eta'][:1]
        params, opt_state = trainer.initialize_model(sample_input)
        param_count = trainer.model.get_parameter_count(params)
        
        print(f"   ✅ Initialization: {param_count:,} parameters")
        
        # Test forward pass
        predictions = trainer.model.apply(params, test_data['eta'][:5], training=False)
        print(f"   ✅ Forward pass: {predictions.shape}")
        
        # Test single training step
        batch = {'eta': train_data['eta'][:16], 'y': train_data['y'][:16]}
        optimizer = trainer.create_optimizer()
        params, opt_state, loss = trainer.train_step(params, opt_state, batch, optimizer)
        print(f"   ✅ Training step: loss={float(loss):.2f}")
        
        # Test evaluation
        metrics = trainer.evaluate(params, test_data, ground_truth)
        print(f"   ✅ Evaluation: MSE={metrics['mse']:.2f}")
        
        test_time = time.time() - start_time
        
        return {
            'status': 'success',
            'parameter_count': param_count,
            'test_time': test_time,
            'metrics': metrics,
            'architecture': config.network.hidden_sizes
        }
        
    except Exception as e:
        test_time = time.time() - start_time
        print(f"   ❌ Failed: {str(e)[:100]}...")
        
        return {
            'status': 'failed',
            'error': str(e),
            'test_time': test_time
        }


def main():
    """Test all individual model implementations."""
    
    print("🔧 TESTING INDIVIDUAL MODEL IMPLEMENTATIONS")
    print("=" * 60)
    print("Quick tests to verify all models work with new framework")
    
    # Load minimal data
    print("\n📊 Loading data...")
    data_dir = Path("data")
    data, ef = load_3d_gaussian_data(data_dir, format="tril")
    
    # Use small subset for fast debugging
    n_debug = 100
    train_data = {"eta": data["train_eta"][:n_debug], "y": data["train_y"][:n_debug]}
    val_data = {"eta": data["val_eta"][:50], "y": data["val_y"][:50]}
    test_data = {"eta": data["val_eta"][50:100], "y": data["val_y"][50:100]}
    
    # Compute ground truth
    ground_truth = compute_ground_truth_3d_tril(test_data['eta'], ef)
    
    print(f"Debug dataset: {n_debug} train, 50 val, 50 test samples")
    
    # Import all model creators
    models_to_test = {}
    
    try:
        from src.models.standard_mlp import create_model_and_trainer
        models_to_test["Standard MLP"] = create_model_and_trainer
    except ImportError as e:
        print(f"⚠️  Could not import Standard MLP: {e}")
    
    try:
        from src.models.deep_flow import create_model_and_trainer as create_flow
        models_to_test["Deep Flow"] = create_flow
    except ImportError as e:
        print(f"⚠️  Could not import Deep Flow: {e}")
    
    try:
        from src.models.quadratic_resnet import create_model_and_trainer as create_quadratic
        models_to_test["Quadratic ResNet"] = create_quadratic
    except ImportError as e:
        print(f"⚠️  Could not import Quadratic ResNet: {e}")
    
    try:
        from src.models.invertible_nn import create_model_and_trainer as create_inn
        models_to_test["Invertible NN"] = create_inn
    except ImportError as e:
        print(f"⚠️  Could not import Invertible NN: {e}")
    
    try:
        from src.models.glu_network import create_model_and_trainer as create_glu
        models_to_test["GLU Network"] = create_glu
    except ImportError as e:
        print(f"⚠️  Could not import GLU Network: {e}")
    
    try:
        from src.models.noprop_ct import create_model_and_trainer as create_noprop
        models_to_test["NoProp-CT"] = create_noprop
    except ImportError as e:
        print(f"⚠️  Could not import NoProp-CT: {e}")
    
    print(f"\nFound {len(models_to_test)} models to test")
    
    # Test each model
    results = {}
    total_start_time = time.time()
    
    for model_name, create_fn in models_to_test.items():
        result = test_model(model_name, create_fn, train_data, val_data, test_data, ground_truth)
        results[model_name] = result
    
    total_time = time.time() - total_start_time
    print(f"\n🏁 All model tests completed in {total_time:.1f}s")
    
    # Summary
    successful_models = [name for name, result in results.items() if result['status'] == 'success']
    failed_models = [name for name, result in results.items() if result['status'] == 'failed']
    
    print(f"\n📊 TEST SUMMARY:")
    print(f"  ✅ Successful models: {len(successful_models)}")
    print(f"  ❌ Failed models: {len(failed_models)}")
    
    if successful_models:
        print(f"\n✅ WORKING MODELS:")
        for name in successful_models:
            result = results[name]
            print(f"  • {name:<20} {result['parameter_count']:>6,} params, {result['test_time']:>5.1f}s, MSE={result['metrics']['mse']:>8.1f}")
    
    if failed_models:
        print(f"\n❌ FAILED MODELS:")
        for name in failed_models:
            error = results[name]['error']
            print(f"  • {name:<20} {error[:80]}...")
    
    # Save debug results
    debug_dir = Path("artifacts/debug_individual_models")
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(debug_dir / "individual_model_tests.json", 'w') as f:
        def convert_for_json(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        debug_data = {
            'test_type': 'individual_model_verification',
            'total_time': total_time,
            'models_tested': len(models_to_test),
            'successful_models': successful_models,
            'failed_models': failed_models,
            'results': convert_for_json(results)
        }
        
        json.dump(debug_data, f, indent=2)
    
    print(f"\n💾 Debug results saved to {debug_dir}/")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if len(successful_models) == len(models_to_test):
        print("  ✅ All models working! Ready for comprehensive comparison")
    else:
        print("  🔧 Fix failed models before running comprehensive experiments")
        for name in failed_models:
            print(f"     - Debug {name}: {results[name]['error'][:50]}...")
    
    print(f"\n✅ Individual model testing completed!")


if __name__ == "__main__":
    main()
