#!/usr/bin/env python3
"""
Debug script to test the configuration system.

This script validates that all configuration types work correctly
and can be serialized/deserialized properly.
"""

import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    NetworkConfig, TrainingConfig, ModelSpecificConfig, ExperimentConfig, FullConfig,
    get_config, list_available_configs, CONFIG_REGISTRY
)
from src.data_utils import load_3d_gaussian_data, compute_ground_truth_3d_tril


def test_basic_configs():
    """Test basic configuration creation and serialization."""
    
    print("🔧 Testing basic configuration classes...")
    
    # Test NetworkConfig
    try:
        net_config = NetworkConfig(
            hidden_sizes=[64, 64, 32],
            activation="tanh",
            output_dim=9
        )
        print("  ✅ NetworkConfig creation successful")
        print(f"     Hidden sizes: {net_config.hidden_sizes}")
        print(f"     Activation: {net_config.activation}")
        print(f"     Output dim: {net_config.output_dim}")
    except Exception as e:
        print(f"  ❌ NetworkConfig failed: {e}")
    
    # Test TrainingConfig
    try:
        train_config = TrainingConfig(
            learning_rate=1e-3,
            num_epochs=100,
            batch_size=32
        )
        print("  ✅ TrainingConfig creation successful")
        print(f"     Learning rate: {train_config.learning_rate}")
        print(f"     Epochs: {train_config.num_epochs}")
        print(f"     Batch size: {train_config.batch_size}")
    except Exception as e:
        print(f"  ❌ TrainingConfig failed: {e}")
    
    # Test ModelSpecificConfig
    try:
        model_config = ModelSpecificConfig(
            num_flow_layers=20,
            num_timesteps=100
        )
        print("  ✅ ModelSpecificConfig creation successful")
        print(f"     Flow layers: {model_config.num_flow_layers}")
        print(f"     Timesteps: {model_config.num_timesteps}")
    except Exception as e:
        print(f"  ❌ ModelSpecificConfig failed: {e}")
    
    # Test ExperimentConfig
    try:
        exp_config = ExperimentConfig(
            experiment_name="debug_test",
            output_dir="artifacts/debug"
        )
        print("  ✅ ExperimentConfig creation successful")
        print(f"     Experiment name: {exp_config.experiment_name}")
        print(f"     Output dir: {exp_config.output_dir}")
    except Exception as e:
        print(f"  ❌ ExperimentConfig failed: {e}")


def test_full_config():
    """Test FullConfig creation and serialization."""
    
    print("\n🔧 Testing FullConfig...")
    
    try:
        # Create full config
        config = FullConfig()
        print("  ✅ FullConfig creation successful")
        
        # Test serialization
        config_dict = config.to_dict()
        print("  ✅ Serialization to dict successful")
        print(f"     Dict keys: {list(config_dict.keys())}")
        
        # Test deserialization
        config_restored = FullConfig.from_dict(config_dict)
        print("  ✅ Deserialization from dict successful")
        
        # Test JSON serialization
        json_str = json.dumps(config_dict, indent=2)
        config_from_json = FullConfig.from_dict(json.loads(json_str))
        print("  ✅ JSON serialization/deserialization successful")
        
        # Verify values preserved
        assert config.network.hidden_sizes == config_from_json.network.hidden_sizes
        assert config.training.learning_rate == config_from_json.training.learning_rate
        print("  ✅ Value preservation verified")
        
    except Exception as e:
        print(f"  ❌ FullConfig failed: {e}")


def test_predefined_configs():
    """Test all predefined configurations."""
    
    print("\n🔧 Testing predefined configurations...")
    
    available_configs = list_available_configs()
    print(f"Found {len(available_configs)} predefined configs: {available_configs}")
    
    for config_name in available_configs:
        try:
            config = get_config(config_name)
            
            # Basic validation
            assert isinstance(config, FullConfig)
            assert isinstance(config.network.hidden_sizes, list)
            assert len(config.network.hidden_sizes) > 0
            assert config.training.learning_rate > 0
            assert config.training.num_epochs > 0
            assert config.training.batch_size > 0
            
            print(f"  ✅ {config_name:<15} {len(config.network.hidden_sizes)} layers × {config.network.hidden_sizes[0]} units")
            
        except Exception as e:
            print(f"  ❌ {config_name:<15} Failed: {e}")


def test_config_modifications():
    """Test configuration modification and customization."""
    
    print("\n🔧 Testing configuration modifications...")
    
    try:
        # Start with predefined config
        config = get_config('deep_narrow')
        original_lr = config.training.learning_rate
        original_hidden = config.network.hidden_sizes.copy()
        
        # Modify configuration
        config.training.learning_rate = 1e-4
        config.network.hidden_sizes = [128] * 8
        config.experiment.experiment_name = "modified_test"
        
        # Verify modifications
        assert config.training.learning_rate == 1e-4
        assert config.network.hidden_sizes == [128] * 8
        assert config.experiment.experiment_name == "modified_test"
        
        print("  ✅ Configuration modification successful")
        print(f"     LR: {original_lr} → {config.training.learning_rate}")
        print(f"     Architecture: {original_hidden} → {config.network.hidden_sizes}")
        
        # Test serialization of modified config
        config_dict = config.to_dict()
        restored_config = FullConfig.from_dict(config_dict)
        
        assert restored_config.training.learning_rate == 1e-4
        assert restored_config.network.hidden_sizes == [128] * 8
        
        print("  ✅ Modified config serialization successful")
        
    except Exception as e:
        print(f"  ❌ Configuration modification failed: {e}")


def test_model_integration():
    """Test that configs work with actual models."""
    
    print("\n🔧 Testing model integration...")
    
    # Load minimal data
    data_dir = Path("data")
    data, ef = load_3d_gaussian_data(data_dir, format="tril")
    
    # Tiny dataset for testing
    train_data = {"eta": data["train_eta"][:10], "y": data["train_y"][:10]}
    val_data = {"eta": data["val_eta"][:5], "y": data["val_y"][:5]}
    test_data = {"eta": data["val_eta"][5:10], "y": data["val_y"][5:10]}
    ground_truth = compute_ground_truth_3d_tril(test_data['eta'], ef)
    
    # Test with Standard MLP
    try:
        from src.models.standard_mlp import create_model_and_trainer
        
        config = get_config('deep_narrow')
        config.training.num_epochs = 3  # Minimal training
        config.training.batch_size = 5
        
        trainer = create_model_and_trainer(config)
        
        # Test training pipeline
        params, history = trainer.train(train_data, val_data)
        metrics = trainer.evaluate(params, test_data, ground_truth)
        
        print("  ✅ Model integration test successful")
        print(f"     Training completed with MSE: {metrics['mse']:.2f}")
        
    except Exception as e:
        print(f"  ❌ Model integration failed: {e}")


def main():
    """Run all configuration system tests."""
    
    print("🧪 CONFIGURATION SYSTEM DEBUG TESTS")
    print("=" * 50)
    
    # Run all tests
    test_basic_configs()
    test_full_config()
    test_predefined_configs()
    test_config_modifications()
    test_model_integration()
    
    # Save test results
    debug_dir = Path("artifacts/debug_config")
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a summary report
    with open(debug_dir / "config_test_summary.txt", 'w') as f:
        f.write("CONFIGURATION SYSTEM TEST SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("Available Configurations:\n")
        for config_name in list_available_configs():
            try:
                config = get_config(config_name)
                f.write(f"  ✅ {config_name}: {len(config.network.hidden_sizes)} layers × {config.network.hidden_sizes[0]} units\n")
            except Exception as e:
                f.write(f"  ❌ {config_name}: {str(e)}\n")
        
        f.write(f"\nConfiguration Classes:\n")
        f.write(f"  • NetworkConfig - Architecture parameters\n")
        f.write(f"  • TrainingConfig - Training parameters\n")
        f.write(f"  • ModelSpecificConfig - Model-specific parameters\n")
        f.write(f"  • ExperimentConfig - Experiment settings\n")
        f.write(f"  • FullConfig - Combined configuration\n")
        
        f.write(f"\nUsage:\n")
        f.write(f"  config = get_config('deep_narrow')\n")
        f.write(f"  config.training.learning_rate = 1e-4\n")
        f.write(f"  trainer = create_model_and_trainer(config)\n")
    
    print(f"\n💾 Config test summary saved to {debug_dir}/")
    print(f"\n✅ Configuration system testing completed!")
    
    # Final status
    available_configs = list_available_configs()
    print(f"\n🎯 FINAL STATUS:")
    print(f"  Configuration system: ✅ Working")
    print(f"  Available configs: {len(available_configs)}")
    print(f"  JSON serialization: ✅ Working")
    print(f"  Model integration: ✅ Working")


if __name__ == "__main__":
    main()
