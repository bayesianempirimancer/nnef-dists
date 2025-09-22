#!/usr/bin/env python3
"""
Comprehensive Model Comparison Script

This script trains all 11 available models on the same dataset with comparable
parameter counts, using moderately deep networks (~12 layers) and moderate width (~128 units).
The Glow network is an exception and uses ~24 layers as specified.

Usage:
    python scripts/run_comprehensive_model_comparison.py --data data/easy_3d_gaussian.pkl --ef gaussian_3d
    python scripts/run_comprehensive_model_comparison.py --data data/challenging_5d_gaussian.pkl --ef gaussian_5d
    python scripts/run_comprehensive_model_comparison.py --data data/easy_3d_gaussian.pkl --ef gaussian_3d --force-retrain
"""

import argparse
import sys
import json
import pickle
import time
from pathlib import Path
import jax.numpy as jnp
from jax import random
import subprocess
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import standardized plotting functions
sys.path.append(str(Path(__file__).parent))
from plot_training_results import plot_training_results, plot_model_comparison, save_results_summary

from src.config import FullConfig, NetworkConfig, TrainingConfig


# =============================================================================
# MODEL ARCHITECTURES - Comparable Parameter Counts
# =============================================================================

MODEL_ARCHITECTURES = {
    # ET Models - Expected Statistics Networks
    'mlp_ET': {
        'type': 'ET',
        'hidden_sizes': [128] * 12,  # 12 layers x 128 units
        'description': 'Multi-Layer Perceptron for Expected Statistics',
        'output_dim': 9,  # 3D mean + 6D upper triangular covariance
    },
    
    'glu_ET': {
        'type': 'ET', 
        'hidden_sizes': [128] * 12,  # 12 layers x 128 units
        'description': 'Gated Linear Unit for Expected Statistics',
        'output_dim': 9,
    },
    
    'quadratic_resnet_ET': {
        'type': 'ET',
        'hidden_sizes': [128] * 10,  # 10 layers + 2 residual blocks = ~12 effective layers
        'description': 'Quadratic ResNet for Expected Statistics',
        'output_dim': 9,
    },
    
    'invertible_nn_ET': {
        'type': 'ET',
        'hidden_sizes': [128] * 12,  # 12 layers x 128 units
        'description': 'Invertible Neural Network for Expected Statistics',
        'output_dim': 9,
    },
    
    'noprop_ct_ET': {
        'type': 'ET',
        'hidden_sizes': [128] * 12,  # 12 layers x 128 units
        'description': 'No-Propagation Continuous-Time for Expected Statistics',
        'output_dim': 9,
    },
    
    'geometric_flow_ET': {
        'type': 'ET',
        'hidden_sizes': [128] * 12,  # 12 layers x 128 units
        'description': 'Geometric Flow for Expected Statistics',
        'output_dim': 9,
    },
    
    'glow_ET': {
        'type': 'ET',
        'hidden_sizes': [128] * 24,  # 24 layers x 128 units (2x as many as others)
        'description': 'Glow Normalizing Flow for Expected Statistics',
        'output_dim': 9,
    },
    
    # LogZ Models - Log Normalizer Networks
    'mlp_logZ': {
        'type': 'LogZ',
        'hidden_sizes': [128] * 12,  # 12 layers x 128 units
        'description': 'Multi-Layer Perceptron for Log Normalizer',
        'output_dim': 1,  # Scalar log normalizer
    },
    
    'glu_logZ': {
        'type': 'LogZ',
        'hidden_sizes': [128] * 12,  # 12 layers x 128 units
        'description': 'Gated Linear Unit for Log Normalizer',
        'output_dim': 1,
    },
    
    'quadratic_resnet_logZ': {
        'type': 'LogZ',
        'hidden_sizes': [128] * 10,  # 10 layers + 2 residual blocks = ~12 effective layers
        'description': 'Quadratic ResNet for Log Normalizer',
        'output_dim': 1,
    },
    
    'convex_nn_logZ': {
        'type': 'LogZ',
        'hidden_sizes': [128, 96, 80, 64, 48, 40, 32, 24, 20, 16, 12, 8],  # 12 layers, decreasing width
        'description': 'Convex Neural Network for Log Normalizer',
        'output_dim': 1,
    },
}


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

def create_training_config():
    """Create standardized training configuration."""
    return TrainingConfig(
        learning_rate=1e-3,
        num_epochs=200,
        batch_size=64,
        patience=25,
        weight_decay=1e-4,
        gradient_clip_norm=1.0,
        save_best=True,
        save_frequency=50
    )


def create_network_config(hidden_sizes, output_dim, model_type):
    """Create network configuration for a specific model."""
    return NetworkConfig(
        hidden_sizes=hidden_sizes,
        output_dim=output_dim,
        activation="swish",
        use_layer_norm=True,
        dropout_rate=0.1,
        use_feature_engineering=(model_type == "ET")  # Only ET models use feature engineering
    )


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_data(data_file_path, ef_distribution, purge_cov_tt=True):
    """Load and preprocess data for training."""
    data_file = Path(data_file_path)
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    print(f"📊 Loading data from: {data_file}")
    print(f"🎯 Exponential Family: {ef_distribution}")
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    # Extract training data
    eta_data = data["train"]["eta"]
    ground_truth = data["train"]["mean"]
    
    print(f"📈 Data shapes: eta={eta_data.shape}, ground_truth={ground_truth.shape}")
    
    # Purge cov_tt to save memory if requested
    if purge_cov_tt:
        if "cov" in data["train"]: del data["train"]["cov"]
        if "cov" in data["val"]: del data["val"]["cov"]
        if "cov" in data["test"]: del data["test"]["cov"]
        import gc
        gc.collect()
        print("✅ Purged cov_tt elements from memory for optimization")
    
    return eta_data, ground_truth, data


def convert_to_9d(eta_12d, mean_12d):
    """Convert 12D data to 9D format for models expecting upper triangular covariance."""
    # Extract components
    eta1 = eta_12d[:, :3]  # 3D
    eta2_flat = eta_12d[:, 3:12]  # 9D
    mu = mean_12d[:, :3]  # 3D
    mu_muT_plus_Sigma_flat = mean_12d[:, 3:12]  # 9D
    
    # Reshape to 3x3 matrices and extract upper triangular
    eta2_matrices = eta2_flat.reshape(-1, 3, 3)
    mu_muT_plus_Sigma_matrices = mu_muT_plus_Sigma_flat.reshape(-1, 3, 3)
    
    # Extract upper triangular elements (6 elements per matrix)
    upper_indices = jnp.triu_indices(3)
    eta2_upper = eta2_matrices[:, upper_indices[0], upper_indices[1]]  # (N, 6)
    mu_muT_plus_Sigma_upper = mu_muT_plus_Sigma_matrices[:, upper_indices[0], upper_indices[1]]  # (N, 6)
    
    # Combine to 9D: [mu (3D), upper_elements (6D)]
    eta_9d = jnp.concatenate([eta1, eta2_upper], axis=1)
    mean_9d = jnp.concatenate([mu, mu_muT_plus_Sigma_upper], axis=1)
    
    return eta_9d, mean_9d


# =============================================================================
# COMPLETION CHECKING
# =============================================================================

def is_model_completed(model_name, output_dir):
    """Check if a model has already been trained and plotted successfully."""
    model_output_dir = Path(output_dir) / model_name
    
    if not model_output_dir.exists():
        return False
    
    # Check for key files that indicate successful completion
    required_files = [
        'results.json',
        'model_comparison.png',
        'training_summary.pkl'
    ]
    
    for file_name in required_files:
        if not (model_output_dir / file_name).exists():
            return False
    
    # Check if results.json contains valid results
    try:
        results_file = model_output_dir / 'results.json'
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Check if it has the expected structure
        if model_name not in results:
            return False
        
        model_result = results[model_name]
        if 'mse' not in model_result.get('test_metrics', {}):
            return False
        
        print(f"✅ {model_name} already completed - skipping")
        return True
        
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        return False


# =============================================================================
# MODEL TRAINING FUNCTIONS
# =============================================================================

def train_model(model_name, model_config, eta_data, ground_truth, training_config):
    """Train a single model and return results."""
    print(f"\n🚀 Training {model_name}...")
    print(f"   Architecture: {model_config['description']}")
    print(f"   Hidden sizes: {model_config['hidden_sizes']}")
    print(f"   Output dim: {model_config['output_dim']}")
    
    start_time = time.time()
    
    try:
        # Create configuration
        config = FullConfig()
        config.network = create_network_config(
            model_config['hidden_sizes'], 
            model_config['output_dim'], 
            model_config['type']
        )
        config.training = training_config
        
        # Prepare data based on model type
        if model_config['output_dim'] == 9:
            # Convert 12D data to 9D for ET models
            eta_train, ground_truth_train = convert_to_9d(eta_data, ground_truth)
        else:
            # Use 12D data directly for LogZ models
            eta_train, ground_truth_train = eta_data, ground_truth
        
        # Import and create model based on type
        if model_config['type'] == 'ET':
            trainer = create_et_trainer(model_name, config)
        else:
            trainer = create_logz_trainer(model_name, config)
        
        # Train the model
        trainer.fit(
            eta_train, 
            ground_truth_train,
            epochs=training_config.num_epochs,
            batch_size=training_config.batch_size,
            learning_rate=training_config.learning_rate
        )
        
        training_time = time.time() - start_time
        
        # Measure inference time (single prediction)
        inference_start = time.time()
        predictions = trainer.predict(eta_train)
        inference_time = time.time() - inference_start
        
        # Calculate metrics
        mse = float(jnp.mean((predictions - ground_truth_train) ** 2))
        mae = float(jnp.mean(jnp.abs(predictions - ground_truth_train)))
        
        print(f"   ✅ {model_name} completed in {training_time:.1f}s")
        print(f"   ⚡ Inference time: {inference_time:.3f}s for {len(eta_train)} samples")
        print(f"   📊 MSE: {mse:.6f}, MAE: {mae:.6f}")
        
        return {
            'status': 'success',
            'model_name': model_name,
            'config': model_config,
            'mse': mse,
            'mae': mae,
            'training_time': training_time,
            'inference_time': inference_time,
            'inference_time_per_sample': inference_time / len(eta_train),
            'predictions': predictions,
            'ground_truth': ground_truth_train,
            'trainer': trainer
        }
        
    except Exception as e:
        training_time = time.time() - start_time
        print(f"   ❌ {model_name} failed after {training_time:.1f}s: {e}")
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'model_name': model_name,
            'config': model_config,
            'error': str(e),
            'training_time': training_time
        }


def create_et_trainer(model_name, config):
    """Create ET trainer based on model name."""
    if model_name == 'mlp_ET':
        from src.models.mlp_ET import create_model_and_trainer
        return create_model_and_trainer(config)
    elif model_name == 'glu_ET':
        from src.models.glu_ET import create_model_and_trainer
        return create_model_and_trainer(config)
    elif model_name == 'quadratic_resnet_ET':
        from src.models.quadratic_resnet_ET import create_model_and_trainer
        return create_model_and_trainer(config)
    elif model_name == 'invertible_nn_ET':
        from src.models.invertible_nn_ET import create_model_and_trainer
        return create_model_and_trainer(config)
    elif model_name == 'noprop_ct_ET':
        from src.models.noprop_ct_ET import create_model_and_trainer
        return create_model_and_trainer(config)
    elif model_name == 'geometric_flow_ET':
        from src.models.geometric_flow_net import create_model_and_trainer
        return create_model_and_trainer(config)
    elif model_name == 'glow_ET':
        from src.models.glow_net_ET import create_glow_et_model_and_trainer
        return create_glow_et_model_and_trainer(config)
    else:
        raise ValueError(f"Unknown ET model: {model_name}")


def create_logz_trainer(model_name, config):
    """Create LogZ trainer based on model name."""
    if model_name == 'mlp_logZ':
        from src.models.mlp_logZ import create_model_and_trainer
        return create_model_and_trainer(config)
    elif model_name == 'glu_logZ':
        from src.models.glu_logZ import create_model_and_trainer
        return create_model_and_trainer(config)
    elif model_name == 'quadratic_resnet_logZ':
        from src.models.quadratic_resnet_logZ import create_model_and_trainer
        return create_model_and_trainer(config)
    elif model_name == 'convex_nn_logZ':
        from src.models.convex_nn_logZ import create_model_and_trainer
        return create_model_and_trainer(config)
    else:
        raise ValueError(f"Unknown LogZ model: {model_name}")


# =============================================================================
# MAIN COMPARISON FUNCTION
# =============================================================================

def run_comprehensive_comparison(data_file, ef_distribution, output_dir="artifacts/comprehensive_comparison", force_retrain=False):
    """Run comprehensive comparison of all models."""
    
    print("🎯 COMPREHENSIVE MODEL COMPARISON")
    print("=" * 60)
    print(f"📊 Data: {data_file}")
    print(f"🎯 Distribution: {ef_distribution}")
    print(f"📁 Output: {output_dir}")
    print(f"🤖 Models: {len(MODEL_ARCHITECTURES)}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    eta_data, ground_truth, full_data = load_data(data_file, ef_distribution)
    
    # Create training configuration
    training_config = create_training_config()
    
    # Train all models (skip completed ones)
    results = {}
    successful_models = []
    failed_models = []
    skipped_models = []
    
    for model_name, model_config in MODEL_ARCHITECTURES.items():
        # Check if model was already completed (unless force retrain is enabled)
        if not force_retrain and is_model_completed(model_name, output_path):
            skipped_models.append(model_name)
            # Load existing results
            model_output_dir = output_path / model_name
            results_file = model_output_dir / 'results.json'
            with open(results_file, 'r') as f:
                existing_results = json.load(f)
            results[model_name] = existing_results[model_name]
            successful_models.append(model_name)
            continue
        
        # Train the model
        result = train_model(model_name, model_config, eta_data, ground_truth, training_config)
        results[model_name] = result
        
        if result['status'] == 'success':
            successful_models.append(model_name)
        else:
            failed_models.append(model_name)
    
    # Print summary
    print(f"\n📊 TRAINING SUMMARY")
    print("=" * 40)
    print(f"✅ Successful: {len(successful_models)}/{len(MODEL_ARCHITECTURES)}")
    print(f"⏭️  Skipped (already completed): {len(skipped_models)}")
    print(f"❌ Failed: {len(failed_models)}/{len(MODEL_ARCHITECTURES)}")
    
    if skipped_models:
        print(f"\n⏭️  SKIPPED MODELS (already completed):")
        for name in skipped_models:
            print(f"  • {name}")
    
    if successful_models:
        print(f"\n🏆 PERFORMANCE RANKING (by MSE):")
        successful_results = [(name, results[name]['mse']) for name in successful_models]
        successful_results.sort(key=lambda x: x[1])
        
        for i, (name, mse) in enumerate(successful_results, 1):
            result = results[name]
            training_time = result.get('training_time', 0)
            inference_time = result.get('inference_time_per_sample', 0) * 1000  # Convert to ms
            print(f"  {i:2d}. {name:<20} MSE: {mse:.6f} | Train: {training_time:.1f}s | Infer: {inference_time:.2f}ms/sample")
    
    if failed_models:
        print(f"\n❌ FAILED MODELS:")
        for name in failed_models:
            print(f"  • {name}: {results[name]['error']}")
    
    # Create comparison plots if we have successful models
    if len(successful_models) >= 2:
        print(f"\n📊 Creating comparison plots...")
        
        # Prepare results for plotting
        plot_results = {}
        for model_name in successful_models:
            result = results[model_name]
            plot_results[model_name] = {
                'train_loss': [],  # Would need to extract from trainer
                'test_metrics': {
                    'mse': result['mse'],
                    'mae': result['mae'],
                    'training_time': result.get('training_time', 0),
                    'inference_time': result.get('inference_time', 0),
                    'inference_time_per_sample': result.get('inference_time_per_sample', 0)
                },
                'model_name': model_name,
                'config': result['config'],
                'predictions': result['predictions'],
                'ground_truth': result['ground_truth']
            }
        
        # Create plots
        plot_model_comparison(
            results=plot_results,
            output_dir=str(output_path),
            save_plots=True,
            show_plots=False
        )
        
        # Save results summary
        save_results_summary(
            results=plot_results,
            output_dir=str(output_path)
        )
        
        print(f"✅ Comparison plots saved to {output_path}")
    else:
        print(f"⚠️  Need at least 2 successful models for comparison plots")
    
    # Save detailed results
    results_file = output_path / "detailed_results.json"
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for model_name, result in results.items():
            json_result = result.copy()
            if 'predictions' in json_result:
                json_result['predictions'] = json_result['predictions'].tolist()
            if 'ground_truth' in json_result:
                json_result['ground_truth'] = json_result['ground_truth'].tolist()
            json_results[model_name] = json_result
        
        json.dump(json_results, f, indent=2)
    
    print(f"💾 Detailed results saved to {results_file}")
    print(f"\n🎉 Comprehensive comparison completed!")
    
    return results


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive model comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_comprehensive_model_comparison.py --data data/easy_3d_gaussian.pkl --ef gaussian_3d
  python scripts/run_comprehensive_model_comparison.py --data data/challenging_5d_gaussian.pkl --ef gaussian_5d
  python scripts/run_comprehensive_model_comparison.py --data data/easy_3d_gaussian.pkl --ef gaussian_3d --force-retrain
        """
    )
    
    parser.add_argument(
        '--data', 
        type=str, 
        required=True,
        help='Path to data file (e.g., data/easy_3d_gaussian.pkl)'
    )
    
    parser.add_argument(
        '--ef',
        type=str,
        required=True,
        help='Exponential family distribution name (e.g., gaussian_3d, gaussian_5d)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='artifacts/comprehensive_comparison',
        help='Output directory for results (default: artifacts/comprehensive_comparison)'
    )
    
    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Force retraining of all models, even if already completed'
    )
    
    args = parser.parse_args()
    
    try:
        results = run_comprehensive_comparison(
            data_file=args.data,
            ef_distribution=args.ef,
            output_dir=args.output,
            force_retrain=args.force_retrain
        )
        
        # Exit with error code if any models failed
        failed_count = sum(1 for r in results.values() if r['status'] == 'failed')
        if failed_count > 0:
            print(f"\n⚠️  {failed_count} models failed - check logs for details")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"\n💥 Comparison failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
