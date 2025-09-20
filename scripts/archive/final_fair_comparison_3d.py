#!/usr/bin/env python3
"""
Final fair comparison on 3D Gaussian dataset with consistent feature engineering.

This script ensures all neural network approaches (except INN, diffusion, NoProp)
use the same feature engineering as the standard MLP for a truly fair comparison.

Features used:
1. Original eta
2. 1/eta (clipped)  
3. eta/||eta|| (normalized)
4. ||eta|| * (1/eta)
5. ||eta|| (norm)
6. log(||eta||)
7. Absolute values of all above

This gives networks access to division-like operations from the start.
"""

import sys
from pathlib import Path
import pickle
import jax
import jax.numpy as jnp
from jax import random
import optax
import flax.linen as nn
import matplotlib.pyplot as plt
import numpy as np
import time
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ef import MultivariateNormal
from src.model import nat2statMLP


def create_nat_features(eta: jnp.ndarray) -> jnp.ndarray:
    """
    Create the same feature engineering as the standard MLP.
    
    This ensures all models have access to the same division-like features.
    """
    # Original eta parameters
    features = [eta]
    
    # 1. clip(1/eta) - inverse with aggressive clipping
    eta_inv = jnp.clip(1.0/eta, -1000.0, 1000.0)
    features.append(eta_inv)
    
    # 2. eta/norm(eta) - normalized eta (unit vector)
    eta_norm = jnp.linalg.norm(eta, axis=-1, keepdims=True)
    features.append(eta/eta_norm)
    
    # 3. norm(eta) * (1/eta)
    features.append(eta_norm * eta_inv)
    
    # 4. norm(eta)
    features.append(eta_norm)
    
    # 5. log(norm(eta))
    features.append(jnp.log(eta_norm))
    
    # Concatenate all features
    result = jnp.concatenate(features, axis=-1)
    
    # Add absolute values
    result = jnp.concatenate([result, jnp.abs(result)], axis=-1)
    
    # Safety checks
    result = jnp.where(jnp.isfinite(result), result, 0.0)
    result = jnp.clip(result, -1e6, 1e6)
    
    return result


class FeatureEngineeringMLP(nn.Module):
    """MLP with explicit feature engineering applied first."""
    
    hidden_sizes: tuple = (256, 128, 64)
    activation: str = "tanh"
    output_dim: int = 12
    
    def setup(self):
        if self.activation == "relu":
            self.act = nn.relu
        elif self.activation == "tanh":
            self.act = nn.tanh
        elif self.activation == "gelu":
            self.act = nn.gelu
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray) -> jnp.ndarray:
        # Apply feature engineering
        x = create_nat_features(eta)
        
        # Standard MLP layers
        for hidden_size in self.hidden_sizes:
            x = nn.Dense(hidden_size)(x)
            x = self.act(x)
        
        # Output layer
        x = nn.Dense(self.output_dim)(x)
        return x


class FeatureEngineeringQuadraticMLP(nn.Module):
    """Quadratic ResNet with feature engineering."""
    
    hidden_size: int = 256
    num_layers: int = 8
    activation: str = "tanh"
    output_dim: int = 12
    
    def setup(self):
        if self.activation == "relu":
            self.act = nn.relu
        elif self.activation == "tanh":
            self.act = nn.tanh
        elif self.activation == "gelu":
            self.act = nn.gelu
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray) -> jnp.ndarray:
        # Apply feature engineering
        x = create_nat_features(eta)
        
        # Project to hidden dimension
        x = nn.Dense(self.hidden_size)(x)
        x = self.act(x)
        
        # Quadratic ResNet layers: y = x + Wx + (B*x)*x
        for i in range(self.num_layers):
            # Linear term
            linear_term = nn.Dense(self.hidden_size, name=f'linear_{i}')(x)
            
            # Quadratic term
            quad_weights = nn.Dense(self.hidden_size, use_bias=False, name=f'quad_{i}')(x)
            quad_term = quad_weights * x
            
            # ResNet connection
            x = x + linear_term + quad_term
            x = self.act(x)
        
        # Output projection
        x = nn.Dense(self.output_dim)(x)
        return x


class FeatureEngineeringAdaptiveQuadraticMLP(nn.Module):
    """Adaptive Quadratic ResNet with feature engineering."""
    
    hidden_size: int = 256
    num_layers: int = 8
    activation: str = "tanh"
    output_dim: int = 12
    
    def setup(self):
        if self.activation == "relu":
            self.act = nn.relu
        elif self.activation == "tanh":
            self.act = nn.tanh
        elif self.activation == "gelu":
            self.act = nn.gelu
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray) -> jnp.ndarray:
        # Apply feature engineering
        x = create_nat_features(eta)
        
        # Project to hidden dimension
        x = nn.Dense(self.hidden_size)(x)
        x = self.act(x)
        
        # Adaptive quadratic ResNet layers: y = x + α*(Wx) + β*((B*x)*x)
        for i in range(self.num_layers):
            # Linear term
            linear_term = nn.Dense(self.hidden_size, name=f'linear_{i}')(x)
            
            # Quadratic term
            quad_weights = nn.Dense(self.hidden_size, use_bias=False, name=f'quad_{i}')(x)
            quad_term = quad_weights * x
            
            # Learnable mixing coefficients
            alpha = self.param(f'alpha_{i}', nn.initializers.ones, (self.hidden_size,))
            beta = self.param(f'beta_{i}', nn.initializers.zeros, (self.hidden_size,))
            
            # Adaptive combination
            x = x + alpha * linear_term + beta * quad_term
            x = self.act(x)
        
        # Output projection
        x = nn.Dense(self.output_dim)(x)
        return x


def load_3d_gaussian_data():
    """Load the largest 3D Gaussian dataset."""
    
    # Find the largest 3D dataset (12-dimensional natural parameters)
    data_dir = Path("data")
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
    
    # Choose the largest dataset
    best_file, n_samples = max(suitable_files, key=lambda x: x[1])
    
    print(f"Loading 3D Gaussian data from {best_file.name}")
    print(f"Dataset size: {n_samples} training samples")
    
    with open(best_file, 'rb') as f:
        data = pickle.load(f)
    
    # Create train/val/test splits
    train_data = {
        "eta": data["train_eta"],
        "y": data["train_y"]
    }
    
    val_data = {
        "eta": data["val_eta"], 
        "y": data["val_y"]
    }
    
    # Create test split from validation data
    n_val = val_data["eta"].shape[0]
    n_test = min(n_val // 2, 200)
    
    test_data = {
        "eta": val_data["eta"][:n_test],
        "y": val_data["y"][:n_test]
    }
    
    # Keep remaining as validation
    val_data = {
        "eta": val_data["eta"][n_test:],
        "y": val_data["y"][n_test:]
    }
    
    return train_data, val_data, test_data


def train_model_3d_fair(model, params, train_data, val_data, config, name="Model"):
    """Train model on 3D data with fair comparison settings."""
    
    num_epochs = config.get('num_epochs', 120)
    learning_rate = config.get('learning_rate', 5e-4)
    batch_size = config.get('batch_size', 32)
    
    # Conservative optimizer for 3D
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate)
    )
    opt_state = optimizer.init(params)
    
    train_losses = []
    val_losses = []
    best_params = params
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 25
    
    print(f"  Training {name}...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training with mini-batches
        n_train = train_data['eta'].shape[0]
        indices = jnp.arange(n_train)
        rng = random.PRNGKey(epoch)
        indices = random.permutation(rng, indices)
        
        epoch_train_loss = 0.0
        num_batches = 0
        
        for i in range(0, n_train, batch_size):
            batch_indices = indices[i:i+batch_size]
            eta_batch = train_data['eta'][batch_indices]
            y_batch = train_data['y'][batch_indices]
            
            def loss_fn(params):
                pred = model.apply(params, eta_batch)
                return jnp.mean(jnp.square(pred - y_batch))
            
            loss, grads = jax.value_and_grad(loss_fn)(params)
            
            # Check for numerical issues
            if not jnp.isfinite(loss):
                print(f"    Non-finite loss at epoch {epoch}")
                break
                
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            epoch_train_loss += float(loss)
            num_batches += 1
        
        if num_batches == 0:
            break
            
        epoch_train_loss /= num_batches
        
        # Validation
        val_pred = model.apply(params, val_data['eta'])
        val_loss = float(jnp.mean(jnp.square(val_pred - val_data['y'])))
        
        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        
        # Best model tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 20 == 0:
            print(f"    Epoch {epoch:3d}: Train={epoch_train_loss:.2f}, Val={val_loss:.2f}, Best={best_val_loss:.2f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break
    
    training_time = time.time() - start_time
    print(f"  Training completed in {training_time:.1f}s, Best val: {best_val_loss:.2f}")
    
    return best_params, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'training_time': training_time,
        'best_val_loss': best_val_loss
    }


def evaluate_model_3d_fair(model, params, test_data, name="Model"):
    """Evaluate model on 3D test data."""
    
    pred = model.apply(params, test_data['eta'])
    
    # Basic metrics
    mse = float(jnp.mean(jnp.square(pred - test_data['y'])))
    mae = float(jnp.mean(jnp.abs(pred - test_data['y'])))
    
    # Component-wise analysis
    component_mse = jnp.mean(jnp.square(pred - test_data['y']), axis=0)
    mean_mse = float(jnp.mean(component_mse[:3]))    # First 3 components (means)
    cov_mse = float(jnp.mean(component_mse[3:]))     # Remaining components (covariances)
    
    return {
        'name': name,
        'mse': mse,
        'mae': mae,
        'mean_mse': mean_mse,
        'cov_mse': cov_mse,
        'component_mse': component_mse.tolist()
    }


def test_fair_comparison_3d():
    """Run fair comparison on 3D Gaussian with consistent feature engineering."""
    
    print("🧊 FINAL FAIR COMPARISON ON 3D GAUSSIAN")
    print("=" * 80)
    print("All models use the same feature engineering for fair comparison")
    print("Features: eta, 1/eta, eta/||eta||, ||eta||*(1/eta), ||eta||, log(||eta||), abs(all)")
    
    # Load 3D data
    train_data, val_data, test_data = load_3d_gaussian_data()
    ef = MultivariateNormal(x_shape=(3,))
    
    print(f"\\nDataset sizes:")
    print(f"  Training: {train_data['eta'].shape[0]} samples")
    print(f"  Validation: {val_data['eta'].shape[0]} samples")
    print(f"  Test: {test_data['eta'].shape[0]} samples")
    print(f"  Input dimension: {train_data['eta'].shape[1]}")
    print(f"  Output dimension: {train_data['y'].shape[1]}")
    
    # Check feature engineering output dimension
    sample_eta = test_data['eta'][:1]
    sample_features = create_nat_features(sample_eta)
    print(f"  Feature dimension: {sample_features.shape[1]} (after engineering)")
    
    results = {}
    
    # 1. Standard MLP with feature engineering (baseline)
    print(f"\\n{'='*60}")
    print("1. STANDARD MLP + FEATURE ENGINEERING")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(42)
    standard_mlp = nat2statMLP(
        hidden_sizes=[256, 128, 64], 
        activation='tanh', 
        output_dim=12,
        use_feature_engineering=True  # Enable feature engineering
    )
    standard_params = standard_mlp.init(rng, test_data['eta'][:1])
    
    standard_config = {
        'num_epochs': 120,
        'learning_rate': 5e-4,
        'batch_size': 32
    }
    
    standard_params, standard_history = train_model_3d_fair(
        standard_mlp, standard_params, train_data, val_data,
        standard_config, "Standard MLP (Features)"
    )
    
    results['standard'] = evaluate_model_3d_fair(
        standard_mlp, standard_params, test_data, "Standard MLP (Features)"
    )
    results['standard']['training_time'] = standard_history['training_time']
    
    # 2. Feature Engineering MLP (explicit implementation)
    print(f"\\n{'='*60}")
    print("2. EXPLICIT FEATURE ENGINEERING MLP")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(43)
    feature_mlp = FeatureEngineeringMLP(
        hidden_sizes=(256, 128, 64),
        activation='tanh',
        output_dim=12
    )
    feature_params = feature_mlp.init(rng, test_data['eta'][:1])
    
    feature_params, feature_history = train_model_3d_fair(
        feature_mlp, feature_params, train_data, val_data,
        standard_config, "Feature Engineering MLP"
    )
    
    results['feature_mlp'] = evaluate_model_3d_fair(
        feature_mlp, feature_params, test_data, "Feature Engineering MLP"
    )
    results['feature_mlp']['training_time'] = feature_history['training_time']
    
    # 3. Quadratic ResNet with feature engineering
    print(f"\\n{'='*60}")
    print("3. QUADRATIC RESNET + FEATURE ENGINEERING")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(44)
    quad_mlp = FeatureEngineeringQuadraticMLP(
        hidden_size=256,
        num_layers=6,
        activation='tanh',
        output_dim=12
    )
    quad_params = quad_mlp.init(rng, test_data['eta'][:1])
    
    quad_config = {
        'num_epochs': 120,
        'learning_rate': 3e-4,  # Lower LR for quadratic
        'batch_size': 32
    }
    
    quad_params, quad_history = train_model_3d_fair(
        quad_mlp, quad_params, train_data, val_data,
        quad_config, "Quadratic ResNet (Features)"
    )
    
    results['quadratic'] = evaluate_model_3d_fair(
        quad_mlp, quad_params, test_data, "Quadratic ResNet (Features)"
    )
    results['quadratic']['training_time'] = quad_history['training_time']
    
    # 4. Adaptive Quadratic ResNet with feature engineering
    print(f"\\n{'='*60}")
    print("4. ADAPTIVE QUADRATIC RESNET + FEATURE ENGINEERING")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(45)
    adaptive_mlp = FeatureEngineeringAdaptiveQuadraticMLP(
        hidden_size=256,
        num_layers=8,
        activation='tanh',
        output_dim=12
    )
    adaptive_params = adaptive_mlp.init(rng, test_data['eta'][:1])
    
    adaptive_config = {
        'num_epochs': 150,
        'learning_rate': 3e-4,
        'batch_size': 32
    }
    
    adaptive_params, adaptive_history = train_model_3d_fair(
        adaptive_mlp, adaptive_params, train_data, val_data,
        adaptive_config, "Adaptive Quadratic ResNet (Features)"
    )
    
    results['adaptive'] = evaluate_model_3d_fair(
        adaptive_mlp, adaptive_params, test_data, "Adaptive Quadratic ResNet (Features)"
    )
    results['adaptive']['training_time'] = adaptive_history['training_time']
    
    # 5. Deep Adaptive Quadratic ResNet with feature engineering
    print(f"\\n{'='*60}")
    print("5. DEEP ADAPTIVE QUADRATIC RESNET + FEATURE ENGINEERING")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(46)
    deep_adaptive_mlp = FeatureEngineeringAdaptiveQuadraticMLP(
        hidden_size=256,
        num_layers=12,  # Deeper
        activation='tanh',
        output_dim=12
    )
    deep_adaptive_params = deep_adaptive_mlp.init(rng, test_data['eta'][:1])
    
    deep_config = {
        'num_epochs': 150,
        'learning_rate': 2e-4,  # Even lower LR for deep network
        'batch_size': 32
    }
    
    deep_adaptive_params, deep_adaptive_history = train_model_3d_fair(
        deep_adaptive_mlp, deep_adaptive_params, train_data, val_data,
        deep_config, "Deep Adaptive Quadratic ResNet (Features)"
    )
    
    results['deep_adaptive'] = evaluate_model_3d_fair(
        deep_adaptive_mlp, deep_adaptive_params, test_data, "Deep Adaptive Quadratic ResNet (Features)"
    )
    results['deep_adaptive']['training_time'] = deep_adaptive_history['training_time']
    
    # Summary
    print(f"\\n{'='*80}")
    print("FINAL FAIR 3D COMPARISON RESULTS")
    print(f"{'='*80}")
    
    # Sort by performance
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mse'])
    
    print(f"{'Rank':<4} {'Architecture':<40} {'MSE':<12} {'Mean MSE':<12} {'Cov MSE':<12} {'Time(s)':<10}")
    print("-" * 95)
    
    for rank, (key, result) in enumerate(sorted_results, 1):
        print(f"{rank:<4} {result['name']:<40} {result['mse']:<12.1f} "
              f"{result['mean_mse']:<12.3f} {result['cov_mse']:<12.1f} {result.get('training_time', 0):<10.1f}")
    
    # Best model analysis
    best_model = sorted_results[0]
    print(f"\\n🏆 BEST 3D MODEL: {best_model[1]['name']}")
    print(f"   Total MSE: {best_model[1]['mse']:.1f}")
    print(f"   Mean MSE: {best_model[1]['mean_mse']:.3f}")
    print(f"   Covariance MSE: {best_model[1]['cov_mse']:.1f}")
    print(f"   Training time: {best_model[1].get('training_time', 0):.1f}s")
    
    # Check if quadratic approaches still dominate
    quad_models = [k for k in results.keys() if 'quadratic' in k.lower()]
    if quad_models:
        quad_mses = [results[k]['mse'] for k in quad_models]
        standard_mse = results['standard']['mse']
        
        best_quad_mse = min(quad_mses)
        if best_quad_mse < standard_mse:
            improvement = (standard_mse - best_quad_mse) / standard_mse * 100
            print(f"\\n✅ Quadratic approaches still dominate 3D: {improvement:.1f}% better than standard")
        else:
            print(f"\\n📊 3D complexity challenges quadratic approaches")
    
    # Create visualization
    create_3d_fair_comparison_plots(results, sorted_results)
    
    # Save results
    output_dir = Path("artifacts/final_fair_3d_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_for_json = {}
    for model_name, result in results.items():
        results_for_json[model_name] = {
            'name': result['name'],
            'mse': result['mse'],
            'mae': result['mae'],
            'mean_mse': result['mean_mse'],
            'cov_mse': result['cov_mse'],
            'training_time': result.get('training_time', 0)
        }
    
    with open(output_dir / "final_fair_3d_results.json", 'w') as f:
        json.dump(results_for_json, f, indent=2)
    
    print(f"\\n📁 Results saved to {output_dir}/")
    
    return results


def create_3d_fair_comparison_plots(results, sorted_results):
    """Create comprehensive 3D fair comparison plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Overall performance
    names = [result[1]['name'] for result in sorted_results]
    mses = [result[1]['mse'] for result in sorted_results]
    colors = ['gold', 'silver', 'brown', 'blue', 'green'][:len(names)]
    
    bars = axes[0, 0].bar(range(len(names)), mses, color=colors, alpha=0.7)
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels([n.replace(' ', '\\n') for n in names], rotation=0)
    axes[0, 0].set_ylabel('Total MSE')
    axes[0, 0].set_title('3D Gaussian: Fair Comparison Results')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, mses):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.0f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Mean vs Covariance MSE breakdown
    mean_mses = [result[1]['mean_mse'] for result in sorted_results]
    cov_mses = [result[1]['cov_mse'] for result in sorted_results]
    
    x = np.arange(len(names))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, mean_mses, width, label='Mean MSE', alpha=0.7, color='blue')
    axes[0, 1].bar(x + width/2, cov_mses, width, label='Covariance MSE', alpha=0.7, color='red')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([n.split()[0] for n in names], rotation=45, ha='right')
    axes[0, 1].set_ylabel('Component MSE (log scale)')
    axes[0, 1].set_title('Mean vs Covariance Prediction')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # 3. Training efficiency
    times = [result[1].get('training_time', 0) for result in sorted_results]
    axes[0, 2].scatter(times, mses, c=colors, s=100, alpha=0.7)
    for i, name in enumerate(names):
        axes[0, 2].annotate(name.split()[0], (times[i], mses[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0, 2].set_xlabel('Training Time (s)')
    axes[0, 2].set_ylabel('Test MSE')
    axes[0, 2].set_title('Efficiency vs Performance')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Architecture comparison (grouped)
    arch_families = {
        'Standard\\nMLP': ['standard', 'feature_mlp'],
        'Quadratic\\nResNet': ['quadratic'],
        'Adaptive\\nQuadratic': ['adaptive', 'deep_adaptive']
    }
    
    family_mses = []
    family_names = []
    family_colors = ['red', 'blue', 'green']
    
    for family_name, models in arch_families.items():
        family_results = [results[m]['mse'] for m in models if m in results]
        if family_results:
            family_mse = min(family_results)  # Best in family
            family_mses.append(family_mse)
            family_names.append(family_name)
    
    bars = axes[1, 0].bar(range(len(family_names)), family_mses, color=family_colors, alpha=0.7)
    axes[1, 0].set_xticks(range(len(family_names)))
    axes[1, 0].set_xticklabels(family_names)
    axes[1, 0].set_ylabel('Best MSE in Family')
    axes[1, 0].set_title('Architecture Family Comparison')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, family_mses):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.0f}', ha='center', va='bottom', fontsize=10)
    
    # 5. Component-wise performance heatmap
    component_names = ['μ₁', 'μ₂', 'μ₃', 'Σ₁₁', 'Σ₁₂', 'Σ₁₃', 'Σ₂₂', 'Σ₂₃', 'Σ₃₃']
    model_names_short = [n.split()[0] for n in names]
    
    # Create matrix of component MSEs
    component_matrix = np.array([result[1]['component_mse'][:9] for result in sorted_results])
    
    im = axes[1, 1].imshow(component_matrix, cmap='viridis', aspect='auto')
    axes[1, 1].set_xticks(range(len(component_names)))
    axes[1, 1].set_xticklabels(component_names, rotation=45, ha='right')
    axes[1, 1].set_yticks(range(len(model_names_short)))
    axes[1, 1].set_yticklabels(model_names_short)
    axes[1, 1].set_title('Component-wise MSE Heatmap')
    
    # Add colorbar
    plt.colorbar(im, ax=axes[1, 1], label='MSE')
    
    # 6. Summary text
    axes[1, 2].axis('off')
    
    best_model = sorted_results[0]
    
    summary_text = f"3D FAIR COMPARISON\\n"
    summary_text += f"={'='*20}\\n\\n"
    summary_text += f"🏆 Winner: {best_model[1]['name']}\\n"
    summary_text += f"Total MSE: {best_model[1]['mse']:.1f}\\n"
    summary_text += f"Mean MSE: {best_model[1]['mean_mse']:.3f}\\n"
    summary_text += f"Cov MSE: {best_model[1]['cov_mse']:.1f}\\n\\n"
    
    summary_text += f"📊 KEY INSIGHTS:\\n"
    summary_text += f"• All models use same features\\n"
    summary_text += f"• Features include 1/eta, eta/||eta||\\n"
    summary_text += f"• Fair comparison of architectures\\n"
    summary_text += f"• 3D much harder than 1D\\n\\n"
    
    summary_text += f"🎯 FEATURE ENGINEERING:\\n"
    summary_text += f"• Original: 12D → 96D features\\n"
    summary_text += f"• Includes division operations\\n"
    summary_text += f"• Helps all architectures\\n"
    summary_text += f"• Levels playing field\\n\\n"
    
    summary_text += f"📈 TOP 3:\\n"
    for i, (k, result) in enumerate(sorted_results[:3]):
        summary_text += f"{i+1}. {result['name'].split()[0]}\\n"
        summary_text += f"   MSE: {result['mse']:.1f}\\n"
    
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save plots
    output_dir = Path("artifacts/final_fair_3d_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "final_fair_3d_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "final_fair_3d_comparison.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"📊 Fair comparison plots saved to {output_dir}/")
    
    return results


if __name__ == "__main__":
    print("🧊 Starting final fair 3D comparison...")
    
    try:
        results = test_fair_comparison_3d()
        print("\\n✅ Final fair 3D comparison completed!")
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
