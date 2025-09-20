#!/usr/bin/env python3
"""
Comprehensive comparison of ALL neural network architectures for learning eta -> y mapping.
This is the definitive evaluation script that runs all models to convergence.

Models tested:
1. Standard MLP (baseline)
2. Division-aware MLP (explicit division operations)
3. GLU-based MLP (gated linear units)
4. Deep GLU with residual connections
5. Quadratic ResNet (y = x + Wx + (B*x)*x)
6. Deep Quadratic ResNet (more layers)
7. Adaptive Quadratic ResNet (learnable coefficients)
8. Improved INN (invertible neural network)
9. NoProp-CT (continuous-time neural ODE)
10. Analytical solution (ground truth)

This script runs extensive training (200+ epochs) for definitive results.
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

# Import all our models
from src.ef import GaussianNatural1D
from src.model import nat2statMLP
from src.division_aware_mlp import DivisionAwareMomentNet
from src.glu_moment_net import GLUMomentNet, DeepGLUMomentNet, create_glu_train_state
from src.quadratic_resnet import QuadraticResNet, DeepAdaptiveQuadraticResNet, create_quadratic_train_state
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


def train_standard_model(model, params, train_data, val_data, config, name="Model"):
    """Train a standard model with extensive epochs."""
    
    num_epochs = config.get('num_epochs', 200)
    learning_rate = config.get('learning_rate', 1e-3)
    batch_size = config.get('batch_size', 64)
    use_scheduler = config.get('use_scheduler', True)
    
    # Create optimizer with optional learning rate scheduling
    if use_scheduler:
        schedule = optax.exponential_decay(learning_rate, transition_steps=50, decay_rate=0.95)
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(schedule)
        )
    else:
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate)
        )
    
    opt_state = optimizer.init(params)
    
    train_losses = []
    val_losses = []
    best_params = params
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.get('patience', 50)
    
    print(f"  Training {name} for up to {num_epochs} epochs...")
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
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            epoch_train_loss += float(loss)
            num_batches += 1
        
        epoch_train_loss /= num_batches
        
        # Validation
        val_pred = model.apply(params, val_data['eta'])
        val_loss = float(jnp.mean(jnp.square(val_pred - val_data['y'])))
        
        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        
        # Early stopping and best model tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 20 == 0:
            print(f"    Epoch {epoch:3d}: Train={epoch_train_loss:.6f}, Val={val_loss:.6f}, Best={best_val_loss:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch} (patience={patience})")
            break
    
    training_time = time.time() - start_time
    print(f"  Training completed in {training_time:.1f}s, Best val loss: {best_val_loss:.6f}")
    
    return best_params, {
        'train_losses': train_losses, 
        'val_losses': val_losses, 
        'training_time': training_time,
        'best_val_loss': best_val_loss,
        'epochs_trained': len(train_losses)
    }


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


def run_comprehensive_comparison():
    """Run the comprehensive model comparison."""
    
    print("🚀 COMPREHENSIVE MODEL COMPARISON")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("This will run all models to convergence - may take several hours!")
    
    # Load data
    ef = GaussianNatural1D()
    train_data, val_data, test_data = load_existing_data("gaussian_1d")
    
    print(f"\nDataset Information:")
    print(f"  Training samples: {train_data['eta'].shape[0]}")
    print(f"  Validation samples: {val_data['eta'].shape[0]}")
    print(f"  Test samples: {test_data['eta'].shape[0]}")
    print(f"  Input dimension: {train_data['eta'].shape[1]}")
    print(f"  Output dimension: {train_data['y'].shape[1]}")
    
    # Store all results
    results = {}
    training_histories = {}
    
    # 1. Analytical Solution (Ground Truth)
    print(f"\n{'='*60}")
    print("1. ANALYTICAL SOLUTION (Ground Truth)")
    print(f"{'='*60}")
    
    analytical_pred = analytical_solution(test_data['eta'])
    analytical_results = {
        'name': 'Analytical',
        'predictions': analytical_pred,
        'mse': float(jnp.mean(jnp.square(analytical_pred - test_data['y']))),
        'mae': float(jnp.mean(jnp.abs(analytical_pred - test_data['y']))),
        'rmse': float(jnp.sqrt(jnp.mean(jnp.square(analytical_pred - test_data['y'])))),
        'r2': 1.0,  # By definition
        'max_error': float(jnp.max(jnp.abs(analytical_pred - test_data['y']))),
        'component_mse': jnp.mean(jnp.square(analytical_pred - test_data['y']), axis=0).tolist(),
        'component_mae': jnp.mean(jnp.abs(analytical_pred - test_data['y']), axis=0).tolist()
    }
    
    results['analytical'] = analytical_results
    training_histories['analytical'] = {'training_time': 0.0, 'epochs_trained': 0}
    
    print(f"  MSE: {analytical_results['mse']:.8f}")
    print(f"  MAE: {analytical_results['mae']:.8f}")
    print(f"  This represents the theoretical lower bound (MCMC sampling error)")
    
    # 2. Standard MLP (Baseline)
    print(f"\n{'='*60}")
    print("2. STANDARD MLP (Baseline)")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(42)
    standard_mlp = nat2statMLP(hidden_sizes=[128, 64, 32], activation='tanh', output_dim=2)
    standard_params = standard_mlp.init(rng, test_data['eta'][:1])
    
    standard_config = {
        'num_epochs': 300,
        'learning_rate': 1e-3,
        'batch_size': 64,
        'use_scheduler': True,
        'patience': 50
    }
    
    standard_params, standard_history = train_standard_model(
        standard_mlp, standard_params, train_data, val_data, standard_config, "Standard MLP"
    )
    
    results['standard_mlp'] = evaluate_model(standard_mlp, standard_params, test_data, "Standard MLP")
    training_histories['standard_mlp'] = standard_history
    
    # 3. Division-Aware MLP
    print(f"\n{'='*60}")
    print("3. DIVISION-AWARE MLP")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(43)
    division_mlp = DivisionAwareMomentNet(
        ef=ef,
        hidden_sizes=(64, 32),
        use_division_layers=True,
        use_reciprocal_layers=True
    )
    division_params = division_mlp.init(rng, test_data['eta'][:1])
    
    division_config = {
        'num_epochs': 300,
        'learning_rate': 8e-4,
        'batch_size': 64,
        'use_scheduler': True,
        'patience': 50
    }
    
    division_params, division_history = train_standard_model(
        division_mlp, division_params, train_data, val_data, division_config, "Division-Aware MLP"
    )
    
    results['division_aware'] = evaluate_model(division_mlp, division_params, test_data, "Division-Aware MLP")
    training_histories['division_aware'] = division_history
    
    # 4. GLU-based MLP
    print(f"\n{'='*60}")
    print("4. GLU-BASED MLP")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(44)
    glu_config = {
        'model_type': 'glu',
        'hidden_sizes': (128, 64),
        'activation': 'tanh',
        'use_glu_layers': True,
        'glu_hidden_ratio': 2.0
    }
    glu_model, glu_params = create_glu_train_state(ef, glu_config, rng)
    
    glu_train_config = {
        'num_epochs': 300,
        'learning_rate': 1e-3,
        'batch_size': 64,
        'use_scheduler': True,
        'patience': 50
    }
    
    glu_params, glu_history = train_standard_model(
        glu_model, glu_params, train_data, val_data, glu_train_config, "GLU MLP"
    )
    
    results['glu'] = evaluate_model(glu_model, glu_params, test_data, "GLU MLP")
    training_histories['glu'] = glu_history
    
    # 5. Deep GLU with Residual Connections
    print(f"\n{'='*60}")
    print("5. DEEP GLU WITH RESIDUAL CONNECTIONS")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(45)
    deep_glu_config = {
        'model_type': 'deep_glu',
        'hidden_size': 128,
        'num_glu_layers': 6,
        'activation': 'tanh'
    }
    deep_glu_model, deep_glu_params = create_glu_train_state(ef, deep_glu_config, rng)
    
    deep_glu_train_config = {
        'num_epochs': 300,
        'learning_rate': 8e-4,
        'batch_size': 64,
        'use_scheduler': True,
        'patience': 50
    }
    
    deep_glu_params, deep_glu_history = train_standard_model(
        deep_glu_model, deep_glu_params, train_data, val_data, deep_glu_train_config, "Deep GLU"
    )
    
    results['deep_glu'] = evaluate_model(deep_glu_model, deep_glu_params, test_data, "Deep GLU")
    training_histories['deep_glu'] = deep_glu_history
    
    # 6. Quadratic ResNet
    print(f"\n{'='*60}")
    print("6. QUADRATIC RESNET")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(46)
    quad_config = {
        'model_type': 'quadratic',
        'hidden_size': 128,
        'num_layers': 10,
        'activation': 'tanh',
        'use_activation_between_layers': True
    }
    quad_model, quad_params = create_quadratic_train_state(ef, quad_config, rng)
    
    quad_train_config = {
        'num_epochs': 300,
        'learning_rate': 8e-4,
        'batch_size': 64,
        'use_scheduler': True,
        'patience': 50
    }
    
    quad_params, quad_history = train_standard_model(
        quad_model, quad_params, train_data, val_data, quad_train_config, "Quadratic ResNet"
    )
    
    results['quadratic'] = evaluate_model(quad_model, quad_params, test_data, "Quadratic ResNet")
    training_histories['quadratic'] = quad_history
    
    # 7. Deep Quadratic ResNet
    print(f"\n{'='*60}")
    print("7. DEEP QUADRATIC RESNET")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(47)
    deep_quad_config = {
        'model_type': 'quadratic',
        'hidden_size': 128,
        'num_layers': 16,
        'activation': 'tanh',
        'use_activation_between_layers': True
    }
    deep_quad_model, deep_quad_params = create_quadratic_train_state(ef, deep_quad_config, rng)
    
    deep_quad_train_config = {
        'num_epochs': 300,
        'learning_rate': 6e-4,
        'batch_size': 64,
        'use_scheduler': True,
        'patience': 60
    }
    
    deep_quad_params, deep_quad_history = train_standard_model(
        deep_quad_model, deep_quad_params, train_data, val_data, deep_quad_train_config, "Deep Quadratic ResNet"
    )
    
    results['deep_quadratic'] = evaluate_model(deep_quad_model, deep_quad_params, test_data, "Deep Quadratic ResNet")
    training_histories['deep_quadratic'] = deep_quad_history
    
    # 8. Adaptive Quadratic ResNet
    print(f"\n{'='*60}")
    print("8. ADAPTIVE QUADRATIC RESNET")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(48)
    adaptive_config = {
        'model_type': 'adaptive_quadratic',
        'hidden_size': 128,
        'num_layers': 10,
        'activation': 'tanh'
    }
    adaptive_model, adaptive_params = create_quadratic_train_state(ef, adaptive_config, rng)
    
    adaptive_train_config = {
        'num_epochs': 300,
        'learning_rate': 8e-4,
        'batch_size': 64,
        'use_scheduler': True,
        'patience': 50
    }
    
    adaptive_params, adaptive_history = train_standard_model(
        adaptive_model, adaptive_params, train_data, val_data, adaptive_train_config, "Adaptive Quadratic ResNet"
    )
    
    results['adaptive_quadratic'] = evaluate_model(adaptive_model, adaptive_params, test_data, "Adaptive Quadratic ResNet")
    training_histories['adaptive_quadratic'] = adaptive_history
    
    # 9. NoProp-CT (Neural ODE)
    print(f"\n{'='*60}")
    print("9. NOPROP-CT (Neural ODE)")
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
    noprop_pred = noprop_model.apply(noprop_state.params, test_data['eta'])
    
    results['noprop_ct'] = evaluate_model(noprop_model, noprop_state.params, test_data, "NoProp-CT")
    training_histories['noprop_ct'] = {
        'train_losses': noprop_history['train_losses'],
        'val_losses': noprop_history['val_losses'],
        'training_time': noprop_training_time,
        'epochs_trained': len(noprop_history['train_losses'])
    }
    
    # 10. Improved INN (Invertible Neural Network)
    print(f"\n{'='*60}")
    print("10. IMPROVED INN (Invertible Neural Network)")
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
    
    results['improved_inn'] = evaluate_model(inn_model, inn_state.params, test_data, "Improved INN")
    results['improved_inn']['predictions'] = inn_pred  # Override with INN predictions
    
    training_histories['improved_inn'] = {
        'train_losses': inn_history['train_losses'],
        'val_losses': inn_history['val_losses'],
        'training_time': inn_training_time,
        'epochs_trained': len(inn_history['train_losses'])
    }
    
    # Generate comprehensive results summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Sort by MSE performance
    model_results = [(k, v) for k, v in results.items()]
    model_results.sort(key=lambda x: x[1]['mse'])
    
    print(f"{'Rank':<4} {'Model':<25} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'R²':<8} {'Time(s)':<10}")
    print("-" * 90)
    
    for rank, (model_name, result) in enumerate(model_results, 1):
        training_time = training_histories[model_name]['training_time']
        print(f"{rank:<4} {result['name']:<25} {result['mse']:<12.8f} {result['mae']:<12.8f} "
              f"{result['rmse']:<12.8f} {result['r2']:<8.4f} {training_time:<10.1f}")
    
    # Find best neural network (excluding analytical)
    neural_results = [(k, v) for k, v in results.items() if k != 'analytical']
    best_neural = min(neural_results, key=lambda x: x[1]['mse'])
    
    print(f"\n🏆 BEST NEURAL NETWORK: {best_neural[1]['name']}")
    print(f"   MSE: {best_neural[1]['mse']:.8f}")
    print(f"   Improvement over analytical: {best_neural[1]['mse'] / results['analytical']['mse']:.1f}x worse")
    
    # Save detailed results
    output_dir = Path("artifacts/comprehensive_comparison_final")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results as JSON
    results_for_json = {}
    for model_name, result in results.items():
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
    
    with open(output_dir / "comprehensive_results.json", 'w') as f:
        json.dump(results_for_json, f, indent=2)
    
    # Generate comprehensive plots
    generate_comprehensive_plots(results, training_histories, test_data, output_dir)
    
    print(f"\n📊 Results saved to {output_dir}/")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results, training_histories


def generate_comprehensive_plots(results, training_histories, test_data, output_dir):
    """Generate comprehensive comparison plots."""
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(24, 16))
    
    # Define colors for each model
    colors = {
        'analytical': 'black',
        'standard_mlp': 'red',
        'division_aware': 'orange',
        'glu': 'yellow',
        'deep_glu': 'green',
        'quadratic': 'blue',
        'deep_quadratic': 'navy',
        'adaptive_quadratic': 'purple',
        'noprop_ct': 'brown',
        'improved_inn': 'pink'
    }
    
    # 1. Training curves
    ax1 = plt.subplot(3, 4, 1)
    for model_name, history in training_histories.items():
        if 'val_losses' in history and len(history['val_losses']) > 1:
            ax1.plot(history['val_losses'], label=results[model_name]['name'], 
                    color=colors.get(model_name, 'gray'), alpha=0.8, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation MSE')
    ax1.set_title('Training Convergence')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)
    ax1.set_yscale('log')
    
    # 2. Final performance comparison (bar chart)
    ax2 = plt.subplot(3, 4, 2)
    model_names = [results[k]['name'] for k in results.keys() if k != 'analytical']
    mse_values = [results[k]['mse'] for k in results.keys() if k != 'analytical']
    model_colors = [colors.get(k, 'gray') for k in results.keys() if k != 'analytical']
    
    bars = ax2.bar(range(len(model_names)), mse_values, color=model_colors, alpha=0.7)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.set_ylabel('Test MSE')
    ax2.set_title('Final Performance Comparison')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, mse_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # 3. Predictions vs targets for best model
    neural_results = {k: v for k, v in results.items() if k != 'analytical'}
    best_model = min(neural_results.items(), key=lambda x: x[1]['mse'])
    
    ax3 = plt.subplot(3, 4, 3)
    ax3.scatter(test_data['y'][:, 0], best_model[1]['predictions'][:, 0], 
               alpha=0.6, s=20, label='E[x]', color='blue')
    ax3.scatter(test_data['y'][:, 1], best_model[1]['predictions'][:, 1], 
               alpha=0.6, s=20, label='E[x²]', color='red')
    ax3.plot([test_data['y'].min(), test_data['y'].max()], 
            [test_data['y'].min(), test_data['y'].max()], 'k--', alpha=0.8)
    ax3.set_xlabel('True Values')
    ax3.set_ylabel('Predicted Values')
    ax3.set_title(f'Best Model: {best_model[1]["name"]}')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Error distribution for best model
    ax4 = plt.subplot(3, 4, 4)
    errors = jnp.abs(best_model[1]['predictions'] - test_data['y'])
    ax4.hist(errors.flatten(), bins=50, alpha=0.7, density=True, color='blue')
    ax4.set_xlabel('Absolute Error')
    ax4.set_ylabel('Density')
    ax4.set_title(f'Error Distribution - {best_model[1]["name"]}')
    ax4.grid(True)
    
    # 5. Component-wise MSE comparison
    ax5 = plt.subplot(3, 4, 5)
    component_0_mse = [results[k]['component_mse'][0] for k in results.keys() if k != 'analytical']
    component_1_mse = [results[k]['component_mse'][1] for k in results.keys() if k != 'analytical']
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax5.bar(x - width/2, component_0_mse, width, label='E[x] MSE', alpha=0.7)
    ax5.bar(x + width/2, component_1_mse, width, label='E[x²] MSE', alpha=0.7)
    ax5.set_xticks(x)
    ax5.set_xticklabels(model_names, rotation=45, ha='right')
    ax5.set_ylabel('Component MSE')
    ax5.set_title('Component-wise Performance')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    # 6. R² comparison
    ax6 = plt.subplot(3, 4, 6)
    r2_values = [results[k]['r2'] for k in results.keys() if k != 'analytical']
    bars = ax6.bar(range(len(model_names)), r2_values, color=model_colors, alpha=0.7)
    ax6.set_xticks(range(len(model_names)))
    ax6.set_xticklabels(model_names, rotation=45, ha='right')
    ax6.set_ylabel('R²')
    ax6.set_title('R² Comparison')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 1)
    
    # 7. Training time comparison
    ax7 = plt.subplot(3, 4, 7)
    training_times = [training_histories[k]['training_time'] for k in results.keys() if k != 'analytical']
    bars = ax7.bar(range(len(model_names)), training_times, color=model_colors, alpha=0.7)
    ax7.set_xticks(range(len(model_names)))
    ax7.set_xticklabels(model_names, rotation=45, ha='right')
    ax7.set_ylabel('Training Time (s)')
    ax7.set_title('Training Time Comparison')
    ax7.grid(True, alpha=0.3)
    
    # 8. Epochs trained comparison
    ax8 = plt.subplot(3, 4, 8)
    epochs_trained = [training_histories[k]['epochs_trained'] for k in results.keys() if k != 'analytical']
    bars = ax8.bar(range(len(model_names)), epochs_trained, color=model_colors, alpha=0.7)
    ax8.set_xticks(range(len(model_names)))
    ax8.set_xticklabels(model_names, rotation=45, ha='right')
    ax8.set_ylabel('Epochs Trained')
    ax8.set_title('Convergence Speed')
    ax8.grid(True, alpha=0.3)
    
    # 9. MSE vs Training Time scatter
    ax9 = plt.subplot(3, 4, 9)
    for k in results.keys():
        if k != 'analytical':
            ax9.scatter(training_histories[k]['training_time'], results[k]['mse'], 
                       s=100, color=colors.get(k, 'gray'), alpha=0.7, label=results[k]['name'])
    ax9.set_xlabel('Training Time (s)')
    ax9.set_ylabel('Test MSE')
    ax9.set_title('Performance vs Efficiency')
    ax9.set_yscale('log')
    ax9.grid(True)
    ax9.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 10. Analytical comparison
    ax10 = plt.subplot(3, 4, 10)
    analytical_ratio = [results[k]['mse'] / results['analytical']['mse'] 
                       for k in results.keys() if k != 'analytical']
    bars = ax10.bar(range(len(model_names)), analytical_ratio, color=model_colors, alpha=0.7)
    ax10.set_xticks(range(len(model_names)))
    ax10.set_xticklabels(model_names, rotation=45, ha='right')
    ax10.set_ylabel('MSE / Analytical MSE')
    ax10.set_title('Distance from Theoretical Optimum')
    ax10.grid(True, alpha=0.3)
    ax10.set_yscale('log')
    
    # 11. Best 3 models detailed comparison
    ax11 = plt.subplot(3, 4, 11)
    sorted_models = sorted([(k, v) for k, v in results.items() if k != 'analytical'], 
                          key=lambda x: x[1]['mse'])[:3]
    
    for i, (model_name, result) in enumerate(sorted_models):
        if model_name in training_histories and 'val_losses' in training_histories[model_name]:
            ax11.plot(training_histories[model_name]['val_losses'], 
                     label=f"{i+1}. {result['name']}", 
                     color=colors.get(model_name, 'gray'), alpha=0.8, linewidth=3)
    
    ax11.set_xlabel('Epoch')
    ax11.set_ylabel('Validation MSE')
    ax11.set_title('Top 3 Models Training')
    ax11.legend()
    ax11.grid(True)
    ax11.set_yscale('log')
    
    # 12. Summary statistics table (as text)
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    summary_text = "SUMMARY STATISTICS\n" + "="*20 + "\n\n"
    summary_text += f"Best Model: {best_model[1]['name']}\n"
    summary_text += f"Best MSE: {best_model[1]['mse']:.8f}\n"
    summary_text += f"Best MAE: {best_model[1]['mae']:.8f}\n"
    summary_text += f"Best R²: {best_model[1]['r2']:.4f}\n\n"
    
    summary_text += f"Analytical MSE: {results['analytical']['mse']:.8f}\n"
    summary_text += f"Gap to optimum: {best_model[1]['mse'] / results['analytical']['mse']:.1f}x\n\n"
    
    summary_text += "Top 3 Models:\n"
    for i, (model_name, result) in enumerate(sorted_models[:3]):
        summary_text += f"{i+1}. {result['name']}\n"
        summary_text += f"   MSE: {result['mse']:.6f}\n"
    
    ax12.text(0.1, 0.9, summary_text, transform=ax12.transAxes, fontsize=10, 
              verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_dir / "comprehensive_comparison_final.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "comprehensive_comparison_final.pdf", bbox_inches='tight')
    plt.close()
    
    print("  Comprehensive plots generated successfully!")


if __name__ == "__main__":
    print("🚀 Starting comprehensive model comparison...")
    print("This will take several hours to complete all models to convergence.")
    print("Press Ctrl+C to interrupt if needed.\n")
    
    try:
        results, histories = run_comprehensive_comparison()
        print("\n✅ Comprehensive comparison completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️  Comparison interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        raise
