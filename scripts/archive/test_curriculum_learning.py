#!/usr/bin/env python3
"""
Test curriculum learning approach for geometric loss.

This script implements and tests the curriculum learning strategy:
1. Warmup: Pure MSE loss (30-50 epochs)
2. Transition: Gradually introduce KL loss (30-50 epochs) 
3. Refinement: Full geometric loss with second-order information

The goal is to get the benefits of geometric loss while maintaining training stability.
"""

import sys
from pathlib import Path
import pickle
import jax
import jax.numpy as jnp
from jax import random, jacfwd
import optax
import matplotlib.pyplot as plt
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ef import GaussianNatural1D
from src.model import nat2statMLP
from src.quadratic_resnet import create_quadratic_train_state
from src.geometric_loss import geometric_loss_fn, evaluate_with_geometric_metrics
from scripts.run_noprop_ct_demo import load_existing_data


def curriculum_kl_weight_schedule(epoch: int, warmup_epochs: int = 30, 
                                 transition_epochs: int = 30, max_kl_weight: float = 0.01) -> float:
    """Smooth curriculum schedule for KL weight."""
    if epoch < warmup_epochs:
        return 0.0
    elif epoch < warmup_epochs + transition_epochs:
        progress = (epoch - warmup_epochs) / transition_epochs
        # Smooth sigmoid transition
        smooth_progress = 0.5 * (1 + jnp.tanh(4 * (progress - 0.5)))
        return float(smooth_progress * max_kl_weight)
    else:
        return max_kl_weight


def train_with_curriculum(model, params, train_data, val_data, config, name="Model"):
    """Train model with curriculum learning."""
    
    num_epochs = config.get('num_epochs', 120)
    learning_rate = config.get('learning_rate', 1e-3)
    batch_size = config.get('batch_size', 64)
    warmup_epochs = config.get('warmup_epochs', 30)
    transition_epochs = config.get('transition_epochs', 30)
    max_kl_weight = config.get('max_kl_weight', 0.01)
    mse_weight = config.get('mse_weight', 1.0)
    regularization = config.get('regularization', 1e-5)
    
    # Learning rate schedule
    schedule = optax.piecewise_constant_schedule(learning_rate, {
        warmup_epochs: 0.7,  # Reduce LR when introducing KL
        warmup_epochs + transition_epochs: 0.5  # Further reduce for refinement
    })
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(schedule)
    )
    opt_state = optimizer.init(params)
    
    # Training tracking
    train_losses = []
    val_losses = []
    mse_losses = []
    kl_losses = []
    kl_weights = []
    
    best_params = params
    best_val_mse = float('inf')
    patience_counter = 0
    patience = 25
    
    print(f"  Training {name} with curriculum...")
    print(f"    Phase 1: Warmup ({warmup_epochs} epochs, MSE only)")
    print(f"    Phase 2: Transition ({transition_epochs} epochs, gradual KL)")
    print(f"    Phase 3: Refinement (remaining epochs, full geometric)")
    print(f"    Max KL weight: {max_kl_weight}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Current curriculum weight
        current_kl_weight = curriculum_kl_weight_schedule(
            epoch, warmup_epochs, transition_epochs, max_kl_weight
        )
        kl_weights.append(current_kl_weight)
        
        # Determine training phase
        if epoch < warmup_epochs:
            phase = "Warmup"
        elif epoch < warmup_epochs + transition_epochs:
            phase = "Transition"
        else:
            phase = "Refinement"
        
        # Training
        n_train = train_data['eta'].shape[0]
        indices = jnp.arange(n_train)
        rng = random.PRNGKey(epoch)
        indices = random.permutation(rng, indices)
        
        epoch_train_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_kl_loss = 0.0
        num_batches = 0
        
        for i in range(0, n_train, batch_size):
            batch_indices = indices[i:i+batch_size]
            eta_batch = train_data['eta'][batch_indices]
            y_batch = train_data['y'][batch_indices]
            cov_batch = train_data['cov'][batch_indices]
            
            def loss_fn(params):
                if current_kl_weight == 0.0:
                    # Pure MSE during warmup
                    pred = model.apply(params, eta_batch)
                    mse_loss = jnp.mean(jnp.square(pred - y_batch))
                    return mse_loss, {
                        'total_loss': mse_loss,
                        'mse_loss': mse_loss,
                        'kl_loss': 0.0
                    }
                else:
                    # Geometric loss with current KL weight
                    return geometric_loss_fn(
                        model, params, eta_batch, y_batch, cov_batch,
                        current_kl_weight, mse_weight, regularization
                    )
            
            (loss, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            
            # Check for NaN
            if jnp.isnan(loss):
                print(f"    NaN loss at epoch {epoch}, stopping")
                break
                
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            epoch_train_loss += float(loss_dict['total_loss'])
            epoch_mse_loss += float(loss_dict['mse_loss'])
            epoch_kl_loss += float(loss_dict['kl_loss'])
            num_batches += 1
        
        if num_batches == 0:
            break
            
        epoch_train_loss /= num_batches
        epoch_mse_loss /= num_batches
        epoch_kl_loss /= num_batches
        
        # Validation (always compute MSE for comparison)
        val_pred = model.apply(params, val_data['eta'])
        val_mse = float(jnp.mean(jnp.square(val_pred - val_data['y'])))
        
        train_losses.append(epoch_train_loss)
        val_losses.append(val_mse)  # Use MSE for validation tracking
        mse_losses.append(epoch_mse_loss)
        kl_losses.append(epoch_kl_loss)
        
        # Best model tracking (use MSE)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_params = params
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d} ({phase:>10}): MSE={epoch_mse_loss:.6f}, KL={epoch_kl_loss:.6f}, "
                  f"KL_w={current_kl_weight:.6f}, Val_MSE={val_mse:.6f}")
        
        # Early stopping (but not during transition)
        if epoch > warmup_epochs + transition_epochs and patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break
    
    training_time = time.time() - start_time
    print(f"  Curriculum training completed in {training_time:.1f}s")
    print(f"  Best validation MSE: {best_val_mse:.6f}")
    
    return best_params, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'mse_losses': mse_losses,
        'kl_losses': kl_losses,
        'kl_weights': kl_weights,
        'training_time': training_time,
        'best_val_mse': best_val_mse,
        'warmup_epochs': warmup_epochs,
        'transition_epochs': transition_epochs
    }


def test_curriculum_learning():
    """Test curriculum learning on our best architectures."""
    
    print("📚 CURRICULUM LEARNING EVALUATION")
    print("=" * 80)
    print("Strategy: MSE warmup → gradual KL introduction → geometric refinement")
    
    # Load data
    ef = GaussianNatural1D()
    train_data, val_data, test_data = load_existing_data("gaussian_1d")
    
    # Add covariance data
    data_file = Path("data/training_data_7a6d32f607c09d157b94129eff71e641.pkl")
    with open(data_file, 'rb') as f:
        full_data = pickle.load(f)
    
    train_data['cov'] = full_data['train_cov']
    val_data['cov'] = full_data['val_cov'][:val_data['eta'].shape[0]]
    test_data['cov'] = full_data['val_cov'][:test_data['eta'].shape[0]]
    
    print(f"Dataset: {train_data['eta'].shape[0]} train, {test_data['eta'].shape[0]} test")
    
    # Analytical baseline
    def analytical_solution(eta):
        eta1, eta2 = eta[:, 0], eta[:, 1]
        mu = -eta1 / (2 * eta2)
        sigma2 = -1 / (2 * eta2)
        E_x = mu
        E_x2 = mu**2 + sigma2
        return jnp.stack([E_x, E_x2], axis=-1)
    
    analytical_pred = analytical_solution(test_data['eta'])
    analytical_mse = float(jnp.mean(jnp.square(analytical_pred - test_data['y'])))
    print(f"Analytical MSE baseline: {analytical_mse:.8f}")
    
    results = {}
    
    # Test 1: Standard MLP with curriculum
    print(f"\n{'='*60}")
    print("1. STANDARD MLP + CURRICULUM LEARNING")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(42)
    standard_mlp = nat2statMLP(hidden_sizes=[64, 32], activation='tanh', output_dim=2)
    standard_params = standard_mlp.init(rng, test_data['eta'][:1])
    
    standard_config = {
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'batch_size': 64,
        'warmup_epochs': 30,
        'transition_epochs': 30,
        'max_kl_weight': 0.005,  # Conservative
        'mse_weight': 1.0,
        'regularization': 1e-5
    }
    
    standard_params_curr, standard_history = train_with_curriculum(
        standard_mlp, standard_params, train_data, val_data,
        standard_config, "Standard MLP"
    )
    
    # Evaluate
    standard_pred = standard_mlp.apply(standard_params_curr, test_data['eta'])
    standard_mse = float(jnp.mean(jnp.square(standard_pred - test_data['y'])))
    
    results['standard'] = {
        'name': 'Standard MLP (Curriculum)',
        'mse': standard_mse,
        'training_time': standard_history['training_time'],
        'history': standard_history
    }
    
    # Test 2: Quadratic ResNet with curriculum  
    print(f"\n{'='*60}")
    print("2. QUADRATIC RESNET + CURRICULUM LEARNING")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(43)
    quad_config = {
        'model_type': 'quadratic',
        'hidden_size': 64,
        'num_layers': 6,
        'activation': 'tanh'
    }
    quad_model, quad_params = create_quadratic_train_state(ef, quad_config, rng)
    
    quad_curriculum_config = {
        'num_epochs': 120,
        'learning_rate': 8e-4,
        'batch_size': 64,
        'warmup_epochs': 40,
        'transition_epochs': 40,
        'max_kl_weight': 0.003,  # Very conservative
        'mse_weight': 1.0,
        'regularization': 1e-6
    }
    
    quad_params_curr, quad_history = train_with_curriculum(
        quad_model, quad_params, train_data, val_data,
        quad_curriculum_config, "Quadratic ResNet"
    )
    
    # Evaluate
    quad_pred = quad_model.apply(quad_params_curr, test_data['eta'])
    quad_mse = float(jnp.mean(jnp.square(quad_pred - test_data['y'])))
    
    results['quadratic'] = {
        'name': 'Quadratic ResNet (Curriculum)',
        'mse': quad_mse,
        'training_time': quad_history['training_time'],
        'history': quad_history
    }
    
    # Test 3: Adaptive Quadratic ResNet with curriculum (our champion)
    print(f"\n{'='*60}")
    print("3. ADAPTIVE QUADRATIC RESNET + CURRICULUM LEARNING")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(44)
    adaptive_config = {
        'model_type': 'adaptive_quadratic',
        'hidden_size': 64,
        'num_layers': 6,
        'activation': 'tanh'
    }
    adaptive_model, adaptive_params = create_quadratic_train_state(ef, adaptive_config, rng)
    
    adaptive_curriculum_config = {
        'num_epochs': 140,
        'learning_rate': 8e-4,
        'batch_size': 64,
        'warmup_epochs': 50,    # Long warmup for our best model
        'transition_epochs': 50,
        'max_kl_weight': 0.002, # Very small KL weight
        'mse_weight': 1.0,
        'regularization': 1e-6
    }
    
    adaptive_params_curr, adaptive_history = train_with_curriculum(
        adaptive_model, adaptive_params, train_data, val_data,
        adaptive_curriculum_config, "Adaptive Quadratic ResNet"
    )
    
    # Evaluate
    adaptive_pred = adaptive_model.apply(adaptive_params_curr, test_data['eta'])
    adaptive_mse = float(jnp.mean(jnp.square(adaptive_pred - test_data['y'])))
    
    results['adaptive'] = {
        'name': 'Adaptive Quadratic ResNet (Curriculum)',
        'mse': adaptive_mse,
        'training_time': adaptive_history['training_time'],
        'history': adaptive_history
    }
    
    # Summary
    print(f"\n{'='*80}")
    print("CURRICULUM LEARNING RESULTS")
    print(f"{'='*80}")
    
    print(f"{'Method':<40} {'MSE':<12} {'vs Analytical':<15} {'Time(s)':<10}")
    print("-" * 80)
    print(f"{'Analytical Solution':<40} {analytical_mse:<12.8f} {'Baseline':<15} {'0.0':<10}")
    
    for key, result in results.items():
        ratio = result['mse'] / analytical_mse
        print(f"{result['name']:<40} {result['mse']:<12.6f} {ratio:<15.1f}x {result['training_time']:<10.1f}")
    
    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]['mse'])
    print(f"\n🏆 BEST CURRICULUM MODEL: {best_model[1]['name']}")
    print(f"   Final MSE: {best_model[1]['mse']:.6f}")
    print(f"   Gap to analytical: {best_model[1]['mse']/analytical_mse:.1f}x")
    print(f"   Training time: {best_model[1]['training_time']:.1f}s")
    
    # Check if curriculum helped vs our previous best results
    # Previous best: Adaptive Quadratic ResNet achieved MSE = 0.00339
    previous_best_mse = 0.00339
    
    if best_model[1]['mse'] < previous_best_mse:
        improvement = (previous_best_mse - best_model[1]['mse']) / previous_best_mse * 100
        print(f"\n🎉 BREAKTHROUGH: {improvement:.1f}% improvement over previous best!")
        print(f"   Previous best: {previous_best_mse:.6f}")
        print(f"   New best: {best_model[1]['mse']:.6f}")
    else:
        print(f"\n📊 Curriculum didn't beat previous best ({previous_best_mse:.6f})")
        print(f"   But provides valuable geometric insights!")
    
    # Create visualization
    create_curriculum_plots(results, analytical_mse)
    
    return results


def create_curriculum_plots(results, analytical_mse):
    """Create comprehensive curriculum learning visualization."""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # 1. Final performance comparison
    names = [r['name'] for r in results.values()]
    mses = [r['mse'] for r in results.values()]
    colors = ['blue', 'red', 'green']
    
    bars = axes[0, 0].bar(range(len(names)), mses, color=colors, alpha=0.7)
    axes[0, 0].axhline(y=analytical_mse, color='black', linestyle='--', linewidth=2, label='Analytical')
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels([n.replace(' ', '\\n') for n in names], rotation=0)
    axes[0, 0].set_ylabel('Test MSE (log scale)')
    axes[0, 0].set_title('Curriculum Learning: Final Results')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, mses):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.6f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # 2. Training time vs performance
    times = [r['training_time'] for r in results.values()]
    axes[0, 1].scatter(times, mses, c=colors, s=150, alpha=0.7)
    for i, name in enumerate(names):
        axes[0, 1].annotate(name.split()[0], (times[i], mses[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[0, 1].set_xlabel('Training Time (s)')
    axes[0, 1].set_ylabel('Test MSE (log scale)')
    axes[0, 1].set_title('Efficiency vs Performance')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Best model curriculum training curves
    best_model = min(results.items(), key=lambda x: x[1]['mse'])
    history = best_model[1]['history']
    epochs = range(len(history['train_losses']))
    
    axes[1, 0].plot(epochs, history['mse_losses'], label='MSE Loss', linewidth=2, color='blue')
    axes[1, 0].plot(epochs, history['kl_losses'], label='KL Loss', linewidth=2, color='red')
    axes[1, 0].axvline(x=history['warmup_epochs'], color='green', linestyle='--', alpha=0.7, 
                      label='Warmup End')
    axes[1, 0].axvline(x=history['warmup_epochs'] + history['transition_epochs'], 
                      color='orange', linestyle='--', alpha=0.7, label='Transition End')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss (log scale)')
    axes[1, 0].set_title(f'Best Model: {best_model[1]["name"]} - Training Curves')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_yscale('log')
    
    # 4. KL weight schedule
    axes[1, 1].plot(epochs, history['kl_weights'], linewidth=3, color='purple')
    axes[1, 1].fill_between(range(history['warmup_epochs']), 0, max(history['kl_weights']), 
                           alpha=0.3, color='blue', label='Warmup (MSE only)')
    axes[1, 1].fill_between(range(history['warmup_epochs'], 
                                 history['warmup_epochs'] + history['transition_epochs']), 
                           0, max(history['kl_weights']), 
                           alpha=0.3, color='orange', label='Transition (gradual KL)')
    axes[1, 1].fill_between(range(history['warmup_epochs'] + history['transition_epochs'], 
                                 len(epochs)), 
                           0, max(history['kl_weights']), 
                           alpha=0.3, color='green', label='Refinement (full geometric)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('KL Weight')
    axes[1, 1].set_title('Curriculum Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 5. Validation MSE evolution
    axes[2, 0].plot(epochs, history['val_losses'], linewidth=2, color='green')
    axes[2, 0].axvline(x=history['warmup_epochs'], color='green', linestyle='--', alpha=0.7)
    axes[2, 0].axvline(x=history['warmup_epochs'] + history['transition_epochs'], 
                      color='orange', linestyle='--', alpha=0.7)
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Validation MSE')
    axes[2, 0].set_title('Validation MSE During Curriculum')
    axes[2, 0].grid(True)
    axes[2, 0].set_yscale('log')
    
    # 6. Phase analysis
    axes[2, 1].axis('off')
    
    warmup_end = history['warmup_epochs']
    transition_end = warmup_end + history['transition_epochs']
    
    warmup_mse = history['mse_losses'][warmup_end-1] if warmup_end > 0 else history['mse_losses'][0]
    transition_mse = history['mse_losses'][transition_end-1] if transition_end < len(history['mse_losses']) else history['mse_losses'][-1]
    final_mse = history['mse_losses'][-1]
    
    summary_text = f"PHASE ANALYSIS\\n"
    summary_text += f"={'='*15}\\n\\n"
    summary_text += f"📚 Warmup Phase:\\n"
    summary_text += f"  MSE: {warmup_mse:.6f}\\n"
    summary_text += f"  Pure MSE learning\\n\\n"
    
    summary_text += f"🔄 Transition Phase:\\n"
    summary_text += f"  MSE: {transition_mse:.6f}\\n"
    summary_text += f"  Change: {(transition_mse-warmup_mse)/warmup_mse*100:+.1f}%\\n\\n"
    
    summary_text += f"🎯 Refinement Phase:\\n"
    summary_text += f"  MSE: {final_mse:.6f}\\n"
    summary_text += f"  Total: {(final_mse-warmup_mse)/warmup_mse*100:+.1f}%\\n\\n"
    
    summary_text += f"🏆 Final vs Analytical:\\n"
    summary_text += f"  {best_model[1]['mse']/analytical_mse:.1f}x worse\\n"
    
    axes[2, 1].text(0.05, 0.95, summary_text, transform=axes[2, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save plots
    output_dir = Path("artifacts/curriculum_learning_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "curriculum_learning_results.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "curriculum_learning_results.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"📊 Curriculum plots saved to {output_dir}/")
    
    return results


if __name__ == "__main__":
    print("📚 Starting curriculum learning evaluation...")
    
    try:
        results = test_curriculum_learning()
        print("\\n✅ Curriculum learning evaluation completed!")
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
