#!/usr/bin/env python3
"""
Test the improved stable curriculum learning with KL(q||p) instead of KL(p||q).

This approach is more numerically stable because we only invert the empirical
covariance matrix (from MCMC data) rather than the network-estimated covariance.

Key improvements:
1. Use KL(q||p) where q=network, p=empirical
2. Only invert stable empirical covariance matrices
3. Better curriculum schedule with more conservative KL weights
4. Improved regularization strategy
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
from src.geometric_loss import geometric_loss_fn
from scripts.run_noprop_ct_demo import load_existing_data


def stable_curriculum_kl_schedule(epoch: int, warmup_epochs: int = 40, 
                                 transition_epochs: int = 60, max_kl_weight: float = 0.001) -> float:
    """
    More conservative curriculum schedule for stable training.
    
    Args:
        epoch: Current epoch
        warmup_epochs: Extended warmup with pure MSE
        transition_epochs: Longer, gentler transition
        max_kl_weight: Much smaller maximum KL weight
        
    Returns:
        KL weight for current epoch
    """
    if epoch < warmup_epochs:
        return 0.0
    elif epoch < warmup_epochs + transition_epochs:
        progress = (epoch - warmup_epochs) / transition_epochs
        # Very smooth exponential transition
        smooth_progress = 1 - jnp.exp(-3 * progress)  # Exponential approach
        return float(smooth_progress * max_kl_weight)
    else:
        return max_kl_weight


def train_with_stable_curriculum(model, params, train_data, val_data, config, name="Model"):
    """Train with improved stable curriculum learning."""
    
    num_epochs = config.get('num_epochs', 150)
    learning_rate = config.get('learning_rate', 1e-3)
    batch_size = config.get('batch_size', 64)
    warmup_epochs = config.get('warmup_epochs', 40)
    transition_epochs = config.get('transition_epochs', 60)
    max_kl_weight = config.get('max_kl_weight', 0.001)
    mse_weight = config.get('mse_weight', 1.0)
    regularization = config.get('regularization', 1e-6)
    
    # Conservative learning rate schedule
    schedule = optax.exponential_decay(
        learning_rate, 
        transition_steps=30, 
        decay_rate=0.95,
        transition_begin=warmup_epochs
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),  # More aggressive clipping
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
    patience = 30
    
    print(f"  Training {name} with stable curriculum...")
    print(f"    Warmup: {warmup_epochs} epochs (MSE only)")
    print(f"    Transition: {transition_epochs} epochs (exponential KL ramp)")
    print(f"    Max KL weight: {max_kl_weight} (conservative)")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Current curriculum weight
        current_kl_weight = stable_curriculum_kl_schedule(
            epoch, warmup_epochs, transition_epochs, max_kl_weight
        )
        kl_weights.append(current_kl_weight)
        
        # Determine phase
        if epoch < warmup_epochs:
            phase = "Warmup"
        elif epoch < warmup_epochs + transition_epochs:
            phase = "Transition"
        else:
            phase = "Refinement"
        
        # Training with mini-batches
        n_train = train_data['eta'].shape[0]
        indices = jnp.arange(n_train)
        rng = random.PRNGKey(epoch)
        indices = random.permutation(rng, indices)
        
        epoch_train_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_kl_loss = 0.0
        num_batches = 0
        nan_detected = False
        
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
                    # Stable geometric loss
                    return geometric_loss_fn(
                        model, params, eta_batch, y_batch, cov_batch,
                        current_kl_weight, mse_weight, regularization
                    )
            
            try:
                (loss, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
                
                # Check for NaN/Inf
                if not jnp.isfinite(loss):
                    print(f"    Non-finite loss at epoch {epoch}, batch {i//batch_size}")
                    nan_detected = True
                    break
                
                # Check gradients
                grad_norm = optax.global_norm(grads)
                if not jnp.isfinite(grad_norm) or grad_norm > 100:
                    print(f"    Bad gradients at epoch {epoch}, batch {i//batch_size}")
                    nan_detected = True
                    break
                    
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                
                epoch_train_loss += float(loss_dict['total_loss'])
                epoch_mse_loss += float(loss_dict['mse_loss'])
                epoch_kl_loss += float(loss_dict['kl_loss'])
                num_batches += 1
                
            except Exception as e:
                print(f"    Training error at epoch {epoch}: {e}")
                nan_detected = True
                break
        
        if nan_detected or num_batches == 0:
            print(f"    Training instability detected, stopping at epoch {epoch}")
            break
            
        epoch_train_loss /= num_batches
        epoch_mse_loss /= num_batches
        epoch_kl_loss /= num_batches
        
        # Validation (always use MSE for comparison)
        try:
            val_pred = model.apply(params, val_data['eta'])
            val_mse = float(jnp.mean(jnp.square(val_pred - val_data['y'])))
        except:
            print(f"    Validation failed at epoch {epoch}")
            break
        
        train_losses.append(epoch_train_loss)
        val_losses.append(val_mse)
        mse_losses.append(epoch_mse_loss)
        kl_losses.append(epoch_kl_loss)
        
        # Best model tracking
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_params = params
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 15 == 0:
            print(f"    Epoch {epoch:3d} ({phase:>10}): MSE={epoch_mse_loss:.6f}, "
                  f"KL={epoch_kl_loss:.6f}, KL_w={current_kl_weight:.6f}, Val={val_mse:.6f}")
        
        # Early stopping (only during refinement phase)
        if epoch > warmup_epochs + transition_epochs and patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break
    
    training_time = time.time() - start_time
    print(f"  Stable curriculum completed in {training_time:.1f}s")
    print(f"  Best validation MSE: {best_val_mse:.8f}")
    
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


def test_stable_curriculum():
    """Test stable curriculum learning approach."""
    
    print("🔬 STABLE CURRICULUM LEARNING EVALUATION")
    print("=" * 80)
    print("Improved approach: KL(q||p) with conservative weights and extended phases")
    
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
    
    # Test 1: Standard MLP with stable curriculum
    print(f"\n{'='*60}")
    print("1. STANDARD MLP + STABLE CURRICULUM")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(42)
    standard_mlp = nat2statMLP(hidden_sizes=[64, 32], activation='tanh', output_dim=2)
    standard_params = standard_mlp.init(rng, test_data['eta'][:1])
    
    standard_config = {
        'num_epochs': 120,
        'learning_rate': 1e-3,
        'batch_size': 64,
        'warmup_epochs': 40,
        'transition_epochs': 60,
        'max_kl_weight': 0.001,  # Very conservative
        'mse_weight': 1.0,
        'regularization': 1e-6
    }
    
    standard_params_curr, standard_history = train_with_stable_curriculum(
        standard_mlp, standard_params, train_data, val_data,
        standard_config, "Standard MLP"
    )
    
    # Evaluate
    standard_pred = standard_mlp.apply(standard_params_curr, test_data['eta'])
    standard_mse = float(jnp.mean(jnp.square(standard_pred - test_data['y'])))
    
    results['standard'] = {
        'name': 'Standard MLP (Stable Curriculum)',
        'mse': standard_mse,
        'training_time': standard_history['training_time'],
        'history': standard_history
    }
    
    # Test 2: Adaptive Quadratic ResNet with stable curriculum
    print(f"\n{'='*60}")
    print("2. ADAPTIVE QUADRATIC RESNET + STABLE CURRICULUM")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(43)
    adaptive_config = {
        'model_type': 'adaptive_quadratic',
        'hidden_size': 64,
        'num_layers': 8,  # Slightly deeper
        'activation': 'tanh'
    }
    adaptive_model, adaptive_params = create_quadratic_train_state(ef, adaptive_config, rng)
    
    adaptive_curriculum_config = {
        'num_epochs': 150,
        'learning_rate': 8e-4,
        'batch_size': 64,
        'warmup_epochs': 50,    # Extended warmup
        'transition_epochs': 70, # Very gradual transition
        'max_kl_weight': 0.0005, # Ultra-conservative KL weight
        'mse_weight': 1.0,
        'regularization': 1e-7   # Better regularization
    }
    
    adaptive_params_curr, adaptive_history = train_with_stable_curriculum(
        adaptive_model, adaptive_params, train_data, val_data,
        adaptive_curriculum_config, "Adaptive Quadratic ResNet"
    )
    
    # Evaluate
    adaptive_pred = adaptive_model.apply(adaptive_params_curr, test_data['eta'])
    adaptive_mse = float(jnp.mean(jnp.square(adaptive_pred - test_data['y'])))
    
    results['adaptive'] = {
        'name': 'Adaptive Quadratic ResNet (Stable Curriculum)',
        'mse': adaptive_mse,
        'training_time': adaptive_history['training_time'],
        'history': adaptive_history
    }
    
    # Test 3: Deep Adaptive Quadratic ResNet (push the limits)
    print(f"\n{'='*60}")
    print("3. DEEP ADAPTIVE QUADRATIC RESNET + STABLE CURRICULUM")
    print(f"{'='*60}")
    
    rng = random.PRNGKey(44)
    deep_adaptive_config = {
        'model_type': 'adaptive_quadratic',
        'hidden_size': 96,
        'num_layers': 12,  # Much deeper
        'activation': 'tanh'
    }
    deep_adaptive_model, deep_adaptive_params = create_quadratic_train_state(ef, deep_adaptive_config, rng)
    
    deep_curriculum_config = {
        'num_epochs': 200,
        'learning_rate': 6e-4,   # Lower LR for deeper network
        'batch_size': 64,
        'warmup_epochs': 70,     # Long warmup for deep network
        'transition_epochs': 80,  # Extended transition
        'max_kl_weight': 0.0003, # Very small for deep network
        'mse_weight': 1.0,
        'regularization': 1e-8   # Tight regularization
    }
    
    deep_params_curr, deep_history = train_with_stable_curriculum(
        deep_adaptive_model, deep_adaptive_params, train_data, val_data,
        deep_curriculum_config, "Deep Adaptive Quadratic ResNet"
    )
    
    # Evaluate
    deep_pred = deep_adaptive_model.apply(deep_params_curr, test_data['eta'])
    deep_mse = float(jnp.mean(jnp.square(deep_pred - test_data['y'])))
    
    results['deep_adaptive'] = {
        'name': 'Deep Adaptive Quadratic ResNet (Stable Curriculum)',
        'mse': deep_mse,
        'training_time': deep_history['training_time'],
        'history': deep_history
    }
    
    # Summary
    print(f"\n{'='*80}")
    print("STABLE CURRICULUM LEARNING RESULTS")
    print(f"{'='*80}")
    
    print(f"{'Method':<50} {'MSE':<12} {'vs Analytical':<15} {'Time(s)':<10}")
    print("-" * 90)
    print(f"{'Analytical Solution':<50} {analytical_mse:<12.8f} {'Baseline':<15} {'0.0':<10}")
    
    # Sort by performance
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mse'])
    
    for key, result in sorted_results:
        ratio = result['mse'] / analytical_mse
        print(f"{result['name']:<50} {result['mse']:<12.8f} {ratio:<15.1f}x {result['training_time']:<10.1f}")
    
    # Find best model
    best_model = sorted_results[0]
    print(f"\n🏆 BEST STABLE CURRICULUM MODEL: {best_model[1]['name']}")
    print(f"   Final MSE: {best_model[1]['mse']:.8f}")
    print(f"   Gap to analytical: {best_model[1]['mse']/analytical_mse:.1f}x")
    
    # Compare with previous best results
    previous_best_mse = 0.00339300  # From comprehensive comparison
    
    if best_model[1]['mse'] < previous_best_mse:
        improvement = (previous_best_mse - best_model[1]['mse']) / previous_best_mse * 100
        print(f"\n🎉 NEW RECORD: {improvement:.1f}% improvement!")
        print(f"   Previous record: {previous_best_mse:.8f}")
        print(f"   New record: {best_model[1]['mse']:.8f}")
        print(f"   Stable curriculum + geometric loss = SUCCESS!")
    else:
        diff = (best_model[1]['mse'] - previous_best_mse) / previous_best_mse * 100
        print(f"\n📊 Result: {diff:+.1f}% vs previous best")
        print(f"   Previous best: {previous_best_mse:.8f}")
        print(f"   Current best: {best_model[1]['mse']:.8f}")
    
    # Create detailed analysis
    create_stable_curriculum_plots(results, analytical_mse, previous_best_mse)
    
    return results


def create_stable_curriculum_plots(results, analytical_mse, previous_best_mse):
    """Create comprehensive stable curriculum visualization."""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # 1. Performance comparison with previous best
    names = [r['name'] for r in results.values()]
    mses = [r['mse'] for r in results.values()]
    colors = ['blue', 'red', 'green']
    
    bars = axes[0, 0].bar(range(len(names)), mses, color=colors, alpha=0.7)
    axes[0, 0].axhline(y=analytical_mse, color='black', linestyle='--', linewidth=2, label='Analytical')
    axes[0, 0].axhline(y=previous_best_mse, color='purple', linestyle=':', linewidth=2, label='Previous Best')
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels([n.replace(' ', '\\n') for n in names], rotation=0)
    axes[0, 0].set_ylabel('Test MSE (log scale)')
    axes[0, 0].set_title('Stable Curriculum: Performance vs Previous Best')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Best model training curves
    best_model = min(results.items(), key=lambda x: x[1]['mse'])
    history = best_model[1]['history']
    epochs = range(len(history['mse_losses']))
    
    axes[0, 1].plot(epochs, history['mse_losses'], label='MSE Loss', linewidth=2, color='blue')
    axes[0, 1].plot(epochs, history['val_losses'], label='Validation MSE', linewidth=2, color='green')
    axes[0, 1].axvline(x=history['warmup_epochs'], color='red', linestyle='--', alpha=0.7, 
                      label='Warmup End')
    axes[0, 1].axvline(x=history['warmup_epochs'] + history['transition_epochs'], 
                      color='orange', linestyle='--', alpha=0.7, label='Transition End')
    axes[0, 1].axhline(y=analytical_mse, color='black', linestyle='--', alpha=0.7, label='Analytical')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE (log scale)')
    axes[0, 1].set_title(f'Best Model Training: {best_model[1]["name"]}')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_yscale('log')
    
    # 3. KL loss evolution
    axes[1, 0].plot(epochs, history['kl_losses'], linewidth=2, color='red')
    axes[1, 0].axvline(x=history['warmup_epochs'], color='red', linestyle='--', alpha=0.7)
    axes[1, 0].axvline(x=history['warmup_epochs'] + history['transition_epochs'], 
                      color='orange', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Loss (log scale)')
    axes[1, 0].set_title('KL Loss Evolution')
    axes[1, 0].grid(True)
    axes[1, 0].set_yscale('log')
    
    # 4. KL weight schedule
    axes[1, 1].plot(epochs, history['kl_weights'], linewidth=3, color='purple')
    axes[1, 1].fill_between(range(history['warmup_epochs']), 0, max(history['kl_weights']), 
                           alpha=0.3, color='blue', label='Warmup')
    axes[1, 1].fill_between(range(history['warmup_epochs'], 
                                 history['warmup_epochs'] + history['transition_epochs']), 
                           0, max(history['kl_weights']), 
                           alpha=0.3, color='orange', label='Transition')
    axes[1, 1].fill_between(range(history['warmup_epochs'] + history['transition_epochs'], 
                                 len(epochs)), 
                           0, max(history['kl_weights']), 
                           alpha=0.3, color='green', label='Refinement')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('KL Weight')
    axes[1, 1].set_title('Stable Curriculum Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 5. All models comparison
    all_names = [r['name'].split('(')[0].strip() for r in results.values()]
    all_mses = [r['mse'] for r in results.values()]
    
    bars = axes[2, 0].bar(range(len(all_names)), all_mses, color=colors, alpha=0.7)
    axes[2, 0].axhline(y=analytical_mse, color='black', linestyle='--', linewidth=2)
    axes[2, 0].axhline(y=previous_best_mse, color='purple', linestyle=':', linewidth=2)
    axes[2, 0].set_xticks(range(len(all_names)))
    axes[2, 0].set_xticklabels(all_names, rotation=45, ha='right')
    axes[2, 0].set_ylabel('Test MSE (log scale)')
    axes[2, 0].set_title('All Models Performance')
    axes[2, 0].set_yscale('log')
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. Summary and insights
    axes[2, 1].axis('off')
    
    summary_text = f"STABLE CURRICULUM RESULTS\\n"
    summary_text += f"={'='*30}\\n\\n"
    summary_text += f"🏆 Best: {best_model[1]['name'].split('(')[0]}\\n"
    summary_text += f"MSE: {best_model[1]['mse']:.8f}\\n"
    summary_text += f"vs Analytical: {best_model[1]['mse']/analytical_mse:.1f}x\\n\\n"
    
    summary_text += f"📊 vs Previous Best:\\n"
    if best_model[1]['mse'] < previous_best_mse:
        improvement = (previous_best_mse - best_model[1]['mse']) / previous_best_mse * 100
        summary_text += f"🎉 NEW RECORD: {improvement:.1f}% better\\n"
    else:
        diff = (best_model[1]['mse'] - previous_best_mse) / previous_best_mse * 100
        summary_text += f"📈 {diff:+.1f}% vs previous\\n"
    
    summary_text += f"\\n🔬 STABLE APPROACH:\\n"
    summary_text += f"• KL(q||p) instead of KL(p||q)\\n"
    summary_text += f"• Only invert empirical covariance\\n"
    summary_text += f"• Conservative KL weights\\n"
    summary_text += f"• Extended curriculum phases\\n"
    summary_text += f"• Better regularization\\n\\n"
    
    summary_text += f"💡 GEOMETRIC INSIGHTS:\\n"
    summary_text += f"• Respects natural geometry\\n"
    summary_text += f"• Uses second-order information\\n"
    summary_text += f"• Network Jacobian → covariance\\n"
    summary_text += f"• KL divergence as natural metric\\n"
    
    axes[2, 1].text(0.05, 0.95, summary_text, transform=axes[2, 1].transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save plots
    output_dir = Path("artifacts/stable_curriculum_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "stable_curriculum_results.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "stable_curriculum_results.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"📊 Stable curriculum plots saved to {output_dir}/")
    
    return results


if __name__ == "__main__":
    print("🔬 Starting stable curriculum learning evaluation...")
    
    try:
        results = test_stable_curriculum()
        print("\\n✅ Stable curriculum evaluation completed!")
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
