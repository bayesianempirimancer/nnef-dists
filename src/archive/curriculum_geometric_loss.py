"""
Curriculum learning approach for geometric loss function.

This module implements a training strategy that starts with standard MSE loss
and gradually introduces the geometric KL loss for refinement. This allows
the network to first learn the basic mapping, then improve using the natural
geometry of the exponential family.

Training phases:
1. Warm-up: Pure MSE loss (epochs 0-30)
2. Transition: Gradually increase KL weight (epochs 30-60) 
3. Refinement: Full geometric loss (epochs 60+)
"""

import jax
import jax.numpy as jnp
from jax import random
import optax
import time
from typing import Dict, Any, Tuple, Callable
from .geometric_loss import geometric_loss_fn, evaluate_with_geometric_metrics

Array = jax.Array


def curriculum_kl_weight_schedule(epoch: int, warmup_epochs: int = 30, 
                                 transition_epochs: int = 30, max_kl_weight: float = 0.01) -> float:
    """
    Curriculum learning schedule for KL weight.
    
    Args:
        epoch: Current epoch
        warmup_epochs: Number of epochs with pure MSE (KL weight = 0)
        transition_epochs: Number of epochs to gradually increase KL weight
        max_kl_weight: Maximum KL weight to reach
        
    Returns:
        KL weight for current epoch
    """
    if epoch < warmup_epochs:
        # Warm-up phase: pure MSE
        return 0.0
    elif epoch < warmup_epochs + transition_epochs:
        # Transition phase: gradually increase KL weight
        progress = (epoch - warmup_epochs) / transition_epochs
        # Use smooth transition (sigmoid-like)
        smooth_progress = 0.5 * (1 + jnp.tanh(4 * (progress - 0.5)))
        return float(smooth_progress * max_kl_weight)
    else:
        # Refinement phase: full KL weight
        return max_kl_weight


def train_with_curriculum_geometric_loss(
    model,
    params,
    train_data: Dict[str, Array],
    val_data: Dict[str, Array],
    config: Dict[str, Any],
    name: str = "Model"
) -> Tuple[Dict, Dict]:
    """
    Train a model using curriculum learning with geometric loss.
    
    Args:
        model: Neural network model
        params: Initial parameters
        train_data: Training data with 'eta', 'y', and 'cov' keys
        val_data: Validation data with 'eta', 'y', and 'cov' keys
        config: Training configuration
        name: Model name for logging
        
    Returns:
        Trained parameters, training history
    """
    
    num_epochs = config.get('num_epochs', 120)
    learning_rate = config.get('learning_rate', 1e-3)
    batch_size = config.get('batch_size', 64)
    
    # Curriculum parameters
    warmup_epochs = config.get('warmup_epochs', 30)
    transition_epochs = config.get('transition_epochs', 30)
    max_kl_weight = config.get('max_kl_weight', 0.01)
    mse_weight = config.get('mse_weight', 1.0)
    regularization = config.get('regularization', 1e-5)
    
    # Learning rate scheduling
    use_lr_schedule = config.get('use_lr_schedule', True)
    if use_lr_schedule:
        # Reduce learning rate when KL loss is introduced
        def lr_schedule(epoch):
            if epoch < warmup_epochs:
                return learning_rate
            elif epoch < warmup_epochs + transition_epochs:
                return learning_rate * 0.5  # Reduce LR during transition
            else:
                return learning_rate * 0.3   # Further reduce for refinement
        
        schedule = optax.piecewise_constant_schedule(learning_rate, {
            warmup_epochs: 0.5,
            warmup_epochs + transition_epochs: 0.6
        })
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
    
    # Training history
    train_losses = []
    val_losses = []
    mse_losses = []
    kl_losses = []
    kl_weights = []
    
    best_params = params
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 25
    
    print(f"  Training {name} with curriculum geometric loss...")
    print(f"    Warmup: {warmup_epochs} epochs (MSE only)")
    print(f"    Transition: {transition_epochs} epochs (gradual KL)")
    print(f"    Refinement: {num_epochs - warmup_epochs - transition_epochs} epochs (full geometric)")
    print(f"    Max KL weight: {max_kl_weight}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Determine current KL weight using curriculum
        current_kl_weight = curriculum_kl_weight_schedule(
            epoch, warmup_epochs, transition_epochs, max_kl_weight
        )
        kl_weights.append(current_kl_weight)
        
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
        
        # Validation (always use current weights)
        if current_kl_weight == 0.0:
            val_pred = model.apply(params, val_data['eta'])
            val_loss = float(jnp.mean(jnp.square(val_pred - val_data['y'])))
        else:
            val_loss, _ = geometric_loss_fn(
                model, params, val_data['eta'], val_data['y'], val_data['cov'],
                current_kl_weight, mse_weight, regularization
            )
            val_loss = float(val_loss)
        
        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        mse_losses.append(epoch_mse_loss)
        kl_losses.append(epoch_kl_loss)
        
        # Best model tracking (use MSE for comparison)
        val_pred = model.apply(params, val_data['eta'])
        val_mse = float(jnp.mean(jnp.square(val_pred - val_data['y'])))
        
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_params = params
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            phase = "Warmup" if current_kl_weight == 0.0 else "Transition" if current_kl_weight < max_kl_weight else "Refinement"
            print(f"    Epoch {epoch:3d} ({phase}): Total={epoch_train_loss:.6f}, MSE={epoch_mse_loss:.6f}, "
                  f"KL={epoch_kl_loss:.6f}, KL_weight={current_kl_weight:.6f}")
        
        # Early stopping (but not during transition)
        if epoch > warmup_epochs + transition_epochs and patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break
    
    training_time = time.time() - start_time
    print(f"  Curriculum training completed in {training_time:.1f}s, Best MSE: {best_val_loss:.6f}")
    
    return best_params, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'mse_losses': mse_losses,
        'kl_losses': kl_losses,
        'kl_weights': kl_weights,
        'training_time': training_time,
        'best_val_loss': best_val_loss,
        'warmup_epochs': warmup_epochs,
        'transition_epochs': transition_epochs
    }


def visualize_curriculum_training(history, name="Model"):
    """Visualize the curriculum learning process."""
    
    import matplotlib.pyplot as plt
    
    epochs = range(len(history['train_losses']))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss components
    axes[0, 0].plot(epochs, history['mse_losses'], label='MSE Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['kl_losses'], label='KL Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['train_losses'], label='Total Loss', linewidth=2, linestyle='--')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f'{name}: Training Loss Components')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_yscale('log')
    
    # KL weight schedule
    axes[0, 1].plot(epochs, history['kl_weights'], linewidth=3, color='purple')
    axes[0, 1].axvline(x=history['warmup_epochs'], color='red', linestyle='--', alpha=0.7, label='Warmup End')
    axes[0, 1].axvline(x=history['warmup_epochs'] + history['transition_epochs'], 
                      color='orange', linestyle='--', alpha=0.7, label='Transition End')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('KL Weight')
    axes[0, 1].set_title('Curriculum: KL Weight Schedule')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Validation loss
    axes[1, 0].plot(epochs, history['val_losses'], linewidth=2, color='green')
    axes[1, 0].axvline(x=history['warmup_epochs'], color='red', linestyle='--', alpha=0.7)
    axes[1, 0].axvline(x=history['warmup_epochs'] + history['transition_epochs'], 
                      color='orange', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Validation Loss')
    axes[1, 0].set_title('Validation Loss During Curriculum')
    axes[1, 0].grid(True)
    axes[1, 0].set_yscale('log')
    
    # Phase analysis
    axes[1, 1].axis('off')
    
    warmup_end = history['warmup_epochs']
    transition_end = warmup_end + history['transition_epochs']
    
    # Calculate phase performance
    warmup_mse = history['mse_losses'][warmup_end-1] if warmup_end > 0 else history['mse_losses'][0]
    transition_mse = history['mse_losses'][transition_end-1] if transition_end < len(history['mse_losses']) else history['mse_losses'][-1]
    final_mse = history['mse_losses'][-1]
    
    summary_text = f"CURRICULUM LEARNING\\n"
    summary_text += f"={'='*20}\\n\\n"
    summary_text += f"📚 PHASE ANALYSIS:\\n"
    summary_text += f"Warmup (MSE only):\\n"
    summary_text += f"  Epochs: 0-{warmup_end}\\n"
    summary_text += f"  Final MSE: {warmup_mse:.6f}\\n\\n"
    
    summary_text += f"Transition (gradual KL):\\n"
    summary_text += f"  Epochs: {warmup_end}-{transition_end}\\n"
    summary_text += f"  Final MSE: {transition_mse:.6f}\\n"
    summary_text += f"  Change: {(transition_mse-warmup_mse)/warmup_mse*100:+.1f}%\\n\\n"
    
    summary_text += f"Refinement (full geometric):\\n"
    summary_text += f"  Epochs: {transition_end}+\\n"
    summary_text += f"  Final MSE: {final_mse:.6f}\\n"
    summary_text += f"  Total change: {(final_mse-warmup_mse)/warmup_mse*100:+.1f}%\\n"
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    return fig


def test_curriculum_on_best_architectures():
    """Test curriculum learning on our best architectures."""
    
    print("📚 CURRICULUM LEARNING WITH GEOMETRIC LOSS")
    print("=" * 70)
    print("Testing curriculum approach: MSE → gradual KL → full geometric")
    
    # Import here to avoid circular imports
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from src.ef import GaussianNatural1D
    from src.model import nat2statMLP
    from src.quadratic_resnet import create_quadratic_train_state
    from scripts.run_noprop_ct_demo import load_existing_data
    
    # Load data
    ef = GaussianNatural1D()
    train_data, val_data, test_data = load_existing_data("gaussian_1d")
    
    # Add covariance data
    import pickle
    data_file = Path("data/training_data_7a6d32f607c09d157b94129eff71e641.pkl")
    with open(data_file, 'rb') as f:
        full_data = pickle.load(f)
    
    train_data['cov'] = full_data['train_cov']
    val_data['cov'] = full_data['val_cov'][:val_data['eta'].shape[0]]
    test_data['cov'] = full_data['val_cov'][:test_data['eta'].shape[0]]
    
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
    print(f"\n{'='*70}")
    print("1. STANDARD MLP + CURRICULUM GEOMETRIC LOSS")
    print(f"{'='*70}")
    
    rng = random.PRNGKey(42)
    standard_mlp = nat2statMLP(hidden_sizes=[64, 32], activation='tanh', output_dim=2)
    standard_params = standard_mlp.init(rng, test_data['eta'][:1])
    
    curriculum_config = {
        'num_epochs': 120,
        'learning_rate': 1e-3,
        'batch_size': 64,
        'warmup_epochs': 30,
        'transition_epochs': 30,
        'max_kl_weight': 0.01,
        'mse_weight': 1.0,
        'regularization': 1e-5,
        'use_lr_schedule': True
    }
    
    standard_params_curr, standard_history_curr = train_with_curriculum_geometric_loss(
        standard_mlp, standard_params, train_data, val_data,
        curriculum_config, "Standard MLP (Curriculum)"
    )
    
    results['standard_curriculum'] = evaluate_with_geometric_metrics(
        standard_mlp, standard_params_curr, test_data, "Standard MLP (Curriculum)"
    )
    results['standard_curriculum']['training_time'] = standard_history_curr['training_time']
    results['standard_curriculum']['history'] = standard_history_curr
    
    # Test 2: Quadratic ResNet with curriculum
    print(f"\n{'='*70}")
    print("2. QUADRATIC RESNET + CURRICULUM GEOMETRIC LOSS")
    print(f"{'='*70}")
    
    rng = random.PRNGKey(43)
    quad_config = {
        'model_type': 'quadratic',
        'hidden_size': 64,
        'num_layers': 6,
        'activation': 'tanh'
    }
    quad_model, quad_params = create_quadratic_train_state(ef, quad_config, rng)
    
    # Use more conservative settings for quadratic ResNet
    quad_curriculum_config = {
        'num_epochs': 120,
        'learning_rate': 8e-4,  # Slightly lower LR
        'batch_size': 64,
        'warmup_epochs': 40,    # Longer warmup
        'transition_epochs': 40, # Longer transition
        'max_kl_weight': 0.005, # Smaller max KL weight
        'mse_weight': 1.0,
        'regularization': 1e-5,
        'use_lr_schedule': True
    }
    
    quad_params_curr, quad_history_curr = train_with_curriculum_geometric_loss(
        quad_model, quad_params, train_data, val_data,
        quad_curriculum_config, "Quadratic ResNet (Curriculum)"
    )
    
    results['quadratic_curriculum'] = evaluate_with_geometric_metrics(
        quad_model, quad_params_curr, test_data, "Quadratic ResNet (Curriculum)"
    )
    results['quadratic_curriculum']['training_time'] = quad_history_curr['training_time']
    results['quadratic_curriculum']['history'] = quad_history_curr
    
    # Test 3: Adaptive Quadratic ResNet with curriculum (our champion)
    print(f"\n{'='*70}")
    print("3. ADAPTIVE QUADRATIC RESNET + CURRICULUM GEOMETRIC LOSS")
    print(f"{'='*70}")
    
    rng = random.PRNGKey(44)
    adaptive_config = {
        'model_type': 'adaptive_quadratic',
        'hidden_size': 64,
        'num_layers': 6,
        'activation': 'tanh'
    }
    adaptive_model, adaptive_params = create_quadratic_train_state(ef, adaptive_config, rng)
    
    # Conservative settings for our best model
    adaptive_curriculum_config = {
        'num_epochs': 150,
        'learning_rate': 8e-4,
        'batch_size': 64,
        'warmup_epochs': 50,    # Longer warmup for stability
        'transition_epochs': 50, # Gradual transition
        'max_kl_weight': 0.003, # Very conservative KL weight
        'mse_weight': 1.0,
        'regularization': 1e-6,
        'use_lr_schedule': True
    }
    
    adaptive_params_curr, adaptive_history_curr = train_with_curriculum_geometric_loss(
        adaptive_model, adaptive_params, train_data, val_data,
        adaptive_curriculum_config, "Adaptive Quadratic ResNet (Curriculum)"
    )
    
    results['adaptive_curriculum'] = evaluate_with_geometric_metrics(
        adaptive_model, adaptive_params_curr, test_data, "Adaptive Quadratic ResNet (Curriculum)"
    )
    results['adaptive_curriculum']['training_time'] = adaptive_history_curr['training_time']
    results['adaptive_curriculum']['history'] = adaptive_history_curr
    
    # Summary
    print(f"\n{'='*70}")
    print("CURRICULUM GEOMETRIC LOSS RESULTS")
    print(f"{'='*70}")
    
    print(f"{'Method':<35} {'MSE':<12} {'vs Analytical':<15} {'Time(s)':<10}")
    print("-" * 75)
    print(f"{'Analytical':<35} {analytical_mse:<12.8f} {'Baseline':<15} {'0.0':<10}")
    
    for key, result in results.items():
        ratio = result['mse'] / analytical_mse
        print(f"{result['name']:<35} {result['mse']:<12.6f} {ratio:<15.1f}x {result['training_time']:<10.1f}")
    
    # Find best curriculum model
    best_curriculum = min(results.items(), key=lambda x: x[1]['mse'])
    print(f"\n🏆 BEST CURRICULUM MODEL: {best_curriculum[1]['name']}")
    print(f"   MSE: {best_curriculum[1]['mse']:.6f}")
    print(f"   Gap to analytical: {best_curriculum[1]['mse']/analytical_mse:.1f}x")
    
    # Create curriculum visualization
    create_curriculum_plots(results, analytical_mse)
    
    return results


def create_curriculum_plots(results, analytical_mse):
    """Create comprehensive curriculum learning plots."""
    
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # 1. Performance comparison
    names = [r['name'] for r in results.values()]
    mses = [r['mse'] for r in results.values()]
    colors = ['blue', 'red', 'green']
    
    bars = axes[0, 0].bar(range(len(names)), mses, color=colors, alpha=0.7)
    axes[0, 0].axhline(y=analytical_mse, color='black', linestyle='--', label='Analytical')
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels([n.replace(' ', '\\n') for n in names], rotation=0)
    axes[0, 0].set_ylabel('Test MSE (log scale)')
    axes[0, 0].set_title('Curriculum Learning Results')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Training time vs performance
    times = [r['training_time'] for r in results.values()]
    axes[0, 1].scatter(times, mses, c=colors, s=100, alpha=0.7)
    for i, name in enumerate(names):
        axes[0, 1].annotate(name.split()[0], (times[i], mses[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[0, 1].set_xlabel('Training Time (s)')
    axes[0, 1].set_ylabel('Test MSE (log scale)')
    axes[0, 1].set_title('Efficiency vs Performance')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Curriculum training curves for best model
    best_model = min(results.items(), key=lambda x: x[1]['mse'])
    history = best_model[1]['history']
    epochs = range(len(history['train_losses']))
    
    axes[1, 0].plot(epochs, history['mse_losses'], label='MSE Loss', linewidth=2)
    axes[1, 0].plot(epochs, history['kl_losses'], label='KL Loss', linewidth=2)
    axes[1, 0].axvline(x=history['warmup_epochs'], color='red', linestyle='--', alpha=0.7, label='Warmup End')
    axes[1, 0].axvline(x=history['warmup_epochs'] + history['transition_epochs'], 
                      color='orange', linestyle='--', alpha=0.7, label='Transition End')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title(f'Best Model: {best_model[1]["name"]} - Loss Components')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_yscale('log')
    
    # 4. KL weight schedule for best model
    axes[1, 1].plot(epochs, history['kl_weights'], linewidth=3, color='purple')
    axes[1, 1].fill_between(range(history['warmup_epochs']), 0, max(history['kl_weights']), 
                           alpha=0.3, color='red', label='Warmup Phase')
    axes[1, 1].fill_between(range(history['warmup_epochs'], 
                                 history['warmup_epochs'] + history['transition_epochs']), 
                           0, max(history['kl_weights']), 
                           alpha=0.3, color='orange', label='Transition Phase')
    axes[1, 1].fill_between(range(history['warmup_epochs'] + history['transition_epochs'], 
                                 len(epochs)), 
                           0, max(history['kl_weights']), 
                           alpha=0.3, color='green', label='Refinement Phase')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('KL Weight')
    axes[1, 1].set_title('Curriculum Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 5. Validation loss evolution
    axes[2, 0].plot(epochs, history['val_losses'], linewidth=2, color='green')
    axes[2, 0].axvline(x=history['warmup_epochs'], color='red', linestyle='--', alpha=0.7)
    axes[2, 0].axvline(x=history['warmup_epochs'] + history['transition_epochs'], 
                      color='orange', linestyle='--', alpha=0.7)
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Validation Loss')
    axes[2, 0].set_title('Validation Loss During Curriculum')
    axes[2, 0].grid(True)
    axes[2, 0].set_yscale('log')
    
    # 6. Summary statistics
    axes[2, 1].axis('off')
    
    summary_text = f"CURRICULUM RESULTS\\n"
    summary_text += f"={'='*25}\\n\\n"
    summary_text += f"🏆 Best: {best_model[1]['name']}\\n"
    summary_text += f"MSE: {best_model[1]['mse']:.6f}\\n"
    summary_text += f"vs Analytical: {best_model[1]['mse']/analytical_mse:.1f}x\\n\\n"
    
    summary_text += f"📚 CURRICULUM STRATEGY:\\n"
    summary_text += f"• Phase 1: Pure MSE learning\\n"
    summary_text += f"• Phase 2: Gradual KL introduction\\n"
    summary_text += f"• Phase 3: Full geometric refinement\\n\\n"
    
    summary_text += f"🎯 KEY BENEFITS:\\n"
    summary_text += f"• Stable initial learning\\n"
    summary_text += f"• Smooth transition to geometry\\n"
    summary_text += f"• Second-order refinement\\n"
    summary_text += f"• Respects natural structure\\n"
    
    axes[2, 1].text(0.05, 0.95, summary_text, transform=axes[2, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save plots
    output_dir = Path("artifacts/curriculum_geometric_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "curriculum_geometric_results.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "curriculum_geometric_results.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"📊 Curriculum plots saved to {output_dir}/")
    
    return results


if __name__ == "__main__":
    print("📚 Starting curriculum geometric loss evaluation...")
    
    try:
        results = test_curriculum_on_best_architectures()
        print("\\n✅ Curriculum evaluation completed!")
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
