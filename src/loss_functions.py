"""
Loss functions for neural network training.

This module provides various loss functions including:
1. Standard losses (MSE, MAE, Huber)
2. Geometric losses (KL divergence using network Jacobian)
3. Curriculum learning strategies
4. Specialized loss functions for exponential families

Combines functionality from geometric_loss.py and curriculum_geometric_loss.py
"""

import jax
import jax.numpy as jnp
from jax import random, grad, jacfwd, jacrev
import flax.linen as nn
from typing import Callable, Tuple, Dict, Any, Optional
import optax
import time
import matplotlib.pyplot as plt

Array = jax.Array


# =============================================================================
# STANDARD LOSS FUNCTIONS
# =============================================================================

def mse_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Standard Mean Squared Error loss."""
    return jnp.mean(jnp.square(predictions - targets))


def mae_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Mean Absolute Error loss."""
    return jnp.mean(jnp.abs(predictions - targets))


def huber_loss(predictions: jnp.ndarray, targets: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    """Huber loss (smooth L1 loss)."""
    residual = jnp.abs(predictions - targets)
    return jnp.mean(jnp.where(
        residual <= delta,
        0.5 * residual ** 2,
        delta * (residual - 0.5 * delta)
    ))


# =============================================================================
# GEOMETRIC LOSS FUNCTIONS
# =============================================================================

def compute_network_jacobian(model, params, eta):
    """
    Compute the Jacobian of the network output w.r.t. natural parameters.
    
    Args:
        model: Flax model
        params: Model parameters
        eta: Natural parameters [batch_size, eta_dim]
    
    Returns:
        Jacobian tensor [batch_size, output_dim, eta_dim]
    """
    def network_fn(eta_single):
        return model.apply(params, eta_single[None, :], training=False)[0]
    
    # Compute Jacobian for each sample in the batch
    jacobian_fn = jacfwd(network_fn)
    jacobians = jax.vmap(jacobian_fn)(eta)
    
    return jacobians


def estimate_covariance_from_jacobian(jacobian, regularization=1e-6):
    """
    Extract covariance matrix from network Jacobian for exponential families.
    
    THEORETICAL FOUNDATION:
    For exponential families: p(x|η) = h(x) exp(η^T T(x) - A(η))
    
    Key relationships:
    - E[T(X)] = ∇A(η)  (our network approximates this)
    - Cov[T(X)] = ∇²A(η) = ∇_η E[T(X)]  (Fisher Information Matrix)
    
    IMPORTANT: The Jacobian ∇_η f(η) where f(η) ≈ E[T(X)] should directly
    give us the covariance structure, NOT J @ J^T as in the original implementation.
    
    However, there's a dimensional challenge: Jacobian is [batch, 9, 12] but
    we need covariance [batch, 9, 9]. This requires careful interpretation of
    the Fisher Information Matrix structure for multivariate Gaussians.
    
    Args:
        jacobian: Jacobian tensor [batch_size, output_dim, eta_dim]
        regularization: Small value added to diagonal for numerical stability
    
    Returns:
        Covariance matrices [batch_size, output_dim, output_dim]
    """
    # TODO: THEORETICAL ISSUE TO RESOLVE
    # The current implementation still has dimensional challenges.
    # For proper exponential family theory:
    # - Jacobian ∇_η E[T(X)] should give Fisher Info Matrix directly
    # - But Jacobian is [batch, 9, 12] and we need Cov[T(X)] as [batch, 9, 9]
    # - Need to properly extract the [9,9] covariance structure from [9,12] Jacobian
    
    # Current implementation uses approximations until this is resolved properly
    
    # For now, let's use a more conservative approach:
    # If jacobian represents ∇_η μ, then we can estimate Cov from the structure
    
    batch_size, output_dim, eta_dim = jacobian.shape
    
    # For 3D Gaussian case with tril format (9D output, 12D input):
    # The jacobian should give us information about the covariance structure
    
    if output_dim == 9 and eta_dim == 12:
        # 3D Gaussian case: extract covariance from Jacobian structure
        # This is a simplified approach - the full Fisher matrix would be more complex
        
        # CORRECTED: For exponential families, the Fisher Information Matrix
        # is the Jacobian ∇_η E[T(X)], not the Jacobian squared.
        # 
        # However, the Jacobian is [batch, 9, 12] and we need [batch, 9, 9].
        # The Fisher Information Matrix for multivariate Gaussian has a specific
        # block structure. For now, we'll use a simplified diagonal approximation.
        
        # Use the norm of each row of the Jacobian as diagonal elements
        jacobian_row_norms = jnp.linalg.norm(jacobian, axis=-1)  # [batch, output_dim]
        
        # Create diagonal covariance matrix
        cov_matrices = jnp.zeros((batch_size, output_dim, output_dim))
        for i in range(output_dim):
            cov_matrices = cov_matrices.at[:, i, i].set(jacobian_row_norms[:, i] + regularization)
        
    else:
        # General case: use a more conservative diagonal approximation
        jacobian_norm = jnp.linalg.norm(jacobian, axis=-1)  # [batch, output_dim]
        
        # Create diagonal covariance matrix
        cov_matrices = jnp.zeros((batch_size, output_dim, output_dim))
        eye = jnp.eye(output_dim)
        
        for b in range(batch_size):
            diag_cov = jnp.diag(jacobian_norm[b] + regularization)
            cov_matrices = cov_matrices.at[b].set(diag_cov)
    
    return cov_matrices


def kl_divergence_multivariate_normal(mu_emp, cov_emp, mu_net, cov_net, regularization=1e-6):
    """
    Stable computation of KL divergence between two multivariate normal distributions.
    
    KL(p || q) = 0.5 * [tr(Σ_q^{-1} Σ_p) + (μ_q - μ_p)^T Σ_q^{-1} (μ_q - μ_p) - k + log(det(Σ_q)/det(Σ_p))]
    
    Args:
        mu_emp: Empirical means [batch_size, dim]
        cov_emp: Empirical covariances [batch_size, dim, dim]
        mu_net: Network means [batch_size, dim]
        cov_net: Network covariances [batch_size, dim, dim]
        regularization: Regularization for numerical stability
    
    Returns:
        KL divergences [batch_size]
    """
    batch_size, dim = mu_emp.shape
    
    # Add regularization to covariances
    eye = jnp.eye(dim)
    cov_emp_reg = cov_emp + regularization * eye[None, :, :]
    cov_net_reg = cov_net + regularization * eye[None, :, :]
    
    try:
        # Compute Cholesky decompositions for numerical stability
        L_emp = jnp.linalg.cholesky(cov_emp_reg)
        L_net = jnp.linalg.cholesky(cov_net_reg)
        
        # Solve linear systems using Cholesky
        # tr(Σ_q^{-1} Σ_p) = tr(L_net^{-T} L_net^{-1} L_emp L_emp^T)
        X = jnp.linalg.solve(L_net, L_emp)
        trace_term = jnp.sum(X ** 2, axis=(1, 2))
        
        # (μ_q - μ_p)^T Σ_q^{-1} (μ_q - μ_p)
        mu_diff = mu_net - mu_emp
        y = jnp.linalg.solve(L_net, mu_diff[..., None])
        mahalanobis_term = jnp.sum(y ** 2, axis=(1, 2))
        
        # log(det(Σ_q)/det(Σ_p)) = 2 * (log(det(L_net)) - log(det(L_emp)))
        log_det_net = 2 * jnp.sum(jnp.log(jnp.diagonal(L_net, axis1=1, axis2=2)), axis=1)
        log_det_emp = 2 * jnp.sum(jnp.log(jnp.diagonal(L_emp, axis1=1, axis2=2)), axis=1)
        log_det_term = log_det_net - log_det_emp
        
        # Combine terms
        kl = 0.5 * (trace_term + mahalanobis_term - dim + log_det_term)
        
        return kl
        
    except Exception:
        # Fallback to regularized computation if Cholesky fails
        cov_net_inv = jnp.linalg.inv(cov_net_reg)
        
        trace_term = jnp.trace(cov_net_inv @ cov_emp_reg, axis1=1, axis2=2)
        
        mu_diff = mu_net - mu_emp
        mahalanobis_term = jnp.sum(mu_diff[:, None, :] @ cov_net_inv @ mu_diff[:, :, None], axis=(1, 2))
        
        log_det_net = jnp.linalg.slogdet(cov_net_reg)[1]
        log_det_emp = jnp.linalg.slogdet(cov_emp_reg)[1]
        log_det_term = log_det_net - log_det_emp
        
        kl = 0.5 * (trace_term + mahalanobis_term - dim + log_det_term)
        
        return kl


def geometric_loss_fn(model, params, eta_batch, y_batch, cov_batch, 
                     kl_weight=0.01, mse_weight=1.0, regularization=1e-6):
    """
    Geometric loss function combining MSE and KL divergence.
    
    Loss = mse_weight * MSE + kl_weight * KL(empirical || network)
    
    Args:
        model: Neural network model
        params: Model parameters
        eta_batch: Natural parameters [batch_size, eta_dim]
        y_batch: Target statistics [batch_size, stat_dim]
        cov_batch: Empirical covariances [batch_size, stat_dim, stat_dim]
        kl_weight: Weight for KL divergence term
        mse_weight: Weight for MSE term
        regularization: Numerical regularization
    
    Returns:
        Combined loss value
    """
    # Standard MSE loss
    predictions = model.apply(params, eta_batch, training=True)
    mse_loss_val = jnp.mean(jnp.square(predictions - y_batch))
    
    if kl_weight > 0:
        # Compute network Jacobian
        jacobian = compute_network_jacobian(model, params, eta_batch)
        
        # Estimate network covariance from Jacobian
        cov_net = estimate_covariance_from_jacobian(jacobian, regularization)
        
        # Extract means (assuming first 3 components are means for 3D case)
        mu_emp = y_batch[:, :3]
        mu_net = predictions[:, :3]
        
        # Extract covariance parts (assuming tril format)
        # For tril format: [mu1, mu2, mu3, cov_11, cov_12, cov_13, cov_22, cov_23, cov_33]
        if y_batch.shape[1] == 9:  # 3D tril format
            # Use empirical covariance from cov_batch
            cov_emp = cov_batch
            
            # For network covariance, use Jacobian estimate
            cov_net_stats = cov_net
        else:
            # Fallback for other formats
            cov_emp = cov_batch
            cov_net_stats = cov_net
        
        # Compute KL divergence
        kl_loss_val = jnp.mean(kl_divergence_multivariate_normal_stable(
            mu_emp, cov_emp, mu_net, cov_net_stats, regularization
        ))
        
        # Handle NaN/Inf in KL loss
        kl_loss_val = jnp.where(jnp.isfinite(kl_loss_val), kl_loss_val, 0.0)
    else:
        kl_loss_val = 0.0
    
    total_loss = mse_weight * mse_loss_val + kl_weight * kl_loss_val
    
    return total_loss


# =============================================================================
# CURRICULUM LEARNING
# =============================================================================

def curriculum_kl_weight_schedule(epoch: int, warmup_epochs: int = 30, 
                                 transition_epochs: int = 30, max_kl_weight: float = 0.01) -> float:
    """
    Curriculum learning schedule for KL weight.
    
    Training phases:
    1. Warm-up (0 to warmup_epochs): KL weight = 0 (pure MSE)
    2. Transition (warmup_epochs to warmup_epochs+transition_epochs): Linear increase
    3. Full (warmup_epochs+transition_epochs+): KL weight = max_kl_weight
    
    Args:
        epoch: Current epoch
        warmup_epochs: Number of pure MSE epochs
        transition_epochs: Number of transition epochs
        max_kl_weight: Maximum KL weight
    
    Returns:
        KL weight for current epoch
    """
    if epoch < warmup_epochs:
        return 0.0
    elif epoch < warmup_epochs + transition_epochs:
        # Linear transition
        progress = (epoch - warmup_epochs) / transition_epochs
        return progress * max_kl_weight
    else:
        return max_kl_weight


def curriculum_geometric_loss_fn(model, params, eta_batch, y_batch, cov_batch, 
                                epoch: int, curriculum_config: Dict[str, Any],
                                regularization: float = 1e-6):
    """
    Curriculum-based geometric loss function.
    
    Args:
        model: Neural network model
        params: Model parameters
        eta_batch: Natural parameters
        y_batch: Target statistics
        cov_batch: Empirical covariances
        epoch: Current training epoch
        curriculum_config: Configuration dict with warmup_epochs, transition_epochs, max_kl_weight
        regularization: Numerical regularization
    
    Returns:
        Loss value and auxiliary information
    """
    # Get current KL weight from curriculum schedule
    kl_weight = curriculum_kl_weight_schedule(
        epoch,
        curriculum_config.get('warmup_epochs', 30),
        curriculum_config.get('transition_epochs', 30),
        curriculum_config.get('max_kl_weight', 0.01)
    )
    
    # Compute geometric loss with current weights
    loss = geometric_loss_fn(
        model, params, eta_batch, y_batch, cov_batch,
        kl_weight=kl_weight,
        mse_weight=curriculum_config.get('mse_weight', 1.0),
        regularization=regularization
    )
    
    # Return loss and auxiliary info
    aux_info = {
        'kl_weight': kl_weight,
        'epoch': epoch,
        'phase': 'warmup' if epoch < curriculum_config.get('warmup_epochs', 30) else
                'transition' if epoch < curriculum_config.get('warmup_epochs', 30) + curriculum_config.get('transition_epochs', 30) else
                'refinement'
    }
    
    return loss, aux_info


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def train_with_geometric_loss(model, params, optimizer, opt_state, train_data, val_data,
                             num_epochs=100, batch_size=64, kl_weight=0.01, 
                             patience=float('inf'), regularization=1e-6):
    """
    Train model with geometric loss function.
    
    Args:
        model: Neural network model
        params: Initial parameters
        optimizer: Optax optimizer
        opt_state: Optimizer state
        train_data: Training data dict with 'eta', 'y', 'cov'
        val_data: Validation data dict
        num_epochs: Number of training epochs
        batch_size: Batch size
        kl_weight: Weight for KL divergence term
        patience: Early stopping patience
        regularization: Numerical regularization
    
    Returns:
        Trained parameters and training history
    """
    best_val_loss = float('inf')
    patience_counter = 0
    best_params = params
    history = {'train_loss': [], 'val_loss': [], 'kl_weight': []}
    
    n_train = train_data['eta'].shape[0]
    
    print(f"Training with geometric loss (KL weight: {kl_weight})")
    
    for epoch in range(num_epochs):
        # Shuffle training data
        rng = random.PRNGKey(epoch)
        perm = random.permutation(rng, n_train)
        
        train_losses = []
        
        # Training batches
        for i in range(0, n_train, batch_size):
            batch_idx = perm[i:i + batch_size]
            
            eta_batch = train_data['eta'][batch_idx]
            y_batch = train_data['y'][batch_idx]
            cov_batch = train_data.get('cov', jnp.eye(y_batch.shape[1])[None, :, :].repeat(len(batch_idx), axis=0))
            
            # Compute loss and gradients
            def loss_fn(params):
                return geometric_loss_fn(model, params, eta_batch, y_batch, cov_batch,
                                       kl_weight=kl_weight, regularization=regularization)
            
            loss, grads = jax.value_and_grad(loss_fn)(params)
            
            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            train_losses.append(loss)
        
        avg_train_loss = float(jnp.mean(jnp.array(train_losses)))
        history['train_loss'].append(avg_train_loss)
        history['kl_weight'].append(kl_weight)
        
        # Validation
        if epoch % 10 == 0:
            val_cov = val_data.get('cov', jnp.eye(val_data['y'].shape[1])[None, :, :].repeat(val_data['y'].shape[0], axis=0))
            val_loss = float(geometric_loss_fn(model, params, val_data['eta'], val_data['y'], val_cov,
                                             kl_weight=kl_weight, regularization=regularization))
            history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch}")
                break
            
            if epoch % 20 == 0:
                print(f"    Epoch {epoch:3d}: Train={avg_train_loss:.2f}, Val={val_loss:.2f}")
    
    return best_params, history


def train_with_curriculum_geometric_loss(model, params, optimizer, opt_state, train_data, val_data,
                                        curriculum_config: Optional[Dict[str, Any]] = None):
    """
    Train model with curriculum learning using geometric loss.
    
    Args:
        model: Neural network model
        params: Initial parameters
        optimizer: Optax optimizer
        opt_state: Optimizer state
        train_data: Training data dict
        val_data: Validation data dict
        curriculum_config: Curriculum configuration
    
    Returns:
        Trained parameters and training history
    """
    if curriculum_config is None:
        curriculum_config = {
            'warmup_epochs': 30,
            'transition_epochs': 30,
            'max_kl_weight': 0.01,
            'mse_weight': 1.0,
            'num_epochs': 100,
            'batch_size': 64,
            'patience': 20,
            'regularization': 1e-6
        }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_params = params
    history = {
        'train_loss': [], 'val_loss': [], 'kl_weight': [], 'mse_loss': [], 'kl_loss': [],
        'phase': []
    }
    
    n_train = train_data['eta'].shape[0]
    batch_size = curriculum_config['batch_size']
    num_epochs = curriculum_config['num_epochs']
    
    print(f"Training with curriculum geometric loss")
    print(f"Phases: Warmup({curriculum_config['warmup_epochs']}) → Transition({curriculum_config['transition_epochs']}) → Refinement")
    
    for epoch in range(num_epochs):
        # Get current curriculum weights
        _, aux_info = curriculum_geometric_loss_fn(
            model, params, train_data['eta'][:1], train_data['y'][:1], 
            jnp.eye(train_data['y'].shape[1])[None, :, :], epoch, curriculum_config
        )
        
        current_kl_weight = aux_info['kl_weight']
        current_phase = aux_info['phase']
        
        # Shuffle training data
        rng = random.PRNGKey(epoch)
        perm = random.permutation(rng, n_train)
        
        train_losses = []
        mse_losses = []
        kl_losses = []
        
        # Training batches
        for i in range(0, n_train, batch_size):
            batch_idx = perm[i:i + batch_size]
            
            eta_batch = train_data['eta'][batch_idx]
            y_batch = train_data['y'][batch_idx]
            cov_batch = train_data.get('cov', jnp.eye(y_batch.shape[1])[None, :, :].repeat(len(batch_idx), axis=0))
            
            # Compute loss and gradients
            def loss_fn(params):
                loss, aux = curriculum_geometric_loss_fn(
                    model, params, eta_batch, y_batch, cov_batch, epoch, curriculum_config
                )
                return loss, aux
            
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            
            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            train_losses.append(loss)
            
            # Track individual loss components
            predictions = model.apply(params, eta_batch, training=False)
            mse_component = float(jnp.mean(jnp.square(predictions - y_batch)))
            mse_losses.append(mse_component)
            
            if current_kl_weight > 0:
                kl_component = (loss - curriculum_config['mse_weight'] * mse_component) / current_kl_weight
                kl_losses.append(float(kl_component))
            else:
                kl_losses.append(0.0)
        
        avg_train_loss = float(jnp.mean(jnp.array(train_losses)))
        avg_mse_loss = float(jnp.mean(jnp.array(mse_losses)))
        avg_kl_loss = float(jnp.mean(jnp.array(kl_losses)))
        
        history['train_loss'].append(avg_train_loss)
        history['mse_loss'].append(avg_mse_loss)
        history['kl_loss'].append(avg_kl_loss)
        history['kl_weight'].append(current_kl_weight)
        history['phase'].append(current_phase)
        
        # Validation
        if epoch % 10 == 0:
            val_cov = val_data.get('cov', jnp.eye(val_data['y'].shape[1])[None, :, :].repeat(val_data['y'].shape[0], axis=0))
            val_loss, _ = curriculum_geometric_loss_fn(
                model, params, val_data['eta'], val_data['y'], val_cov, epoch, curriculum_config
            )
            val_loss = float(val_loss)
            history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= curriculum_config['patience']:
                print(f"    Early stopping at epoch {epoch}")
                break
            
            if epoch % 20 == 0:
                print(f"    Epoch {epoch:3d}: Train={avg_train_loss:.2f}, Val={val_loss:.2f}, "
                      f"Phase={current_phase}, KL_w={current_kl_weight:.4f}")
    
    return best_params, history


# =============================================================================
# EVALUATION UTILITIES
# =============================================================================

def evaluate_with_geometric_metrics(model, params, test_data, name="Model"):
    """
    Evaluate model with both standard and geometric metrics.
    
    Args:
        model: Neural network model
        params: Model parameters
        test_data: Test data dict
        name: Model name for printing
    
    Returns:
        Dictionary of evaluation metrics
    """
    predictions = model.apply(params, test_data['eta'], training=False)
    
    # Standard metrics
    mse = float(jnp.mean(jnp.square(predictions - test_data['y'])))
    mae = float(jnp.mean(jnp.abs(predictions - test_data['y'])))
    
    metrics = {'mse': mse, 'mae': mae}
    
    # Geometric metrics if covariance data available
    if 'cov' in test_data:
        try:
            jacobian = compute_network_jacobian(model, params, test_data['eta'])
            cov_net = estimate_covariance_from_jacobian(jacobian)
            
            # Extract means
            mu_emp = test_data['y'][:, :3]
            mu_net = predictions[:, :3]
            cov_emp = test_data['cov']
            
            # Compute KL divergence
            kl_div = jnp.mean(kl_divergence_multivariate_normal_stable(
                mu_emp, cov_emp, mu_net, cov_net
            ))
            
            metrics['kl_divergence'] = float(kl_div)
            metrics['geometric_loss'] = float(mse + 0.01 * kl_div)
            
        except Exception as e:
            print(f"Warning: Could not compute geometric metrics: {e}")
    
    print(f"{name} Evaluation:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return metrics


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_curriculum_training(history: Dict[str, list], save_path: Optional[str] = None):
    """Plot curriculum training progress showing phase transitions."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(len(history['train_loss']))
    
    # 1. Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], label='Total Loss', alpha=0.8)
    if 'mse_loss' in history:
        axes[0, 0].plot(epochs, history['mse_loss'], label='MSE Component', alpha=0.8)
    if 'val_loss' in history and history['val_loss']:
        val_epochs = range(0, len(epochs), len(epochs) // len(history['val_loss']))[:len(history['val_loss'])]
        axes[0, 0].plot(val_epochs, history['val_loss'], label='Validation', marker='o', alpha=0.8)
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 2. KL weight schedule
    if 'kl_weight' in history:
        axes[0, 1].plot(epochs, history['kl_weight'], color='red', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('KL Weight')
        axes[0, 1].set_title('Curriculum Schedule')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Loss components
    if 'mse_loss' in history and 'kl_loss' in history:
        axes[1, 0].plot(epochs, history['mse_loss'], label='MSE Loss', alpha=0.8)
        axes[1, 0].plot(epochs, history['kl_loss'], label='KL Loss', alpha=0.8)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss Component')
        axes[1, 0].set_title('Loss Components')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    
    # 4. Training phases
    if 'phase' in history:
        phase_colors = {'warmup': 'blue', 'transition': 'orange', 'refinement': 'green'}
        phases = history['phase']
        
        current_phase = phases[0]
        phase_start = 0
        
        for i, phase in enumerate(phases + ['end']):  # Add 'end' to close last phase
            if phase != current_phase or i == len(phases):
                # End of current phase
                color = phase_colors.get(current_phase, 'gray')
                axes[1, 1].axvspan(phase_start, i-1, alpha=0.3, color=color, label=current_phase)
                
                if i < len(phases):
                    current_phase = phase
                    phase_start = i
        
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Training Phase')
        axes[1, 1].set_title('Curriculum Phases')
        axes[1, 1].legend()
        axes[1, 1].set_xlim(0, len(epochs))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# =============================================================================
# LOSS FUNCTION REGISTRY
# =============================================================================

LOSS_FUNCTIONS = {
    'mse': mse_loss,
    'mae': mae_loss,
    'huber': huber_loss,
    'geometric': geometric_loss_fn,
    'curriculum_geometric': curriculum_geometric_loss_fn,
}


def get_loss_function(loss_name: str, **kwargs) -> Callable:
    """Get a loss function by name with optional configuration."""
    if loss_name not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss function: {loss_name}. Available: {list(LOSS_FUNCTIONS.keys())}")
    
    base_fn = LOSS_FUNCTIONS[loss_name]
    
    if kwargs:
        # Create a partial function with the given kwargs
        def configured_loss(*args, **call_kwargs):
            merged_kwargs = {**kwargs, **call_kwargs}
            return base_fn(*args, **merged_kwargs)
        return configured_loss
    
    return base_fn


def list_available_loss_functions() -> list:
    """List all available loss functions."""
    return list(LOSS_FUNCTIONS.keys())
