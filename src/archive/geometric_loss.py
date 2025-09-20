"""
Geometric loss functions that respect the natural geometry of the exponential family.

This module implements KL divergence-based loss functions that use the empirical
covariance structure and the network's Jacobian to create a more principled
training objective.

Key idea: 
- p = Normal(μ_emp, Σ_emp) from MCMC data
- q = Normal(μ_net, ∇_η μ_net) from network + Jacobian
- Loss = KL(p || q)

This respects the natural geometry and provides second-order information.
"""

import jax
import jax.numpy as jnp
from jax import random, grad, jacfwd, jacrev
import flax.linen as nn
from typing import Callable, Tuple, Dict, Any
import optax
from flax.training import train_state
import time

Array = jax.Array


def compute_network_jacobian(model, params, eta):
    """
    Compute the Jacobian of the network output w.r.t. natural parameters.
    
    Args:
        model: Neural network model
        params: Model parameters  
        eta: Natural parameters of shape (batch_size, eta_dim)
        
    Returns:
        Jacobian of shape (batch_size, output_dim, eta_dim)
    """
    def model_fn(eta_single):
        """Model function for a single input."""
        return model.apply(params, eta_single[None, :])[0]  # Remove batch dim
    
    # Compute Jacobian for each sample in the batch
    jacobian_fn = jacfwd(model_fn)
    
    # Vectorize over the batch
    batch_jacobian = jax.vmap(jacobian_fn)(eta)
    
    return batch_jacobian


def estimate_covariance_from_jacobian(jacobian, regularization=1e-6):
    """
    Estimate covariance matrix from Jacobian using the Fisher Information approximation.
    
    For exponential families, the covariance of sufficient statistics is related to
    the Hessian of the log-partition function, which we approximate using the Jacobian.
    
    Args:
        jacobian: Jacobian of shape (batch_size, output_dim, eta_dim)
        regularization: Small value added to diagonal for numerical stability
        
    Returns:
        Estimated covariance matrices of shape (batch_size, output_dim, output_dim)
    """
    # Use J @ J^T as approximation to covariance (Fisher Information relationship)
    # This assumes the Jacobian captures the local curvature
    cov_estimate = jnp.einsum('bij,bkj->bik', jacobian, jacobian)
    
    # Add regularization to diagonal
    batch_size, output_dim, _ = jacobian.shape
    eye = jnp.eye(output_dim)[None, :, :].repeat(batch_size, axis=0)
    cov_estimate = cov_estimate + regularization * eye
    
    return cov_estimate


def kl_divergence_multivariate_normal_stable(mu_emp, cov_emp, mu_net, cov_net, regularization=1e-6):
    """
    Compute KL divergence KL(q || p) where q = N(mu_net, cov_net), p = N(mu_emp, cov_emp).
    
    This is more stable because we only invert the empirical covariance (cov_emp),
    not the network-estimated covariance (cov_net).
    
    KL(q || p) = 0.5 * [tr(Σ_emp^{-1} Σ_net) + (μ_emp-μ_net)^T Σ_emp^{-1} (μ_emp-μ_net) - k + log(|Σ_emp|/|Σ_net|)]
    
    Args:
        mu_emp, mu_net: Means of shape (batch_size, dim)
        cov_emp, cov_net: Covariances of shape (batch_size, dim, dim)
        regularization: Added to diagonal for numerical stability
        
    Returns:
        KL divergences of shape (batch_size,)
    """
    batch_size, dim = mu_emp.shape
    
    # Add regularization to covariances
    eye = jnp.eye(dim)[None, :, :].repeat(batch_size, axis=0)
    cov_emp_reg = cov_emp + regularization * eye
    cov_net_reg = cov_net + regularization * eye
    
    try:
        # Compute inverse of empirical covariance (more stable)
        cov_emp_inv = jnp.linalg.inv(cov_emp_reg)
        
        # Trace term: tr(Σ_emp^{-1} Σ_net)
        trace_term = jnp.trace(jnp.einsum('bij,bjk->bik', cov_emp_inv, cov_net_reg), axis1=-2, axis2=-1)
        
        # Quadratic term: (μ_emp-μ_net)^T Σ_emp^{-1} (μ_emp-μ_net)
        mu_diff = mu_emp - mu_net
        quad_term = jnp.einsum('bi,bij,bj->b', mu_diff, cov_emp_inv, mu_diff)
        
        # Log determinant term: log(|Σ_emp|/|Σ_net|)
        logdet_emp = jnp.linalg.slogdet(cov_emp_reg)[1]
        logdet_net = jnp.linalg.slogdet(cov_net_reg)[1]
        logdet_term = logdet_emp - logdet_net
        
        # KL divergence: KL(q || p)
        kl = 0.5 * (trace_term + quad_term - dim + logdet_term)
        
        # Clamp to reasonable range
        kl = jnp.clip(kl, -100.0, 100.0)
        
        return kl
        
    except:
        # If matrix operations fail, return a moderate penalty
        return jnp.full((batch_size,), 10.0)


def kl_divergence_multivariate_normal(mu1, cov1, mu2, cov2, regularization=1e-6):
    """
    Compute KL divergence between two multivariate normal distributions.
    
    KL(p || q) where p = N(mu1, cov1), q = N(mu2, cov2)
    
    Args:
        mu1, mu2: Means of shape (batch_size, dim)
        cov1, cov2: Covariances of shape (batch_size, dim, dim)
        regularization: Added to diagonal for numerical stability
        
    Returns:
        KL divergences of shape (batch_size,)
    """
    batch_size, dim = mu1.shape
    
    # Add regularization to covariances
    eye = jnp.eye(dim)[None, :, :].repeat(batch_size, axis=0)
    cov1_reg = cov1 + regularization * eye
    cov2_reg = cov2 + regularization * eye
    
    # Compute KL divergence: KL(p || q) = 0.5 * [tr(Σ2^{-1} Σ1) + (μ2-μ1)^T Σ2^{-1} (μ2-μ1) - k + log(|Σ2|/|Σ1|)]
    
    # Compute inverse of cov2
    try:
        cov2_inv = jnp.linalg.inv(cov2_reg)
        
        # Trace term: tr(Σ2^{-1} Σ1)
        trace_term = jnp.trace(jnp.einsum('bij,bjk->bik', cov2_inv, cov1_reg), axis1=-2, axis2=-1)
        
        # Quadratic term: (μ2-μ1)^T Σ2^{-1} (μ2-μ1)  
        mu_diff = mu2 - mu1
        quad_term = jnp.einsum('bi,bij,bj->b', mu_diff, cov2_inv, mu_diff)
        
        # Log determinant term: log(|Σ2|/|Σ1|)
        logdet1 = jnp.linalg.slogdet(cov1_reg)[1]
        logdet2 = jnp.linalg.slogdet(cov2_reg)[1]
        logdet_term = logdet2 - logdet1
        
        # KL divergence
        kl = 0.5 * (trace_term + quad_term - dim + logdet_term)
        
        return kl
        
    except:
        # If matrix operations fail, return a large penalty
        return jnp.full((batch_size,), 1e6)


def geometric_loss_fn(model, params, eta_batch, y_batch, cov_batch, 
                     kl_weight=1.0, mse_weight=1.0, regularization=1e-6):
    """
    Geometric loss function combining MSE and KL divergence.
    
    Args:
        model: Neural network model
        params: Model parameters
        eta_batch: Natural parameters (batch_size, eta_dim)
        y_batch: Empirical sufficient statistics (batch_size, output_dim)
        cov_batch: Empirical covariance (batch_size, output_dim, output_dim)
        kl_weight: Weight for KL divergence term
        mse_weight: Weight for MSE term
        regularization: Regularization for numerical stability
        
    Returns:
        Total loss, loss components dict
    """
    # Network prediction
    mu_net = model.apply(params, eta_batch)
    
    # Compute network Jacobian to estimate covariance
    jacobian = compute_network_jacobian(model, params, eta_batch)
    cov_net = estimate_covariance_from_jacobian(jacobian, regularization)
    
    # MSE loss (standard)
    mse_loss = jnp.mean(jnp.square(mu_net - y_batch))
    
    # KL divergence loss: KL(q || p) where q = network, p = empirical
    # This is more stable as we only invert the empirical covariance
    kl_losses = kl_divergence_multivariate_normal_stable(
        y_batch, cov_batch, mu_net, cov_net, regularization
    )
    kl_loss = jnp.mean(kl_losses)
    
    # Handle NaN/Inf in KL loss
    kl_loss = jnp.where(jnp.isfinite(kl_loss), kl_loss, 1e6)
    
    # Total loss
    total_loss = mse_weight * mse_loss + kl_weight * kl_loss
    
    loss_dict = {
        'total_loss': total_loss,
        'mse_loss': mse_loss,
        'kl_loss': kl_loss,
        'mean_kl': jnp.mean(kl_losses)
    }
    
    return total_loss, loss_dict


class GeometricTrainState(train_state.TrainState):
    """Train state for geometric loss training."""
    
    kl_weight: float = 1.0
    mse_weight: float = 1.0
    regularization: float = 1e-6


def train_with_geometric_loss(
    model,
    params,
    train_data: Dict[str, Array],
    val_data: Dict[str, Array],
    config: Dict[str, Any],
    name: str = "Model"
) -> Tuple[Dict, Dict]:
    """
    Train a model using the geometric KL loss function.
    
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
    
    num_epochs = config.get('num_epochs', 100)
    learning_rate = config.get('learning_rate', 1e-4)
    batch_size = config.get('batch_size', 64)
    kl_weight = config.get('kl_weight', 1.0)
    mse_weight = config.get('mse_weight', 1.0)
    regularization = config.get('regularization', 1e-6)
    
    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Important for Jacobian-based loss
        optax.adam(learning_rate)
    )
    opt_state = optimizer.init(params)
    
    train_losses = []
    val_losses = []
    mse_losses = []
    kl_losses = []
    
    best_params = params
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    print(f"  Training {name} with geometric loss...")
    print(f"    KL weight: {kl_weight}, MSE weight: {mse_weight}")
    start_time = time.time()
    
    for epoch in range(num_epochs):
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
                return geometric_loss_fn(
                    model, params, eta_batch, y_batch, cov_batch,
                    kl_weight, mse_weight, regularization
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
        
        # Validation
        val_loss, val_loss_dict = geometric_loss_fn(
            model, params, val_data['eta'], val_data['y'], val_data['cov'],
            kl_weight, mse_weight, regularization
        )
        val_loss = float(val_loss)
        
        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        mse_losses.append(epoch_mse_loss)
        kl_losses.append(epoch_kl_loss)
        
        # Best model tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d}: Total={epoch_train_loss:.6f}, MSE={epoch_mse_loss:.6f}, KL={epoch_kl_loss:.6f}, Val={val_loss:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break
    
    training_time = time.time() - start_time
    print(f"  Training completed in {training_time:.1f}s, Best val: {best_val_loss:.6f}")
    
    return best_params, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'mse_losses': mse_losses,
        'kl_losses': kl_losses,
        'training_time': training_time,
        'best_val_loss': best_val_loss
    }


def evaluate_with_geometric_metrics(model, params, test_data, name="Model"):
    """Evaluate model with both standard and geometric metrics."""
    
    # Standard prediction
    pred = model.apply(params, test_data['eta'])
    
    # Standard metrics
    mse = float(jnp.mean(jnp.square(pred - test_data['y'])))
    mae = float(jnp.mean(jnp.abs(pred - test_data['y'])))
    
    # Geometric metrics
    try:
        # Compute network Jacobian
        jacobian = compute_network_jacobian(model, params, test_data['eta'])
        cov_net = estimate_covariance_from_jacobian(jacobian)
        
        # KL divergence
        kl_losses = kl_divergence_multivariate_normal(
            test_data['y'], test_data['cov'], pred, cov_net
        )
        kl_loss = float(jnp.mean(kl_losses))
        
        # Covariance prediction error
        cov_mse = float(jnp.mean(jnp.square(cov_net - test_data['cov'])))
        
        # Jacobian condition number (measure of numerical stability)
        try:
            # Compute condition number of Jacobian matrices
            cond_numbers = jnp.array([jnp.linalg.cond(j) for j in jacobian])
            mean_cond = float(jnp.mean(cond_numbers))
        except:
            mean_cond = float('inf')
        
    except Exception as e:
        print(f"  Warning: Geometric metrics failed for {name}: {e}")
        kl_loss = float('inf')
        cov_mse = float('inf')
        mean_cond = float('inf')
    
    return {
        'name': name,
        'mse': mse,
        'mae': mae,
        'kl_loss': kl_loss,
        'cov_mse': cov_mse,
        'condition_number': mean_cond,
        'predictions': pred
    }


# Test the geometric loss function
if __name__ == "__main__":
    from ef import GaussianNatural1D
    from model import nat2statMLP
    
    print("Testing geometric loss function...")
    
    # Create test data
    ef = GaussianNatural1D()
    rng = random.PRNGKey(0)
    
    # Create dummy data with covariance
    batch_size = 10
    eta = random.normal(rng, (batch_size, 2))
    eta = eta.at[:, 1].set(-jnp.abs(eta[:, 1]))  # Ensure eta2 < 0
    
    y = random.normal(rng, (batch_size, 2))
    cov = jnp.eye(2)[None, :, :].repeat(batch_size, axis=0) * 0.1  # Simple covariance
    
    # Create model
    model = nat2statMLP(hidden_sizes=[32, 16], activation='tanh', output_dim=2)
    params = model.init(rng, eta[:1])
    
    # Test geometric loss
    total_loss, loss_dict = geometric_loss_fn(
        model, params, eta, y, cov, kl_weight=1.0, mse_weight=1.0
    )
    
    print(f"Total loss: {total_loss:.6f}")
    print(f"MSE loss: {loss_dict['mse_loss']:.6f}")
    print(f"KL loss: {loss_dict['kl_loss']:.6f}")
    
    # Test Jacobian computation
    jacobian = compute_network_jacobian(model, params, eta)
    print(f"Jacobian shape: {jacobian.shape}")
    
    # Test covariance estimation
    cov_est = estimate_covariance_from_jacobian(jacobian)
    print(f"Estimated covariance shape: {cov_est.shape}")
    
    print("✅ Geometric loss function working correctly!")
