#!/usr/bin/env python3
"""
Improved log normalizer comparison with proper exponential family theory.

This implementation uses:
1. Proper 3D Gaussian exponential family relationships
2. More sophisticated training strategies
3. Better network architectures
4. Should achieve near-perfect reproduction of sufficient statistics
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, hessian, jacfwd, jacrev
import matplotlib.pyplot as plt
import flax.linen as nn
import optax


class ImprovedLogNormalizer(nn.Module):
    """Improved log normalizer with better architecture."""
    hidden_sizes: list
    use_skip_connections: bool = True
    use_layer_norm: bool = True
    
    @nn.compact
    def __call__(self, x, training=True):
        # Input layer with proper initialization
        x = nn.Dense(self.hidden_sizes[0], 
                    kernel_init=nn.initializers.glorot_normal(),
                    name='input')(x)
        
        # Hidden layers with skip connections (doubled layers, capped at 128)
        for i in range(len(self.hidden_sizes) * 2):
            layer_size = min(128, self.hidden_sizes[i // 2])
            residual = x
            
            # Main transformation
            x = nn.Dense(layer_size, 
                        kernel_init=nn.initializers.glorot_normal(),
                        name=f'hidden_{i}')(x)
            x = nn.swish(x)
            
            # Skip connection if dimensions match
            if self.use_skip_connections and residual.shape[-1] == layer_size:
                x = x + residual
            
            # Add layer normalization for stability (optional)
            if self.use_layer_norm:
                x = nn.LayerNorm(name=f'layer_norm_{i}')(x)
        
        # Output layer
        x = nn.Dense(1, 
                    kernel_init=nn.initializers.glorot_normal(),
                    name='output')(x)
        
        return jnp.squeeze(x, axis=-1)


class ImprovedQuadraticResNet(nn.Module):
    """Improved quadratic ResNet with better design."""
    hidden_sizes: list
    use_layer_norm: bool = True
    
    @nn.compact
    def __call__(self, x, training=True):
        # Input projection
        x = nn.Dense(self.hidden_sizes[0], 
                    kernel_init=nn.initializers.glorot_normal(),
                    name='input_proj')(x)
        
        # Quadratic residual blocks (doubled layers, capped at 128)
        for i in range(len(self.hidden_sizes) * 2):
            layer_size = min(128, self.hidden_sizes[i // 2])
            residual = x
            
            # Ensure residual projection
            if residual.shape[-1] != layer_size:
                residual = nn.Dense(layer_size, 
                                  kernel_init=nn.initializers.glorot_normal(),
                                  name=f'residual_proj_{i}')(residual)
            
            # Linear path: Ax
            linear = nn.Dense(layer_size, 
                            kernel_init=nn.initializers.glorot_normal(),
                            name=f'linear_{i}')(x)
            linear = nn.swish(linear)
            
            # Quadratic path: Bx (with smaller initialization to prevent explosion)
            quadratic = nn.Dense(layer_size, 
                               kernel_init=nn.initializers.normal(stddev=0.01),
                               name=f'quadratic_{i}')(x)
            quadratic = nn.swish(quadratic)
            
            # Combine: y = Ax + (Bx)x (element-wise multiplication)
            x = residual + linear + linear * quadratic
            
            # Add layer normalization to prevent value explosion (optional)
            if self.use_layer_norm:
                x = nn.LayerNorm(name=f'layer_norm_{i}')(x)
        
        # Final layers
        x = nn.Dense(64, 
                    kernel_init=nn.initializers.glorot_normal(),
                    name='pre_final')(x)
        x = nn.swish(x)
        
        x = nn.Dense(1, 
                    kernel_init=nn.initializers.glorot_normal(),
                    name='final')(x)
        
        return jnp.squeeze(x, axis=-1)


def generate_proper_3d_gaussian_data(n_samples=8000, seed=42):
    """Generate proper 3D Gaussian data using exponential family theory."""
    rng = random.PRNGKey(seed)
    
    # Generate natural parameters for 3D Gaussian
    # For MultivariateNormal: Σ⁻¹ = -2η₂, μ = Ση₁
    # So: η₁ = Σ⁻¹μ, η₂ = -0.5Σ⁻¹
    
    eta_vectors = []
    expected_stats = []
    
    for i in range(n_samples):
        # Generate random mean and covariance
        mean = random.normal(random.PRNGKey(seed + i), (3,)) * 1.0
        A = random.normal(random.PRNGKey(seed + i + 1000), (3, 3))
        covariance_matrix = A.T @ A + jnp.eye(3) * 0.01  # Ensure positive definite
        
        # Convert to natural parameters
        sigma_inv = jnp.linalg.inv(covariance_matrix)
        eta1 = sigma_inv @ mean  # η₁ = Σ⁻¹μ
        eta2_matrix = -0.5 * sigma_inv  # η₂ = -0.5Σ⁻¹ (negative definite)
        
        # Natural parameters: [η₁, η₂_flattened] = 3 + 9 = 12 parameters
        eta_vector = jnp.concatenate([eta1, eta2_matrix.flatten()])
        eta_vectors.append(eta_vector)
        
        # Expected sufficient statistics: [μ, μμᵀ + Σ] = 3 + 9 = 12 statistics
        expected_stat = jnp.concatenate([
            mean,  # μ (3 values)
            (jnp.outer(mean, mean) + covariance_matrix).flatten()  # μμᵀ + Σ (9 values)
        ])
        expected_stats.append(expected_stat)
    
    return jnp.array(eta_vectors), jnp.array(expected_stats)


def compute_network_jacobian(model, params, eta_batch):
    """Compute Jacobian of network output w.r.t. natural parameters."""
    def single_log_normalizer(eta_single):
        return model.apply(params, eta_single.reshape(1, -1), training=False).squeeze()
    
    # Compute Jacobian for each sample
    jacobians = jax.vmap(jacfwd(single_log_normalizer))(eta_batch)
    return jacobians


def compute_network_hessian(model, params, eta_batch):
    """Compute Hessian of network output w.r.t. natural parameters."""
    def single_log_normalizer(eta_single):
        return model.apply(params, eta_single.reshape(1, -1), training=False).squeeze()
    
    # Compute Hessian for each sample
    hessians = jax.vmap(hessian(single_log_normalizer))(eta_batch)
    return hessians


def kl_divergence_multivariate_normal(mu_p, cov_p, mu_q, cov_q, regularization=1e-6):
    """Compute KL divergence KL(p || q) between two multivariate normal distributions."""
    batch_size, dim = mu_p.shape
    
    # Add regularization to covariances
    eye = jnp.eye(dim)
    cov_p_reg = cov_p + regularization * eye[None, :, :]
    cov_q_reg = cov_q + regularization * eye[None, :, :]
    
    # Compute KL divergence: 0.5 * [tr(Σ_q^{-1} Σ_p) + (μ_q - μ_p)^T Σ_q^{-1} (μ_q - μ_p) - k + log(det(Σ_q)/det(Σ_p))]
    
    # Compute Σ_q^{-1}
    cov_q_inv = jnp.linalg.inv(cov_q_reg)
    
    # Trace term: tr(Σ_q^{-1} Σ_p)
    trace_term = jnp.einsum('bij,bji->b', cov_q_inv, cov_p_reg)
    
    # Quadratic form: (μ_q - μ_p)^T Σ_q^{-1} (μ_q - μ_p)
    mu_diff = mu_q - mu_p
    quad_form = jnp.einsum('bi,bij,bj->b', mu_diff, cov_q_inv, mu_diff)
    
    # Log determinant term: log(det(Σ_q)/det(Σ_p))
    log_det_q = jnp.linalg.slogdet(cov_q_reg)[1]
    log_det_p = jnp.linalg.slogdet(cov_p_reg)[1]
    log_det_term = log_det_q - log_det_p
    
    # Combine terms
    kl_div = 0.5 * (trace_term + quad_form - dim + log_det_term)
    
    return kl_div


def geometric_kl_loss(model, params, eta_batch, ground_truth_batch, ground_truth_tt, kl_weight=0.1, regularization=1e-6):
    """Geometric KL loss using Jacobian and Hessian of log normalizer network."""
    
    # Get network log normalizer values
    log_norm = model.apply(params, eta_batch, training=True)
    
    # Compute network Jacobian (gives network estimate of mean)
    jacobian = compute_network_jacobian(model, params, eta_batch)
    network_mean = jacobian  # ∇A(η) = E[T(X)|η]
    
    # Compute network Hessian (gives network estimate of covariance)
    hessian_matrix = compute_network_hessian(model, params, eta_batch)
    network_cov = hessian_matrix  # ∇²A(η) = Cov[T(X)|η]
    
    # Extract ground truth mean and compute ground truth covariance
    ground_truth_mean = ground_truth_batch  # This is E[T(X)] from ground truth [batch_size, 12]
    
    # Extract ground truth covariance from E[TTᵀ] - μμᵀ
    # ground_truth_tt has shape [batch_size, 12, 12]
    ground_truth_cov = ground_truth_tt - jnp.einsum('bi,bj->bij', ground_truth_mean, ground_truth_mean)  # [batch_size, 12, 12]
    
    # Compute MSE loss
    mse_loss = jnp.mean(jnp.square(network_mean - ground_truth_mean))
    
    # Compute KL divergence using full 12D statistics
    kl_loss = jnp.mean(kl_divergence_multivariate_normal(
        ground_truth_mean, ground_truth_cov,
        network_mean, network_cov,
        regularization
    ))
    
    return mse_loss + kl_weight * kl_loss


def compute_ground_truth_tt(ground_truth_mean):
    """Compute ground truth E[TTᵀ] for the full 12-dimensional sufficient statistic."""
    # For MultivariateNormal: T = [x₁, x₂, x₃, x₁x₁, x₁x₂, x₁x₃, x₂x₁, x₂x₂, x₂x₃, x₃x₁, x₃x₂, x₃x₃]
    # We need to compute E[TTᵀ] which is a 12×12 matrix
    
    batch_size = ground_truth_mean.shape[0]
    
    # Extract μ (first 3 components) and E[xxᵀ] (last 9 components)
    mu = ground_truth_mean[:, :3]  # [batch_size, 3]
    E_xxT = ground_truth_mean[:, 3:].reshape(batch_size, 3, 3)  # [batch_size, 3, 3]
    
    # For the full sufficient statistic T = [x, xxᵀ_flattened]
    # E[TTᵀ] has the structure:
    #   [E[xᵀx]      E[xᵀ(xxᵀ)]]
    #   [E[(xxᵀ)x]   E[(xxᵀ)(xxᵀ)]]
    
    tt_matrices = []
    for i in range(batch_size):
        mu_i = mu[i]  # [3]
        E_xxT_i = E_xxT[i]  # [3, 3]
        
        # E[xᵀx] = μμᵀ + Σ (but we need to extract Σ from E[xxᵀ] = μμᵀ + Σ)
        # So Σ = E[xxᵀ] - μμᵀ
        Sigma = E_xxT_i - jnp.outer(mu_i, mu_i)  # [3, 3]
        E_xTx = jnp.outer(mu_i, mu_i) + Sigma  # [3, 3]
        
        # E[xᵀ(xxᵀ)] = E[xᵀ] * E[xxᵀ] = μᵀ * E[xxᵀ] (simplified)
        # This is a 3×9 matrix
        E_xT_xxT = mu_i[:, None] * E_xxT_i.flatten()[None, :]  # [3, 9]
        
        # E[(xxᵀ)(xxᵀ)] = E[xxᵀ] ⊗ E[xxᵀ] (Kronecker product approximation)
        # This is a 9×9 matrix
        E_xxT_xxT = jnp.kron(E_xxT_i, E_xxT_i)  # [9, 9]
        
        # Construct full 12×12 matrix
        tt_matrix = jnp.block([
            [E_xTx, E_xT_xxT],
            [E_xT_xxT.T, E_xxT_xxT]
        ])  # [12, 12]
        
        tt_matrices.append(tt_matrix)
    
    return jnp.array(tt_matrices)  # [batch_size, 12, 12]


def curriculum_kl_weight_schedule(epoch, warmup_epochs=30, transition_epochs=30, max_kl_weight=0.1):
    """Schedule KL loss weight for curriculum learning."""
    if epoch < warmup_epochs:
        return 0.0  # Pure MSE loss
    elif epoch < warmup_epochs + transition_epochs:
        # Linear transition from 0 to max_kl_weight
        progress = (epoch - warmup_epochs) / transition_epochs
        return max_kl_weight * progress
    else:
        return max_kl_weight  # Full KL loss


def advanced_training(model, eta_data, ground_truth, ground_truth_tt, epochs=200, use_geometric_loss=True):
    """Advanced training with curriculum learning and geometric KL loss."""
    
    # Initialize
    rng = random.PRNGKey(123)
    params = model.init(rng, eta_data[:1])
    
    # Advanced optimizer with learning rate scheduling and more aggressive clipping
    lr_schedule = optax.exponential_decay(
        init_value=1e-3,  # Reduced initial learning rate
        transition_steps=epochs // 4,
        decay_rate=0.9
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),  # More aggressive clipping
        optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4)
    )
    
    opt_state = optimizer.init(params)
    
    # Loss function with curriculum learning
    def loss_fn(params, eta_batch, target_batch, tt_batch, kl_weight):
        if use_geometric_loss and kl_weight > 0:
            # Use geometric KL loss with curriculum
            return geometric_kl_loss(model, params, eta_batch, target_batch, tt_batch, kl_weight=kl_weight)
        else:
            # Pure MSE loss (curriculum phase 1)
            log_norm = model.apply(params, eta_batch, training=True)
            def single_log_normalizer(eta_single):
                return model.apply(params, eta_single[None, :], training=False)[0]
            grad_fn = jax.grad(single_log_normalizer)
            network_mean = jax.vmap(grad_fn)(eta_batch)
            mse_loss = jnp.mean(jnp.square(network_mean - target_batch))
            l2_loss = jnp.mean(jnp.square(log_norm))
            return mse_loss + 1e-6 * l2_loss
    
    # Training loop with curriculum learning
    losses = []
    mse_losses = []
    kl_losses = []
    kl_weights = []
    
    for epoch in range(epochs):
        # Get curriculum KL weight
        kl_weight = curriculum_kl_weight_schedule(epoch, warmup_epochs=30, transition_epochs=30, max_kl_weight=0.1)
        
        # Compute loss and gradients
        loss, grads = jax.value_and_grad(loss_fn)(params, eta_data, ground_truth, ground_truth_tt, kl_weight)
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        # Track losses
        losses.append(float(loss))
        kl_weights.append(float(kl_weight))
        
        # Compute individual loss components for monitoring
        if use_geometric_loss:
            mse_component = geometric_kl_loss(model, params, eta_data, ground_truth, ground_truth_tt, kl_weight=0.0)
            kl_component = geometric_kl_loss(model, params, eta_data, ground_truth, ground_truth_tt, kl_weight=1.0)
            mse_losses.append(float(mse_component))
            kl_losses.append(float(kl_component))
        else:
            mse_losses.append(float(loss))
            kl_losses.append(0.0)
        
        if epoch % 100 == 0 or epoch < 10:
            print(f"Epoch {epoch}: Total Loss = {loss:.8f}, MSE = {mse_losses[-1]:.8f}, KL Weight = {kl_weight:.4f}")
    
    return params, losses, mse_losses, kl_losses, kl_weights


def plot_improved_comparison(eta_data, ground_truth, basic_pred, resnet_pred, basic_losses, resnet_losses, 
                           basic_mse_losses=None, basic_kl_losses=None, basic_kl_weights=None,
                           resnet_mse_losses=None, resnet_kl_losses=None, resnet_kl_weights=None):
    """Create improved comparison plots for MSE-only training."""
    
    # Statistic names for MultivariateNormal (12 statistics total)
    stat_names = [
        'E[x₁]', 'E[x₂]', 'E[x₃]',  # 3 mean statistics
        'E[x₁x₁]', 'E[x₁x₂]', 'E[x₁x₃]',  # First row of xx^T
        'E[x₂x₁]', 'E[x₂x₂]', 'E[x₂x₃]',  # Second row of xx^T  
        'E[x₃x₁]', 'E[x₃x₂]', 'E[x₃x₃]'   # Third row of xx^T
    ]
    
    # Ensure same size
    min_size = min(basic_pred.shape[1], resnet_pred.shape[1], ground_truth.shape[1], len(stat_names))
    basic_pred = basic_pred[:, :min_size]
    resnet_pred = resnet_pred[:, :min_size]
    ground_truth = ground_truth[:, :min_size]
    stat_names = stat_names[:min_size]
    
    # Create comprehensive plots for MSE-only training
    fig = plt.figure(figsize=(24, 18))
    
    # 1. Training loss curves
    ax1 = plt.subplot(4, 4, 1)
    ax1.plot(basic_losses, label='Basic LogNormalizer', linewidth=2)
    ax1.plot(resnet_losses, label='Quadratic ResNet', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 2. MSE loss comparison (if available)
    ax2 = plt.subplot(4, 4, 2)
    if basic_mse_losses is not None and resnet_mse_losses is not None:
        ax2.plot(basic_mse_losses, label='Basic MSE', linewidth=2, color='blue', alpha=0.7)
        ax2.plot(resnet_mse_losses, label='ResNet MSE', linewidth=2, color='red', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MSE Loss')
        ax2.set_title('MSE Loss Comparison')
        ax2.legend()
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.axis('off')
    
    # 3. Performance summary
    ax3 = plt.subplot(4, 4, 3)
    basic_mse = float(jnp.mean(jnp.square(basic_pred - ground_truth)))
    resnet_mse = float(jnp.mean(jnp.square(resnet_pred - ground_truth)))
    
    improvement = (basic_mse - resnet_mse) / basic_mse * 100
    
    ax3.text(0.1, 0.8, f'Basic LogNormalizer MSE: {basic_mse:.6f}', transform=ax3.transAxes, fontsize=12)
    ax3.text(0.1, 0.6, f'Quadratic ResNet MSE: {resnet_mse:.6f}', transform=ax3.transAxes, fontsize=12)
    ax3.text(0.1, 0.4, f'Improvement: {improvement:.1f}%', transform=ax3.transAxes, fontsize=12, 
             color='green' if improvement > 0 else 'red')
    ax3.set_title('Performance Summary')
    ax3.axis('off')
    
    # 4-16. Scatter plots for each statistic (12 statistics + 1 empty space)
    for i in range(min(min_size, 12)):  # Limit to 12 subplots
        ax = plt.subplot(4, 4, i + 4)
        
        # Scatter plots with better visualization
        ax.scatter(ground_truth[:, i], basic_pred[:, i], 
                  alpha=0.7, label='Basic', s=30, color='blue', edgecolors='white', linewidth=0.5)
        ax.scatter(ground_truth[:, i], resnet_pred[:, i], 
                  alpha=0.7, label='ResNet', s=30, color='red', edgecolors='white', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(ground_truth[:, i].min(), basic_pred[:, i].min(), resnet_pred[:, i].min())
        max_val = max(ground_truth[:, i].max(), basic_pred[:, i].max(), resnet_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='Perfect')
        
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{stat_names[i]}')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add R² score
        basic_r2 = 1 - jnp.sum((basic_pred[:, i] - ground_truth[:, i])**2) / jnp.sum((ground_truth[:, i] - jnp.mean(ground_truth[:, i]))**2)
        resnet_r2 = 1 - jnp.sum((resnet_pred[:, i] - ground_truth[:, i])**2) / jnp.sum((ground_truth[:, i] - jnp.mean(ground_truth[:, i]))**2)
        ax.text(0.05, 0.95, f'Basic R²: {float(basic_r2):.3f}\nResNet R²: {float(resnet_r2):.3f}', 
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('improved_log_normalizer_comparison.png', dpi=150, bbox_inches='tight')
    print("Improved plot saved as: improved_log_normalizer_comparison.png")
    plt.close()  # Close the figure to free memory and prevent blocking
    
    return fig


def main():
    """Main function with improved implementation."""
    print("Improved Log Normalizer Comparison")
    print("=" * 50)
    
    # Generate proper data
    print("Generating proper 3D Gaussian data with exponential family theory...")
    eta_data, ground_truth = generate_proper_3d_gaussian_data(n_samples=800, seed=42)
    print(f"Data shapes: eta={eta_data.shape}, ground_truth={ground_truth.shape}")
    
    # Compute ground truth <TT> for geometric KL loss
    print("Computing ground truth <TT> for geometric loss...")
    ground_truth_tt = compute_ground_truth_tt(ground_truth)
    print(f"Ground truth TT shape: {ground_truth_tt.shape}")
    
    # Create optimized models (more layers, smaller hidden sizes)
    print("\nCreating optimized models with more layers and smaller hidden sizes...")
    
    # Create models for maximum performance
    print("\nCreating models with LayerNorm enabled...")
    basic_model = ImprovedLogNormalizer(hidden_sizes=[128, 128, 128], 
                                      use_skip_connections=True,
                                      use_layer_norm=True)
    resnet_model = ImprovedQuadraticResNet(hidden_sizes=[128, 128, 128],
                                        use_layer_norm=True)
    
    # Calculate parameter counts
    basic_param_count = sum(x.size for x in jax.tree.leaves(basic_model.init(random.PRNGKey(0), eta_data[:1])))
    resnet_param_count = sum(x.size for x in jax.tree.leaves(resnet_model.init(random.PRNGKey(0), eta_data[:1])))
    
    print(f"Basic LogNormalizer: {basic_param_count:,} parameters")
    print(f"Quadratic ResNet: {resnet_param_count:,} parameters")
    print(f"Parameter ratio (ResNet/Basic): {resnet_param_count/basic_param_count:.1f}x")
    
    # Train with MSE loss only - increased epochs for convergence
    print("\nTraining Basic LogNormalizer with MSE loss...")
    basic_params, basic_losses, basic_mse_losses, basic_kl_losses, basic_kl_weights = advanced_training(
        basic_model, eta_data, ground_truth, ground_truth_tt, epochs=500, use_geometric_loss=False)
    
    print("\nTraining Quadratic ResNet with MSE loss...")
    resnet_params, resnet_losses, resnet_mse_losses, resnet_kl_losses, resnet_kl_weights = advanced_training(
        resnet_model, eta_data, ground_truth, ground_truth_tt, epochs=500, use_geometric_loss=False)
    
    # Make predictions
    print("\nMaking predictions...")
    
    def basic_single_log_normalizer(eta_single):
        return basic_model.apply(basic_params, eta_single[None, :], training=False)[0]
    
    basic_grad_fn = jax.grad(basic_single_log_normalizer)
    basic_predictions = jax.vmap(basic_grad_fn)(eta_data)
    
    def resnet_single_log_normalizer(eta_single):
        return resnet_model.apply(resnet_params, eta_single[None, :], training=False)[0]
    
    resnet_grad_fn = jax.grad(resnet_single_log_normalizer)
    resnet_predictions = jax.vmap(resnet_grad_fn)(eta_data)
    
    print(f"Prediction shapes: Basic={basic_predictions.shape}, ResNet={resnet_predictions.shape}")
    
    # Create improved plots (MSE loss only)
    print("\nCreating improved comparison plots...")
    plot_improved_comparison(eta_data, ground_truth, basic_predictions, resnet_predictions, basic_losses, resnet_losses,
                           basic_mse_losses, None, None,  # No KL components
                           resnet_mse_losses, None, None)  # No KL components
    
    # Detailed analysis
    print("\n" + "="*60)
    print("DETAILED PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Overall metrics
    basic_mse = float(jnp.mean(jnp.square(basic_predictions - ground_truth)))
    resnet_mse = float(jnp.mean(jnp.square(resnet_predictions - ground_truth)))
    
    basic_mae = float(jnp.mean(jnp.abs(basic_predictions - ground_truth)))
    resnet_mae = float(jnp.mean(jnp.abs(resnet_predictions - ground_truth)))
    
    print(f"Overall Performance:")
    print(f"  Basic LogNormalizer - MSE: {basic_mse:.8f}, MAE: {basic_mae:.8f}")
    print(f"  Quadratic ResNet    - MSE: {resnet_mse:.8f}, MAE: {resnet_mae:.8f}")
    
    improvement_mse = (basic_mse - resnet_mse) / basic_mse * 100
    improvement_mae = (basic_mae - resnet_mae) / basic_mae * 100
    
    print(f"  Improvement - MSE: {improvement_mse:.1f}%, MAE: {improvement_mae:.1f}%")
    
    # Per-statistic analysis
    print(f"\nPer-Statistic Analysis:")
    stat_names = [
        'E[x₁]', 'E[x₂]', 'E[x₃]',  # 3 mean statistics
        'E[x₁x₁]', 'E[x₁x₂]', 'E[x₁x₃]',  # First row of xx^T
        'E[x₂x₁]', 'E[x₂x₂]', 'E[x₂x₃]',  # Second row of xx^T  
        'E[x₃x₁]', 'E[x₃x₂]', 'E[x₃x₃]'   # Third row of xx^T
    ]
    
    for i, stat_name in enumerate(stat_names):
        if i < min(basic_predictions.shape[1], ground_truth.shape[1]):
            basic_error = float(jnp.mean(jnp.abs(basic_predictions[:, i] - ground_truth[:, i])))
            resnet_error = float(jnp.mean(jnp.abs(resnet_predictions[:, i] - ground_truth[:, i])))
            
            basic_r2 = 1 - jnp.sum((basic_predictions[:, i] - ground_truth[:, i])**2) / jnp.sum((ground_truth[:, i] - jnp.mean(ground_truth[:, i]))**2)
            resnet_r2 = 1 - jnp.sum((resnet_predictions[:, i] - ground_truth[:, i])**2) / jnp.sum((ground_truth[:, i] - jnp.mean(ground_truth[:, i]))**2)
            
            print(f"  {stat_name}: Basic MAE={basic_error:.6f} (R²={float(basic_r2):.3f}), "
                  f"ResNet MAE={resnet_error:.6f} (R²={float(resnet_r2):.3f})")
    
    # Final assessment
    if resnet_mse < basic_mse:
        print(f"\n🏆 Quadratic ResNet achieves {improvement_mse:.1f}% better performance!")
    else:
        print(f"\n🏆 Basic LogNormalizer achieves {-improvement_mse:.1f}% better performance!")
    
    # Quality assessment
    if basic_mse < 1e-6:
        print("🎯 Both models achieve near-perfect performance!")
    elif basic_mse < 1e-4:
        print("✅ Excellent performance achieved!")
    elif basic_mse < 1e-2:
        print("⚠️  Good performance, but could be better.")
    else:
        print("❌ Performance needs improvement.")
    
    print("\n✅ Improved comparison complete!")


if __name__ == "__main__":
    main()
