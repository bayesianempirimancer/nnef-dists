#!/usr/bin/env python3
"""
Debug script to test the theoretical relationship between Jacobian and covariance
in exponential families.

This script investigates the correct relationship between:
1. Network Jacobian ∇_η f(η) where f(η) ≈ E[T(X)]
2. Fisher Information Matrix / Covariance of sufficient statistics
3. Ground truth analytical covariance

Goal: Fix the theoretical issue in estimate_covariance_from_jacobian()
"""

import sys
from pathlib import Path
import jax
import jax.numpy as jnp
from jax import jacfwd, random
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_deep_narrow_config
from src.models.standard_mlp import create_model_and_trainer
from src.data_utils import load_3d_gaussian_data, compute_ground_truth_3d_tril
from src.ef import MultivariateNormal_tril


def compute_analytical_fisher_matrix(eta: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the analytical Fisher Information Matrix for 3D Gaussian.
    
    For multivariate Gaussian with natural parameters η = [η₁, η₂]:
    - η₁ ∈ ℝ³ (related to mean)
    - η₂ ∈ ℝ⁹ (related to precision matrix, flattened)
    
    The Fisher Information Matrix has a specific block structure.
    """
    batch_size = eta.shape[0]
    
    # Extract η₁ (first 3) and η₂ (last 9, reshaped to 3x3)
    eta1 = eta[:, :3]  # [batch, 3]
    eta2 = eta[:, 3:].reshape(batch_size, 3, 3)  # [batch, 3, 3]
    
    # Precision matrix: Λ = -2η₂
    Lambda = -2.0 * eta2  # [batch, 3, 3]
    
    # Covariance matrix: Σ = Λ⁻¹
    Sigma = jnp.linalg.inv(Lambda)  # [batch, 3, 3]
    
    # For multivariate Gaussian, the Fisher Information Matrix has block structure:
    # I = [[Σ⁻¹,     -Σ⁻¹μ ⊗ I],
    #      [-μ ⊗ Σ⁻¹,  Σ⁻¹ ⊗ Σ⁻¹]]
    # 
    # But this is for the full parameter space. For the sufficient statistics
    # covariance, we need the covariance of [X, vec(XX^T)].
    
    # Mean: μ = Σ η₁
    mu = jnp.einsum('bij,bj->bi', Sigma, eta1)  # [batch, 3]
    
    # For sufficient statistics T(X) = [X, tril(XX^T)], the covariance is complex.
    # Let's compute it analytically for comparison.
    
    # Simplified approach: return Sigma as the primary covariance structure
    # This is not the full Fisher matrix but gives us a baseline for comparison
    return Sigma


def compute_network_jacobian_detailed(model, params, eta):
    """
    Compute network Jacobian with detailed analysis.
    
    Returns both the Jacobian and analysis of its structure.
    """
    def network_fn(eta_single):
        return model.apply(params, eta_single[None, :], training=False)[0]
    
    # Compute Jacobian for each sample
    jacobian_fn = jacfwd(network_fn)
    jacobians = jax.vmap(jacobian_fn)(eta)
    
    # Analyze Jacobian structure
    batch_size, output_dim, eta_dim = jacobians.shape
    
    analysis = {
        'shape': jacobians.shape,
        'mean_magnitude': float(jnp.mean(jnp.abs(jacobians))),
        'max_magnitude': float(jnp.max(jnp.abs(jacobians))),
        'min_magnitude': float(jnp.min(jnp.abs(jacobians))),
        'frobenius_norm': float(jnp.mean(jnp.linalg.norm(jacobians.reshape(batch_size, -1), axis=1))),
    }
    
    # Check for specific structure in 3D Gaussian case
    if output_dim == 9 and eta_dim == 12:
        # Split Jacobian by parameter type
        jac_wrt_eta1 = jacobians[:, :, :3]   # ∇_{η₁} E[T(X)] [batch, 9, 3]
        jac_wrt_eta2 = jacobians[:, :, 3:]   # ∇_{η₂} E[T(X)] [batch, 9, 9]
        
        analysis.update({
            'jac_eta1_norm': float(jnp.mean(jnp.linalg.norm(jac_wrt_eta1, axis=(1,2)))),
            'jac_eta2_norm': float(jnp.mean(jnp.linalg.norm(jac_wrt_eta2, axis=(1,2)))),
            'eta1_vs_eta2_ratio': float(jnp.mean(jnp.linalg.norm(jac_wrt_eta1, axis=(1,2))) / 
                                       jnp.mean(jnp.linalg.norm(jac_wrt_eta2, axis=(1,2))))
        })
    
    return jacobians, analysis


def test_covariance_estimation_methods(jacobian, ground_truth_cov=None):
    """
    Test different methods for extracting covariance from Jacobian.
    """
    batch_size, output_dim, eta_dim = jacobian.shape
    
    methods = {}
    
    # Method 1: Original (incorrect) - J @ J^T
    methods['J_JT'] = jnp.einsum('bij,bkj->bik', jacobian, jacobian)
    
    # Method 2: Diagonal from Jacobian row norms  
    jacobian_row_norms = jnp.linalg.norm(jacobian, axis=-1)
    methods['diagonal_norms'] = jnp.zeros((batch_size, output_dim, output_dim))
    for i in range(output_dim):
        methods['diagonal_norms'] = methods['diagonal_norms'].at[:, i, i].set(jacobian_row_norms[:, i])
    
    # Method 3: Use only the η₂ part (precision-related parameters)
    if output_dim == 9 and eta_dim == 12:
        jac_eta2 = jacobian[:, :, 3:]  # [batch, 9, 9]
        methods['eta2_part'] = jac_eta2
    
    # Method 4: Symmetrized version of η₂ part
    if output_dim == 9 and eta_dim == 12:
        jac_eta2 = jacobian[:, :, 3:]  # [batch, 9, 9]
        methods['eta2_symmetrized'] = 0.5 * (jac_eta2 + jnp.swapaxes(jac_eta2, 1, 2))
    
    # Evaluate methods if ground truth available
    if ground_truth_cov is not None:
        print("\n🔍 COVARIANCE ESTIMATION METHODS COMPARISON:")
        print("-" * 60)
        
        for name, estimated_cov in methods.items():
            if estimated_cov.shape == ground_truth_cov.shape:
                # Compute error metrics
                mse = float(jnp.mean(jnp.square(estimated_cov - ground_truth_cov)))
                mae = float(jnp.mean(jnp.abs(estimated_cov - ground_truth_cov)))
                
                # Frobenius norm error
                frob_error = float(jnp.mean(jnp.linalg.norm(
                    estimated_cov - ground_truth_cov, ord='fro', axis=(1,2)
                )))
                
                print(f"{name:20s}: MSE={mse:8.2f}, MAE={mae:8.2f}, Frob={frob_error:8.2f}")
            else:
                print(f"{name:20s}: Shape mismatch {estimated_cov.shape} vs {ground_truth_cov.shape}")
    
    return methods


def main():
    """Debug the Jacobian-covariance relationship."""
    
    print("🔬 DEBUGGING JACOBIAN-COVARIANCE THEORY")
    print("=" * 60)
    print("Investigating the correct relationship between network Jacobian")
    print("and covariance matrix for exponential families")
    
    # Load small dataset for debugging
    print("\n📊 Loading data...")
    data_dir = Path("data")
    data, ef = load_3d_gaussian_data(data_dir, format="tril")
    
    # Use small subset for debugging
    n_debug = 10
    debug_eta = data["train_eta"][:n_debug]
    debug_y = data["train_y"][:n_debug]
    
    print(f"Debug dataset: {n_debug} samples")
    print(f"eta shape: {debug_eta.shape}")
    print(f"y shape: {debug_y.shape}")
    
    # Compute analytical ground truth
    print("\n🎯 Computing analytical ground truth...")
    ground_truth_stats = compute_ground_truth_3d_tril(debug_eta, ef)
    
    # For the covariance comparison, we need the actual covariance of sufficient statistics
    # This is complex for multivariate Gaussian, so let's use a simplified version
    
    # Extract η₁ and η₂
    eta1 = debug_eta[:, :3]
    eta2 = debug_eta[:, 3:].reshape(n_debug, 3, 3)
    
    # Compute Σ = (-2η₂)⁻¹
    Sigma = jnp.linalg.inv(-2.0 * eta2)  # [batch, 3, 3]
    
    print(f"Analytical covariance (Σ) shape: {Sigma.shape}")
    print(f"Sample Σ[0]:\n{Sigma[0]}")
    
    # Train a simple model for Jacobian analysis
    print("\n🏗️  Training simple model for Jacobian analysis...")
    config = get_deep_narrow_config()
    config.training.num_epochs = 20  # Quick training for debugging
    config.training.batch_size = 16
    config.network.hidden_sizes = [64, 64]  # Simple architecture for debugging
    
    trainer = create_model_and_trainer(config)
    
    # Quick training
    train_data = {"eta": debug_eta, "y": debug_y}
    val_data = {"eta": debug_eta, "y": debug_y}  # Same for debugging
    
    params, history = trainer.train(train_data, val_data)
    
    # Compute network Jacobian
    print("\n🧮 Computing network Jacobian...")
    jacobians, jac_analysis = compute_network_jacobian_detailed(trainer.model, params, debug_eta)
    
    print(f"Jacobian analysis:")
    for key, value in jac_analysis.items():
        print(f"  {key}: {value}")
    
    # Test different covariance estimation methods
    print("\n🔬 Testing covariance estimation methods...")
    
    # For comparison, we'll use Σ as a proxy for the "true" covariance structure
    # In reality, the covariance of [X, tril(XX^T)] is more complex
    
    # Create a simplified ground truth covariance for comparison
    # Use the diagonal structure of Σ extended to the 9D tril space
    simple_gt_cov = jnp.zeros((n_debug, 9, 9))
    
    # Fill diagonal with elements from Σ (simplified approximation)
    for i in range(3):  # Mean components
        simple_gt_cov = simple_gt_cov.at[:, i, i].set(Sigma[:, i, i])
    
    # For tril components (indices 3-8), use scaled versions of Σ elements
    tril_indices = [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]
    for idx, (i, j) in enumerate(tril_indices):
        diag_idx = 3 + idx
        simple_gt_cov = simple_gt_cov.at[:, diag_idx, diag_idx].set(Sigma[:, i, j])
    
    methods_results = test_covariance_estimation_methods(jacobians, simple_gt_cov)
    
    # Analyze network predictions vs ground truth
    print("\n📊 Network predictions analysis:")
    predictions = trainer.model.apply(params, debug_eta, training=False)
    
    pred_mse = float(jnp.mean(jnp.square(predictions - ground_truth_stats)))
    pred_mae = float(jnp.mean(jnp.abs(predictions - ground_truth_stats)))
    
    print(f"Prediction MSE vs ground truth: {pred_mse:.4f}")
    print(f"Prediction MAE vs ground truth: {pred_mae:.4f}")
    
    # Create visualization
    print("\n📈 Creating visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Jacobian magnitude heatmap
    jac_magnitude = jnp.mean(jnp.abs(jacobians), axis=0)  # Average over batch
    im1 = axes[0, 0].imshow(jac_magnitude, aspect='auto', cmap='viridis')
    axes[0, 0].set_title('Jacobian Magnitude\n∇_η E[T(X)]')
    axes[0, 0].set_xlabel('Natural Parameters (η)')
    axes[0, 0].set_ylabel('Sufficient Statistics')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. Ground truth covariance structure
    gt_cov_mean = jnp.mean(simple_gt_cov, axis=0)
    im2 = axes[0, 1].imshow(gt_cov_mean, aspect='auto', cmap='coolwarm')
    axes[0, 1].set_title('Ground Truth Covariance\n(Simplified)')
    axes[0, 1].set_xlabel('Statistic Index')
    axes[0, 1].set_ylabel('Statistic Index')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. Best method covariance estimate
    best_method = 'eta2_symmetrized' if 'eta2_symmetrized' in methods_results else 'diagonal_norms'
    best_cov_mean = jnp.mean(methods_results[best_method], axis=0)
    im3 = axes[0, 2].imshow(best_cov_mean, aspect='auto', cmap='coolwarm')
    axes[0, 2].set_title(f'Estimated Covariance\n({best_method})')
    axes[0, 2].set_xlabel('Statistic Index')
    axes[0, 2].set_ylabel('Statistic Index')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # 4. Predictions vs Ground Truth
    axes[1, 0].scatter(ground_truth_stats.flatten(), predictions.flatten(), alpha=0.6, s=20)
    min_val = min(ground_truth_stats.min(), predictions.min())
    max_val = max(ground_truth_stats.max(), predictions.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[1, 0].set_xlabel('Ground Truth')
    axes[1, 0].set_ylabel('Network Predictions')
    axes[1, 0].set_title('Predictions vs Ground Truth')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Jacobian structure analysis
    if jacobians.shape[1] == 9 and jacobians.shape[2] == 12:
        # Split by parameter type
        jac_eta1 = jacobians[:, :, :3]   # ∇_{η₁} E[T(X)]
        jac_eta2 = jacobians[:, :, 3:]   # ∇_{η₂} E[T(X)]
        
        eta1_norms = jnp.linalg.norm(jac_eta1, axis=(1,2))
        eta2_norms = jnp.linalg.norm(jac_eta2, axis=(1,2))
        
        axes[1, 1].scatter(eta1_norms, eta2_norms, alpha=0.7)
        axes[1, 1].set_xlabel('||∇_{η₁} E[T(X)]||')
        axes[1, 1].set_ylabel('||∇_{η₂} E[T(X)]||')
        axes[1, 1].set_title('Jacobian Structure Analysis')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Method comparison
    if len(methods_results) > 1:
        method_names = list(methods_results.keys())
        method_errors = []
        
        for name, estimated_cov in methods_results.items():
            if estimated_cov.shape == simple_gt_cov.shape:
                error = float(jnp.mean(jnp.square(estimated_cov - simple_gt_cov)))
                method_errors.append(error)
            else:
                method_errors.append(float('inf'))
        
        bars = axes[1, 2].bar(range(len(method_names)), method_errors, alpha=0.7)
        axes[1, 2].set_xticks(range(len(method_names)))
        axes[1, 2].set_xticklabels(method_names, rotation=45, ha='right')
        axes[1, 2].set_ylabel('MSE vs Ground Truth')
        axes[1, 2].set_title('Covariance Estimation Methods')
        axes[1, 2].set_yscale('log')
        
        # Add value labels
        for bar, error in zip(bars, method_errors):
            if error != float('inf'):
                height = bar.get_height()
                axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                               f'{error:.1e}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save debug plots
    debug_dir = Path("artifacts/debug_jacobian")
    debug_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(debug_dir / "jacobian_covariance_debug.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Debug plots saved to {debug_dir}/")
    
    # Summary and recommendations
    print("\n🎯 THEORETICAL ANALYSIS SUMMARY:")
    print("-" * 50)
    print("Current Issue:")
    print("  • Original implementation used J @ J^T (incorrect for exp families)")
    print("  • Jacobian ∇_η E[T(X)] should directly relate to Fisher Info Matrix")
    print("  • Dimensional challenge: Jacobian [batch,9,12] vs Covariance [batch,9,9]")
    
    print("\nKey Findings:")
    print(f"  • Network achieves {pred_mse:.4f} MSE vs ground truth")
    print(f"  • Jacobian mean magnitude: {jac_analysis['mean_magnitude']:.4f}")
    print(f"  • Jacobian max magnitude: {jac_analysis['max_magnitude']:.4f}")
    
    if 'eta1_vs_eta2_ratio' in jac_analysis:
        print(f"  • η₁ vs η₂ Jacobian ratio: {jac_analysis['eta1_vs_eta2_ratio']:.4f}")
    
    print("\nRecommendations:")
    print("  1. Use the η₂ part of Jacobian (∇_{η₂} E[T(X)]) for covariance structure")
    print("  2. Consider symmetrization since covariance matrices are symmetric")
    print("  3. Add proper regularization for numerical stability")
    print("  4. Validate against known analytical Fisher Information Matrix")
    
    # Save detailed debug results
    debug_results = {
        'jacobian_analysis': jac_analysis,
        'prediction_metrics': {'mse': pred_mse, 'mae': pred_mae},
        'methods_tested': list(methods_results.keys()),
        'recommendations': [
            "Use eta2 part of Jacobian for covariance structure",
            "Apply symmetrization for covariance matrices", 
            "Add proper regularization",
            "Validate against analytical Fisher matrix"
        ]
    }
    
    import json
    with open(debug_dir / "debug_results.json", 'w') as f:
        def convert_for_json(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_for_json(debug_results), f, indent=2)
    
    print(f"\n💾 Debug results saved to {debug_dir}/")
    print("\n✅ Jacobian theory debugging completed!")


if __name__ == "__main__":
    main()
