"""
Centralized eta feature engineering for neural networks.

This module provides consistent feature transformations of natural parameters η
across all neural network models in the nnef-dists package. The features are
designed to improve learning of the mapping between natural parameters and
sufficient statistics for exponential family distributions.

Features implemented:
1. Polynomial features (quadratic, cubic)
2. Cross-product terms
3. Inverse features with numerical stability
4. Normalized features (unit vectors)
5. Logarithmic features
6. Combined interaction terms

Usage:
    from eta_features import compute_eta_features
    
    # Basic polynomial features
    features = compute_eta_features(eta, method='polynomial')
    
    # Advanced features with all transformations
    features = compute_eta_features(eta, method='advanced')
    
    # Custom feature selection
    features = compute_eta_features(eta, method='custom', 
                                  include_polynomial=True,
                                  include_inverse=True,
                                  include_normalized=False)
"""

import jax.numpy as jnp
from typing import Optional, List, Dict, Any
import warnings


def compute_eta_features(eta: jnp.ndarray, 
                        method: str = 'default',
                        include_polynomial: bool = False,
                        include_cross_terms: bool = True,
                        include_inverse: bool = False,
                        include_normalized: bool = True,
                        include_doubly_normalized: bool = True,
                        include_norm_features: bool = True,
                        include_absolute: bool = False,
                        max_polynomial_degree: int = 3,
                        numerical_stability: bool = True,
                        clip_bounds: tuple = (-1e6, 1e6)) -> jnp.ndarray:
    """
    Compute feature transformations of natural parameters η.
    
    New approach focuses on normalized eta features:
    - All polynomial, cross-product, and inverse features are computed on eta/||eta||
    - Includes norm(eta) and 1/norm(eta) as separate features
    - Default method includes: eta, eta/||eta||, eta/||eta||^2, cross-products of normalized eta
    
    Args:
        eta: Natural parameters [batch_size, eta_dim]
        method: Predefined method ('default', 'polynomial', 'advanced', 'minimal', 'custom')
        include_polynomial: Include polynomial features of normalized eta
        include_cross_terms: Include cross-product terms of normalized eta
        include_inverse: Include inverse features of normalized eta
        include_normalized: Include normalized features (eta/||eta||)
        include_doubly_normalized: Include doubly normalized features (eta/||eta||^2)
        include_norm_features: Include norm(eta), 1/norm(eta), and log features
        include_absolute: Include absolute value features
        max_polynomial_degree: Maximum degree for polynomial features
        numerical_stability: Apply numerical stability measures
        clip_bounds: Bounds for clipping extreme values
        
    Returns:
        Feature matrix [batch_size, feature_dim]
    """
    
    # Select predefined method settings
    if method == 'default':
        # Default: eta, eta/||eta||, eta/||eta||^2, cross-products of normalized eta
        include_polynomial = False
        include_cross_terms = False
        include_inverse = False
        include_normalized = True
        include_doubly_normalized = True
        include_norm_features = True
        include_absolute = False
        
    elif method == 'polynomial':
        include_polynomial = True
        include_cross_terms = False
        include_inverse = False
        include_normalized = True
        include_doubly_normalized = True
        include_norm_features = False
        include_logarithmic = True
        include_absolute = False
        
    elif method == 'advanced':
        include_polynomial = True
        include_cross_terms = True
        include_inverse = True
        include_normalized = True
        include_doubly_normalized = True
        include_norm_features = True
        include_absolute = False
        
    elif method == 'minimal':
        include_polynomial = False
        include_cross_terms = False
        include_inverse = False
        include_normalized = True
        include_doubly_normalized = True
        include_norm_features = False
        include_logarithmic = True
        include_absolute = False
        max_polynomial_degree = 2

    elif method == 'noprop':
        include_polynomial = False
        include_cross_terms = False
        include_inverse = False
        include_normalized = True
        include_doubly_normalized = True
        include_norm_features = True
        include_absolute = False
        max_polynomial_degree = 2
        
    elif method == 'convex_only':
        # Convex-only features for convex neural networks
        include_polynomial = True  # Polynomial features of eta are convex if degree is even and coeffs > 0
        include_cross_terms = False  # Cross-products of eta can be convex
        include_inverse = False     # 1/x is not convex
        include_normalized = False  # eta/||eta|| is not convex
        include_doubly_normalized = False  # eta/||eta||^2 is not convex
        include_norm_features = True  # ||eta|| is convex, but 1/||eta|| and log(1+||eta||) are not
        include_logarithmic = False # log functions are typically concave
        include_absolute = True     # |eta| is convex
        max_polynomial_degree = 2   # Only even degrees for convexity
        
    elif method == 'custom':
        # Use the provided boolean flags
        pass
    else:
        raise ValueError(f"Unknown method: {method}. Use 'default', 'polynomial', 'advanced', 'minimal', 'convex_only', or 'custom'")
    
    features = []
    
    # Always include original eta
    features.append(eta)
    
    # Compute eta norm and normalized eta (used by multiple features)
    eta_norm = jnp.linalg.norm(eta, axis=-1, keepdims=True)
    eta_norm_safe = jnp.maximum(eta_norm, 1e-8)  # Avoid division by zero
    eta_normalized = eta / eta_norm_safe
    
    # 1. Normalized eta features (eta/||eta||)
    if include_normalized:
        features.append(eta_normalized)
    
    # 1.1. Doubly normalized eta features (eta/||eta||^2)
    if include_doubly_normalized:
        eta_doubly_normalized = eta / (eta_norm_safe ** 2)
        features.append(eta_doubly_normalized)

    # 2. Norm features: ||eta||, 1/||eta||, 1/||eta||^2, and log(1+||eta||)
    if include_norm_features:
        features.append(eta_norm)  # norm(eta) - CONVEX     
        features.append(1/eta_norm) # 1/norm(eta) - CONVEX because eta is never 0 vector
        features.append(1/(eta_norm** 2)) # 1/norm(eta)^2 - CONVEX because eta is never 0 vector
        features.append(-jnp.log(1.0+eta_norm)) # -log(1+norm(eta)) - CONVEX with the minus sign
    
    # 3. Cross-product terms
    if include_cross_terms:
        if method != 'convex_only':
            eta_for_cross = eta_normalized
            eta_dim = eta_for_cross.shape[-1]            
            for i in range(eta_dim):
                for j in range(i + 1, eta_dim):
                    cross_term = eta_for_cross[..., i:i+1] * eta_for_cross[..., j:j+1]
                    features.append(cross_term)
    
    # 4. Polynomial features
    if include_polynomial:
        # For convex_only, use original eta and only even degrees (convex)
        # Otherwise use normalized eta
        eta_for_poly = eta if method == 'convex_only' else eta_normalized
        
        for degree in range(2, max_polynomial_degree + 1):
            # For convex_only, only include even degrees (convex)
            if method == 'convex_only' and degree % 2 != 0:
                continue
            poly_features = eta_for_poly ** degree
            features.append(poly_features)
    
    # 5. Inverse features of NORMALIZED eta (1/normalized_eta) - NOT CONVEX
    if include_inverse and method != 'convex_only':
        if numerical_stability:
            # Avoid division by very small numbers
            eta_norm_safe_inv = jnp.where(jnp.abs(eta_normalized) < 1e-8, 
                                        jnp.sign(eta_normalized) * 1e-8, eta_normalized)
            eta_norm_inv = jnp.clip(1.0 / eta_norm_safe_inv, -1000.0, 1000.0)
        else:
            eta_norm_inv = 1.0 / eta_normalized
        features.append(eta_norm_inv)
    
    # 6. Logarithmic features (now integrated into norm features)
    # These are handled in the norm features section above
    
    # 7. Absolute value features
    if include_absolute:
        # For convex_only, use original eta (|eta| is convex)
        # Otherwise use normalized eta
        eta_for_abs = eta if method == 'convex_only' else eta_normalized
        abs_eta = jnp.abs(eta_for_abs)
        features.append(abs_eta)
    
    # Concatenate all features
    result = jnp.concatenate(features, axis=-1)
    
    # Apply numerical stability measures
    if numerical_stability:
        # Replace any remaining NaN or Inf values
        result = jnp.where(jnp.isfinite(result), result, 0.0)
        
        # Clip extreme values
        result = jnp.clip(result, clip_bounds[0], clip_bounds[1])
    
    return result


def get_feature_dimension(eta_dim: int,
                         method: str = 'default',
                         include_polynomial: bool = False,
                         include_cross_terms: bool = True,
                         include_inverse: bool = False,
                         include_normalized: bool = True,
                         include_doubly_normalized: bool = True,
                         include_norm_features: bool = True,
                         include_absolute: bool = False,
                         max_polynomial_degree: int = 2) -> int:
    """
    Compute the output dimension of eta features without actually computing them.
    
    This is useful for initializing neural networks with the correct input size.
    
    Args:
        eta_dim: Dimension of input eta
        method: Feature method (same options as compute_eta_features)
        **kwargs: Same feature flags as compute_eta_features
        
    Returns:
        Total feature dimension
    """
    
    # Apply method presets
    if method == 'default':
        include_polynomial = False
        include_cross_terms = False
        include_inverse = False
        include_normalized = True
        include_doubly_normalized = True
        include_norm_features = False
        include_logarithmic = True
        include_absolute = False
        
    elif method == 'polynomial':
        include_polynomial = True
        include_cross_terms = True
        include_inverse = False
        include_normalized = True
        include_norm_features = True
        include_logarithmic = False
        include_absolute = False
        
    elif method == 'advanced':
        include_polynomial = True
        include_cross_terms = True
        include_inverse = True
        include_normalized = True
        include_norm_features = True
        include_logarithmic = True
        include_absolute = True
        
    elif method == 'minimal':
        include_polynomial = True
        include_cross_terms = False
        include_inverse = False
        include_normalized = True
        include_norm_features = True
        include_logarithmic = False
        include_absolute = False
        max_polynomial_degree = 2
        
    elif method == 'noprop':
        include_polynomial = False
        include_cross_terms = False
        include_inverse = False
        include_normalized = True
        include_doubly_normalized = True
        include_norm_features = True  # For log(1+norm^2)
        include_logarithmic = True
        include_absolute = False
        max_polynomial_degree = 2
        
    elif method == 'convex_only':
        include_polynomial = True
        include_cross_terms = False
        include_inverse = False
        include_normalized = False
        include_doubly_normalized = False
        include_norm_features = True
        include_logarithmic = False
        include_absolute = True
        max_polynomial_degree = 2
    
    feature_count = 0
    
    # Original eta
    feature_count += eta_dim
    
    # Normalized eta features (eta/||eta||)
    if include_normalized:
        feature_count += eta_dim
    
    # Doubly normalized eta features (eta/||eta||^2)
    if include_doubly_normalized:
        feature_count += eta_dim
    
    # Norm features: ||eta||, 1/||eta||, 1/||eta||^2, -log(1+||eta||) - ALL CONVEX
    if include_norm_features:
        feature_count += 4  # norm, inverse norm, inverse norm squared, -log(1+norm)
    
    # Cross-product terms of normalized eta
    if include_cross_terms and eta_dim > 1:
        feature_count += eta_dim * (eta_dim - 1) // 2  # Upper triangular combinations
    
    # Polynomial features
    if include_polynomial:
        if method == 'convex_only':
            # Only even degrees (2, 4, 6, ...)
            even_degrees = [d for d in range(2, max_polynomial_degree + 1) if d % 2 == 0]
            feature_count += eta_dim * len(even_degrees)
        else:
            feature_count += eta_dim * (max_polynomial_degree - 1)  # degrees 2 through max_degree
    
    # Inverse features of normalized eta - NOT CONVEX
    if include_inverse:
        feature_count += eta_dim
    
    # Logarithmic features (now integrated into norm features)
    # These are handled in the norm features section above
    
    # Absolute value features of normalized eta
    if include_absolute:
        feature_count += eta_dim
    
    return feature_count


def get_feature_names(eta_dim: int,
                     method: str = 'default',
                     include_polynomial: bool = False,
                     include_cross_terms: bool = True,
                     include_inverse: bool = False,
                     include_normalized: bool = True,
                     include_doubly_normalized: bool = True,
                     include_norm_features: bool = True,
                     include_logarithmic: bool = False,
                     include_absolute: bool = False,
                     max_polynomial_degree: int = 3) -> List[str]:
    """
    Get descriptive names for each feature dimension.
    
    Useful for debugging and interpretability.
    
    Returns:
        List of feature names
    """
    
    # Apply method presets (same logic as above)
    if method == 'default':
        include_polynomial = False
        include_cross_terms = False
        include_inverse = False
        include_normalized = True
        include_doubly_normalized = True
        include_norm_features = False
        include_logarithmic = True
        include_absolute = False
        
    elif method == 'polynomial':
        include_polynomial = True
        include_cross_terms = False
        include_inverse = False
        include_normalized = True
        include_doubly_normalized = True
        include_norm_features = False
        include_logarithmic = True
        include_absolute = False
        
    elif method == 'advanced':
        include_polynomial = True
        include_cross_terms = True
        include_inverse = True
        include_normalized = True
        include_doubly_normalized = True
        include_norm_features = True
        include_absolute = False
        
    elif method == 'minimal':
        include_polynomial = False
        include_cross_terms = False
        include_inverse = False
        include_normalized = True
        include_doubly_normalized = True
        include_norm_features = False
        include_logarithmic = True
        include_absolute = False
        max_polynomial_degree = 2
        
    elif method == 'noprop':
        include_polynomial = False
        include_cross_terms = False
        include_inverse = False
        include_normalized = True
        include_doubly_normalized = True
        include_norm_features = True  # For log(1+norm^2)
        include_logarithmic = True
        include_absolute = False
        max_polynomial_degree = 2
        
    elif method == 'convex_only':
        include_polynomial = True
        include_cross_terms = False
        include_inverse = False
        include_normalized = False
        include_doubly_normalized = False
        include_norm_features = True
        include_logarithmic = False
        include_absolute = True
        max_polynomial_degree = 2
    
    names = []
    
    # Original eta
    for i in range(eta_dim):
        names.append(f'eta_{i}')
    
    # Normalized eta features (eta/||eta||)
    if include_normalized:
        for i in range(eta_dim):
            names.append(f'eta_{i}/||eta||')
    
    # Doubly normalized eta features (eta/||eta||^2)
    if include_doubly_normalized:
        for i in range(eta_dim):
            names.append(f'eta_{i}/||eta||^2')
    
    # Norm features: ||eta||, 1/||eta||, 1/||eta||^2, -log(1+||eta||) - ALL CONVEX
    if include_norm_features:
        names.append('||eta||')
        names.append('1/||eta||')
        names.append('1/||eta||^2')
        names.append('-log(1+||eta||)')
    
    # Cross-product terms
    if include_cross_terms and eta_dim > 1:
        for i in range(eta_dim):
            for j in range(i + 1, eta_dim):
                if method == 'convex_only':
                    names.append(f'eta_{i}*eta_{j}')
                else:
                    names.append(f'(eta_{i}/||eta||)*(eta_{j}/||eta||)')
    
    # Polynomial features
    if include_polynomial:
        for degree in range(2, max_polynomial_degree + 1):
            # For convex_only, only include even degrees
            if method == 'convex_only' and degree % 2 != 0:
                continue
            for i in range(eta_dim):
                if method == 'convex_only':
                    names.append(f'eta_{i}^{degree}')
                else:
                    names.append(f'(eta_{i}/||eta||)^{degree}')
    
    # Inverse features of normalized eta - NOT CONVEX
    if include_inverse:
        for i in range(eta_dim):
            names.append(f'1/(eta_{i}/||eta||)')
    
    # Logarithmic features (now integrated into norm features)
    # These are handled in the norm features section above
    
    # Absolute value features
    if include_absolute:
        for i in range(eta_dim):
            if method == 'convex_only':
                names.append(f'|eta_{i}|')
            else:
                names.append(f'|eta_{i}/||eta|||')
    
    return names


# Convenience functions for common use cases
def default_features(eta: jnp.ndarray) -> jnp.ndarray:
    """Compute default feature set: eta, eta/||eta||, cross-products, norm features."""
    return compute_eta_features(eta, method='default')


def polynomial_features(eta: jnp.ndarray, degree: int = 3) -> jnp.ndarray:
    """Compute polynomial features of normalized eta up to specified degree."""
    return compute_eta_features(eta, method='polynomial', max_polynomial_degree=degree)


def advanced_features(eta: jnp.ndarray) -> jnp.ndarray:
    """Compute comprehensive feature set with all transformations."""
    return compute_eta_features(eta, method='advanced')


def minimal_features(eta: jnp.ndarray) -> jnp.ndarray:
    """Compute minimal feature set (normalized eta + quadratic terms)."""
    return compute_eta_features(eta, method='minimal')


def convex_features(eta: jnp.ndarray) -> jnp.ndarray:
    """Compute convex-only feature set for convex neural networks."""
    return compute_eta_features(eta, method='convex_only')


# Example usage and testing
if __name__ == "__main__":
    import jax
    
    # Test with sample data
    rng = jax.random.PRNGKey(42)
    eta_test = jax.random.normal(rng, (10, 4))  # 10 samples, 4D eta
    
    print("Testing eta feature engineering:")
    print(f"Original eta shape: {eta_test.shape}")
    
    # Test different methods
    methods = ['default', 'convex_only', 'minimal', 'polynomial', 'advanced']
    for method in methods:
        features = compute_eta_features(eta_test, method=method)
        feature_dim = get_feature_dimension(4, method=method)
        feature_names = get_feature_names(4, method=method)
        
        print(f"\nMethod '{method}':")
        print(f"  Features shape: {features.shape}")
        print(f"  Expected dimension: {feature_dim}")
        print(f"  Feature names: {feature_names[:8]}...")  # Show first 8 names
        
        # Check for numerical issues
        finite_check = jnp.all(jnp.isfinite(features))
        print(f"  All finite: {finite_check}")
        print(f"  Value range: [{jnp.min(features):.3f}, {jnp.max(features):.3f}]")
