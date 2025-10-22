"""
Eta embedding layer for neural networks.

This module provides a non-learnable embedding layer that transforms
natural parameters η into rich feature representations for neural networks.
The embedding is applied as part of the model forward pass, enabling
gradient computation for logZ networks.
"""

import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, List


class EtaEmbedding(nn.Module):
    """
    Non-learnable eta embedding layer.
    
    This layer transforms raw natural parameters η into rich feature
    representations. The embedding is applied during the forward pass, 
    enabling gradient computation.
    
    Args:
        embedding_type: Type of embedding ('default', 'polynomial', 'advanced', etc.)
        eta_dim: Dimension of input eta (for validation)
    """
    
    embedding_type: str = 'default'
    eta_dim: Optional[int] = None
    
    def __call__(self, eta: jnp.ndarray) -> jnp.ndarray:
        """
        Apply eta embedding transformation.
        
        Args:
            eta: Natural parameters [batch_size, eta_dim]
            
        Returns:
            Embedded features [batch_size, feature_dim]
        """
        # Validate input dimension if specified
        if self.eta_dim is not None and eta.shape[-1] != self.eta_dim:
            raise ValueError(f"Expected eta_dim={self.eta_dim}, got {eta.shape[-1]}")
        
        # Apply eta feature transformation
        embedded_eta = self._compute_eta_features(eta, method=self.embedding_type)
        
        return embedded_eta
    
    def _compute_eta_features(self, eta: jnp.ndarray, method: str = 'default') -> jnp.ndarray:
        """
        Compute feature transformations of natural parameters η.
        
        Args:
            eta: Natural parameters [batch_size, eta_dim]
            method: Predefined method ('default', 'polynomial', 'advanced', 'minimal', 'convex_only')
            
        Returns:
            Feature matrix [batch_size, feature_dim]
        """
        if method == 'none':
            return eta

        # Compute eta norm and normalized eta (used by multiple features)
        eta_norm = jnp.linalg.norm(eta, axis=-1, keepdims=True)
        eta_norm_safe = jnp.maximum(eta_norm, 1e-6)  # Avoid division by zero
        eta_normalized = eta / eta_norm_safe
        eta_inverse_safe = jnp.clip(1./eta,-1000.0,1000.0)

        features = []

        if method == 'minimal':
            # Minimal: just normalized eta + quadratic terms
            features.append(eta_normalized)
            features.append(eta_norm)
        
        # Apply method-specific feature selection
        elif method == 'default':
            # Default: eta, eta/||eta||, eta/||eta||^2, norm features
            features.append(nn.sigmoid(eta_norm)*eta_normalized)  # This feature captures the limit of large eta.
            features.append(nn.softplus(eta))                     # These features show up alot in 1d distributions.
            features.append(nn.softplus(-eta))
            features.append(nn.sigmoid(eta))                      
            features.append(eta*jnp.exp(-jnp.abs(eta)))
            features.append(jnp.exp(-jnp.abs(eta)))
            features.append(nn.tanh(eta)/eta_inverse_safe)
            features.append(eta_inverse_safe*jnp.exp(-jnp.abs(eta_inverse_safe)))
            features.append(eta_normalized)  # eta/||eta||
            features.append(eta_normalized/eta_norm_safe)
            features.append(eta_norm)  # ||eta||
            features.append(1.0/(eta_norm + 1e-6))  # 1/||eta||
            features.append(-jnp.log(1.0 + eta_norm))  # -log(1+||eta||)
            
        elif method == 'polynomial':
            # Polynomial: default + polynomial features
            features.append(eta)
            features.append(eta_normalized)
            features.append(eta / (eta_norm_safe ** 2))
            features.append(eta_norm)
            features.append(1/eta_norm_safe)
            features.append(1/(eta_norm_safe ** 2))
            features.append(-jnp.log(1.0 + eta_norm))
            
            # Add polynomial features (degrees 2 and 3)
            features.append(eta_normalized ** 2)
            features.append(eta_normalized ** 3)
            
        elif method == 'advanced':
            # Advanced: all features including cross-products and inverse
            features.append(eta)
            features.append(eta_normalized)
            features.append(eta / (eta_norm_safe ** 2))
            features.append(eta_norm)
            features.append(1/eta_norm_safe)
            features.append(1/(eta_norm_safe ** 2))
            features.append(-jnp.log(1.0 + eta_norm))
            
            # Polynomial features
            features.append(eta_normalized ** 2)
            features.append(eta_normalized ** 3)
            
            # Cross-product terms (for 2D eta: eta_0 * eta_1)
            if eta.shape[-1] > 1:
                for i in range(eta.shape[-1]):
                    for j in range(i + 1, eta.shape[-1]):
                        cross_term = eta_normalized[..., i:i+1] * eta_normalized[..., j:j+1]
                        features.append(cross_term)
            
            # Inverse features (with numerical stability)
            eta_norm_safe_inv = jnp.where(jnp.abs(eta_normalized) < 1e-8, 
                                        jnp.sign(eta_normalized) * 1e-8, eta_normalized)
            eta_norm_inv = jnp.clip(1.0 / eta_norm_safe_inv, -1000.0, 1000.0)
            features.append(eta_norm_inv)
            
        elif method == 'minimal':
            # Minimal: just normalized eta + quadratic terms
            features.append(eta_normalized)
            features.append(eta_norm)
            
        elif method == 'convex_only':
            # Convex-only features for convex neural networks
            features.append(eta)                
            features.append(eta_norm)  # ||eta|| is convex
            features.append(1/eta_norm_safe)  # 1/||eta|| is convex
            features.append(1/(eta_norm_safe ** 2))  # 1/||eta||^2 is convex
            features.append(-jnp.log(1.0 + eta_norm))  # -log(1+||eta||) is convex
            features.append(jnp.abs(eta))  # |eta| is convex
            features.append(eta ** 2)  # eta^2 is convex (only even degrees)
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'default', 'polynomial', 'advanced', 'minimal', or 'convex_only'")
        
        # Concatenate all features
        result = jnp.concatenate(features, axis=-1)
        
        # Apply numerical stability measures
        result = jnp.where(jnp.isfinite(result), result, 0.0)
        result = jnp.clip(result, -1e6, 1e6)
        
        return result
    
    def get_output_dim(self, eta_dim: int) -> int:
        """
        Get the output dimension of the embedding.
        
        Args:
            eta_dim: Input eta dimension
            
        Returns:
            Output feature dimension
        """
        # Create dummy input to compute output shape
        dummy_eta = jnp.zeros((1, eta_dim))
        dummy_output = self.__call__(dummy_eta)
        return dummy_output.shape[-1]


def create_eta_embedding(embedding_type: str = 'default', eta_dim: Optional[int] = None) -> EtaEmbedding:
    """
    Factory function to create eta embedding layer.
    
    Args:
        embedding_type: Type of embedding to create
        eta_dim: Input eta dimension (optional, for validation)
        
    Returns:
        EtaEmbedding layer
    """
    return EtaEmbedding(embedding_type=embedding_type, eta_dim=eta_dim)


# Convenience functions for common embedding types
def default_eta_embedding(eta_dim: Optional[int] = None) -> EtaEmbedding:
    """Create default eta embedding (eta, normalized, doubly normalized, norm features)."""
    return create_eta_embedding('default', eta_dim)


def polynomial_eta_embedding(eta_dim: Optional[int] = None) -> EtaEmbedding:
    """Create polynomial eta embedding."""
    return create_eta_embedding('polynomial', eta_dim)


def advanced_eta_embedding(eta_dim: Optional[int] = None) -> EtaEmbedding:
    """Create advanced eta embedding with all features."""
    return create_eta_embedding('advanced', eta_dim)


def minimal_eta_embedding(eta_dim: Optional[int] = None) -> EtaEmbedding:
    """Create minimal eta embedding."""
    return create_eta_embedding('minimal', eta_dim)


def convex_eta_embedding(eta_dim: Optional[int] = None) -> EtaEmbedding:
    """Create convex-only eta embedding."""
    return create_eta_embedding('convex_only', eta_dim)


# Example usage and testing
if __name__ == "__main__":
    import jax
    
    # Test with sample data
    rng = jax.random.PRNGKey(42)
    eta_test = jax.random.normal(rng, (10, 2))  # 10 samples, 2D eta
    
    print("Testing eta embedding:")
    print(f"Original eta shape: {eta_test.shape}")
    
    # Test different methods
    methods = ['default', 'convex_only', 'minimal', 'polynomial', 'advanced']
    for method in methods:
        embedding = EtaEmbedding(embedding_type=method)
        features = embedding(eta_test)
        
        print(f"\nMethod '{method}':")
        print(f"  Features shape: {features.shape}")
        print(f"  Value range: [{jnp.min(features):.3f}, {jnp.max(features):.3f}]")
        
        # Check for numerical issues
        finite_check = jnp.all(jnp.isfinite(features))
        print(f"  All finite: {finite_check}")