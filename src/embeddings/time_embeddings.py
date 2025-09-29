"""
Time embedding implementations for neural networks.

This module provides standardized time embedding functions for continuous-time
neural networks, particularly useful for flow-based models and NoProp training.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Union, Callable


class TimeEmbedding(nn.Module):
    """
    Base class for time embeddings.
    
    Provides a standardized interface for creating time embeddings
    from continuous time values t ∈ [0, 1].
    """
    
    embed_dim: int
    max_freq: float = 10.0
    include_linear: bool = True
    linear_weight: float = 0.1
    
    @nn.compact
    def __call__(self, t: Union[float, jnp.ndarray]) -> jnp.ndarray:
        """
        Create time embedding from continuous time.
        
        Args:
            t: Time value(s) ∈ [0, 1]
            
        Returns:
            Time embedding [embed_dim,] or [batch_size, embed_dim]
        """
        raise NotImplementedError("Subclasses must implement __call__")


class FourierTimeEmbedding(TimeEmbedding):
    """
    Fourier-based time embedding using sinusoidal functions.
    
    Creates embeddings using sin and cos functions with different frequencies,
    optionally combined with a linear component for time progression.
    """
    
    def __call__(self, t: Union[float, jnp.ndarray]) -> jnp.ndarray:
        """
        Create Fourier time embedding.
        
        Args:
            t: Time value(s) ∈ [0, 1]
            
        Returns:
            Time embedding [embed_dim,] or [batch_size, embed_dim]
        """
        # Ensure t is a JAX array
        if isinstance(t, float):
            t = jnp.array(t)
        
        # Create frequency schedule
        n_freqs = max(1, self.embed_dim // 4)  # Use 1/4 of dimension for frequencies
        freqs = jnp.logspace(0, jnp.log10(self.max_freq), n_freqs)
        
        # Create sin and cos embeddings
        sin_embeddings = jnp.sin(2 * jnp.pi * freqs * t)
        cos_embeddings = jnp.cos(2 * jnp.pi * freqs * t)
        
        # Combine sin and cos
        freq_embeddings = jnp.concatenate([sin_embeddings, cos_embeddings])
        
        # Adjust to target dimension
        if len(freq_embeddings) < self.embed_dim:
            # Repeat with different phase shifts if needed
            n_repeats = (self.embed_dim + len(freq_embeddings) - 1) // len(freq_embeddings)
            repeated_embeddings = jnp.tile(freq_embeddings, n_repeats)
            embeddings = repeated_embeddings[:self.embed_dim]
        else:
            # Truncate if too many
            embeddings = freq_embeddings[:self.embed_dim]
        
        # Add linear component if requested
        if self.include_linear:
            linear_component = jnp.linspace(0, t, self.embed_dim)
            embeddings = embeddings + self.linear_weight * linear_component
        
        return embeddings


class SimpleTimeEmbedding(TimeEmbedding):
    """
    Simple time embedding using basic sinusoidal functions.
    
    A simpler version that uses evenly spaced frequencies and doesn't
    include the linear component or complex frequency scheduling.
    """
    
    def __call__(self, t: Union[float, jnp.ndarray]) -> jnp.ndarray:
        """
        Create simple time embedding.
        
        Args:
            t: Time value(s) ∈ [0, 1]
            
        Returns:
            Time embedding [embed_dim,] or [batch_size, embed_dim]
        """
        # Ensure t is a JAX array
        if isinstance(t, float):
            t = jnp.array(t)
        
        # Use evenly spaced frequencies
        embed_dim = min(self.embed_dim, 16)  # Cap at 16 for simplicity
        freqs = jnp.linspace(0, 1, embed_dim // 2)
        
        # Create sin and cos embeddings
        sin_embeddings = jnp.sin(2 * jnp.pi * freqs * t)
        cos_embeddings = jnp.cos(2 * jnp.pi * freqs * t)
        
        # Combine and pad/truncate to target dimension
        time_embed = jnp.concatenate([sin_embeddings, cos_embeddings])
        
        if len(time_embed) < self.embed_dim:
            # Pad with zeros
            padding = jnp.zeros(self.embed_dim - len(time_embed))
            time_embed = jnp.concatenate([time_embed, padding])
        else:
            # Truncate
            time_embed = time_embed[:self.embed_dim]
        
        return time_embed


# Convenience functions for creating time embeddings
def create_time_embedding(embed_dim: int, 
                         method: str = "fourier",
                         max_freq: float = 10.0,
                         include_linear: bool = True,
                         linear_weight: float = 0.1) -> TimeEmbedding:
    """
    Create a time embedding instance.
    
    Args:
        embed_dim: Dimension of the time embedding
        method: Method to use ("fourier" or "simple")
        max_freq: Maximum frequency for Fourier embeddings
        include_linear: Whether to include linear component
        linear_weight: Weight for linear component
        
    Returns:
        TimeEmbedding instance
    """
    if method == "fourier":
        return FourierTimeEmbedding(
            embed_dim=embed_dim,
            max_freq=max_freq,
            include_linear=include_linear,
            linear_weight=linear_weight
        )
    elif method == "simple":
        return SimpleTimeEmbedding(
            embed_dim=embed_dim,
            max_freq=max_freq,
            include_linear=include_linear,
            linear_weight=linear_weight
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'fourier' or 'simple'")


def fourier_time_embedding(t: Union[float, jnp.ndarray], 
                          embed_dim: int,
                          max_freq: float = 10.0,
                          include_linear: bool = True,
                          linear_weight: float = 0.1) -> jnp.ndarray:
    """
    Create Fourier time embedding directly without class instantiation.
    
    Args:
        t: Time value(s) ∈ [0, 1]
        embed_dim: Dimension of the embedding
        max_freq: Maximum frequency
        include_linear: Whether to include linear component
        linear_weight: Weight for linear component
        
    Returns:
        Time embedding [embed_dim,] or [batch_size, embed_dim]
    """
    embedding = FourierTimeEmbedding(
        embed_dim=embed_dim,
        max_freq=max_freq,
        include_linear=include_linear,
        linear_weight=linear_weight
    )
    return embedding(t)


def simple_time_embedding(t: Union[float, jnp.ndarray], 
                         embed_dim: int) -> jnp.ndarray:
    """
    Create simple time embedding directly without class instantiation.
    
    Args:
        t: Time value(s) ∈ [0, 1]
        embed_dim: Dimension of the embedding
        
    Returns:
        Time embedding [embed_dim,] or [batch_size, embed_dim]
    """
    embedding = SimpleTimeEmbedding(embed_dim=embed_dim)
    return embedding(t)


# Example usage and testing
if __name__ == "__main__":
    import jax
    
    print("Testing time embeddings:")
    
    # Test with single time value
    t_single = 0.5
    print(f"\nSingle time value: {t_single}")
    
    # Test Fourier embedding
    fourier_embed = fourier_time_embedding(t_single, embed_dim=8, max_freq=5.0)
    print(f"Fourier embedding shape: {fourier_embed.shape}")
    print(f"Fourier embedding values: {fourier_embed}")
    
    # Test simple embedding
    simple_embed = simple_time_embedding(t_single, embed_dim=8)
    print(f"Simple embedding shape: {simple_embed.shape}")
    print(f"Simple embedding values: {simple_embed}")
    
    # Test with batch of time values
    t_batch = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    print(f"\nBatch time values: {t_batch}")
    
    fourier_batch = fourier_time_embedding(t_batch, embed_dim=6, max_freq=3.0)
    print(f"Fourier batch shape: {fourier_batch.shape}")
    print(f"Fourier batch values:\n{fourier_batch}")
    
    # Test class-based usage
    print(f"\nClass-based usage:")
    embedding = create_time_embedding(embed_dim=4, method="fourier")
    class_embed = embedding(t_single)
    print(f"Class embedding shape: {class_embed.shape}")
    print(f"Class embedding values: {class_embed}")
