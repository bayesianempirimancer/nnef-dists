#!/usr/bin/env python
"""JAX-compatible matrix utility functions for upper triangular operations."""

import jax.numpy as jnp
from typing import Union

Array = Union[jnp.ndarray]

def vector_to_strict_upper_triangular(vector: Array, n: int) -> Array:
    """
    Convert a vector of length n*(n-1)/2 to an upper triangular matrix with zeros on diagonal.
    
    Args:
        vector: Array of length n*(n-1)/2 containing upper triangular elements (excluding diagonal)
        n: Size of the resulting n×n matrix
        
    Returns:
        n×n matrix with zeros on diagonal and upper triangular elements from vector
        
    Example:
        >>> vector = [1, 2, 3]  # n=3, so length = 3*(3-1)/2 = 3
        >>> matrix = vector_to_strict_upper_triangular(vector, 3)
        >>> print(matrix)
        [[0 1 2]
         [0 0 3]
         [0 0 0]]
    """
    # Create zero matrix
    matrix = jnp.zeros((n, n))
    
    # Create mask for strict upper triangular (k=1 excludes diagonal)
    upper_mask = jnp.triu(jnp.ones((n, n)), k=1)
    
    # Get indices of upper triangular elements (excluding diagonal)
    upper_indices = jnp.where(upper_mask.flatten())[0]
    
    # Set the upper triangular elements
    flat_matrix = matrix.flatten()
    flat_matrix = flat_matrix.at[upper_indices].set(vector)
    
    # Reshape back to matrix
    return flat_matrix.reshape(n, n)

def strict_upper_triangular_to_vector(matrix: Array) -> Array:
    """
    Extract upper triangular elements (excluding diagonal) from a matrix into a vector.
    
    Args:
        matrix: n×n matrix
        
    Returns:
        Vector of length n*(n-1)/2 containing upper triangular elements (excluding diagonal)
    """
    n = matrix.shape[-1]  # Handle both single matrices and batched matrices
    
    # Create mask for strict upper triangular (k=1 excludes diagonal)
    upper_mask = jnp.triu(jnp.ones((n, n)), k=1)
    
    # Extract upper triangular elements
    upper_elements = matrix * upper_mask
    
    # Get indices of upper triangular elements (excluding diagonal)
    upper_indices = jnp.where(upper_mask.flatten())[0]
    
    # Extract the values at those indices
    flat_matrix = upper_elements.reshape(-1, n * n)
    vector = flat_matrix[..., upper_indices]
    
    return vector

def test_functions():
    """Test the functions with both single matrices and batched operations."""
    print("Testing strict upper triangular functions...")
    
    # Test case 1: Single 3x3 matrix
    print("\n=== Single Matrix Test ===")
    vector_3d = jnp.array([1, 2, 3])  # 3*(3-1)/2 = 3 elements
    n_3d = 3
    
    matrix_3d = vector_to_strict_upper_triangular(vector_3d, n_3d)
    print(f"Input vector: {vector_3d}")
    print(f"Output matrix:\n{matrix_3d}")
    
    vector_back = strict_upper_triangular_to_vector(matrix_3d)
    print(f"Extracted vector: {vector_back}")
    print(f"Round-trip successful: {jnp.allclose(vector_3d, vector_back)}")
    
    # Test case 2: Single 4x4 matrix
    print("\n=== 4x4 Matrix Test ===")
    vector_4d = jnp.array([1, 2, 3, 4, 5, 6])  # 4*(4-1)/2 = 6 elements
    n_4d = 4
    
    matrix_4d = vector_to_strict_upper_triangular(vector_4d, n_4d)
    print(f"Input vector: {vector_4d}")
    print(f"Output matrix:\n{matrix_4d}")
    
    vector_back_4d = strict_upper_triangular_to_vector(matrix_4d)
    print(f"Extracted vector: {vector_back_4d}")
    print(f"Round-trip successful: {jnp.allclose(vector_4d, vector_back_4d)}")
    
    # Test case 3: Batched operations with vmap
    print("\n=== Batched Operations Test ===")
    batch_vectors = jnp.array([
        [1, 2, 3],      # First matrix
        [4, 5, 6],      # Second matrix
        [7, 8, 9]       # Third matrix
    ])  # Shape: (3, 3) - 3 vectors of length 3
    
    # Use vmap to apply to batch of vectors
    from jax import vmap
    batched_to_matrix = vmap(lambda v: vector_to_strict_upper_triangular(v, 3))
    batch_matrices = batched_to_matrix(batch_vectors)
    
    print(f"Batch vectors shape: {batch_vectors.shape}")
    print(f"Batch matrices shape: {batch_matrices.shape}")
    print(f"First matrix:\n{batch_matrices[0]}")
    print(f"Second matrix:\n{batch_matrices[1]}")
    
    # Test reverse operation
    batched_to_vector = vmap(lambda m: strict_upper_triangular_to_vector(m))
    batch_vectors_back = batched_to_vector(batch_matrices)
    
    print(f"Reconstructed batch vectors: {batch_vectors_back}")
    print(f"Batch round-trip successful: {jnp.allclose(batch_vectors, batch_vectors_back)}")
    
    # Test case 4: Higher dimensional batch
    print("\n=== Higher Dimensional Batch Test ===")
    large_batch_vectors = jnp.array([
        [[1, 2, 3], [4, 5, 6]],  # First batch
        [[7, 8, 9], [10, 11, 12]] # Second batch
    ])  # Shape: (2, 2, 3)
    
    # Apply vmap twice for 2D batch
    batched_to_matrix_2d = vmap(vmap(lambda v: vector_to_strict_upper_triangular(v, 3)))
    large_batch_matrices = batched_to_matrix_2d(large_batch_vectors)
    
    print(f"Large batch vectors shape: {large_batch_vectors.shape}")
    print(f"Large batch matrices shape: {large_batch_matrices.shape}")
    print(f"First batch, first matrix:\n{large_batch_matrices[0, 0]}")

if __name__ == "__main__":
    test_functions()
