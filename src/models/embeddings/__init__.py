"""
Embeddings module for neural network feature extraction and time embeddings.

This module provides standardized implementations for:
- Eta feature engineering (natural parameter transformations)
- Time embeddings (Fourier embeddings for continuous time)
- Other specialized embeddings used across neural network models
"""

from .eta_features import (
    compute_eta_features,
    get_feature_dimension,
    get_feature_names,
    default_features,
    polynomial_features,
    advanced_features,
    minimal_features,
    convex_features
)

from .time_embeddings import (
    TimeEmbedding,
    FourierTimeEmbedding,
    create_time_embedding
)

__all__ = [
    # Eta features
    'compute_eta_features',
    'get_feature_dimension', 
    'get_feature_names',
    'default_features',
    'polynomial_features',
    'advanced_features',
    'minimal_features',
    'convex_features',
    
    # Time embeddings
    'TimeEmbedding',
    'FourierTimeEmbedding',
    'create_time_embedding'
]
