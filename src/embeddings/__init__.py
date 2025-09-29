"""
Embeddings module for neural network feature extraction and time embeddings.

This module provides standardized implementations for:
- Eta feature engineering (natural parameter transformations)
- Time embeddings (Fourier embeddings for continuous time)
- Other specialized embeddings used across neural network models
"""

from .eta_embedding import (
    EtaEmbedding,
    create_eta_embedding,
    default_eta_embedding,
    polynomial_eta_embedding,
    advanced_eta_embedding,
    minimal_eta_embedding,
    convex_eta_embedding
)

from .time_embeddings import (
    TimeEmbedding,
    FourierTimeEmbedding,
    create_time_embedding
)

__all__ = [
    # Eta embeddings
    'EtaEmbedding',
    'create_eta_embedding',
    'default_eta_embedding',
    'polynomial_eta_embedding',
    'advanced_eta_embedding',
    'minimal_eta_embedding',
    'convex_eta_embedding',
    
    # Time embeddings
    'TimeEmbedding',
    'FourierTimeEmbedding',
    'create_time_embedding'
]
