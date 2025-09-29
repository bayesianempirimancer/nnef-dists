"""
Custom neural network layers and building blocks.

This module provides custom neural network layers and building blocks
for the nnef-dists package, including specialized layers for exponential
family distributions and neural network architectures.
"""

# Import all layer classes
from .affine import AffineCouplingLayer
from .bilinear import BilinearLayer, BilinearResidualBlock
from .convex import (
    ConvexHiddenLayer, SimpleConvexBlock, ICNNBlock,
    ConvexResNetWrapper, ConvexResNetWrapperBivariate
)
from .glu import GLUBlock
from .mlp import MLPBlock
from .normalization import WeakLayerNorm
from .quadratic import QuadraticLayer, QuadraticBlock, QuadraticProjectionLayer, QuadraticProjectionBlock
from .resnet_wrapper import (
    ResNetWrapper, ResNetWrapperBivariate
)

# Public API
__all__ = [
    # Affine layers
    "AffineCouplingLayer",
    
    # Bilinear layers
    "BilinearLayer",
    "BilinearResidualBlock",
    
    # Convex layers
    "ConvexHiddenLayer",
    "SimpleConvexBlock",
    "ICNNBlock", 
    "ConvexResNetWrapper",
    "ConvexResNetWrapperBivariate",
    
    # GLU layers
    "GLUBlock",
    
    # MLP layers
    "MLPBlock",
    
    # Normalization layers
    "WeakLayerNorm",
    
    # Quadratic layers
    "QuadraticLayer",
    "QuadraticBlock",
    "QuadraticProjectionLayer",
    "QuadraticProjectionBlock",
    
    # ResNet wrappers
    "ResNetWrapper",
    "ResNetWrapperBivariate",
]
