"""
Utilities module for neural network models.

This module provides utility functions and classes for:
- Gradient and Hessian computation
- Other model-related utilities
"""

from .gradient_hessian_utils import (
    GradientHessianUtils,
    LogNormalizerDerivatives,
    create_gradient_utils,
    create_log_normalizer_derivatives,
    compile_gradient_function,
    compile_hessian_function
)

__all__ = [
    'GradientHessianUtils',
    'LogNormalizerDerivatives', 
    'create_gradient_utils',
    'create_log_normalizer_derivatives',
    'compile_gradient_function',
    'compile_hessian_function'
]
