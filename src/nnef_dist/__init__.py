"""nnef_dist: Neural networks for non-named exponential family inference."""

from .ef import PolynomialEF1D, log_unnormalized_density
from .model import MomentMLP

__all__ = [
    "PolynomialEF1D",
    "log_unnormalized_density",
    "MomentMLP",
]

__version__ = "0.0.1"


