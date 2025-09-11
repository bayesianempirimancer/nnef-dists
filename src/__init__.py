"""nnef_dist: Neural networks for non-named exponential family inference."""

from ef import ExponentialFamily, GaussianNatural1D, MultivariateNormal, ef_factory
from model import MomentMLP

__all__ = [
    "ExponentialFamily",
    "GaussianNatural1D",
    "MultivariateNormal",
    "ef_factory",
    "MomentMLP",
]

__version__ = "0.0.1"


