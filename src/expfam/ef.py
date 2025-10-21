from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Union, Optional
from functools import cached_property

import jax
import jax.numpy as jnp
from .ef_base import ExponentialFamily

Array = jax.Array


@dataclass(frozen=True)
class Gaussian(ExponentialFamily):
    """
    Univariate Gaussian in natural parameterization with T(x) = [x, x^2].
    log p(x | eta) ∝ eta1 * x + eta2 * x^2  (base measure constant 0)
    Integrability requires eta2 < 0.
    """
    x_shape: Tuple[int, ...] = (1,)
    name: str = "gaussian_1d"

    @cached_property
    def stat_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {"x": (1,), "x2": (1,)}

    def _compute_stats(self, x: Array) -> Dict[str, Array]:
        return {"x": x, "x2": x ** 2}

    def eta_bounds(self) -> Optional[Dict[str, Tuple[float, float]]]:
        """Gaussian constraints: x unbounded, x2 < 0 (so x2 bounded above by 0)"""
        return {
            "x": (-jnp.inf, jnp.inf),   # x coefficient unbounded
            "x2": (-jnp.inf, -0.5)      # x2 coefficient < 0, with some margin
        }


@dataclass(frozen=True)
class RectifiedGaussian(ExponentialFamily):
    """
    Univariate Rectified Gaussian in natural parameterization with T(x) = [x, x^2].
    log p(x | eta) ∝ eta1 * x + eta2 * x^2  (base measure constant 0)
    Integrability requires eta2 < 0.
    Constraint: x >= 0 (rectified to non-negative values).
    """
    x_shape: Tuple[int, ...] = (1,)
    name: str = "rectified_gaussian"

    @cached_property
    def stat_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {"x": (1,), "x2": (1,)}

    def _compute_stats(self, x: Array) -> Dict[str, Array]:
        return {"x": x, "x2": x ** 2}
    
    @property
    def x_bounds(self) -> Optional[Tuple[Array, Array]]:
        """No bounds for rectified Gaussian"""
        return None
    
    @property
    def x_constraint_fn(self) -> Callable[[Array], bool]:
        """Constraint function for rectified Gaussian: x >= 0"""
        return lambda x: x >= 0
    
    def eta_bounds(self) -> Optional[Dict[str, Tuple[float, float]]]:
        """RectifiedGaussian constraints: x > -1, x2 < 0 (so x2 bounded above by 0)"""
        return {
            "x": (-1.0, jnp.inf),       # x coefficient > -1
            "x2": (-jnp.inf, -0.5)      # x2 coefficient < 0, with some margin
        }


@dataclass(frozen=True)
class MultivariateNormal(ExponentialFamily):
    """
    Multivariate normal in natural parameterization with T(x) = [x, x^T x].
    log p(x | eta) ∝ eta1 . x + eta2 . (x *x^T)
    Dictionary keys are sensibly named 'x' and 'xxT' and x is assumed to be in vector format, 

    Integrability requires eta2 to be negative definite.  
    """
    x_shape: Tuple[int,]
    name: str = "multivariate_normal"

    @cached_property
    def stat_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {"x": (self.x_shape[-1],), "xxT": (self.x_shape[-1], self.x_shape[-1])}

    def _compute_stats(self, x: Array) -> Dict[str, Array]:
        return {"x": x, "xxT": x[...,None] * x[...,None,:]}

    def _reparam_eta(self, eta: Dict[str,Array]) -> Dict[str,Array]:
        norm = jnp.sqrt(jnp.sum(eta['xxT']**2, axis = (-2,-1), keepdims=True))
        eta2 = -eta['xxT']@eta['xxT'].mT/eta['xxT'].shape[-1]/norm
        return {'x': eta['x'], 'xxT': eta2 - jnp.eye(eta['xxT'].shape[-1])*0.5}
    
    def find_nearest_analytical_point(self, eta_target: Union[Array, Dict[str, Array]]) -> Tuple[Union[Array, Dict[str, Array]], Union[Array, Dict[str, Array]]]:
        """
        Find the nearest analytical reference point (η₀, μ₀) for flow-based computation.
        
        For multivariate Gaussian, we use the same mean but diagonal covariance matrix.
        This gives us a point where μ₀ can be computed analytically while being close to the target.
        
        Args:
            eta_target: Target natural parameters (flattened array or dict)
            
        Returns:
            Tuple of (eta_0, mu_0) in the same format as input:
                - If input is array: returns (eta_0_array, mu_0_array)
                - If input is dict: returns (eta_0_dict, mu_0_dict)
        """
        # Check if input is array or dict
        input_is_array = isinstance(eta_target, (jnp.ndarray, Array))
        
        # Convert to dict format for processing
        eta_target_dict = self.unflatten_stats_or_eta(eta_target)
        
        eta1_nearest = eta_target_dict['x']  # Linear part [batch_size, d] or [d,]
        eta2_matrix = eta_target_dict['xxT']  # [batch_size, d, d] or [d, d]
        
        # Handle batch dimension for diagonal extraction
        if eta2_matrix.ndim >= 3:  # Batch case: [..., d, d]
            # Get the last two dimensions for diagonal extraction
            eta2_nearest_diag = jnp.diagonal(eta2_matrix, axis1=-2, axis2=-1)  # [..., d]
        else:  # Single case: [d, d]
            eta2_nearest_diag = jnp.diag(eta2_matrix)  # [d,]
        
        Sigma_target_diag = -0.5 / eta2_nearest_diag
        mu_nearest = Sigma_target_diag * eta1_nearest

        # Handle batch dimension for matrix construction
        if eta2_matrix.ndim >= 3:  # Batch case
            d = eta2_matrix.shape[-1]  # Get the dimension from the last axis
            # Create diagonal matrices for arbitrary batch dimensions
            # Use einsum with ellipsis to handle arbitrary batch dimensions
            Sigma_diag_matrices = jnp.einsum('...i,ij->...ij', Sigma_target_diag, jnp.eye(d))
            eta2_diag_matrices = jnp.einsum('...i,ij->...ij', eta2_nearest_diag, jnp.eye(d))
            mu_outer_products = jnp.einsum('...i,...j->...ij', mu_nearest, mu_nearest)
            
            mu_nearest_dict = {
                'x': mu_nearest,
                'xxT': Sigma_diag_matrices + mu_outer_products
            }
            eta_nearest_dict = {
                'x': eta1_nearest,
                'xxT': eta2_diag_matrices
            }
        else:  # Single case
            mu_nearest_dict = {
                'x': mu_nearest,
                'xxT': jnp.diag(Sigma_target_diag) + jnp.outer(mu_nearest, mu_nearest)
            }
            eta_nearest_dict = {
                'x': eta1_nearest,
                'xxT': jnp.diag(eta2_nearest_diag)
            }

        # Return in the same format as input
        if input_is_array:
            eta_0_array = self.flatten_stats_or_eta(eta_nearest_dict)
            mu_0_array = self.flatten_stats_or_eta(mu_nearest_dict)
            return eta_0_array, mu_0_array
        else:
            return eta_nearest_dict, mu_nearest_dict
    

@dataclass(frozen=True)
class MultivariateNormal_tril(MultivariateNormal):
    """
    Multivariate normal in natural parameterization with T(x) = [x, lower_triangular(xx^T)].
    This avoids overparameterization by storing only the lower triangular part of xx^T.
    Uses standard lower triangular convention compatible with JAX/NumPy.
    
    Simplified implementation: inherits _compute_stats from MultivariateNormal (full xxT),
    then uses _reparam_eta to ensure negative definite matrices, and only converts to
    lower triangular format for storage efficiency.
    """
    x_shape: Tuple[int,]
    name: str = "multivariate_normal_tril"

    def _reparam_eta(self, eta: Dict[str,Array]) -> Dict[str,Array]:
        # Convert lower triangular to full matrix for internal processing
        eta_xxT_full = self.unflatten_LT(eta['xxT_tril'])
        
        # Use parent's reparam method with full matrix
        eta_full_dict = {'x': eta['x'], 'xxT': eta_xxT_full}
        eta_reparam_full = super()._reparam_eta(eta_full_dict)
        
        # Convert back to lower triangular format for user interface consistency
        eta_xxT_tril_reparam = self.flatten_LT(eta_reparam_full['xxT'])
        
        return {'x': eta['x'], 'xxT_tril': eta_xxT_tril_reparam}

    def eta_bounds(self) -> Optional[Dict[str, Tuple[float, float]]]:
        """Handled by reparam"""
        return None

    @cached_property
    def stat_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {"x": (self.x_shape[-1],), "xxT_tril": (self.x_shape[-1]*(self.x_shape[-1]+1)//2,)}

    @cached_property
    def tril_mask(self) -> Array:
        n = self.x_shape[-1]
        return jnp.tril(jnp.ones((n,n), dtype=bool))
    
    @cached_property
    def tril_indices(self) -> Array:
        return jnp.where(self.tril_mask.flatten())[0]

    def flatten_LT(self, x: Array) -> Array:
        n = self.x_shape[-1]
        batch_shape = x.shape[:-2]
        assert x.shape[-2:] == (n, n), f"Shape mismatch: {x.shape[-2:]} != {n, n}"
        flat_x = x.reshape(-1, n * n)
        flat_x = flat_x[:,self.tril_indices]
        return flat_x.reshape(batch_shape + (n*(n+1)//2,))

    def unflatten_LT(self, x: Array) -> Array:
        n = self.x_shape[-1]
        expected_shape = n*(n+1)//2
        batch_shape = x.shape[:-1]
        assert expected_shape == x.shape[-1], f"Shape mismatch: Terminal shape of {x.shape} != {n*(n+1)//2}"

        flat_x = x.reshape(-1, expected_shape)

        def lt_vec_to_full_matrix(v):
            flat = jnp.zeros((n * n,))
            flat = flat.at[self.tril_indices].set(v)
            return flat.reshape((n, n))

        full_mats = jax.vmap(lt_vec_to_full_matrix)(flat_x)
        return full_mats.reshape(batch_shape + (n, n))

    def _compute_stats(self, x: Array) -> Dict[str, Array]:
        # Compute full xxT matrix
        xxT = x[...,None] * x[...,None,:]
        # Convert to lower triangular format for storage
        return {"x": x, "xxT_tril": self.flatten_LT(xxT)}
        
    def LTnat_to_standard_nat(self, eta: Union[Dict[str, Array], Array]) -> Union[Dict[str, Array], Array]:
        """
        Convert lower triangular natural parameters to standard format.
        
        Args:
            eta: Either a dictionary with 'x' and 'xxT_tril' keys, or a flattened array
            
        Returns:
            Either a dictionary with 'x' and 'xxT' keys, or a flattened array
        """
        if isinstance(eta, dict):
            # Dictionary input - return dictionary output
            eta_unflattened = self.unflatten_stats_or_eta(eta)
            xxT = self.unflatten_LT(eta_unflattened["xxT_tril"])
            xxT = xxT + xxT.T - xxT*jnp.eye(xxT.shape[-1])
            return {"x": eta_unflattened["x"], "xxT": xxT}
        else:
            # Array input - unflatten using tril format, then convert to standard
            eta_dict = self.unflatten_stats_or_eta(eta)
            # Convert to standard format
            eta_standard_dict = self.LTnat_to_standard_nat(eta_dict)
            # Flatten back to array
            from .ef import MultivariateNormal
            mvn_std = MultivariateNormal(x_shape=self.x_shape)
            return mvn_std.flatten_stats_or_eta(eta_standard_dict)

    def standard_nat_to_LTnat(self, eta: Union[Dict[str, Array], Array]) -> Union[Dict[str, Array], Array]:
        """
        Convert standard natural parameters to lower triangular format.
        
        Args:
            eta: Either a dictionary with 'x' and 'xxT' keys, or a flattened array
            
        Returns:
            Either a dictionary with 'x' and 'xxT_tril' keys, or a flattened array
        """
        if isinstance(eta, dict):
            # Dictionary input - return dictionary output
            eta_unflattened = MultivariateNormal(x_shape=self.x_shape).unflatten_stats_or_eta(eta)
            # Use jax.vmap to apply diag operation to each matrix in the batch
            def extract_diag_matrix(matrix):
                return jnp.diag(jnp.diag(matrix))
            
            xxT = 2*eta_unflattened["xxT"] - jax.vmap(extract_diag_matrix)(2*eta_unflattened["xxT"])
            xxT = xxT*self.tril_mask
            return {"x": eta_unflattened["x"], "xxT_tril": self.flatten_LT(eta_unflattened["xxT"])}
        else:
            # Array input - unflatten using standard format, then convert to tril
            eta_dict = MultivariateNormal(x_shape=self.x_shape).unflatten_stats_or_eta(eta)
            # Convert to tril format
            eta_tril_dict = self.standard_nat_to_LTnat(eta_dict)
            # Flatten back to array
            return self.flatten_stats_or_eta(eta_tril_dict)

    def find_nearest_analytical_point(self, eta_target: Union[Array, Dict[str, Array]]) -> Tuple[Dict[str, Array], Dict[str, Array]]:
        """
        Find the nearest analytical reference point (η₀, μ₀) for flow-based computation.
        
        For multivariate Gaussian, we use the same mean but diagonal covariance matrix.
        This gives us a point where μ₀ can be computed analytically while being close to the target.
        """
        eta_standard = self.LTnat_to_standard_nat(eta_target)
        tempdist = MultivariateNormal(x_shape=self.x_shape)
        eta_nearest_dict, mu_nearest_dict = tempdist.find_nearest_analytical_point(eta_standard)
        eta_nearest = self.standard_nat_to_LTnat(eta_nearest_dict)
        mu_nearest = self.standard_nat_to_LTnat(mu_nearest_dict)

        return eta_nearest, mu_nearest


@dataclass(frozen=True)
class ClippedMultivariateNormal(MultivariateNormal):
    """
    Multivariate normal in natural parameterization with T(x) = [x, diag(x^T x)].
    This avoids overparameterization by storing only the diagonal elements of xx^T.
    """

    x_shape: Tuple[int,]
    name: str = "clipped_multivariate_normal"
        
    @cached_property
    def stat_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {"x": (self.x_shape[-1],), "xxT": (self.x_shape[-1], self.x_shape[-1])}

    def _compute_stats(self, x: Array) -> Dict[str, Array]:
        return {"x": x, "xxT": x[...,None] * x[...,None,:]}

    def eta_bounds(self) -> Optional[Dict[str, Tuple[float, float]]]:
        """Handled by reparam"""
        return None

@dataclass(frozen=True)
class LaplaceProduct(ExponentialFamily):
    """ 
    Laplace product in natural parameterization with T(x) = -abs(x-1), -abs(x+1) where x is a vector.
    """
    x_shape: Tuple[int,]
    name: str = "laplace_product"

    @cached_property
    def stat_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {"xm1": (self.x_shape[-1],), "xp1": (self.x_shape[-1],)}

    def eta_bounds(self) -> Optional[Dict[str, Tuple[float, float]]]:
        """LaplaceProduct constraints: all eta dimensions must be > 0"""
        return {
            "xm1": (0.5, jnp.inf),  # xm1 coefficient > 0
            "xp1": (0.5, jnp.inf)   # xp1 coefficient > 0
        }

    def _compute_stats(self, x: Array) -> Dict[str, Array]:
        return {"xm1": -jnp.abs(x-1), "xp1": -jnp.abs(x+1)}

    def find_nearest_analytical_point(self, eta_target: Union[Array, Dict[str, Array]]) -> Tuple[Dict[str, Array], Dict[str, Array]]:
        """
        Find the nearest analytical reference point (η₀, μ₀) for flow-based computation.
        
        For Laplace product, we use the same mean but diagonal covariance matrix.
        This gives us a point where μ₀ can be computed analytically while being close to the target.
        """
        eta_max = jnp.max(eta_target, axis=-1, keepdims=True)
        idx = jnp.argmax(eta_target, axis=-1)
        from flax.linen import one_hot
        mask = one_hot(idx, eta_target.shape[-1], axis=-1, dtype=bool)
        eta_nearest = eta_target * mask
        mu_nearest = jnp.repeat(1.0/eta_max, eta_target.shape[-1], axis=-1)

        mu_nearest = mu_nearest*mask + (1 - mask) * (2 + mu_nearest*jnp.exp(-2*mu_nearest))
        mu_nearest = -mu_nearest

        # logic: if idx ==0 then mu[:,0] = -1.0/eta_max
        #        if idx ==0 then mu[:,1] = -(2 + 1/eta_max*exp(-2*eta_max))
        #        if idx ==1 then mu[:,0] = -(2 + 1/eta_max*exp(-2*eta_max))
        #        if idx ==1 then mu[:,1] = -1.0/eta_max)
        
        return eta_nearest, mu_nearest


def ef_factory(name: str, **kwargs) -> ExponentialFamily:
    n = name.lower()
    
    if n in {"gaussian_1d", "gauss1d", "gaussian"}:
        return Gaussian()
    elif n in {"mv_normal", "multivariate_normal"}:
        x_shape = kwargs.get("x_shape", (2,))  # default 2D
        # Convert list to tuple if needed
        if isinstance(x_shape, list):
            x_shape = tuple(x_shape)
        return MultivariateNormal(x_shape=x_shape)
    elif n in {"mv_normal_tril", "multivariate_normal_tril"}:
        x_shape = kwargs.get("x_shape", (2,))  # default 2D
        # Convert list to tuple if needed
        if isinstance(x_shape, list):
            x_shape = tuple(x_shape)
        return MultivariateNormal_tril(x_shape=x_shape)
    elif n in {"clipped_mv_normal", "clipped_multivariate_normal"}:
        x_shape = kwargs.get("x_shape", (2,))  # default 2D
        # Convert list to tuple if needed
        if isinstance(x_shape, list):
            x_shape = tuple(x_shape)
        return ClippedMultivariateNormal(x_shape=x_shape)
    elif n in {"laplace_product", "product_laplace"}:
        x_shape = kwargs.get("x_shape", (1,))  # default 1D
        # Convert list to tuple if needed
        if isinstance(x_shape, list):
            x_shape = tuple(x_shape)
        return LaplaceProduct(x_shape=x_shape)
    elif n in {"rectified_gaussian", "rectifiedgaussian"}:
        return RectifiedGaussian()
    raise ValueError(f"Unknown EF name: {name}")

