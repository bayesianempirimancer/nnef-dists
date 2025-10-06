"""
Flow field layer for continuous ode and continuous time flow flow models.

Flow fields net are that operate on a state vector (z), time (t), and 
an input or driving force (u).  The output is a change in the state vector.
In a continuous time setting we think of a NN flow field as playing the role of 
phi(.) in equation dz/dt = phi(z,u,t)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple, Callable
from .mlp import MLPBlock
from .concatsquash import ConcatSquash
from ..embeddings.time_embeddings import LogFreqTimeEmbedding

class FlowFieldMLP(nn.Module):
    """
    Flow field net for continuous ode and continuous time flow flow models.
    """
    dim: int
    features: [int, ...]
    t_embed_dim: int 
    t_embedding_fn: Callable = LogFreqTimeEmbedding
    activation: Callable = nn.swish # because it is fun to say
    use_layer_norm: bool = False
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: float) -> jnp.ndarray:
        """
        Compute flow field dz/dt using an MLP network.
        
        Args:
            z: State vector with shape (..., dim_z) - this shape is preserved
            x: Input/driving force with shape (..., dim_x) where ... must be broadcast-compatible with z's batch shape
            t: Time scalar
            
        Returns:
            Flow field output with shape (..., dim) where ... matches z's batch shape
        """
        batch_shape = z.shape[:-1]
        # Try to broadcast x's shape to z's batch shape
        try:
            x_broadcasted = jnp.broadcast_to(x, batch_shape + x.shape[-1:])
        except ValueError as e:
            raise AssertionError(f"x must be broadcast-compatible with z's batch shape. "
                               f"z.shape[:-1] = {batch_shape}, x.shape[:-1] = {x.shape[:-1]}. "
                               f"Broadcast error: {e}")
                               
        # Time embedding must broadcast to the batch shape
        t_embed = self.t_embedding_fn(embed_dim=self.t_embed_dim)(t)
        t_embed_batch = jnp.broadcast_to(t_embed, batch_shape + (self.t_embed_dim,))

        # Broadcast x to match z's batch shape (z keeps its original shape)
        x_broadcasted = jnp.broadcast_to(x, batch_shape + (x.shape[-1],))
        
        # Use ConcatSquash for efficient multi-input processing
        output = ConcatSquash(self.features[0])(z, x_broadcasted, t_embed_batch)
        output = self.activation(output)

        if len(self.features) > 1:
            mlp = MLPBlock(
                features=self.features[1:],
                activation=self.activation,
                use_layer_norm=self.use_layer_norm,
                dropout_rate=self.dropout_rate
            )
            output=mlp(output)
        output = nn.Dense(self.dim, name='flow_field_output')(output)
        return output

class FisherFlowFieldMLP(nn.Module):
    """
    A Fisher flow field relates the flow of natural parameters to the flow of the expectations of the sufficient statistics via the 
    equation $d\\mu/dt = F(\\eta) d\\eta/dt$, where $F(\\eta) = \\Sigma_{TT}(\\eta)$ is both the Fisher information matrix and the covariance 
    of the associated sufficient statistics.  Rather than compute $\\Sigma_{TT}(\\eta)$ via sampling, we can parameterize an approximate flow field
    that has the right structure.  Note that this approach requires an additional input deta_dt, hence the input is (z, x, t, deta_dt)
    """
    dim: int  # the dimension of the natural parameters (could be inferred from z and deta_dt)
    features: [int, ...]
    t_embed_dim: int 
    matrix_rank: Optional[int] = None
    t_embedding_fn: Callable = LogFreqTimeEmbedding
    activation: Callable = nn.swish # because it is fun to say
    use_layer_norm: bool = False
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray, deta_dt: jnp.ndarray, training: bool = True, rngs: dict = None) -> jnp.ndarray:
        """
        Compute Fisher flow field using an MLP network.
        
        Args:
            z: State vector with shape (..., dim)
            x: Input/driving force with shape batch_shape + (dim_x,) or broadcast-compatible with z's batch shape
            t: Time scalar or array with batch_shape
            deta_dt: Natural parameter derivative with shape (..., dim)
            training: Whether in training mode
            rngs: Random number generators for dropout
            
        Returns:
            Fisher flow field output with shape (..., dim)
        """
        # Extract and embed inputs with proper broadcasting
        # Handle both scalar and array time inputs
        t_shape = t.shape if hasattr(t, 'shape') else ()
        batch_shape = jnp.broadcast_shapes(z.shape[:-1], x.shape[:-1], t_shape)
        
        # Create time embedding - optimized version
        # The time embedding functions already support batched input, expects t.shape = batch_shape 
        t_embed = self.t_embedding_fn(embed_dim=self.t_embed_dim)(t)
        
        # Optimize broadcasting - only broadcast if necessary
        if t_embed.shape[:-1] != batch_shape:
            t_embed = jnp.broadcast_to(t_embed, batch_shape + (self.t_embed_dim,))
        
        # Only broadcast if shapes don't already match
        if x.shape[:-1] != batch_shape:
            x = jnp.broadcast_to(x, batch_shape + (x.shape[-1],))
        if z.shape[:-1] != batch_shape:
            z = jnp.broadcast_to(z, batch_shape + (z.shape[-1],))

        # Use ConcatSquash for efficient multi-input processing
        output = ConcatSquash(self.features[0])(z, x, t_embed)
        output = self.activation(output)

        if len(self.features) > 1:
            mlp = MLPBlock(
                features=self.features[1:],
                activation=self.activation,
                use_layer_norm=self.use_layer_norm,
                dropout_rate=self.dropout_rate
            )
            output = mlp(output, training=training, rngs=rngs)
            
        matrix_rank = self.dim if self.matrix_rank is None else self.matrix_rank
        
        output = nn.Dense(self.dim*matrix_rank, name='flow_field_output')(output)
        output = output.reshape(batch_shape + (self.dim, matrix_rank))
        
        # Optimized: Compute (output @ output.T) @ deta_dt in one step
        # This avoids creating the intermediate fisher_matrix tensor
        result = (output @ output.mT) @ deta_dt[..., None]  # Shape: (..., dim, 1)
        return result.squeeze(-1)  # Shape: (..., dim)

class GeodesicFlowFieldMLP(nn.Module):
    """
    Geodesic flow relates to the Riemanian metric on the space of natural parameters.  In the hamiltonian fomrulation of a geodesic flow
    The momentum is generated by a kinetic term that depends upon the spatial coordinate, $H = 1/2 p^T g^{-1}(x) p$, where $g(x)$ is the 
    metric. In the neuralized version $g(x)$ is a matrix paramteriszed by a neural network and the flow field is given by 
    
    $dx/dt = g^{-1}(x) p$
    $dp/dt = -p^T\\nabla g^{-1}(x) p$ or equivalently $dp_k/dt = -\\sum _{ij} p_i d/dx_k g^{-1}(x) p_j$
    
    Here we assume that z = (x, p)$
    """
    dim: int # the dimension of x so z.shape[-1] = dim + dim
    features: [int, ...]
    t_embed_dim: int
    t_embedding_fn: Callable = LogFreqTimeEmbedding
    activation: Callable = nn.swish # because it is fun to say
    use_layer_norm: bool = False
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, z: jnp.ndarray, x: jnp.ndarray, t: float) -> jnp.ndarray:
        # Extract and embed inputs with proper broadcasting
        q, p, t_embed, batch_shape = self._extract_and_embed(z, x, t)

        # Create the network components once
        concat_squash = ConcatSquash(self.features[0])
        if len(self.features) > 1:
            mlp = MLPBlock(features=self.features[1:], 
                        activation=self.activation, 
                        use_layer_norm=self.use_layer_norm, 
                        dropout_rate=self.dropout_rate)
        else:
            mlp = None
        dense_layer = nn.Dense(self.dim*self.dim, name='ginv_dense')
        
        # Define the single Ginv function once
        def single_ginv_fn(q_single, p_single, t_embed_single):
            ginv = concat_squash(q_single, p_single, t_embed_single)
            ginv = self.activation(ginv)
            if mlp is not None:
                ginv = mlp(ginv)
            ginv = dense_layer(ginv)
            # Reshape to matrix form (works for both single and batched inputs)
            ginv = ginv.reshape(ginv.shape[:-1] + (self.dim, self.dim))
            ginv = ginv@ginv.mT
            return ginv.reshape(ginv.shape[:-2] + (-1,))  # Return flattened matrix
        
        # Compute Ginv directly (single_ginv_fn already handles batches)
        Ginv = single_ginv_fn(q, p, t_embed).reshape(batch_shape + (self.dim, self.dim))
        
        # Compute gradient of Ginv with respect to q using vmap
        # Use jax.jit to cache the compiled jacobian function and avoid recompilation
        @jax.jit
        def compute_gradients(q_batch, p_batch, t_embed_batch):
            jacobian_fn = jax.jacobian(single_ginv_fn, argnums=0)
            return jax.vmap(jacobian_fn)(q_batch, p_batch, t_embed_batch)
        
        gradGinv = compute_gradients(q, p, t_embed)
        gradGinv = gradGinv.reshape(batch_shape + (self.dim, self.dim, self.dim))

        dqdt = jnp.einsum('...ij,...j->...i', Ginv, p)
        dpdt = -jnp.einsum('...i,...j,...ijk->...k', p, p, gradGinv)

        return jnp.concatenate([dqdt, dpdt], axis=-1)
        
    @nn.compact
    def _extract_and_embed(self, z: jnp.ndarray, x: jnp.ndarray, t: float) -> tuple:

        assert z.shape[-1]%2 == 0, "Primary input to GeodesicFlowFlield must be even-dimensional"
        q = z[..., :self.dim]  # position part of z
        p = z[..., self.dim:]  # momentum part of z
        batch_shape = jnp.broadcast_shapes(z.shape[:-1], x.shape[:-1])
        t_embed = self.t_embedding_fn(embed_dim=self.t_embed_dim)(t)
        t_embed = jnp.broadcast_to(t_embed, batch_shape + (self.t_embed_dim,))
        return q, p, t_embed, batch_shape

    @nn.compact
    def get_inverse_metric(self, z: jnp.ndarray, x: jnp.ndarray, t: float) -> jnp.ndarray:
        """
        Helper function to compute just the inverse metric matrix Ginv for a batch of inputs.
        This reuses the same network structure as the main __call__ method.
        
        Args:
            z: State vector with shape (..., 2*dim) where first dim elements are position q, last dim elements are momentum p
            x: External input with shape (..., dim_x) where ... must be broadcast-compatible with z's batch shape
            t: Time scalar
            
        Returns:
            Inverse metric matrix Ginv with shape (..., dim, dim)
        """
        # Extract and embed inputs with proper broadcasting
        q, p, t_embed, batch_shape = self._extract_and_embed(z, x, t)

        # Create the same network components as in __call__
        concat_squash = ConcatSquash(self.features[0])
        if len(self.features) > 1:
            mlp = MLPBlock(features=self.features[1:], 
                        activation=self.activation, 
                        use_layer_norm=self.use_layer_norm, 
                        dropout_rate=self.dropout_rate)
        else:
            mlp = None
        dense_layer = nn.Dense(self.dim*self.dim, name='ginv_dense')
        
        # Follow the format of the same single Ginv function as in __call__
        ginv = concat_squash(q, p, t_embed)
        ginv = self.activation(ginv)
        if mlp is not None:
            ginv = mlp(ginv)
        ginv = dense_layer(ginv)
        ginv = ginv.reshape(batch_shape + (self.dim, self.dim))
        return ginv@ginv.mT
        



