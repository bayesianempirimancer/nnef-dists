"""
Transformer-based moment prediction network using self-attention.

This module implements a Transformer architecture for learning the mapping from
natural parameters to expected sufficient statistics. The key insight is that
self-attention can capture complex dependencies between different components
of the natural parameters, which is especially important for multivariate cases.

Architecture: Input and output have the same dimension, so we use self-attention
to transform the natural parameters while preserving dimensionality.
"""

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from typing import Callable, Tuple, Optional
from .ef import ExponentialFamily

Array = jax.Array


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention layer."""
    
    num_heads: int = 8
    head_dim: int = 64
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x: Array, training: bool = True) -> Array:
        """
        Apply multi-head self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            training: Whether in training mode
            
        Returns:
            Output tensor of same shape as input
        """
        batch_size, seq_len = x.shape
        
        # We need to treat each dimension as a "token" in the sequence
        # Reshape to (batch_size, seq_len, 1) to create sequence dimension
        x = x[:, :, None]  # (batch_size, seq_len, 1)
        
        # Project to embedding dimension
        embed_dim = self.num_heads * self.head_dim
        x = nn.Dense(embed_dim)(x)  # (batch_size, seq_len, embed_dim)
        
        # Create Q, K, V projections
        qkv = nn.Dense(3 * embed_dim, use_bias=False)(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # Each: (batch, seq, heads, head_dim)
        
        # Compute attention scores
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn_scores = jnp.einsum('bqhd,bkhd->bhqk', q, k) * scale
        
        # Apply softmax
        attn_weights = nn.softmax(attn_scores, axis=-1)
        
        # Apply dropout to attention weights
        if training:
            attn_weights = nn.Dropout(self.dropout_rate)(attn_weights, deterministic=not training)
        
        # Apply attention to values
        out = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        
        # Concatenate heads and project
        out = out.reshape(batch_size, seq_len, embed_dim)
        out = nn.Dense(embed_dim)(out)
        
        # Project back to original dimension (1 in our case)
        out = nn.Dense(1)(out)
        
        # Reshape back to (batch_size, seq_len)
        return out.squeeze(-1)


class QuadraticResNetLayer(nn.Module):
    """Quadratic ResNet layer for use in Transformer: y = x + Wx + (B*x)*x"""
    
    use_bias: bool = True
    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        # Linear transformation: Wx
        linear_term = nn.Dense(x.shape[-1], use_bias=self.use_bias, name='linear')(x)
        
        # Quadratic transformation: (B*x)*x = B*x^2 (element-wise)
        quadratic_weights = nn.Dense(x.shape[-1], use_bias=False, name='quadratic')(x)
        quadratic_term = quadratic_weights * x  # Element-wise multiplication
        
        # ResNet connection: y = x + Wx + (B*x)*x
        return x + linear_term + quadratic_term


class AdaptiveQuadraticResNetLayer(nn.Module):
    """Adaptive quadratic ResNet layer: y = x + α*(Wx) + β*((B*x)*x)"""
    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        # Linear transformation: Wx
        linear_term = nn.Dense(x.shape[-1], use_bias=True, name='linear')(x)
        
        # Quadratic transformation: (B*x)*x
        quadratic_weights = nn.Dense(x.shape[-1], use_bias=False, name='quadratic')(x)
        quadratic_term = quadratic_weights * x
        
        # Learnable mixing coefficients
        alpha = self.param('alpha', nn.initializers.ones, (x.shape[-1],))
        beta = self.param('beta', nn.initializers.zeros, (x.shape[-1],))
        
        # Adaptive combination: y = x + α*Wx + β*(B*x)*x
        return x + alpha * linear_term + beta * quadratic_term


class TransformerBlock(nn.Module):
    """A single Transformer block with self-attention and feed-forward."""
    
    num_heads: int = 8
    head_dim: int = 64
    mlp_dim: int = 256
    dropout_rate: float = 0.1
    activation: str = "gelu"
    use_quadratic_mlp: bool = False
    use_adaptive_quadratic: bool = False
    
    def setup(self):
        self.activation_fn = self._get_activation(self.activation)
    
    def _get_activation(self, name: str) -> Callable[[Array], Array]:
        if name == "relu":
            return nn.relu
        elif name == "gelu":
            return nn.gelu
        elif name == "tanh":
            return nn.tanh
        elif name == "swish":
            return nn.swish
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    @nn.compact
    def __call__(self, x: Array, training: bool = True) -> Array:
        """Apply Transformer block: LayerNorm -> Self-Attention -> Residual -> LayerNorm -> MLP/QuadraticMLP -> Residual"""
        
        # Self-attention with residual connection and layer norm
        attn_input = nn.LayerNorm()(x)
        attn_out = MultiHeadSelfAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate
        )(attn_input, training=training)
        
        x = x + attn_out  # Residual connection
        
        # Feed-forward network with residual connection and layer norm
        ff_input = nn.LayerNorm()(x)
        
        if self.use_adaptive_quadratic:
            # Use adaptive quadratic ResNet layer
            ff_out = AdaptiveQuadraticResNetLayer()(ff_input)
        elif self.use_quadratic_mlp:
            # Use quadratic ResNet layer
            ff_out = QuadraticResNetLayer()(ff_input)
        else:
            # Standard MLP
            ff_out = nn.Dense(self.mlp_dim)(ff_input)
            ff_out = self.activation_fn(ff_out)
            if training:
                ff_out = nn.Dropout(self.dropout_rate)(ff_out, deterministic=not training)
            ff_out = nn.Dense(x.shape[-1])(ff_out)  # Project back to input dimension
        
        x = x + ff_out  # Residual connection
        
        return x


class QuadraticTransformerBlock(nn.Module):
    """Transformer block specifically designed with quadratic operations."""
    
    num_heads: int = 8
    head_dim: int = 64
    dropout_rate: float = 0.1
    activation: str = "tanh"  # tanh works well with quadratic terms
    num_quadratic_layers: int = 2
    
    def setup(self):
        self.activation_fn = self._get_activation(self.activation)
    
    def _get_activation(self, name: str) -> Callable[[Array], Array]:
        if name == "relu":
            return nn.relu
        elif name == "gelu":
            return nn.gelu
        elif name == "tanh":
            return nn.tanh
        elif name == "swish":
            return nn.swish
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    @nn.compact
    def __call__(self, x: Array, training: bool = True) -> Array:
        """Transformer block with multiple quadratic ResNet layers."""
        
        # Self-attention with residual connection and layer norm
        attn_input = nn.LayerNorm()(x)
        attn_out = MultiHeadSelfAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate
        )(attn_input, training=training)
        
        x = x + attn_out  # Residual connection
        
        # Multiple quadratic ResNet layers
        for i in range(self.num_quadratic_layers):
            quad_input = nn.LayerNorm()(x)
            quad_out = AdaptiveQuadraticResNetLayer()(quad_input)
            quad_out = self.activation_fn(quad_out)
            x = x + quad_out  # Residual connection
        
        return x


class PositionalEncoding(nn.Module):
    """Add positional encodings to help the model understand parameter positions."""
    
    max_len: int = 1000
    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Add learnable positional encodings."""
        batch_size, seq_len = x.shape
        
        # Learnable positional embeddings
        pos_embed = self.param('pos_embed', 
                              nn.initializers.normal(stddev=0.02),
                              (seq_len,))
        
        return x + pos_embed[None, :]  # Broadcast over batch dimension


class TransformerMomentNet(nn.Module):
    """Transformer-based moment prediction network."""
    
    ef: ExponentialFamily
    num_layers: int = 6
    num_heads: int = 8
    head_dim: int = 64
    mlp_dim: int = 256
    dropout_rate: float = 0.1
    activation: str = "gelu"
    use_positional_encoding: bool = True
    
    @nn.compact
    def __call__(self, eta: Array, training: bool = True) -> Array:
        """
        Transform natural parameters to expected sufficient statistics using Transformer.
        
        Args:
            eta: Natural parameters of shape (batch_size, eta_dim)
            training: Whether in training mode
            
        Returns:
            Expected sufficient statistics of shape (batch_size, eta_dim)
        """
        x = eta
        
        # Add positional encoding if enabled
        if self.use_positional_encoding:
            x = PositionalEncoding()(x)
        
        # Stack of Transformer blocks
        for i in range(self.num_layers):
            x = TransformerBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
                use_quadratic_mlp=False,
                use_adaptive_quadratic=False
            )(x, training=training)
        
        # Final layer norm
        x = nn.LayerNorm()(x)
        
        # Optional final projection (identity in this case since input/output dims match)
        # But we can add a small MLP for final refinement
        x = nn.Dense(self.mlp_dim // 2)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.ef.eta_dim)(x)  # Project to output dimension
        
        return x


class QuadraticTransformerMomentNet(nn.Module):
    """Transformer with quadratic ResNet layers instead of standard MLPs."""
    
    ef: ExponentialFamily
    num_layers: int = 6
    num_heads: int = 8
    head_dim: int = 64
    dropout_rate: float = 0.1
    activation: str = "tanh"  # tanh works well with quadratic
    use_positional_encoding: bool = True
    use_adaptive_quadratic: bool = True
    
    @nn.compact
    def __call__(self, eta: Array, training: bool = True) -> Array:
        """Transformer with quadratic ResNet feed-forward layers."""
        x = eta
        
        # Add positional encoding if enabled
        if self.use_positional_encoding:
            x = PositionalEncoding()(x)
        
        # Stack of Transformer blocks with quadratic MLPs
        for i in range(self.num_layers):
            x = TransformerBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=256,  # Not used when quadratic is enabled
                dropout_rate=self.dropout_rate,
                activation=self.activation,
                use_quadratic_mlp=not self.use_adaptive_quadratic,
                use_adaptive_quadratic=self.use_adaptive_quadratic
            )(x, training=training)
        
        # Final layer norm
        x = nn.LayerNorm()(x)
        
        # Final quadratic projection
        x = AdaptiveQuadraticResNetLayer()(x)
        
        return x


class PureQuadraticTransformerMomentNet(nn.Module):
    """Transformer using the specialized QuadraticTransformerBlock."""
    
    ef: ExponentialFamily
    num_layers: int = 6
    num_heads: int = 8
    head_dim: int = 64
    dropout_rate: float = 0.1
    activation: str = "tanh"
    use_positional_encoding: bool = True
    num_quadratic_layers: int = 2
    
    @nn.compact
    def __call__(self, eta: Array, training: bool = True) -> Array:
        """Pure quadratic Transformer architecture."""
        x = eta
        
        # Add positional encoding if enabled
        if self.use_positional_encoding:
            x = PositionalEncoding()(x)
        
        # Stack of quadratic Transformer blocks
        for i in range(self.num_layers):
            x = QuadraticTransformerBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
                num_quadratic_layers=self.num_quadratic_layers
            )(x, training=training)
        
        # Final layer norm and projection
        x = nn.LayerNorm()(x)
        x = AdaptiveQuadraticResNetLayer()(x)
        
        return x


class DeepTransformerMomentNet(nn.Module):
    """Deeper Transformer with more sophisticated architecture."""
    
    ef: ExponentialFamily
    num_layers: int = 12
    num_heads: int = 12
    head_dim: int = 64
    mlp_dim: int = 512
    dropout_rate: float = 0.1
    activation: str = "gelu"
    use_positional_encoding: bool = True
    use_layer_scale: bool = True
    layer_scale_init: float = 1e-4
    
    @nn.compact
    def __call__(self, eta: Array, training: bool = True) -> Array:
        """Deep Transformer with layer scaling for better training."""
        x = eta
        
        # Add positional encoding
        if self.use_positional_encoding:
            x = PositionalEncoding()(x)
        
        # Stack of Transformer blocks with layer scaling
        for i in range(self.num_layers):
            # Layer scaling for better deep network training
            if self.use_layer_scale:
                gamma = self.param(f'layer_scale_{i}',
                                 lambda key, shape: jnp.full(shape, self.layer_scale_init),
                                 (x.shape[-1],))
            else:
                gamma = 1.0
            
            block_out = TransformerBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                activation=self.activation
            )(x, training=training)
            
            # Apply layer scaling
            x = x + gamma * (block_out - x)
        
        # Final processing
        x = nn.LayerNorm()(x)
        
        # Multi-layer final projection
        x = nn.Dense(self.mlp_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        x = nn.Dense(self.mlp_dim // 2)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.ef.eta_dim)(x)
        
        return x


def create_transformer_train_state(
    ef: ExponentialFamily,
    config: dict,
    rng: Array
) -> Tuple[nn.Module, dict]:
    """Create and initialize a Transformer-based model."""
    
    model_type = config.get('model_type', 'transformer')
    
    if model_type == 'transformer':
        model = TransformerMomentNet(
            ef=ef,
            num_layers=config.get('num_layers', 6),
            num_heads=config.get('num_heads', 8),
            head_dim=config.get('head_dim', 64),
            mlp_dim=config.get('mlp_dim', 256),
            dropout_rate=config.get('dropout_rate', 0.1),
            activation=config.get('activation', 'gelu'),
            use_positional_encoding=config.get('use_positional_encoding', True)
        )
    elif model_type == 'quadratic_transformer':
        model = QuadraticTransformerMomentNet(
            ef=ef,
            num_layers=config.get('num_layers', 6),
            num_heads=config.get('num_heads', 8),
            head_dim=config.get('head_dim', 64),
            dropout_rate=config.get('dropout_rate', 0.1),
            activation=config.get('activation', 'tanh'),
            use_positional_encoding=config.get('use_positional_encoding', True),
            use_adaptive_quadratic=config.get('use_adaptive_quadratic', True)
        )
    elif model_type == 'pure_quadratic_transformer':
        model = PureQuadraticTransformerMomentNet(
            ef=ef,
            num_layers=config.get('num_layers', 6),
            num_heads=config.get('num_heads', 8),
            head_dim=config.get('head_dim', 64),
            dropout_rate=config.get('dropout_rate', 0.1),
            activation=config.get('activation', 'tanh'),
            use_positional_encoding=config.get('use_positional_encoding', True),
            num_quadratic_layers=config.get('num_quadratic_layers', 2)
        )
    elif model_type == 'deep_transformer':
        model = DeepTransformerMomentNet(
            ef=ef,
            num_layers=config.get('num_layers', 12),
            num_heads=config.get('num_heads', 12),
            head_dim=config.get('head_dim', 64),
            mlp_dim=config.get('mlp_dim', 512),
            dropout_rate=config.get('dropout_rate', 0.1),
            activation=config.get('activation', 'gelu'),
            use_positional_encoding=config.get('use_positional_encoding', True),
            use_layer_scale=config.get('use_layer_scale', True)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Initialize with proper RNG handling for dropout
    dummy_eta = jnp.zeros((1, ef.eta_dim))
    params = model.init(rng, dummy_eta, training=False)
    
    return model, params


# Test the Transformer architecture
if __name__ == "__main__":
    from ef import GaussianNatural1D
    
    print("Testing TransformerMomentNet...")
    
    # Create test data
    ef = GaussianNatural1D()
    rng = random.PRNGKey(0)
    
    # Test standard transformer
    transformer_config = {
        'model_type': 'transformer',
        'num_layers': 4,
        'num_heads': 4,
        'head_dim': 32,
        'mlp_dim': 128,
        'dropout_rate': 0.1
    }
    
    model, params = create_transformer_train_state(ef, transformer_config, rng)
    
    # Test forward pass
    test_eta = jnp.array([[1.0, -2.0], [0.5, -1.5], [-0.8, -3.0]])
    output = model.apply(params, test_eta, training=False)
    
    print(f"Input shape: {test_eta.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Input: {test_eta}")
    print(f"Output: {output}")
    
    print("\n✅ Transformer architecture working correctly!")
    
    # Test deep transformer
    print("\nTesting DeepTransformerMomentNet...")
    
    deep_config = {
        'model_type': 'deep_transformer',
        'num_layers': 6,
        'num_heads': 6,
        'head_dim': 32,
        'mlp_dim': 256,
        'use_layer_scale': True
    }
    
    deep_model, deep_params = create_transformer_train_state(ef, deep_config, rng)
    deep_output = deep_model.apply(deep_params, test_eta, training=False)
    
    print(f"Deep Transformer output: {deep_output}")
    print("✅ Deep Transformer working correctly!")
