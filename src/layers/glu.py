"""
GLU (Gated Linear Unit) layer implementations.

This module provides standardized GLU layer implementations that can be used
with ResNet wrappers for residual connections.
"""

import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Callable


class GLULayer(nn.Module):
    """
    Gated Linear Unit (GLU) layer.
    
    Implements the GLU activation: GLU(x) = sigmoid(W1*x) * (W2*x)
    where W1 and W2 are learned linear transformations.
    """
    
    features: int
    use_bias: bool = True
    gate_activation: Callable = nn.sigmoid
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through GLU layer.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            training: Whether in training mode
            
        Returns:
            GLU output [batch_size, features]
        """
        # First linear transformation for gating
        gate = nn.Dense(self.features, use_bias=self.use_bias, name='gate')(x)
        gate = self.gate_activation(gate)
        
        # Second linear transformation for values
        values = nn.Dense(self.features, use_bias=self.use_bias, name='values')(x)
        
        # Apply gating
        output = gate * values
        
        return output


class GLUBlock(nn.Module):
    """
    GLU block containing multiple GLU layers in sequence.
    
    This is a true "block" that can contain multiple layers, similar to MLPBlock.
    Can be used directly or wrapped with ResNetWrapper for residual connections.
    
    Examples:
        GLUBlock(features=(64, 64, 64))  # 3 GLU layers: 64 -> 64 -> 64
        GLUBlock(features=(128, 64))     # 2 GLU layers: 128 -> 64
        GLUBlock(features=(64,))         # 1 GLU layer: 64
    """
    
    features: tuple  # Tuple of feature sizes for each layer in the block
    use_bias: bool = True
    use_layer_norm: bool = True
    dropout_rate: float = 0.0
    gate_activation: Callable = nn.sigmoid
    activation: Optional[Callable] = nn.swish
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through GLU block.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            training: Whether in training mode
            
        Returns:
            GLU block output [batch_size, features[-1]]
        """
        # Apply each GLU layer in the block
        for i, layer_features in enumerate(self.features):
            # Apply GLU layer
            glu_layer = GLULayer(
                features=layer_features,
                use_bias=self.use_bias,
                gate_activation=self.gate_activation,
                name=f'glu_layer_{i}'
            )
            x = glu_layer(x, training=training)
            
            # Apply activation if specified
            if self.activation is not None:
                x = self.activation(x)
            
            # Apply layer normalization if enabled
            if self.use_layer_norm:
                x = nn.LayerNorm(name=f'layer_norm_{i}')(x)
            
            # Apply dropout if enabled
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        return x


# Convenience functions
def create_glu_layer(features: int, 
                    use_bias: bool = True,
                    gate_activation: Callable = nn.sigmoid) -> GLULayer:
    """Create a GLU layer."""
    return GLULayer(
        features=features,
        use_bias=use_bias,
        gate_activation=gate_activation
    )


def create_glu_block(features: tuple,
                    use_bias: bool = True,
                    use_layer_norm: bool = True,
                    dropout_rate: float = 0.0,
                    gate_activation: Callable = nn.sigmoid,
                    activation: Optional[Callable] = nn.swish) -> GLUBlock:
    """Create a GLU block."""
    return GLUBlock(
        features=features,
        use_bias=use_bias,
        use_layer_norm=use_layer_norm,
        dropout_rate=dropout_rate,
        gate_activation=gate_activation,
        activation=activation
    )


# Example usage and testing
if __name__ == "__main__":
    import jax
    
    print("Testing GLU layers:")
    
    # Test parameters
    batch_size = 4
    input_dim = 8
    output_dim = 6
    
    # Create test input
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (batch_size, input_dim))
    
    print(f"Input shape: {x.shape}")
    
    # Test GLU layer
    print("\n1. Testing GLULayer:")
    glu_layer = GLULayer(features=output_dim)
    params = glu_layer.init(key, x)
    output = glu_layer.apply(params, x)
    print(f"GLU layer output shape: {output.shape}")
    print(f"GLU layer output range: [{jnp.min(output):.3f}, {jnp.max(output):.3f}]")
    
    # Test GLU block
    print("\n2. Testing GLUBlock:")
    glu_block = GLUBlock(
        features=output_dim,
        use_layer_norm=True,
        dropout_rate=0.1
    )
    params_block = glu_block.init(key, x)
    output_block = glu_block.apply(params_block, x, training=True)
    print(f"GLU block output shape: {output_block.shape}")
    print(f"GLU block output range: [{jnp.min(output_block):.3f}, {jnp.max(output_block):.3f}]")
    
    # Test with ResNet wrapper
    print("\n3. Testing GLU with ResNetWrapper:")
    from .resnet_wrapper import ResNetWrapper
    
    glu_resnet = ResNetWrapper(
        base_module=GLUBlock(features=output_dim),
        num_blocks=2,
        use_projection=True,
        activation=nn.swish
    )
    params_resnet = glu_resnet.init(key, x)
    output_resnet = glu_resnet.apply(params_resnet, x, training=True)
    print(f"GLU ResNet output shape: {output_resnet.shape}")
    print(f"GLU ResNet output range: [{jnp.min(output_resnet):.3f}, {jnp.max(output_resnet):.3f}]")
    
    print("\n✅ All GLU tests passed!")
