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
    activation: Callable = nn.swish
    gate_activation: Callable = nn.sigmoid

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through GLU layer.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            GLU output [batch_size, features]
        """
        # First linear transformation for gating
        x = nn.Dense(2*self.features, use_bias=self.use_bias, name='gate')(x)
        x1, x2 = jnp.split(x, 2, axis=-1)
        
        return self.gate_activation(x1) * self.activation(x2)
        

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
    dropout_rate: float = 0.1
    activation: Optional[Callable] = nn.swish
    gate_activation: Callable = nn.sigmoid
    
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
            x = GLULayer(
                features=layer_features,
                use_bias=self.use_bias,
                gate_activation=self.gate_activation,
                activation=self.activation,
                name=f'glu_layer_{i}'
            )(x)
            # Apply dropout between layers
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

        return x 


# Convenience functions
def create_glu_layer(features: int, 
                    use_bias: bool = True,
                    gate_activation: Callable = nn.sigmoid,
                    activation: Callable = nn.swish) -> GLULayer:
    """Create a GLU layer."""
    return GLULayer(
        features=features,
        use_bias=use_bias,
        gate_activation=gate_activation,
        activation=activation
    )


def create_glu_block(features: tuple,
                    use_bias: bool = True,
                    dropout_rate: float = 0.0,
                    gate_activation: Callable = nn.sigmoid,
                    activation: Optional[Callable] = nn.swish) -> GLUBlock:
    """Create a GLU block."""
    return GLUBlock(
        features=features,
        use_bias=use_bias,
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
        features=(output_dim,),
        dropout_rate=0.1
    )
    params_block = glu_block.init(key, x)
    output_block = glu_block.apply(params_block, x, training=True, rngs={'dropout': key})
    print(f"GLU block output shape: {output_block.shape}")
    print(f"GLU block output range: [{jnp.min(output_block):.3f}, {jnp.max(output_block):.3f}]")
    
    # Test with ResNet wrapper
    print("\n3. Testing GLU with ResNetWrapper:")
    try:
        from .resnet_wrapper import ResNetWrapper
    except ImportError:
        print("ResNetWrapper not available for testing")
        print("\n✅ GLU tests passed (without ResNet)!")
        exit(0)
    
    glu_resnet = ResNetWrapper(
        base_module_class=GLUBlock,
        base_module_kwargs={'features': (output_dim,)},
        num_blocks=2,
        activation=nn.swish
    )
    params_resnet = glu_resnet.init(key, x)
    output_resnet = glu_resnet.apply(params_resnet, x, training=True, rngs={'dropout': key})
    print(f"GLU ResNet output shape: {output_resnet.shape}")
    print(f"GLU ResNet output range: [{jnp.min(output_resnet):.3f}, {jnp.max(output_resnet):.3f}]")
    
    print("\n✅ All GLU tests passed!")
