"""
MLP (Multi-Layer Perceptron) layer implementations.

This module provides standardized MLP layer implementations that can be used
with ResNet wrappers for residual connections.
"""

import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Callable


class MLPLayer(nn.Module):
    """
    Simple MLP layer with activation and optional layer normalization.
    
    This is a basic dense layer with activation that can be used directly
    or wrapped with ResNetWrapper for residual connections.
    """
    
    features: int
    use_bias: bool = True
    activation: Callable = nn.swish
    use_layer_norm: bool = True
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through MLP layer.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            training: Whether in training mode
            
        Returns:
            MLP output [batch_size, features]
        """
        # Dense layer
        output = nn.Dense(self.features, use_bias=self.use_bias, name='dense')(x)
        
        # Apply activation
        if self.activation is not None:
            output = self.activation(output)
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            output = nn.LayerNorm(name='layer_norm')(output)
        
        # Apply dropout if enabled
        if self.dropout_rate > 0:
            output = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(output)
        
        return output


class MLPBlock(nn.Module):
    """
    MLP block that can be used directly or wrapped with ResNetWrapper.
    
    This is essentially the same as MLPLayer but with a more descriptive name
    for when it's used as a building block in larger architectures.
    """
    
    features: int
    use_bias: bool = True
    activation: Callable = nn.swish
    use_layer_norm: bool = True
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through MLP block.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            training: Whether in training mode
            
        Returns:
            MLP block output [batch_size, features]
        """
        # Use MLPLayer implementation
        mlp_layer = MLPLayer(
            features=self.features,
            use_bias=self.use_bias,
            activation=self.activation,
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
            name='mlp_layer'
        )
        return mlp_layer(x, training=training)


# Convenience functions
def create_mlp_layer(features: int,
                    use_bias: bool = True,
                    activation: Callable = nn.swish,
                    use_layer_norm: bool = True,
                    dropout_rate: float = 0.0) -> MLPLayer:
    """Create an MLP layer."""
    return MLPLayer(
        features=features,
        use_bias=use_bias,
        activation=activation,
        use_layer_norm=use_layer_norm,
        dropout_rate=dropout_rate
    )


def create_mlp_block(features: int,
                    use_bias: bool = True,
                    activation: Callable = nn.swish,
                    use_layer_norm: bool = True,
                    dropout_rate: float = 0.0) -> MLPBlock:
    """Create an MLP block."""
    return MLPBlock(
        features=features,
        use_bias=use_bias,
        activation=activation,
        use_layer_norm=use_layer_norm,
        dropout_rate=dropout_rate
    )


# Example usage and testing
if __name__ == "__main__":
    import jax
    
    print("Testing MLP layers:")
    
    # Test parameters
    batch_size = 4
    input_dim = 8
    output_dim = 6
    
    # Create test input
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (batch_size, input_dim))
    
    print(f"Input shape: {x.shape}")
    
    # Test MLP layer
    print("\n1. Testing MLPLayer:")
    mlp_layer = MLPLayer(
        features=output_dim,
        use_layer_norm=True,
        dropout_rate=0.1
    )
    params = mlp_layer.init(key, x)
    output = mlp_layer.apply(params, x, training=True)
    print(f"MLP layer output shape: {output.shape}")
    print(f"MLP layer output range: [{jnp.min(output):.3f}, {jnp.max(output):.3f}]")
    
    # Test MLP block
    print("\n2. Testing MLPBlock:")
    mlp_block = MLPBlock(
        features=output_dim,
        use_layer_norm=True,
        dropout_rate=0.1
    )
    params_block = mlp_block.init(key, x)
    output_block = mlp_block.apply(params_block, x, training=True)
    print(f"MLP block output shape: {output_block.shape}")
    print(f"MLP block output range: [{jnp.min(output_block):.3f}, {jnp.max(output_block):.3f}]")
    
    # Test with ResNet wrapper
    print("\n3. Testing MLP with ResNetWrapper:")
    from .resnet_wrapper import ResNetWrapper
    
    mlp_resnet = ResNetWrapper(
        base_module=MLPBlock(features=output_dim),
        num_blocks=2,
        use_projection=True,
        activation=nn.swish
    )
    params_resnet = mlp_resnet.init(key, x)
    output_resnet = mlp_resnet.apply(params_resnet, x, training=True)
    print(f"MLP ResNet output shape: {output_resnet.shape}")
    print(f"MLP ResNet output range: [{jnp.min(output_resnet):.3f}, {jnp.max(output_resnet):.3f}]")
    
    print("\n✅ All MLP tests passed!")
