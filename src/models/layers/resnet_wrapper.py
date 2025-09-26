"""
ResNet wrapper for arbitrary neural network modules.

This module provides a ResNet wrapper that can take any nn.Module and wrap it
with residual connections, creating a ResNet-style architecture with skip connections.
"""

import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Optional, Union


class ResNetWrapper(nn.Module):
    """
    ResNet wrapper that creates one or more ResNet blocks from any nn.Module.
    
    This wrapper takes any nn.Module and creates a ResNet-style architecture
    with the specified number of blocks, each with residual connections.
    
    When num_blocks=1, this acts as a single ResNet block.
    When num_blocks>1, this creates multiple ResNet blocks in sequence.
    
    Example:
        # Single ResNet block
        base_module = nn.Dense(features=64)
        resnet = ResNetWrapper(base_module, num_blocks=1)
        
        # Multiple ResNet blocks
        resnet = ResNetWrapper(base_module, num_blocks=3)
    """
    
    base_module: nn.Module  # The module to wrap with residual connections
    num_blocks: int = 3  # Number of ResNet blocks to create
    use_projection: bool = True  # Whether to use projection layers for dimension mismatch
    activation: Optional[Callable] = None  # Optional activation between blocks
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        """
        Forward pass through the ResNet wrapper.
        
        Args:
            x: Input tensor
            *args: Additional arguments passed to each block
            **kwargs: Additional keyword arguments passed to each block
            
        Returns:
            Output tensor after passing through all ResNet blocks
        """
        current_input = x
        
        # Apply each ResNet block
        for i in range(self.num_blocks):
            # Apply the wrapped module with a unique name to avoid parameter sharing
            output = self.base_module(current_input, *args, **kwargs, name=f'block_{i}')
            
            # Handle dimension mismatch for the residual connection
            if self.use_projection and output.shape[-1] != current_input.shape[-1]:
                # Project residual to match output dimension
                projected_residual = nn.Dense(output.shape[-1], name=f'residual_proj_{i}')(current_input)
                current_input = output + projected_residual
            else:
                # Direct residual connection (assumes same output dimension)
                current_input = output + current_input
            
            # Apply activation if specified (except for the last block)
            if self.activation is not None and i < self.num_blocks - 1:
                current_input = self.activation(current_input)
        
        return current_input


class ResNetWrapperBivariate(nn.Module):
    """
    ResNet wrapper that creates one or more ResNet blocks from any nn.Module that takes two inputs.
    
    This wrapper takes any nn.Module that accepts (x, y) inputs and creates a ResNet-style 
    architecture with the specified number of blocks, each with residual connections with respect to x.
    
    When num_blocks=1, this acts as a single bivariate ResNet block.
    When num_blocks>1, this creates multiple bivariate ResNet blocks in sequence.
    
    The output shape will always match x.shape, making it residual with respect to x.
    
    Example:
        # Single bivariate ResNet block
        class BilinearModule(nn.Module):
            @nn.compact
            def __call__(self, x, y):
                return nn.Dense(64)(x) + nn.Dense(64)(y)
        
        resnet = ResNetWrapperBivariate(BilinearModule(), num_blocks=1)
        output = resnet.apply(params, x, y)  # output.shape == x.shape
        
        # Multiple bivariate ResNet blocks
        resnet = ResNetWrapperBivariate(BilinearModule(), num_blocks=3)
        output = resnet.apply(params, x, y)  # output.shape == x.shape
    """
    
    base_module: nn.Module  # The module to wrap with residual connections
    num_blocks: int = 3  # Number of ResNet blocks to create
    use_projection: bool = True  # Whether to use projection layers for dimension mismatch
    activation: Optional[Callable] = None  # Optional activation between blocks
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        """
        Forward pass through the ResNet wrapper with two inputs.
        
        Args:
            x: First input tensor (residual connection applied to this)
            y: Second input tensor
            *args: Additional arguments passed to each block
            **kwargs: Additional keyword arguments passed to each block
            
        Returns:
            Output tensor with shape matching x.shape
        """
        current_x = x
        
        # Apply each ResNet block
        for i in range(self.num_blocks):
            # Apply the wrapped module with a unique name to avoid parameter sharing
            output = self.base_module(current_x, y, *args, **kwargs, name=f'block_{i}')
            
            # Handle dimension mismatch for the residual connection
            if self.use_projection and output.shape[-1] != current_x.shape[-1]:
                # Project residual to match output dimension
                projected_residual = nn.Dense(output.shape[-1], name=f'residual_proj_{i}')(current_x)
                current_x = output + projected_residual
            else:
                # Direct residual connection (assumes same output dimension as x)
                current_x = output + current_x
            
            # Apply activation if specified (except for the last block)
            if self.activation is not None and i < self.num_blocks - 1:
                current_x = self.activation(current_x)
        
        return current_x


# Convenience functions for creating ResNet wrappers
def create_resnet_wrapper(base_module: nn.Module, 
                         num_blocks: int = 1,
                         use_projection: bool = True,
                         activation: Optional[Callable] = None) -> ResNetWrapper:
    """
    Create a ResNet wrapper for any nn.Module.
    
    Args:
        base_module: The module to wrap with residual connections
        num_blocks: Number of ResNet blocks to create
        use_projection: Whether to use projection layers for dimension mismatch
        activation: Optional activation function between blocks
        
    Returns:
        ResNetWrapper instance
    """
    return ResNetWrapper(
        base_module=base_module,
        num_blocks=num_blocks,
        use_projection=use_projection,
        activation=activation
    )


def create_resnet_wrapper_bivariate(base_module: nn.Module, 
                                   num_blocks: int = 1,
                                   use_projection: bool = True,
                                   activation: Optional[Callable] = None) -> ResNetWrapperBivariate:
    """
    Create a ResNet wrapper for any nn.Module that takes two inputs (x, y).
    
    Args:
        base_module: The module to wrap with residual connections (must accept x, y inputs)
        num_blocks: Number of ResNet blocks to create
        use_projection: Whether to use projection layers for dimension mismatch
        activation: Optional activation function between blocks
        
    Returns:
        ResNetWrapperBivariate instance
    """
    return ResNetWrapperBivariate(
        base_module=base_module,
        num_blocks=num_blocks,
        use_projection=use_projection,
        activation=activation
    )
