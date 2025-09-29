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
    
    IMPORTANT: The base module must have matching input and output dimensions
    for the residual connection to work properly.
    
    When num_blocks=1, this acts as a single ResNet block.
    When num_blocks>1, this creates multiple ResNet blocks in sequence.
    
    Example:
        # Single ResNet block
        base_module = nn.Dense(features=64)  # 64D -> 64D
        resnet = ResNetWrapper(base_module, num_blocks=1)
        
        # Multiple ResNet blocks with separate parameters
        resnet = ResNetWrapper(base_module, num_blocks=3, share_parameters=False)
        
        # Multiple ResNet blocks with shared parameters
        resnet = ResNetWrapper(base_module, num_blocks=3, share_parameters=True)
    """
    
    base_module: nn.Module  # The module to wrap with residual connections
    num_blocks: int = 3  # Number of ResNet blocks to create
    activation: Optional[Callable] = None  # Optional activation between blocks
    share_parameters: bool = False  # Whether to share parameters between blocks
    weight_residual: bool = False  # Whether to weight the residual connection
    residual_weight: float = 1.0  # Weight for residual connection
    
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
        # Assert that the base module maintains input/output dimension matching for residual connections
        # This is a fundamental requirement for ResNet blocks
        assert hasattr(self.base_module, 'features'), \
            f"ResNet wrapper requires base_module to have 'features' attribute. " \
            f"Base module type: {type(self.base_module)}"
        
        # For MLPBlock, check that the first and last features match
        if hasattr(self.base_module, 'features') and isinstance(self.base_module.features, tuple):
            first_feature = self.base_module.features[0]
            last_feature = self.base_module.features[-1]
            assert first_feature == last_feature, \
                f"ResNet wrapper requires base_module to have matching input/output dimensions. " \
                f"Base module features: {self.base_module.features} " \
                f"(first: {first_feature}, last: {last_feature}). " \
                f"ResNet blocks require input_dim == output_dim for residual connections. " \
                f"Consider using a projection layer in your base module or adjusting the architecture."
        
        current_input = x
        
        # Apply each ResNet block
        for i in range(self.num_blocks):
            # Apply the wrapped module with optional parameter sharing
            if self.share_parameters:
                # Reuse the same base_module parameters for all blocks
                output = self.base_module(current_input, *args, **kwargs)
            else:
                # Create separate parameters for each block by creating a new instance inline
                # This ensures each block gets its own parameter set
                if hasattr(self.base_module, '__class__') and hasattr(self.base_module, 'features'):
                    # Handle nn.Dense, MLPBlock and other custom modules with features attribute
                    # Create a new instance of the same class with the same parameters
                    module_class = self.base_module.__class__
                    
                    # Copy known constructor parameters from the base module
                    # This is safer than trying to copy all attributes
                    constructor_kwargs = {}
                    
                    # Always copy features (required)
                    constructor_kwargs['features'] = self.base_module.features
                    
                    # Copy optional parameters if they exist (excluding name, which we set manually)
                    optional_params = ['use_bias', 'activation', 'use_layer_norm', 'dropout_rate', 
                                     'dtype', 'dot_general', 'dot_general_cls']
                    for param in optional_params:
                        if hasattr(self.base_module, param):
                            constructor_kwargs[param] = getattr(self.base_module, param)
                    
                    # Append block index to the name to ensure unique parameter names
                    # Get the original name from the base module, not from constructor_kwargs
                    original_name = getattr(self.base_module, 'name', None) or 'module'
                    constructor_kwargs['name'] = f'{original_name}_block_{i}'
                    
                    new_module = module_class(**constructor_kwargs)
                    output = new_module(current_input, *args, **kwargs)
                else:
                    # For other module types, we need to handle them case by case
                    # For now, fall back to parameter sharing
                    output = self.base_module(current_input, *args, **kwargs)
            
            # Check that input and output dimensions match for residual connection
            if output.shape[-1] != current_input.shape[-1]:
                raise ValueError(
                    f"ResNet block {i}: Input dimension {current_input.shape[-1]} and output dimension {output.shape[-1]} must match for residual connection. "
                    f"Consider using a projection layer in your base module or adjusting the architecture."
                )
            
            # Direct residual connection (input and output dimensions must match)
            if self.weight_residual:
                current_input = output + self.residual_weight * current_input
            else:
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
    
    IMPORTANT: The base module must output the same dimension as x for the residual connection
    to work properly.
    
    When num_blocks=1, this acts as a single bivariate ResNet block.
    When num_blocks>1, this creates multiple bivariate ResNet blocks in sequence.
    
    The output shape will always match x.shape, making it residual with respect to x.
    
    Example:
        # Single bivariate ResNet block
        class BilinearModule(nn.Module):
            @nn.compact
            def __call__(self, x, y):
                return nn.Dense(x.shape[-1])(x) + nn.Dense(x.shape[-1])(y)  # Output matches x.shape
        
        resnet = ResNetWrapperBivariate(BilinearModule(), num_blocks=1)
        output = resnet.apply(params, x, y)  # output.shape == x.shape
        
        # Multiple bivariate ResNet blocks with separate parameters
        resnet = ResNetWrapperBivariate(BilinearModule(), num_blocks=3, share_parameters=False)
        output = resnet.apply(params, x, y)  # output.shape == x.shape
        
        # Multiple bivariate ResNet blocks with shared parameters
        resnet = ResNetWrapperBivariate(BilinearModule(), num_blocks=3, share_parameters=True)
        output = resnet.apply(params, x, y)  # output.shape == x.shape
    """
    
    base_module: nn.Module  # The module to wrap with residual connections
    num_blocks: int = 3  # Number of ResNet blocks to create
    activation: Optional[Callable] = None  # Optional activation between blocks
    share_parameters: bool = False  # Whether to share parameters between blocks
    weight_residual: bool = False  # Whether to weight the residual connection
    residual_weight: float = 1.0  # Weight for residual connection
    
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
        # Assert that the base module maintains input/output dimension matching for residual connections
        # This is a fundamental requirement for ResNet blocks
        assert hasattr(self.base_module, 'features'), \
            f"ResNet wrapper requires base_module to have 'features' attribute. " \
            f"Base module type: {type(self.base_module)}"
        
        # For MLPBlock, check that the first and last features match
        if hasattr(self.base_module, 'features') and isinstance(self.base_module.features, tuple):
            first_feature = self.base_module.features[0]
            last_feature = self.base_module.features[-1]
            assert first_feature == last_feature, \
                f"ResNet wrapper requires base_module to have matching input/output dimensions. " \
                f"Base module features: {self.base_module.features} " \
                f"(first: {first_feature}, last: {last_feature}). " \
                f"ResNet blocks require input_dim == output_dim for residual connections. " \
                f"Consider using a projection layer in your base module or adjusting the architecture."
        
        current_x = x
        
        # Apply each ResNet block
        for i in range(self.num_blocks):
            # Apply the wrapped module with optional parameter sharing
            if self.share_parameters:
                # Reuse the same base_module parameters for all blocks
                output = self.base_module(current_x, y, *args, **kwargs)
            else:
                # Create separate parameters for each block by creating a new instance inline
                # This ensures each block gets its own parameter set
                if hasattr(self.base_module, '__class__') and hasattr(self.base_module, 'features'):
                    # Handle nn.Dense, MLPBlock and other custom modules with features attribute
                    # Create a new instance of the same class with the same parameters
                    module_class = self.base_module.__class__
                    
                    # Copy known constructor parameters from the base module
                    # This is safer than trying to copy all attributes
                    constructor_kwargs = {}
                    
                    # Always copy features (required)
                    constructor_kwargs['features'] = self.base_module.features
                    
                    # Copy optional parameters if they exist (excluding name, which we set manually)
                    optional_params = ['use_bias', 'activation', 'use_layer_norm', 'dropout_rate', 
                                     'dtype', 'dot_general', 'dot_general_cls']
                    for param in optional_params:
                        if hasattr(self.base_module, param):
                            constructor_kwargs[param] = getattr(self.base_module, param)
                    
                    # Append block index to the name to ensure unique parameter names
                    # Get the original name from the base module, not from constructor_kwargs
                    original_name = getattr(self.base_module, 'name', None) or 'module'
                    constructor_kwargs['name'] = f'{original_name}_block_{i}'
                    
                    new_module = module_class(**constructor_kwargs)
                    output = new_module(current_x, y, *args, **kwargs)
                else:
                    # For other module types, we need to handle them case by case
                    # For now, fall back to parameter sharing
                    output = self.base_module(current_x, y, *args, **kwargs)
            
            # Check that input and output dimensions match for residual connection
            if output.shape[-1] != current_x.shape[-1]:
                raise ValueError(
                    f"ResNet block {i}: Input dimension {current_x.shape[-1]} and output dimension {output.shape[-1]} must match for residual connection. "
                    f"Consider using a projection layer in your base module or adjusting the architecture."
                )
            
            # Direct residual connection (input and output dimensions must match)
            if self.weight_residual:
                current_x = output + self.residual_weight * current_x
            else:
                current_x = output + current_x
            
            # Apply activation if specified (except for the last block)
            if self.activation is not None and i < self.num_blocks - 1:
                current_x = self.activation(current_x)
        
        return current_x


# Convenience functions for creating ResNet wrappers
def create_resnet_wrapper(base_module: nn.Module, 
                         num_blocks: int = 1,
                         activation: Optional[Callable] = None,
                         share_parameters: bool = False,
                         weight_residual: bool = False,
                         residual_weight: float = 1.0) -> ResNetWrapper:
    """
    Create a ResNet wrapper for any nn.Module.
    
    Args:
        base_module: The module to wrap with residual connections (must have matching input/output dimensions)
        num_blocks: Number of ResNet blocks to create
        activation: Optional activation function between blocks
        share_parameters: Whether to share parameters between blocks
        weight_residual: Whether to weight the residual connection (default: False)
        residual_weight: Weight for residual connection (default: 1.0)
        
    Returns:
        ResNetWrapper instance
    """
    return ResNetWrapper(
        base_module=base_module,
        num_blocks=num_blocks,
        activation=activation,
        share_parameters=share_parameters,
        weight_residual=weight_residual,
        residual_weight=residual_weight
    )


def create_resnet_wrapper_bivariate(base_module: nn.Module, 
                                   num_blocks: int = 1,
                                   activation: Optional[Callable] = None,
                                   share_parameters: bool = False,
                                   weight_residual: bool = False,
                                   residual_weight: float = 1.0) -> ResNetWrapperBivariate:
    """
    Create a ResNet wrapper for any nn.Module that takes two inputs (x, y).
    
    Args:
        base_module: The module to wrap with residual connections (must accept x, y inputs and output same dimension as x)
        num_blocks: Number of ResNet blocks to create
        activation: Optional activation function between blocks
        share_parameters: Whether to share parameters between blocks
        weight_residual: Whether to weight the residual connection (default: False)
        residual_weight: Weight for residual connection (default: 1.0)
        
    Returns:
        ResNetWrapperBivariate instance
    """
    return ResNetWrapperBivariate(
        base_module=base_module,
        num_blocks=num_blocks,
        activation=activation,
        share_parameters=share_parameters,
        weight_residual=weight_residual,
        residual_weight=residual_weight
    )
