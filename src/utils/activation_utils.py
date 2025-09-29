"""
Utilities for activation functions.

This module provides utilities for converting activation function strings
to callable activation functions, which is commonly needed across different layers.
"""

from typing import Callable, Union
import flax.linen as nn


def get_activation_function(activation: Union[str, Callable]) -> Callable:
    """
    Convert activation string to callable activation function.
    
    Args:
        activation: Either a string name of the activation function or a callable
        
    Returns:
        Callable activation function
        
    Raises:
        ValueError: If activation string is not recognized
    """
    if callable(activation):
        return activation
    
    activation_map = {
        "linear": lambda x: x,
        "none": lambda x: x,
        "identity": lambda x: x,
        "relu": nn.relu,
        "leaky_relu": nn.leaky_relu,
        "gelu": nn.gelu,
        "swish": nn.swish,
        "sigmoid": nn.sigmoid,
        "tanh": nn.tanh,
        "softplus": nn.softplus,
        "softmax": nn.softmax,
        "log_softmax": nn.log_softmax,
        "elu": nn.elu,
        "selu": nn.selu,
    }
    
    if activation.lower() in activation_map:
        return activation_map[activation.lower()]
    else:
        raise ValueError(f"Unknown activation function: {activation}. "
                        f"Supported activations: {list(activation_map.keys())}")


def get_activation_name(activation: Union[str, Callable]) -> str:
    """
    Get the string name of an activation function.
    
    Args:
        activation: Either a string name or a callable activation function
        
    Returns:
        String name of the activation function
    """
    if isinstance(activation, str):
        return activation.lower()
    
    # Map callable functions back to their names
    callable_map = {
        nn.relu: "relu",
        nn.leaky_relu: "leaky_relu", 
        nn.gelu: "gelu",
        nn.swish: "swish",
        nn.sigmoid: "sigmoid",
        nn.tanh: "tanh",
        nn.softplus: "softplus",
        nn.softmax: "softmax",
        nn.log_softmax: "log_softmax",
        nn.elu: "elu",
        nn.selu: "selu",
    }
    
    # Check for lambda functions
    if activation == (lambda x: x):
        return "linear"
    
    return callable_map.get(activation, "unknown")
