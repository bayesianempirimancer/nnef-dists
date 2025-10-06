"""
Configuration for NoProp MLP Network

This module provides configuration classes for the NoProp MLP Network,
which is a simplified version of the NoProp Geometric Flow ET Network
that uses a regular 3-layer MLP instead of the Fisher flow field.
"""

import argparse
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .base_training_config import BaseTrainingConfig


@dataclass
class NoProp_MLP_Config:
    """Configuration for NoProp MLP Network."""
    
    # Model dimensions
    input_dim: int = 0  # Must be set by caller based on actual data
    output_dim: int = 0  # Must be set by caller based on actual data
    
    # Network architecture
    hidden_sizes: list = None
    activation: str = "swish"
    use_layer_norm: bool = False
    layer_norm_type: str = "weak_layer_norm"
    dropout_rate: float = 0.1
    
    # Embedding parameters
    time_embed_dim: int = 10
    time_embed_min_freq: float = 0.25
    time_embed_max_freq: float = 4.0
    eta_embed_dim: Optional[int] = 8  # Default eta embedding dimension
    eta_embedding_type: str = "default"  # Type of eta embedding to use
    
    # Loss configuration
    loss_type: str = "noprop"  # "noprop" or "flow_matching"
    
    # Model capabilities
    supports_dropout: bool = True
    supports_batch_norm: bool = False
    supports_layer_norm: bool = True
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.hidden_sizes is None:
            self.hidden_sizes = [32, 32, 32]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model_type': 'noprop_mlp',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'use_layer_norm': self.use_layer_norm,
            'layer_norm_type': self.layer_norm_type,
            'dropout_rate': self.dropout_rate,
            'time_embed_dim': self.time_embed_dim,
            'time_embed_min_freq': self.time_embed_min_freq,
            'time_embed_max_freq': self.time_embed_max_freq,
            'eta_embed_dim': self.eta_embed_dim,
            'eta_embedding_type': self.eta_embedding_type,
            'loss_type': self.loss_type,
            'supports_dropout': self.supports_dropout,
            'supports_batch_norm': self.supports_batch_norm,
            'supports_layer_norm': self.supports_layer_norm,
        }
    
    def get_architecture_summary(self) -> str:
        """Get a summary of the architecture."""
        hidden_str = " -> ".join(map(str, self.hidden_sizes))
        return f"MLP: {self.input_dim} -> {hidden_str} -> {self.output_dim} ({self.activation})"


def create_noprop_mlp_config(input_dim: int, 
                            output_dim: int,
                            hidden_sizes: List[int] = None,
                            activation: str = "swish",
                            use_layer_norm: bool = False,
                            layer_norm_type: str = "weak_layer_norm",
                            dropout_rate: float = 0.1,
                            time_embed_dim: int = 10,
                            time_embed_min_freq: float = 0.25,
                            time_embed_max_freq: float = 4.0,
                            eta_embed_dim: Optional[int] = 8,
                            eta_embedding_type: str = "default",
                            loss_type: str = "noprop") -> NoProp_MLP_Config:
    """
    Create a NoProp MLP configuration with specified parameters.
    
    Args:
        input_dim: Input dimension (eta dimension)
        output_dim: Output dimension (mu_T dimension)
        hidden_sizes: List of hidden layer sizes
        activation: Activation function
        use_layer_norm: Whether to use layer normalization
        layer_norm_type: Type of layer normalization
        dropout_rate: Dropout rate
        time_embed_dim: Time embedding dimension
        time_embed_min_freq: Minimum frequency for time embedding
        time_embed_max_freq: Maximum frequency for time embedding
        eta_embed_dim: Eta embedding dimension (None = no embedding)
        eta_embedding_type: Type of eta embedding to use
        loss_type: Loss type ("noprop" or "flow_matching")
        
    Returns:
        NoProp_MLP_Config instance
    """
    if hidden_sizes is None:
        hidden_sizes = [32, 32, 32]
    
    return NoProp_MLP_Config(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
        use_layer_norm=use_layer_norm,
        layer_norm_type=layer_norm_type,
        dropout_rate=dropout_rate,
        time_embed_dim=time_embed_dim,
        time_embed_min_freq=time_embed_min_freq,
        time_embed_max_freq=time_embed_max_freq,
        eta_embed_dim=eta_embed_dim,
        eta_embedding_type=eta_embedding_type,
        loss_type=loss_type
    )


def add_noprop_mlp_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add NoProp MLP specific arguments to an argument parser.
    
    Args:
        parser: Argument parser to add arguments to
        
    Returns:
        Updated argument parser
    """
    # Model architecture arguments
    parser.add_argument("--hidden-sizes", type=int, nargs="+",
                       help="Hidden layer sizes (overrides config default)")
    parser.add_argument("--activation", type=str, 
                       choices=["relu", "gelu", "swish", "tanh", "none", "linear"],
                       help="Activation function (overrides config default)")
    parser.add_argument("--use-layer-norm", action="store_true",
                       help="Use layer normalization (overrides config default)")
    parser.add_argument("--layer-norm-type", type=str,
                       choices=["none", "weak_layer_norm", "rms_norm", "group_norm", 
                               "instance_norm", "weight_norm", "spectral_norm", 
                               "adaptive_norm", "pre_norm", "post_norm"],
                       help="Type of layer normalization (overrides config default)")
    parser.add_argument("--dropout-rate", type=float,
                       help="Dropout rate (overrides config default)")
    
    # Embedding arguments
    parser.add_argument("--time-embed-dim", type=int,
                       help="Time embedding dimension (overrides config default)")
    parser.add_argument("--time-embed-min-freq", type=float,
                       help="Minimum frequency for time embedding (overrides config default)")
    parser.add_argument("--time-embed-max-freq", type=float,
                       help="Maximum frequency for time embedding (overrides config default)")
    parser.add_argument("--eta-embed-dim", type=int,
                       help="Eta embedding dimension (overrides config default, None = no embedding)")
    parser.add_argument("--eta-embedding-type", type=str,
                       choices=["default", "polynomial", "advanced", "minimal", "convex_only"],
                       help="Type of eta embedding (overrides config default)")
    
    # Loss arguments
    parser.add_argument("--loss-type", type=str,
                       choices=["noprop", "flow_matching"],
                       help="Loss type (overrides config default)")
    
    return parser


def create_noprop_mlp_config_from_args(args: argparse.Namespace, 
                                      input_dim: int, 
                                      output_dim: int) -> NoProp_MLP_Config:
    """
    Create NoProp MLP configuration from command line arguments.
    
    Starts with config defaults and only overrides values when explicitly provided
    in command line arguments.
    
    Args:
        args: Parsed command line arguments
        input_dim: Input dimension
        output_dim: Output dimension
        
    Returns:
        NoProp_MLP_Config instance
    """
    # Start with default config
    config = create_noprop_mlp_config(input_dim=input_dim, output_dim=output_dim)
    
    # Override only if arguments are explicitly provided
    model_attributes = ['hidden_sizes', 'activation', 'use_layer_norm', 
                       'layer_norm_type', 'dropout_rate', 'time_embed_dim',
                       'time_embed_min_freq', 'time_embed_max_freq', 'eta_embed_dim',
                       'eta_embedding_type', 'loss_type']
    
    for attribute in model_attributes:
        if hasattr(args, attribute) and getattr(args, attribute) is not None:
            setattr(config, attribute, getattr(args, attribute))
    
    return config
