"""
Training factory for creating appropriate trainers based on model type.

This module provides a factory pattern for creating the correct trainer
for each model type, following Hugging Face conventions.
"""

from typing import Union, Type
from .base_et_trainer import BaseETTrainer
from ..models.mlp_et_net import MLP_ET_Network
from ..models.glu_et_net import GLU_ET_Network
from ..models.glow_et_net import Glow_ET_Network
from ..models.quadratic_et_net import Quadratic_ET_Network
from ..configs.mlp_et_config import MLP_ET_Config
from ..configs.glu_et_config import GLU_ET_Config
from ..configs.glow_et_config import Glow_ET_Config
from ..configs.quadratic_et_config import Quadratic_ET_Config


def create_trainer(model, config):
    """
    Create the appropriate trainer for a given model and config.
    
    Args:
        model: The model to train
        config: The model configuration
        
    Returns:
        Appropriate trainer instance
        
    Raises:
        ValueError: If model/config combination is not supported
    """
    # MLP ET
    if isinstance(model, MLP_ET_Network) and isinstance(config, MLP_ET_Config):
        return BaseETTrainer(model, config)
    
    # GLU ET
    elif isinstance(model, GLU_ET_Network) and isinstance(config, GLU_ET_Config):
        return BaseETTrainer(model, config)
    
    # GLOW ET
    elif isinstance(model, Glow_ET_Network) and isinstance(config, Glow_ET_Config):
        return BaseETTrainer(model, config)
    
    # Quadratic ET
    elif isinstance(model, Quadratic_ET_Network) and isinstance(config, Quadratic_ET_Config):
        return BaseETTrainer(model, config)
    
    else:
        raise ValueError(f"Unsupported model/config combination: {type(model)} / {type(config)}")


def create_model_and_trainer(model_type: str, config_dict: dict):
    """
    Create both model and trainer from configuration.
    
    Args:
        model_type: Type of model ('mlp_et', 'glu_et', 'glow_et', 'quadratic_et')
        config_dict: Configuration dictionary
        
    Returns:
        Tuple of (model, trainer)
    """
    if model_type == "mlp_et":
        config = MLP_ET_Config(**config_dict)
        model = MLP_ET_Network(config=config)
        trainer = BaseETTrainer(model, config)
        return model, trainer
    
    elif model_type == "glu_et":
        config = GLU_ET_Config(**config_dict)
        model = GLU_ET_Network(config=config)
        trainer = BaseETTrainer(model, config)
        return model, trainer
    
    elif model_type == "glow_et":
        config = Glow_ET_Config(**config_dict)
        model = Glow_ET_Network(config=config)
        trainer = BaseETTrainer(model, config)
        return model, trainer
    
    elif model_type == "quadratic_et":
        config = Quadratic_ET_Config(**config_dict)
        model = Quadratic_ET_Network(config=config)
        trainer = BaseETTrainer(model, config)
        return model, trainer
    
    elif model_type == "geometric_flow_et":
        from ..configs.geometric_flow_et_config import Geometric_Flow_ET_Config
        from ..models.geometric_flow_et_net import Geometric_Flow_ET_Network
        config = Geometric_Flow_ET_Config(**config_dict)
        model = Geometric_Flow_ET_Network(config=config)
        trainer = BaseETTrainer(model, config)
        return model, trainer
    
    elif model_type == "noprop_geometric_flow_et":
        from ..configs.noprop_geometric_flow_et_config import NoProp_Geometric_Flow_ET_Config
        from ..models.noprop_geometric_flow_et_net import NoProp_Geometric_Flow_ET_Network
        from .noprop_geometric_flow_et_trainer import NoPropGeometricFlowETTrainer
        config = NoProp_Geometric_Flow_ET_Config(**config_dict)
        model = NoProp_Geometric_Flow_ET_Network(config=config)
        trainer = NoPropGeometricFlowETTrainer(model, config)
        return model, trainer
    
    elif model_type == "noprop_mlp":
        from ..configs.noprop_mlp_config import NoProp_MLP_Config
        from ..models.noprop_mlp_net import NoProp_MLP_Network
        from .noprop_mlp_trainer import NoPropMLPTrainer
        # Filter out model_type from config_dict as it's not expected by NoProp_MLP_Config
        filtered_config = {k: v for k, v in config_dict.items() if k != 'model_type'}
        config = NoProp_MLP_Config(**filtered_config)
        model = NoProp_MLP_Network(config=config)
        trainer = NoPropMLPTrainer(model, config)
        return model, trainer
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# Convenience functions for common use cases
def create_mlp_et_trainer(config_dict: dict):
    """Create an MLP ET model and trainer."""
    return create_model_and_trainer("mlp_et", config_dict)

def create_glu_et_trainer(config_dict: dict):
    """Create a GLU ET model and trainer."""
    return create_model_and_trainer("glu_et", config_dict)

def create_glow_et_trainer(config_dict: dict):
    """Create a GLOW ET model and trainer."""
    return create_model_and_trainer("glow_et", config_dict)

def create_quadratic_et_trainer(config_dict: dict):
    """Create a Quadratic ET model and trainer."""
    return create_model_and_trainer("quadratic_et", config_dict)
