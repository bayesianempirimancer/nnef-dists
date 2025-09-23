"""
Glow ET implementation with dedicated architecture.

This module provides a standalone Glow-based ET model for directly predicting expected statistics.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any, Tuple, Optional, Union, List

from ..base_model import BaseNeuralNetwork
from .ET_Net import ETTrainer
from ..config import FullConfig


class Glow_ET_Network(BaseNeuralNetwork):
    """
    Glow-based ET Network for directly predicting expected statistics.
    
    This network uses a Glow architecture with affine coupling layers
    to directly predict expected sufficient statistics E[T(X)|η] from natural parameters η.
    """
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True, 
                 num_flow_layers: int = 20, flow_hidden_size: int = 64) -> jnp.ndarray:
        """
        Forward pass through the Glow ET network with proper affine coupling layers.
        
        Args:
            eta: Natural parameters of shape (batch_size, input_dim)
            training: Whether in training mode (affects dropout)
            num_flow_layers: Number of flow layers to use
            flow_hidden_size: Hidden size for flow layers
            
        Returns:
            Predicted expected statistics of shape (batch_size, output_dim)
        """
        x = eta
        
        # Use proper Glow architecture with affine coupling layers
        # Apply multiple flow layers for better expressiveness
        for i in range(num_flow_layers):
            x = self._affine_coupling_layer(x, flow_hidden_size, i, training)
        
        # Final output layer - sufficient statistics
        x = nn.Dense(self.config.output_dim, name='et_output')(x)
        return x
    
    def compute_internal_loss(self, params: Dict, eta: jnp.ndarray, 
                            predicted_stats: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Compute internal losses (e.g., smoothness penalties, regularization).
        
        Args:
            params: Model parameters
            eta: Natural parameters
            predicted_stats: Predicted statistics
            training: Whether in training mode
            
        Returns:
            Internal loss value
        """
        return 0.0
    
    def _affine_coupling_layer(self, x: jnp.ndarray, hidden_size: int, 
                              layer_idx: int, training: bool) -> jnp.ndarray:
        """Affine coupling layer for flow-based architecture with numerical stability."""
        input_dim = x.shape[-1]
        
        # Use different random splits for each layer to improve expressiveness
        # Create a frozen random permutation for this layer
        rng = jax.random.PRNGKey(layer_idx)  # Use layer_idx as seed for reproducibility
        perm = jax.random.permutation(rng, input_dim)
        
        # Apply permutation
        x_permuted = x[..., perm]
        
        # Split the permuted input
        split_idx = input_dim // 2
        x1, x2 = x_permuted[..., :split_idx], x_permuted[..., split_idx:]
        
        # Neural network for scaling and translation with conservative initialization
        # Use smaller initial weights to prevent explosion
        net_out = nn.Dense(hidden_size, name=f'flow_net_{layer_idx}', 
                          kernel_init=nn.initializers.normal(0.01))(x1)
        net_out = nn.tanh(net_out)  # Use tanh instead of swish for bounded output
        net_out = nn.Dense(x2.shape[-1] * 2, name=f'flow_params_{layer_idx}',
                          kernel_init=nn.initializers.normal(0.01))(net_out)
        
        # Split into scale and translation
        params_dim = net_out.shape[-1]
        params_split_idx = params_dim // 2
        log_scale, translation = net_out[..., :params_split_idx], net_out[..., params_split_idx:]
        
        # Apply transformation with very conservative numerical stability
        # Use very tight clipping to prevent any overflow
        log_scale_clamped = jnp.clip(log_scale, -2.0, 2.0)  # Very conservative clipping
        scale = jnp.exp(log_scale_clamped)
        
        # Also clip the translation to prevent large shifts
        translation_clamped = jnp.clip(translation, -1.0, 1.0)
        
        x2_transformed = x2 * scale + translation_clamped
        
        # Add residual connection for stability (mix original and transformed)
        x2_final = 0.8 * x2_transformed + 0.2 * x2
        
        # Concatenate back in permuted space
        x_permuted_out = jnp.concatenate([x1, x2_final], axis=-1)
        
        # Apply inverse permutation to get back to original order
        inv_perm = jnp.argsort(perm)
        output = x_permuted_out[..., inv_perm]
        
        return output


class Glow_ET_Trainer(ETTrainer):
    """Trainer for Glow ET Network."""
    
    def __init__(self, config: FullConfig):
        # Create model with both network and model_specific configs
        model = Glow_ET_Network(config=config.network)
        super().__init__(model, config)
        
        # Store flow configuration for use in forward passes
        self.num_flow_layers = config.model_specific.num_flow_layers
        self.flow_hidden_size = config.model_specific.flow_hidden_size
    
    def loss_fn(self, params: Dict, eta: jnp.ndarray, target_mu_T: jnp.ndarray) -> jnp.ndarray:
        """
        Compute loss function with flow configuration parameters.
        
        Args:
            params: Model parameters
            eta: Natural parameters
            target_mu_T: Target expected sufficient statistics
            
        Returns:
            Loss value
        """
        # Compute network predictions with flow configuration
        predicted_mu_T = self.model.apply(
            params, eta, training=True, 
            num_flow_layers=self.num_flow_layers,
            flow_hidden_size=self.flow_hidden_size
        )
        
        # MSE loss
        mse_loss = jnp.mean((predicted_mu_T - target_mu_T) ** 2)
        total_loss = mse_loss
        
        # Add model-specific internal losses (e.g., smoothness penalties, regularization)
        internal_loss = self.model.compute_internal_loss(params, eta, predicted_mu_T, training=True)
        total_loss += internal_loss
        
        # L1 regularization on parameters (configurable, default off)
        if self.l1_reg_weight > 0.0:
            l1_reg = 0.0
            for param in jax.tree.leaves(params):
                l1_reg += jnp.sum(jnp.abs(param))
            total_loss += self.l1_reg_weight * l1_reg
        
        return total_loss
    
    def predict(self, params: Dict, eta: jnp.ndarray) -> jnp.ndarray:
        """
        Make predictions with flow configuration parameters.
        
        Args:
            params: Model parameters
            eta: Natural parameters
            
        Returns:
            Predicted expected statistics
        """
        return self.model.apply(
            params, eta, training=False,
            num_flow_layers=self.num_flow_layers,
            flow_hidden_size=self.flow_hidden_size
        )
    
    def train(self, train_data: Dict[str, jnp.ndarray], val_data: Dict[str, jnp.ndarray] = None, 
              epochs: int = 100, learning_rate: float = 1e-3, batch_size: int = 32, 
              patience: int = float('inf')) -> Tuple[Dict, Dict]:
        """
        Train the model with flow configuration parameters.
        
        Args:
            train_data: Training data dictionary with 'eta' and 'mu_T' keys
            val_data: Validation data dictionary (optional)
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            patience: Early stopping patience
            
        Returns:
            Tuple of (best_params, training_history)
        """
        import optax
        from tqdm import tqdm
        import jax.random as random
        
        # Initialize optimizer
        optimizer = optax.adam(learning_rate)
        
        # Initialize parameters with flow configuration
        self.rng, init_rng = random.split(self.rng)
        params = self.model.init(
            init_rng, train_data['eta'][:1],
            num_flow_layers=self.num_flow_layers,
            flow_hidden_size=self.flow_hidden_size
        )
        opt_state = optimizer.init(params)
        
        # Training loop
        best_params = params
        best_loss = float('inf')
        patience_counter = 0
        training_history = {'train_loss': [], 'val_loss': []}
        
        pbar = tqdm(range(epochs), desc="Training ET Network")
        for epoch in pbar:
            # Training step
            params, opt_state, train_loss = self.train_step(
                params, opt_state, train_data, optimizer
            )
            training_history['train_loss'].append(train_loss)
            
            # Validation
            if val_data is not None:
                val_loss = float(self.loss_fn(params, val_data['eta'], val_data['mu_T']))
                training_history['val_loss'].append(val_loss)
                
                # Update progress bar
                pbar.set_postfix({
                    'train_loss': f'{train_loss:.6f}',
                    'val_loss': f'{val_loss:.6f}'
                })
                
                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = params
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                        break
            else:
                # Update progress bar without validation
                pbar.set_postfix({'train_loss': f'{train_loss:.6f}'})
                
                # Update best params based on training loss
                if train_loss < best_loss:
                    best_loss = train_loss
                    best_params = params
        
        return best_params, training_history


def create_glow_et_model_and_trainer(config: FullConfig) -> Glow_ET_Trainer:
    """Factory function to create Glow ET model and trainer."""
    return Glow_ET_Trainer(config)