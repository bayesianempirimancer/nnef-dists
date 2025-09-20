"""
ET Network - A unified neural network for directly predicting expected statistics.

This module provides a flexible neural network architecture specifically designed
for directly predicting the expected values of sufficient statistics E[T(X)|η]
from natural parameters η. Unlike LogZ networks, these models directly output
the statistics without requiring gradient/Hessian computation.

Key features:
- Flexible architecture (MLP, GLU, Quadratic ResNet, Deep Flow, Invertible NN)
- Direct prediction of sufficient statistics
- Numerically stable training with regularization
- Support for various activation functions
- Layer normalization for stability
- Skip connections and residual blocks
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import random
from typing import Dict, Any, Tuple, Optional, Union, List
import optax
import time
from tqdm import tqdm

from ..base_model import BaseNeuralNetwork, BaseTrainer
from ..config import FullConfig
from ..ef import ExponentialFamily


class ETNetwork(BaseNeuralNetwork):
    """
    Unified ET Network for directly predicting expected statistics.
    
    This network can be configured to use different architectures:
    - MLP: Standard multi-layer perceptron
    - GLU: Gated linear unit architecture
    - Quadratic: Quadratic ResNet architecture
    - DeepFlow: Glow network architecture (normalizing flows with affine coupling)
    - Invertible: Invertible neural network architecture
    - NoPropCT: Non-propagating continuous-time architecture
    
    The network directly outputs expected sufficient statistics E[T(X)|η]
    without requiring gradient/Hessian computation.
    """
    
    architecture: str = "mlp"  # "mlp", "glu", "quadratic", "deepflow", "invertible", "nopropct"
    
    @nn.compact
    def __call__(self, eta: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through the ET network.
        
        Args:
            eta: Natural parameters [batch_size, eta_dim]
            training: Whether in training mode
            
        Returns:
            Expected sufficient statistics [batch_size, stats_dim]
        """
        # Apply feature engineering if enabled
        if hasattr(self.config, 'use_feature_engineering') and self.config.use_feature_engineering:
            from ..eta_features import compute_eta_features
            x = compute_eta_features(eta, method='default')
        else:
            x = eta
        
        if self.architecture == "mlp":
            x = self._mlp_forward(x, training)
        elif self.architecture == "glu":
            x = self._glu_forward(x, training)
        elif self.architecture == "quadratic":
            x = self._quadratic_forward(x, training)
        elif self.architecture == "deepflow":
            x = self._deepflow_forward(x, training)
        elif self.architecture == "invertible":
            x = self._invertible_forward(x, training)
        elif self.architecture == "nopropct":
            x = self._nopropct_forward(x, training)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        # Final output layer - sufficient statistics
        x = nn.Dense(self.config.output_dim, name='et_output')(x)
        return x
    
    def _mlp_forward(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        """MLP architecture forward pass."""
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            x = nn.Dense(hidden_size, name=f'mlp_hidden_{i}')(x)
            x = nn.swish(x)
            if getattr(self.config, 'use_layer_norm', True):
                x = nn.LayerNorm(name=f'mlp_layer_norm_{i}')(x)
            
            dropout_rate = getattr(self.config, 'dropout_rate', 0.0)
            if dropout_rate > 0:
                x = nn.Dropout(rate=dropout_rate, deterministic=not training)(x)
        
        return x
    
    def _glu_forward(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        """GLU architecture forward pass."""
        # Input projection
        x = nn.Dense(self.config.hidden_sizes[0], name='glu_input_proj')(x)
        x = nn.swish(x)
        
        # GLU blocks with residual connections
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            residual = x
            if residual.shape[-1] != hidden_size:
                residual = nn.Dense(hidden_size, name=f'glu_residual_proj_{i}')(residual)
            
            # GLU layer
            linear1 = nn.Dense(hidden_size, name=f'glu_linear1_{i}')(x)
            linear2 = nn.Dense(hidden_size, name=f'glu_linear2_{i}')(x)
            gate = nn.sigmoid(linear1)
            glu_out = gate * linear2
            
            # Residual connection
            x = residual + glu_out
            
            # Layer normalization
            if getattr(self.config, 'use_layer_norm', True):
                x = nn.LayerNorm(name=f'glu_layer_norm_{i}')(x)
            
            dropout_rate = getattr(self.config, 'dropout_rate', 0.0)
            if dropout_rate > 0:
                x = nn.Dropout(rate=dropout_rate, deterministic=not training)(x)
        
        return x
    
    def _quadratic_forward(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        """Quadratic ResNet architecture forward pass."""
        # Input projection
        if len(self.config.hidden_sizes) > 0:
            x = nn.Dense(self.config.hidden_sizes[0], name='quad_input_proj')(x)
        else:
            x = nn.Dense(64, name='quad_input_proj')(x)
        
        # Quadratic residual blocks
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            x = self._quadratic_residual_block(x, hidden_size, i, training)
        
        return x
    
    def _quadratic_residual_block(self, x: jnp.ndarray, hidden_size: int, 
                                 block_idx: int, training: bool) -> jnp.ndarray:
        """Single quadratic residual block."""
        # Store input for residual connection
        residual = x
        if residual.shape[-1] != hidden_size:
            residual = nn.Dense(hidden_size, name=f'quad_residual_proj_{block_idx}')(residual)
        
        # Linear transformation
        linear_out = nn.Dense(hidden_size, name=f'quad_linear_{block_idx}')(x)
        linear_out = nn.swish(linear_out)
        
        # Quadratic transformation with smaller initialization
        quadratic_out = nn.Dense(hidden_size, 
                                kernel_init=nn.initializers.normal(stddev=0.01),
                                name=f'quad_quadratic_{block_idx}')(x)
        quadratic_out = nn.swish(quadratic_out)
        
        # Combine: y = residual + Ax + (Bx)x (updated formula)
        output = residual + linear_out - residual * quadratic_out
        
        # Layer normalization
        if getattr(self.config, 'use_layer_norm', True):
            output = nn.LayerNorm(name=f'quad_layer_norm_{block_idx}')(output)
        
        return output
    
    def _deepflow_forward(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        """Deep flow architecture forward pass."""
        # Flow-based transformations
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            # Affine coupling layer
            x = self._affine_coupling_layer(x, hidden_size, i, training)
        
        return x
    
    def _affine_coupling_layer(self, x: jnp.ndarray, hidden_size: int, 
                              layer_idx: int, training: bool) -> jnp.ndarray:
        """Affine coupling layer for flow-based architecture."""
        # Split input
        x1, x2 = jnp.split(x, 2, axis=-1)
        
        # Neural network for scaling and translation
        net_out = nn.Dense(hidden_size, name=f'flow_net_{layer_idx}')(x1)
        net_out = nn.swish(net_out)
        net_out = nn.Dense(x2.shape[-1] * 2, name=f'flow_params_{layer_idx}')(net_out)
        
        # Split into scale and translation
        log_scale, translation = jnp.split(net_out, 2, axis=-1)
        
        # Apply transformation
        x2_transformed = x2 * jnp.exp(log_scale) + translation
        
        # Concatenate back
        output = jnp.concatenate([x1, x2_transformed], axis=-1)
        
        return output
    
    def _invertible_forward(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        """Invertible neural network forward pass."""
        # Real NVP style invertible layers
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            x = self._invertible_layer(x, hidden_size, i, training)
        
        return x
    
    def _invertible_layer(self, x: jnp.ndarray, hidden_size: int, 
                         layer_idx: int, training: bool) -> jnp.ndarray:
        """Single invertible layer."""
        # Permutation to mix dimensions
        perm = jnp.arange(x.shape[-1])
        if layer_idx % 2 == 1:
            perm = jnp.roll(perm, x.shape[-1] // 2)
        x_perm = x[..., perm]
        
        # Neural network for transformation
        net_out = nn.Dense(hidden_size, name=f'inv_net1_{layer_idx}')(x_perm)
        net_out = nn.swish(net_out)
        net_out = nn.Dense(hidden_size, name=f'inv_net2_{layer_idx}')(net_out)
        net_out = nn.swish(net_out)
        net_out = nn.Dense(x.shape[-1], name=f'inv_transform_{layer_idx}')(net_out)
        
        # Apply transformation
        output = x + net_out
        
        return output
    
    def _nopropct_forward(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        """Non-propagating continuous-time architecture forward pass."""
        # Continuous-time inspired transformations
        for i, hidden_size in enumerate(self.config.hidden_sizes):
            x = self._ct_layer(x, hidden_size, i, training)
        
        return x
    
    def _ct_layer(self, x: jnp.ndarray, hidden_size: int, 
                  layer_idx: int, training: bool) -> jnp.ndarray:
        """Continuous-time inspired layer."""
        # Time-like parameter
        t = jnp.ones((x.shape[0], 1)) * (layer_idx + 1)
        
        # Concatenate time with input
        x_with_time = jnp.concatenate([x, t], axis=-1)
        
        # Neural ODE-like transformation
        net_out = nn.Dense(hidden_size, name=f'ct_net1_{layer_idx}')(x_with_time)
        net_out = nn.swish(net_out)
        net_out = nn.Dense(hidden_size, name=f'ct_net2_{layer_idx}')(net_out)
        net_out = nn.swish(net_out)
        net_out = nn.Dense(x.shape[-1], name=f'ct_ode_{layer_idx}')(net_out)
        
        # Euler step integration
        dt = 0.1
        output = x + dt * net_out
        
        return output


class ETTrainer(BaseTrainer):
    """
    Trainer for ET Networks that directly predict expected statistics.
    
    This trainer specializes in training networks that directly output
    expected sufficient statistics E[T(X)|η] without requiring gradient
    or Hessian computation.
    """
    
    def __init__(self, config: FullConfig, architecture: str = "mlp", l1_reg_weight: float = 0.0):
        model = ETNetwork(config=config.network, architecture=architecture)
        super().__init__(model, config)
        self.architecture = architecture
        self.l1_reg_weight = l1_reg_weight
    
    def et_loss_fn(self, params: Dict, eta: jnp.ndarray, 
                   target_stats: jnp.ndarray) -> jnp.ndarray:
        """
        Compute loss for expected statistics prediction.
        
        Args:
            params: Model parameters
            eta: Natural parameters
            target_stats: Target expected sufficient statistics
            
        Returns:
            Loss value
        """
        # Compute network predictions
        predicted_stats = self.model.apply(params, eta, training=True)
        
        # MSE loss
        mse_loss = jnp.mean((predicted_stats - target_stats) ** 2)
        
        # L1 regularization on parameters (configurable, default off)
        if self.l1_reg_weight > 0.0:
            l1_reg = 0.0
            for param in jax.tree_leaves(params):
                l1_reg += jnp.sum(jnp.abs(param))
            total_loss = mse_loss + self.l1_reg_weight * l1_reg
        else:
            total_loss = mse_loss
        
        return total_loss
    
    def train_step(self, params: Dict, opt_state: Any, batch: Dict[str, jnp.ndarray], 
                   optimizer: Any) -> Tuple[Dict, Any, float]:
        """Single training step."""
        loss_value, grads = jax.value_and_grad(self.et_loss_fn)(
            params, batch['eta'], batch['stats']
        )
        
        # Gradient clipping for stability
        grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, float(loss_value)
    
    def train(self, train_data: Dict[str, jnp.ndarray], 
              val_data: Optional[Dict[str, jnp.ndarray]] = None,
              epochs: int = 300, learning_rate: float = 1e-3) -> Tuple[Dict, Dict]:
        """Train the ET network."""
        # Initialize model
        rng = random.PRNGKey(42)
        params = self.model.init(rng, train_data['eta'][:1])
        
        # Optimizer
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)
        
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        best_params = params
        best_loss = float('inf')
        
        print(f"Training ET Network ({self.architecture}) for {epochs} epochs")
        
        for epoch in range(epochs):
            # Training step
            train_loss = 0.0
            batch_size = 32
            n_train = train_data['eta'].shape[0]
            
            # Mini-batch training
            for i in range(0, n_train, batch_size):
                batch = {
                    'eta': train_data['eta'][i:i+batch_size],
                    'stats': train_data['stats'][i:i+batch_size]
                }
                
                params, opt_state, batch_loss = self.train_step(params, opt_state, batch, optimizer)
                train_loss += batch_loss
            
            train_loss /= (n_train // batch_size)
            history['train_loss'].append(train_loss)
            
            # Validation
            if val_data is not None:
                val_loss = self.et_loss_fn(params, val_data['eta'], val_data['stats'])
                history['val_loss'].append(float(val_loss))
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = params
            
            # Progress reporting
            if epoch % 50 == 0 or epoch < 10:
                if val_data is not None:
                    print(f"  Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
                else:
                    print(f"  Epoch {epoch}: Train Loss = {train_loss:.6f}")
        
        return best_params, history
    
    def predict(self, params: Dict, eta: jnp.ndarray) -> jnp.ndarray:
        """Make predictions using the trained model."""
        return self.model.apply(params, eta, training=False)
    
    def evaluate(self, params: Dict, test_data: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """Evaluate the model on test data."""
        predictions = self.predict(params, test_data['eta'])
        
        # MSE and MAE evaluation
        mse = jnp.mean((predictions - test_data['stats']) ** 2)
        mae = jnp.mean(jnp.abs(predictions - test_data['stats']))
        
        # Per-component analysis
        component_mse = jnp.mean((predictions - test_data['stats']) ** 2, axis=0)
        
        results = {
            'mse': float(mse),
            'mae': float(mae),
            'component_mse': component_mse.tolist() if hasattr(component_mse, 'tolist') else float(component_mse)
        }
        
        return results


def create_et_network_and_trainer(config: FullConfig, architecture: str = "mlp", l1_reg_weight: float = 0.0) -> ETTrainer:
    """Factory function to create ET network and trainer."""
    return ETTrainer(config, architecture=architecture, l1_reg_weight=l1_reg_weight)


# Architecture-specific factory functions
def create_mlp_et(config: FullConfig) -> ETTrainer:
    """Create MLP-based ET network."""
    return ETTrainer(config, architecture="mlp")

def create_glu_et(config: FullConfig) -> ETTrainer:
    """Create GLU-based ET network."""
    return ETTrainer(config, architecture="glu")

def create_quadratic_et(config: FullConfig) -> ETTrainer:
    """Create Quadratic ResNet-based ET network."""
    return ETTrainer(config, architecture="quadratic")

def create_deepflow_et(config: FullConfig) -> ETTrainer:
    """Create Deep Flow-based ET network."""
    return ETTrainer(config, architecture="deepflow")

def create_invertible_et(config: FullConfig) -> ETTrainer:
    """Create Invertible NN-based ET network."""
    return ETTrainer(config, architecture="invertible")

def create_nopropct_et(config: FullConfig) -> ETTrainer:
    """Create Non-Propagating CT-based ET network."""
    return ETTrainer(config, architecture="nopropct")
